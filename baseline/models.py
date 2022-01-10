import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dnc import DNC
from layers import GraphConvolution

class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, adj, dropout, device=torch.device('cpu:0')):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        adj = self.normalize(adj + np.eye(adj.shape[0]))

        self.adj = torch.FloatTensor(adj).to(device)
        self.x = torch.eye(voc_size).to(device)

        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        node_embedding = self.gcn1(self.x, self.adj)
        node_embedding = F.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2(node_embedding, self.adj)
        return node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

class GAMENet(nn.Module):
    def __init__(self, vocab_size, ehr_adj, ddi_adj, emb_dim=64, dropout=0.4, device=torch.device('cpu:0'), ddi_in_memory=True):
        super(GAMENet, self).__init__()
        K = len(vocab_size)
        self.K = K
        self.vocab_size = vocab_size
        self.device = device
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.ddi_in_memory = ddi_in_memory
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], emb_dim) for i in range(K-1)])
        self.dropout = nn.Dropout(p=dropout)

        self.encoders = nn.ModuleList([nn.GRU(emb_dim, emb_dim*2, batch_first=True) for _ in range(K-1)])

        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )

        self.ehr_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ehr_adj, dropout=dropout, device=device)
        self.ddi_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ddi_adj, dropout=dropout, device=device)
        self.inter = nn.Parameter(torch.FloatTensor(1))

        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 3, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, vocab_size[2])
        )

        self.init_weights()

    def forward(self, inputs):
        # inputs (adm, 3, codes)

        # generate medical embeddings and queries
        i1_seq = []
        i2_seq = []
        def mean_embedding(embedding):
            return embedding.mean(dim=1).unsqueeze(dim=0)  # (1,1,dim)
        for adm in inputs:
            i1 = mean_embedding(self.dropout(self.embeddings[0](torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device)))) # (1,1,dim)
            i2 = mean_embedding(self.dropout(self.embeddings[1](torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device))))
            i1_seq.append(i1)
            i2_seq.append(i2)
        i1_seq = torch.cat(i1_seq, dim=1) #(1,seq,dim)
        i2_seq = torch.cat(i2_seq, dim=1) #(1,seq,dim)

        o1, h1 = self.encoders[0](
            i1_seq
        ) # o1:(1, seq, dim*2) hi:(1,1,dim*2)
        o2, h2 = self.encoders[1](
            i2_seq
        )
        patient_representations = torch.cat([o1, o2], dim=-1).squeeze(dim=0) # (seq, dim*4)
        queries = self.query(patient_representations) # (seq, dim)

        # graph memory module
        '''I:generate current inputs'''
        query = queries[-1:] # (1,dim)

        '''G:generate graph memory bank and insert history information'''
        if self.ddi_in_memory:
            drug_memory = self.ehr_gcn() - self.ddi_gcn() * self.inter  # (size, dim)
        else:
            drug_memory = self.ehr_gcn()

        if len(inputs) > 1:
            history_keys = queries[:(queries.size(0)-1)] # (seq-1, dim)

            history_values = np.zeros((len(inputs)-1, self.vocab_size[2]))
            for idx, adm in enumerate(inputs):
                if idx == len(inputs)-1:
                    break
                history_values[idx, adm[2]] = 1
            history_values = torch.FloatTensor(history_values).to(self.device) # (seq-1, size)

        '''O:read from global memory bank and dynamic memory bank'''
        key_weights1 = F.softmax(torch.mm(query, drug_memory.t()), dim=-1)  # (1, size)
        fact1 = torch.mm(key_weights1, drug_memory)  # (1, dim)

        if len(inputs) > 1:
            visit_weight = F.softmax(torch.mm(query, history_keys.t()), dim=1) # (1, seq-1)
            weighted_values = visit_weight.mm(history_values) # (1, size)
            fact2 = torch.mm(weighted_values, drug_memory) # (1, dim)
        else:
            fact2 = fact1
        '''R:convert O and predict'''
        output = self.output(torch.cat([query, fact1, fact2], dim=-1)) # (1, dim)

        if self.training:
            neg_pred_prob = torch.sigmoid(output)
            neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)
            batch_neg = neg_pred_prob.mul(self.tensor_ddi_adj).mean()

            return output, batch_neg
        else:
            return output

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)

        self.inter.data.uniform_(-initrange, initrange)

'''
DMNC
'''
class DMNC(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, dropout=0.3, device=torch.device('cpu:0')):
        super(DMNC, self).__init__()
        K = len(vocab_size)
        self.K = K
        self.vocab_size = vocab_size
        self.device = device

        self.token_start = vocab_size[2]
        self.token_end = vocab_size[2] + 1

        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i] if i != 2 else vocab_size[2] + 2, emb_dim) for i in range(K)])
        self.dropout = nn.Dropout(p=dropout)

        self.encoders = nn.ModuleList([DNC(
            input_size=emb_dim,
            hidden_size=emb_dim,
            rnn_type='gru',
            num_layers=1,
            num_hidden_layers=1,
            nr_cells=16,
            cell_size=emb_dim,
            read_heads=1,
            batch_first=True,
            gpu_id=0,
            independent_linears=False
        ) for _ in range(K - 1)])

        self.decoder = nn.GRU(emb_dim + emb_dim * 2, emb_dim * 2,
                              batch_first=True)  # inputs: (y, r1, r2,) hidden: (hidden1, hidden2)
        self.interface_weighting = nn.Linear(emb_dim * 2, 2 * (emb_dim + 1 + 3))  # 2 read head (key, str, mode)
        self.decoder_r2o = nn.Linear(2 * emb_dim, emb_dim * 2)

        self.output = nn.Linear(emb_dim * 2, vocab_size[2] + 2)

    def forward(self, inputs, i1_state=None, i2_state=None, h_n=None, max_len=20):
        # inputs (3, code)
        i1_inputs_tensor = self.embeddings[0](
            torch.LongTensor(inputs[0]).unsqueeze(dim=0).to(self.device))  # (1, seq, codes)
        i2_inputs_tensor = self.embeddings[1](
            torch.LongTensor(inputs[1]).unsqueeze(dim=0).to(self.device))  # (1, seq, codes)

        o1, (ch1, m1, r1) = \
            self.encoders[0](i1_inputs_tensor, (None, None, None) if i1_state is None else i1_state)
        o2, (ch2, m2, r2) = \
            self.encoders[1](i2_inputs_tensor, (None, None, None) if i2_state is None else i2_state)

        # save memory state
        i1_state = (ch1, m1, r1)
        i2_state = (ch2, m2, r2)

        predict_sequence = [self.token_start] + inputs[2]
        if h_n is None:
            h_n = torch.cat([ch1[0], ch2[0]], dim=-1)

        output_logits = []
        r1 = r1.unsqueeze(dim=0)
        r2 = r2.unsqueeze(dim=0)

        if self.training:
            for item in predict_sequence:
                # teacher force predict drug
                item_tensor = self.embeddings[2](
                    torch.LongTensor([item]).unsqueeze(dim=0).to(self.device))  # (1, seq, codes)

                o3, h_n = self.decoder(torch.cat([item_tensor, r1, r2], dim=-1), h_n)
                read_keys, read_strengths, read_modes = self.decode_read_variable(h_n.squeeze(0))

                # read from i1_mem, i2_mem and i3_mem
                r1, _ = self.read_from_memory(self.encoders[0],
                                              read_keys.select(dim=1, index=0).unsqueeze(dim=1),
                                              read_strengths.select(dim=1, index=0).unsqueeze(dim=1),
                                              read_modes.select(dim=1, index=0).unsqueeze(dim=1), i1_state[1])

                r2, _ = self.read_from_memory(self.encoders[1],
                                              read_keys.select(dim=1, index=1).unsqueeze(dim=1),
                                              read_strengths.select(dim=1, index=1).unsqueeze(dim=1),
                                              read_modes.select(dim=1, index=1).unsqueeze(dim=1), i2_state[1])

                output = self.decoder_r2o(torch.cat([r1, r2], dim=-1))
                output = self.output(output + o3).squeeze(dim=0)
                output_logits.append(output)
        else:
            item_tensor = self.embeddings[2](
                torch.LongTensor([self.token_start]).unsqueeze(dim=0).to(self.device))  # (1, seq, codes)
            for idx in range(max_len):
                # predict
                # teacher force predict drug
                o3, h_n = self.decoder(torch.cat([item_tensor, r1, r2], dim=-1), h_n)
                read_keys, read_strengths, read_modes = self.decode_read_variable(h_n.squeeze(0))

                # read from i1_mem, i2_mem and i3_mem
                r1, _ = self.read_from_memory(self.encoders[0],
                                              read_keys.select(dim=1, index=0).unsqueeze(dim=1),
                                              read_strengths.select(dim=1, index=0).unsqueeze(dim=1),
                                              read_modes.select(dim=1, index=0).unsqueeze(dim=1), i1_state[1])

                r2, _ = self.read_from_memory(self.encoders[1],
                                              read_keys.select(dim=1, index=1).unsqueeze(dim=1),
                                              read_strengths.select(dim=1, index=1).unsqueeze(dim=1),
                                              read_modes.select(dim=1, index=1).unsqueeze(dim=1), i2_state[1])

                output = self.decoder_r2o(torch.cat([r1, r2], dim=-1))
                output = self.output(output + o3).squeeze(dim=0)
                output = F.softmax(output, dim=-1)
                output_logits.append(output)

                inputs_token = torch.argmax(output, dim=-1)
                inputs_token = inputs_token.item()
                item_tensor = self.embeddings[2](
                    torch.LongTensor([inputs_token]).unsqueeze(dim=0).to(self.device))  # (1, seq, codes)

        return torch.cat(output_logits, dim=0), i1_state, i2_state, h_n

    def read_from_memory(self, dnc, read_key, read_str, read_mode, m_hidden):
        read_vectors, hidden = dnc.memories[0].read(read_key, read_str, read_mode, m_hidden)
        return read_vectors, hidden

    def decode_read_variable(self, inputs):
        w = 64
        r = 2
        b = inputs.size(0)

        inputs = self.interface_weighting(inputs)
        # r read keys (b * w * r)

        read_keys = torch.tanh(inputs.narrow(1, 0, r * w).contiguous().view(b, r, w))
        # r read strengths (b * r)
        read_strengths = F.softplus(inputs.narrow(1, r * w, r).contiguous().view(b, r))
        # read modes (b * 3*r)
        read_modes = F.softmax(inputs.narrow(1, r * w + r, inputs.size(1) - r * w - r).contiguous().view(b, r, 3), -1)
        return read_keys, read_strengths, read_modes


'''
Leap
'''
class Leap(nn.Module):
    def __init__(self, voc_size, emb_dim=128, dropout=0.3, device=torch.device('cpu:0')):
        super(Leap, self).__init__()
        self.voc_size = voc_size
        self.device = device
        self.SOS_TOKEN = voc_size[2]
        self.END_TOKEN = voc_size[2]+1

        self.enc_embedding = nn.Sequential(
            nn.Embedding(voc_size[0], emb_dim, ),
            nn.Dropout(dropout)
        )
        self.dec_embedding = nn.Sequential(
            nn.Embedding(voc_size[2] + 2, emb_dim, ),
            nn.Dropout(dropout)
        )

        self.dec_gru = nn.GRU(emb_dim*2, emb_dim, batch_first=True)

        self.attn = nn.Linear(emb_dim*2, 1)

        self.output = nn.Linear(emb_dim, voc_size[2]+2)


    def forward(self, inputs, max_len=20):
        device = self.device
        # inputs (3, codes)
        inputs_tensor = torch.LongTensor(inputs[0]).to(device)
        # (len, dim)
        inputs_embedding = self.enc_embedding(inputs_tensor.unsqueeze(dim=0)).squeeze(dim=0)

        output_logits = []
        hidden_state = None
        if self.training:
            for med_code in [self.SOS_TOKEN] + inputs[2]:
                dec_inputs = torch.LongTensor([med_code]).unsqueeze(dim=0).to(device)
                dec_inputs = self.dec_embedding(dec_inputs).squeeze(dim=0) # (1,dim)

                if hidden_state is None:
                    hidden_state = dec_inputs

                hidden_state_repeat = hidden_state.repeat(inputs_embedding.size(0), 1) # (len, dim)
                combined_inputs = torch.cat([hidden_state_repeat, inputs_embedding], dim=-1) # (len, dim*2)
                attn_weight = F.softmax(self.attn(combined_inputs).t(), dim=-1) # (1, len)
                inputs_embedding = attn_weight.mm(inputs_embedding) # (1, dim)

                _, hidden_state = self.dec_gru(torch.cat([inputs_embedding, dec_inputs], dim=-1).unsqueeze(dim=0), hidden_state.unsqueeze(dim=0))
                hidden_state = hidden_state.squeeze(dim=0) # (1,dim)

                output_logits.append(self.output(F.relu(hidden_state)))

            return torch.cat(output_logits, dim=0)

        else:
            for di in range(max_len):
                if di == 0:
                    dec_inputs = torch.LongTensor([[self.SOS_TOKEN]]).to(device)
                dec_inputs = self.dec_embedding(dec_inputs).squeeze(dim=0) # (1,dim)
                if hidden_state is None:
                    hidden_state = dec_inputs
                hidden_state_repeat = hidden_state.repeat(inputs_embedding.size(0), 1)  # (len, dim)
                combined_inputs = torch.cat([hidden_state_repeat, inputs_embedding], dim=-1)  # (len, dim*2)
                attn_weight = F.softmax(self.attn(combined_inputs).t(), dim=-1)  # (1, len)
                inputs_embedding = attn_weight.mm(inputs_embedding)  # (1, dim)
                _, hidden_state = self.dec_gru(torch.cat([inputs_embedding, dec_inputs], dim=-1).unsqueeze(dim=0),
                                               hidden_state.unsqueeze(dim=0))
                hidden_state = hidden_state.squeeze(dim=0)  # (1,dim)
                output = self.output(F.relu(hidden_state))
                topv, topi = output.data.topk(1)
                output_logits.append(F.softmax(output, dim=-1))
                dec_inputs = topi.detach()
            return torch.cat(output_logits, dim=0)

'''
Retain
'''
class Retain(nn.Module):
    def __init__(self, voc_size, emb_dim=64, dropout=0.3, device=torch.device('cpu:0')):
        super(Retain, self).__init__()
        self.device = device
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.inputs_len = voc_size[0] + voc_size[1] + voc_size[2]
        self.output_len = voc_size[2]

        self.embedding = nn.Sequential(
            nn.Embedding(self.inputs_len + 1, self.emb_dim, padding_idx=self.inputs_len),
            nn.Dropout(dropout)
        )

        self.alpha_gru = nn.GRU(emb_dim, emb_dim, batch_first=True)
        self.beta_gru = nn.GRU(emb_dim, emb_dim, batch_first=True)

        self.alpha_li = nn.Linear(emb_dim, 1)
        self.beta_li = nn.Linear(emb_dim, emb_dim)

        self.output = nn.Linear(emb_dim, self.output_len)

    def forward(self, inputs):
        device = self.device
        # inputs: (visit, 3, codes )
        max_len = max([(len(v[0]) + len(v[1]) + len(v[2])) for v in inputs])
        inputs_np = []
        for visit in inputs:
            inputs_tmp = []
            inputs_tmp.extend(visit[0])
            inputs_tmp.extend(list(np.array(visit[1]) + self.voc_size[0]))
            inputs_tmp.extend(list(np.array(visit[2]) + self.voc_size[0] + self.voc_size[1]))
            if len(inputs_tmp) < max_len:
                inputs_tmp.extend( [self.inputs_len]*(max_len - len(inputs_tmp)) )

            inputs_np.append(inputs_tmp)

        visit_emb = self.embedding(torch.LongTensor(inputs_np).to(device)) # (visit, max_len, emb)
        visit_emb = torch.sum(visit_emb, dim=1) # (visit, emb)

        g, _ = self.alpha_gru(visit_emb.unsqueeze(dim=0)) # g: (1, visit, emb)
        h, _ = self.beta_gru(visit_emb.unsqueeze(dim=0)) # h: (1, visit, emb)

        g = g.squeeze(dim=0) # (visit, emb)
        h = h.squeeze(dim=0) # (visit, emb)
        attn_g = F.softmax(self.alpha_li(g), dim=-1) # (visit, 1)
        attn_h = torch.tanh(self.beta_li(h)) # (visit, emb)

        c = attn_g * attn_h * visit_emb # (visit, emb)
        c = torch.sum(c, dim=0).unsqueeze(dim=0) # (1, emb)

        return self.output(c)

'''
MLP
'''
class MLP(nn.Module):

    def __init__(self, vocab_size, emb_dim=64, seq_len=32, hidden_size=1024, num_layers=3, dropout=0.5, history=False, 
                 device=torch.device('cpu:0')):
        super().__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.seq_len = seq_len
        self.output_size = vocab_size[2]
        self.history = history
        if history:
            self.PAD_TOKEN = self.vocab_size[0] + self.vocab_size[1] + self.vocab_size[2] + 6
        else:
            self.PAD_TOKEN = self.vocab_size[0] + self.vocab_size[1] + 3

        # add <cls> <diag> <proc> <pad> (<adm> <cur> <med> if self.history)
        self.dropout = nn.Dropout(p=dropout)
        if history:
            self.embedding = nn.Embedding(vocab_size[0] + vocab_size[1] + vocab_size[2] + 7, emb_dim) 
        else:
            self.embedding = nn.Embedding(vocab_size[0] + vocab_size[1] + 4, emb_dim) 
        self.linear = nn.Sequential(
                        nn.Linear(emb_dim * seq_len, hidden_size),
                        nn.ReLU()
                    )
        self.layers = nn.Sequential(*[nn.Sequential(nn.Linear(hidden_size, hidden_size), 
                                                    nn.ReLU()) 
                                      for _ in range(num_layers - 1)])
        self.output = nn.Linear(hidden_size, self.output_size)

    def convert_to_embedding(self, inputs):
        if len(inputs) >= self.seq_len:
            inputs = inputs[:self.seq_len]
        else:
            inputs = inputs + [self.PAD_TOKEN for _ in range(self.seq_len - len(inputs))]
        embedding = self.embedding(torch.LongTensor(inputs).to(self.device))
        embedding = F.dropout(embedding, p=0.1)
        return embedding


    def forward(self, inputs):
        device = self.device

        embedding = self.convert_to_embedding(inputs)
        hidden_state = self.dropout(self.linear(embedding.reshape(-1)))
        hidden_state = self.dropout(self.layers(hidden_state))
        output = self.output(hidden_state)
        
        return output.unsqueeze(dim=0)

class DualMLP(nn.Module):

    def __init__(self, vocab_size, emb_dim=64, seq_len=16, hidden_size=512, num_layers=3, dropout=0.5, history=False, 
                 device=torch.device('cpu:0')):
        super().__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.seq_len = seq_len
        self.output_size = vocab_size[2]
        self.history = history
        if history:
            self.PAD_TOKEN = self.vocab_size[0] + self.vocab_size[1] + 2
        else:
            self.PAD_TOKEN = self.vocab_size[0] + self.vocab_size[1]

        # add <pad>
        self.dropout = nn.Dropout(p=dropout)
        if history:
            self.embedding = nn.Embedding(vocab_size[0] + vocab_size[1] + 3, emb_dim) 
        else:
            self.embedding = nn.Embedding(vocab_size[0] + vocab_size[1] + 1, emb_dim) 
        self.linear_diag = nn.Sequential(
                               nn.Linear(emb_dim * seq_len, hidden_size),
                               nn.ReLU()
                           )
        self.linear_proc = nn.Sequential(
                               nn.Linear(emb_dim * seq_len, hidden_size),
                               nn.ReLU()
                           )
        self.layers_diag = nn.Sequential(*[nn.Sequential(nn.Linear(hidden_size, hidden_size), 
                                                         nn.ReLU()) 
                                           for _ in range(num_layers - 1)])
        self.layers_proc = nn.Sequential(*[nn.Sequential(nn.Linear(hidden_size, hidden_size), 
                                                         nn.ReLU()) 
                                           for _ in range(num_layers - 1)])
        self.output = nn.Linear(hidden_size * 2, self.output_size)

    def convert_to_embedding(self, inputs):
        if len(inputs) >= self.seq_len:
            inputs = inputs[:self.seq_len]
        else:
            inputs = inputs + [self.PAD_TOKEN for _ in range(self.seq_len - len(inputs))]
        embedding = self.embedding(torch.LongTensor(inputs).to(self.device))
        embedding = F.dropout(embedding, p=0.1)
        return embedding


    def forward(self, inputs):
        device = self.device

        diags, procs = inputs
        diag_embedding = self.convert_to_embedding(diags)
        proc_embedding = self.convert_to_embedding(procs)

        diag_hidden_state = self.dropout(self.linear_diag(diag_embedding.reshape(-1)))
        diag_hidden_state = self.dropout(self.layers_diag(diag_hidden_state))
        proc_hidden_state = self.dropout(self.linear_proc(proc_embedding.reshape(-1)))
        proc_hidden_state = self.dropout(self.layers_proc(proc_hidden_state))

        output = self.output(torch.cat((diag_hidden_state, proc_hidden_state)))
        
        return output.unsqueeze(dim=0)


# Original THUMT version
class PositionalEmbedding(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        if inputs.dim() != 3:
            raise ValueError("The rank of input must be 3.")

        length = inputs.shape[1]
        channels = inputs.shape[2]
        half_dim = channels // 2

        positions = torch.arange(length, dtype=inputs.dtype,
                                 device=inputs.device)
        dimensions = torch.arange(half_dim, dtype=inputs.dtype,
                                  device=inputs.device)

        scale = math.log(10000.0) / float(half_dim - 1)
        dimensions.mul_(-scale).exp_()

        scaled_time = positions.unsqueeze(1) * dimensions.unsqueeze(0)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)],
                           dim=1)

        if channels % 2 == 1:
            pad = torch.zeros([signal.shape[0], 1], dtype=inputs.dtype,
                              device=inputs.device)
            signal = torch.cat([signal, pad], axis=1)

        return inputs + torch.reshape(signal, [1, -1, channels]).to(inputs)


class MultiHeadAttention(nn.Module):

    def __init__(self, hidden_size, head_size, num_heads, dropout=0.0):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_size = head_size
        self.num_heads = num_heads

        attention_hidden_size = num_heads * head_size
        self.attention_hidden_size = attention_hidden_size

        self.dropout = nn.Dropout(p=dropout)
        self.q_transform = nn.Linear(hidden_size, attention_hidden_size)
        self.k_transform = nn.Linear(hidden_size, attention_hidden_size)
        self.v_transform = nn.Linear(hidden_size, attention_hidden_size)
        self.o_transform = nn.Linear(attention_hidden_size, hidden_size)

        self.reset_parameters()

    @staticmethod
    def split_heads(x, heads):
        batch = x.shape[0]
        length = x.shape[1]
        channels = x.shape[2]

        y = torch.reshape(x, [batch, length, heads, channels // heads])
        return torch.transpose(y, 2, 1)

    @staticmethod
    def combine_heads(x):
        batch = x.shape[0]
        heads = x.shape[1]
        length = x.shape[2]
        channels = x.shape[3]

        y = torch.transpose(x, 2, 1)

        return torch.reshape(y, [batch, length, heads * channels])

    def forward(self, x):
        q = self.q_transform(x)
        k = self.k_transform(x)
        v = self.v_transform(x)

        qh = self.split_heads(q, self.num_heads)
        kh = self.split_heads(k, self.num_heads)
        vh = self.split_heads(v, self.num_heads)

        qh = qh * (self.head_size ** -0.5)
        
        kh = kh.transpose(-2, -1)
        logits = torch.matmul(qh, kh)

        weights = self.dropout(torch.softmax(logits, dim=-1))
        
        y = torch.matmul(weights, vh)

        outputs = self.o_transform(self.combine_heads(y))

        return outputs

    def reset_parameters(self, initializer="uniform_scaling", **kwargs):
        if initializer == "uniform_scaling":
            # 6 / (4 * hidden_size) -> 6 / (2 * hidden_size)
            nn.init.xavier_uniform_(self.q_transform.weight, 2 ** -0.5)
            nn.init.xavier_uniform_(self.k_transform.weight, 2 ** -0.5)
            nn.init.xavier_uniform_(self.v_transform.weight, 2 ** -0.5)
            nn.init.xavier_uniform_(self.o_transform.weight)
            nn.init.constant_(self.q_transform.bias, 0.0)
            nn.init.constant_(self.k_transform.bias, 0.0)
            nn.init.constant_(self.v_transform.bias, 0.0)
            nn.init.constant_(self.o_transform.bias, 0.0)
        else:
            raise ValueError("Unknown initializer %d" % initializer)

class FeedForward(nn.Module):

    def __init__(self, input_size, hidden_size, output_size=None, dropout=0.0):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size or input_size

        self.dropout = nn.Dropout(p=dropout)
        self.input_transform = nn.Linear(input_size, hidden_size)
        self.output_transform = nn.Linear(hidden_size, self.output_size)

        self.reset_parameters()

    def forward(self, x):
        h = F.relu(self.input_transform(x))
        h = self.dropout(h)
        output = self.output_transform(h)
        return output

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_transform.weight)
        nn.init.xavier_uniform_(self.output_transform.weight)
        nn.init.constant_(self.input_transform.bias, 0.0)
        nn.init.constant_(self.output_transform.bias, 0.0)


class AttentionSubLayer(nn.Module):

    def __init__(self, hidden_size, head_size, num_heads, dropout):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.attention = MultiHeadAttention(hidden_size=hidden_size,
                                            head_size=head_size,
                                            num_heads=num_heads,
                                            dropout=dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        y = self.attention(x)
        y = self.dropout(y)
        return self.layer_norm(x + y)


class FFNSubLayer(nn.Module):

    def __init__(self, hidden_size, filter_size, dropout):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.ffn_layer = FeedForward(input_size=hidden_size,
                                     hidden_size=filter_size,
                                     dropout=dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        y = self.ffn_layer(x)
        y = self.dropout(y)
        return self.layer_norm(x + y)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, hidden_size, head_size, num_heads, filter_size, dropout):
        super().__init__()
        self.self_attention = AttentionSubLayer(hidden_size, head_size, num_heads, dropout)
        self.feed_forward = FFNSubLayer(hidden_size, filter_size, dropout)

    def forward(self, x):
        x = self.self_attention(x)
        x = self.feed_forward(x)
        return x


class Transformer(nn.Module):

    def __init__(self, vocab_size, hidden_size=256, head_size=32, num_heads=8, filter_size=1024, num_layers=3, 
                 dropout=0.1, history=False, device=torch.device('cpu:0')):
        super().__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.output_size = vocab_size[2]
        self.history = history

        # add <cls> <diag> <proc> (<adm> <cur> <med> if self.history)
        if history:
            self.embedding = nn.Embedding(vocab_size[0] + vocab_size[1] + vocab_size[2] + 6, hidden_size) 
        else:
            self.embedding = nn.Embedding(vocab_size[0] + vocab_size[1] + 3, hidden_size) 

        self.encoding = PositionalEmbedding()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_size, head_size, num_heads, filter_size, dropout)
            for i in range(num_layers)])

        self.output = nn.Linear(hidden_size, self.output_size)

    def convert_to_embedding(self, inputs):
        embedding = self.embedding(torch.LongTensor(inputs).to(self.device))
        embedding = F.dropout(embedding, p=0.1)
        return embedding
    
    def forward(self, inputs):
        x = self.convert_to_embedding(inputs).unsqueeze(0)
        x = self.encoding(x)

        for layer in self.layers:
            x = layer(x)

        cls_output = x.squeeze(0).select(dim=0, index=0)
        output = self.output(cls_output)

        return output.unsqueeze(dim=0)


class DualTransformer(nn.Module):

    def __init__(self, vocab_size, hidden_size=128, head_size=32, num_heads=8, filter_size=512, num_layers=3, 
                 dropout=0.1, history=False, device=torch.device('cpu:0')):
        super().__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.output_size = vocab_size[2]

        if history:
            self.embedding = nn.Embedding(vocab_size[0] + vocab_size[1] + 2, hidden_size) 
        else:
            self.embedding = nn.Embedding(vocab_size[0] + vocab_size[1], hidden_size) 

        self.encoding = PositionalEmbedding()
        self.layers_diag = nn.ModuleList([
            TransformerEncoderLayer(hidden_size, head_size, num_heads, filter_size, dropout)
            for i in range(num_layers)])
        self.layers_proc = nn.ModuleList([
            TransformerEncoderLayer(hidden_size, head_size, num_heads, filter_size, dropout)
            for i in range(num_layers)])

        self.output = nn.Linear(hidden_size * 2, self.output_size)

    def convert_to_embedding(self, inputs):
        embedding = self.embedding(torch.LongTensor(inputs).to(self.device))
        embedding = F.dropout(embedding, p=0.1)
        return embedding
    
    def forward(self, inputs):
        diags, procs = inputs
        diag_x = self.convert_to_embedding(diags).unsqueeze(0)
        proc_x = self.convert_to_embedding(procs).unsqueeze(0)
        diag_x = self.encoding(diag_x)
        proc_x = self.encoding(proc_x)

        for layer in self.layers_diag:
            diag_x = layer(diag_x)
        for layer in self.layers_proc:
            proc_x = layer(proc_x)

        diag_mean_output = diag_x.mean(dim=1)
        proc_mean_output = proc_x.mean(dim=1)
        mean_output = torch.cat((diag_mean_output, proc_mean_output), dim=1)
        output = self.output(mean_output)

        return output


class MaskLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, mask):
        weight = torch.mul(self.weight, mask)
        output = torch.mm(input, weight)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class MolecularGraphNeuralNetwork(nn.Module):
    def __init__(self, N_fingerprint, dim, layer_hidden, device):
        super(MolecularGraphNeuralNetwork, self).__init__()
        self.device = device
        self.embed_fingerprint = nn.Embedding(N_fingerprint, dim).to(self.device)
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim).to(self.device)
                                            for _ in range(layer_hidden)])
        self.layer_hidden = layer_hidden

    def pad(self, matrices, pad_value):
        """Pad the list of matrices
        with a pad_value (e.g., 0) for batch proc essing.
        For example, given a list of matrices [A, B, C],
        we obtain a new matrix [A00, 0B0, 00C],
        where 0 is the zero (i.e., pad value) matrix.
        """
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = torch.FloatTensor(np.zeros((M, N))).to(self.device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i:i+m, j:j+n] = matrix
            i += m
            j += n
        return pad_matrices

    def update(self, matrix, vectors, layer):
        hidden_vectors = torch.relu(self.W_fingerprint[layer](vectors))
        return hidden_vectors + torch.mm(matrix, hidden_vectors)

    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)

    def mean(self, vectors, axis):
        mean_vectors = [torch.mean(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(mean_vectors)

    def forward(self, inputs):

        """Cat or pad each input data for batch processing."""
        fingerprints, adjacencies, molecular_sizes = inputs
        fingerprints = torch.cat(fingerprints)
        adjacencies = self.pad(adjacencies, 0)

        """MPNN layer (update the fingerprint vectors)."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        for l in range(self.layer_hidden):
            hs = self.update(adjacencies, fingerprint_vectors, l)
            # fingerprint_vectors = F.normalize(hs, 2, 1)  # normalize.
            fingerprint_vectors = hs

        """Molecular vector by sum or mean of the fingerprint vectors."""
        molecular_vectors = self.sum(fingerprint_vectors, molecular_sizes)
        # molecular_vectors = self.mean(fingerprint_vectors, molecular_sizes)

        return molecular_vectors



class SafeDrugModel(nn.Module):

    def __init__(self, vocab_size, ddi_adj, ddi_mask_H, MPNNSet, N_fingerprints, average_projection, emb_dim=256, device=torch.device('cpu:0')):
        super(SafeDrugModel, self).__init__()

        self.device = device

        # pre-embedding
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], emb_dim) for i in range(2)])
        self.dropout = nn.Dropout(p=0.5)
        self.encoders = nn.ModuleList([nn.GRU(emb_dim, emb_dim, batch_first=True) for _ in range(2)])
        self.query = nn.Sequential(
                nn.ReLU(),
                nn.Linear(2 * emb_dim, emb_dim)
        )

        # bipartite local embedding
        self.bipartite_transform = nn.Sequential(
            nn.Linear(emb_dim, ddi_mask_H.shape[1])
        )
        self.bipartite_output = MaskLinear(ddi_mask_H.shape[1], vocab_size[2], False)
        
        # MPNN global embedding
        self.MPNN_molecule_Set = list(zip(*MPNNSet))

        self.MPNN_emb = MolecularGraphNeuralNetwork(N_fingerprints, emb_dim, layer_hidden=2, device=device).forward(self.MPNN_molecule_Set)
        self.MPNN_emb = torch.mm(average_projection.to(device=self.device), self.MPNN_emb.to(device=self.device))
        self.MPNN_emb.to(device=self.device)
        # self.MPNN_emb = torch.tensor(self.MPNN_emb, requires_grad=True)
        self.MPNN_output = nn.Linear(vocab_size[2], vocab_size[2])
        self.MPNN_layernorm = nn.LayerNorm(vocab_size[2])
        
        # graphs, bipartite matrix
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.tensor_ddi_mask_H = torch.FloatTensor(ddi_mask_H).to(device)
        self.init_weights()

    def forward(self, input):

	    # patient health representation
        i1_seq = []
        i2_seq = []
        def sum_embedding(embedding):
            return embedding.sum(dim=1).unsqueeze(dim=0)  # (1,1,dim)
        for adm in input:
            i1 = sum_embedding(self.dropout(self.embeddings[0](torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device)))) # (1,1,dim)
            i2 = sum_embedding(self.dropout(self.embeddings[1](torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device))))
            i1_seq.append(i1)
            i2_seq.append(i2)
        i1_seq = torch.cat(i1_seq, dim=1) #(1,seq,dim)
        i2_seq = torch.cat(i2_seq, dim=1) #(1,seq,dim)

        o1, h1 = self.encoders[0](
            i1_seq
        )
        o2, h2 = self.encoders[1](
            i2_seq
        )
        patient_representations = torch.cat([o1, o2], dim=-1).squeeze(dim=0) # (seq, dim*2)
        query = self.query(patient_representations)[-1:, :] # (seq, dim)
        
	    # MPNN embedding
        MPNN_match = torch.sigmoid(torch.mm(query, self.MPNN_emb.t()))
        MPNN_att = self.MPNN_layernorm(MPNN_match + self.MPNN_output(MPNN_match))
        
	    # local embedding
        bipartite_emb = self.bipartite_output(torch.sigmoid(self.bipartite_transform(query)), self.tensor_ddi_mask_H.t())
        
        result = torch.mul(bipartite_emb, MPNN_att)
        
        neg_pred_prob = torch.sigmoid(result)
        neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)

        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()

        return result, batch_neg

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)




