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
            visit_weight = F.softmax(torch.mm(query, history_keys.t())) # (1, seq-1)
            weighted_values = visit_weight.mm(history_values) # (1, size)
            fact2 = torch.mm(weighted_values, drug_memory) # (1, dim)
        else:
            fact2 = fact1
        '''R:convert O and predict'''
        output = self.output(torch.cat([query, fact1, fact2], dim=-1)) # (1, dim)

        if self.training:
            neg_pred_prob = F.sigmoid(output)
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
            inputs_size=emb_dim,
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
                                              read_keys[:, 0, :].unsqueeze(dim=1),
                                              read_strengths[:, 0].unsqueeze(dim=1),
                                              read_modes[:, 0, :].unsqueeze(dim=1), i1_state[1])

                r2, _ = self.read_from_memory(self.encoders[1],
                                              read_keys[:, 1, :].unsqueeze(dim=1),
                                              read_strengths[:, 1].unsqueeze(dim=1),
                                              read_modes[:, 1, :].unsqueeze(dim=1), i2_state[1])

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
                                              read_keys[:, 0, :].unsqueeze(dim=1),
                                              read_strengths[:, 0].unsqueeze(dim=1),
                                              read_modes[:, 0, :].unsqueeze(dim=1), i1_state[1])

                r2, _ = self.read_from_memory(self.encoders[1],
                                              read_keys[:, 1, :].unsqueeze(dim=1),
                                              read_strengths[:, 1].unsqueeze(dim=1),
                                              read_modes[:, 1, :].unsqueeze(dim=1), i2_state[1])

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
        read_keys = F.tanh(inputs[:, :r * w].contiguous().view(b, r, w))
        # r read strengths (b * r)
        read_strengths = F.softplus(inputs[:, r * w:r * w + r].contiguous().view(b, r))
        # read modes (b * 3*r)
        read_modes = F.softmax(inputs[:, (r * w + r):].contiguous().view(b, r, 3), -1)
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
    def __init__(self, voc_size, emb_dim=64, device=torch.device('cpu:0')):
        super(Retain, self).__init__()
        self.device = device
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.inputs_len = voc_size[0] + voc_size[1] + voc_size[2]
        self.output_len = voc_size[2]

        self.embedding = nn.Sequential(
            nn.Embedding(self.inputs_len + 1, self.emb_dim, padding_idx=self.inputs_len),
            nn.Dropout(0.3)
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
        attn_h = F.tanh(self.beta_li(h)) # (visit, emb)

        c = attn_g * attn_h * visit_emb # (visit, emb)
        c = torch.sum(c, dim=0).unsqueeze(dim=0) # (1, emb)

        return self.output(c)

'''
MLP
'''
class MLP(nn.Module):

    def __init__(self, vocab_size, emb_dim=64, seq_len=32, hidden_size=1024, num_layers=3, dropout=0.5, 
                 device=torch.device('cpu:0')):
        super().__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.seq_len = seq_len
        self.output_size = vocab_size[2]
        self.PAD_TOKEN = self.vocab_size[0] + self.vocab_size[1] + 3

        # add <cls> <diag> <proc> <pad>
        self.dropout = nn.Dropout(p=dropout)
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

    def __init__(self, vocab_size, emb_dim=64, seq_len=16, hidden_size=512, num_layers=3, dropout=0.5, 
                 device=torch.device('cpu:0')):
        super().__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.seq_len = seq_len
        self.output_size = vocab_size[2]
        self.PAD_TOKEN = self.vocab_size[0] + self.vocab_size[1]

        # add <pad>
        self.dropout = nn.Dropout(p=dropout)
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

    def __init__(self, vocab_size, hidden_size=256, head_size=32, num_heads=8, filter_size=1024, num_layers=6, 
                 dropout=0.5, device=torch.device('cpu:0')):
        super().__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.output_size = vocab_size[2]

        # add <cls> <diag> <proc>
        self.embedding = nn.Embedding(vocab_size[0] + vocab_size[1] + 3, hidden_size) 
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
        for layer in self.layers:
            x = layer(x)

        cls_output = x.squeeze(0)[0, :]
        output = self.output(cls_output)

        return output.unsqueeze(dim=0)



