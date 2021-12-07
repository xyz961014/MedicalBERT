import os
import sys
import dill
import random
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score, roc_auc_score, precision_score, f1_score, average_precision_score



from data.build_vocab_and_records import build_records, build_records_for_pretrain_vocab


class MedicalPretrainingDataset(Dataset):

    def __init__(self, input_file):
        self.input_file = input_file
        f = h5py.File(input_file, "r")
        keys = ['input_ids', 'input_mask', 'segment_ids', 'masked_lm_positions', 'masked_lm_ids',
                'seq_level_labels']
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs[0])

    def __getitem__(self, index):

        [input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, seq_level_labels] = [
            torch.from_numpy(inputs[index].astype(np.int64)) if indice < 5 else torch.from_numpy(
                np.asarray(inputs[index].astype(np.int64))) for indice, inputs in enumerate(self.inputs)]

        masked_lm_labels = torch.ones(input_ids.shape, dtype=torch.long) * -1
        # store number of masked tokens in index
        #padded_mask_indices = masked_lm_positions.eq(0).nonzero()
        padded_mask_indices = torch.nonzero(masked_lm_positions.eq(0))
        if len(padded_mask_indices) > 0:
            index = padded_mask_indices[0].item()
        else:
            index = len(masked_lm_positions)
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]

        return [input_ids, segment_ids, input_mask,
                masked_lm_labels, seq_level_labels]


class MedicalRecommendationDataset(object):
    def __init__(self, data_dir, data_file, vocab, train_set_ratio=2/3):

        self.data_dir = data_dir
        self.data_file = data_file
        self.data = dill.load(open(os.path.join(data_dir, data_file), "rb"))
        self.vocab = vocab

        if type(self.vocab) == dict:
            # baseline vocab, get vocab and vocab_size
            self.diag_vocab = self.vocab['diag_vocab']
            self.proc_vocab = self.vocab['proc_vocab']
            self.med_vocab = self.vocab['med_vocab']
            self.diag_vocab_size = len(self.diag_vocab.idx2word)
            self.proc_vocab_size = len(self.proc_vocab.idx2word)
            self.med_vocab_size = len(self.med_vocab.idx2word)
            self.vocab_size = (self.diag_vocab_size, self.proc_vocab_size, self.med_vocab_size)
        else:
            # pretrain vocab
            # add MED-ATC token to vocab
            for idx, row in self.data.iterrows():
                for atc in row["NDC"]:
                    if not vocab.normalize_word(atc, "MED-ATC") in vocab.word2idx.keys():
                        vocab.add_word(atc, "MED-ATC")
            vocab.build_intra_type_index("MED-ATC")
            self.vocab_size = vocab.vocab_size
            self.med_vocab_size = vocab.get_type_vocab_size("MED-ATC")

        # split data
        split_point = int(len(self.data) * train_set_ratio)
        self.data_train = self.data[:split_point]
        eval_len = int(len(self.data[split_point:]) / 2)
        self.data_test = self.data[split_point:split_point + eval_len]
        self.data_eval = self.data[split_point+eval_len:]

    def get_dataloader(self, model_name, shuffle=False, history=False):
        print("-" * 10 + " Getting train_loader " + '-' * 10)
        train_loader = MedicalRecommendationDataloader(self.data_train, model_name, self.vocab, 
                                                       shuffle=shuffle, history=history)
        print("-" * 10 + " Getting eval_loader  " + '-' * 10)
        eval_loader = MedicalRecommendationDataloader(self.data_eval, model_name, self.vocab, 
                                                      evaluate=True, history=history)
        print("-" * 10 + " Getting test_loader  " + '-' * 10)
        test_loader = MedicalRecommendationDataloader(self.data_test, model_name, self.vocab, 
                                                      evaluate=True, history=history)
        return train_loader, eval_loader, test_loader

    def get_train_eval_loader(self, model_name, shuffle=False, history=False):
        print("-" * 10 + " Getting eval_loader on training set  " + '-' * 10)
        return MedicalRecommendationDataloader(self.data_train, model_name, self.vocab, 
                                               evaluate=True, history=history)


    def get_extra_data(self, model_name):
        if model_name == "GAMENet":
            if "_" in self.data_file:
                data_prefix = "{}_".format(self.data_file.split("_")[0])
            else:
                data_prefix = ""
            ehr_adj = dill.load(open(os.path.join(self.data_dir, "{}ehr_adj_final.pkl".format(data_prefix)), "rb"))
            ddi_adj = dill.load(open(os.path.join(self.data_dir, "{}ddi_A_final.pkl".format(data_prefix)), "rb"))
            return ehr_adj, ddi_adj
        if model_name == "SafeDrug":
            ddi_adj = dill.load(open(os.path.join(self.data_dir, "ddi_A_final_safedrug.pkl"), "rb"))
            ddi_mask_H = dill.load(open(os.path.join(self.data_dir, "ddi_mask_H_safedrug.pkl"), 'rb'))
            molecule = dill.load(open(os.path.join(self.data_dir, "idx2SMILES_safedrug.pkl"), 'rb')) 
            return ddi_adj, ddi_mask_H, molecule


class MedicalRecommendationDataloader(object):
    def __init__(self, data, model_name, vocab, shuffle=False, 
                 batch_size=1, evaluate=False, history=False):
        self.data = data
        self.vocab = vocab
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.model_name = model_name
        if type(self.vocab) == dict:
            # baseline vocab
            self.diag_vocab_size = len(vocab["diag_vocab"].idx2word)
            self.proc_vocab_size = len(vocab["proc_vocab"].idx2word)
            self.med_vocab_size = len(vocab["med_vocab"].idx2word)
            # build records
            self.records = build_records(data, vocab)
        else:
            # pretrain vocab build records
            self.records = build_records_for_pretrain_vocab(data, vocab)
            self.med_vocab_size = vocab.get_type_vocab_size("MED-ATC")

        self.evaluate = evaluate
        self.history = history
        if evaluate:
            self.shuffle = False

        if model_name in ["Leap", "DMNC"]:
            self.END_TOKEN = self.med_vocab_size + 1
        elif model_name in ["MLP", "Transformer"]:
            if history:
                self.CLS_TOKEN = self.diag_vocab_size + self.proc_vocab_size + self.med_vocab_size
                self.DIAG_TOKEN = self.diag_vocab_size + self.proc_vocab_size + self.med_vocab_size + 1
                self.PROC_TOKEN = self.diag_vocab_size + self.proc_vocab_size + self.med_vocab_size + 2
                self.MED_TOKEN = self.diag_vocab_size + self.proc_vocab_size + self.med_vocab_size + 3
                self.ADM_TOKEN = self.diag_vocab_size + self.proc_vocab_size + self.med_vocab_size + 4
                self.CUR_TOKEN = self.diag_vocab_size + self.proc_vocab_size + self.med_vocab_size + 5
            else:
                self.CLS_TOKEN = self.diag_vocab_size + self.proc_vocab_size
                self.DIAG_TOKEN = self.diag_vocab_size + self.proc_vocab_size + 1
                self.PROC_TOKEN = self.diag_vocab_size + self.proc_vocab_size + 2
        elif model_name in ["DualMLP", "DualTransformer"]:
            if history:
                self.ADM_TOKEN = self.diag_vocab_size + self.proc_vocab_size
                self.CUR_TOKEN = self.diag_vocab_size + self.proc_vocab_size + 1
        elif model_name in ["MedicalBert"]:
            self.CLS_TOKEN = vocab.get_word_id("<CLS>", "SPECIAL")
            self.ADM_TOKEN = vocab.get_word_id("<ADMISSION>", "SPECIAL")
            self.CUR_TOKEN = vocab.get_word_id("<SPECIAL0>", "SPECIAL")
            self.DIAG_TOKEN = vocab.get_word_id("<DIAG>", "TYPE")
            self.PROC_TOKEN = vocab.get_word_id("<PROC>", "TYPE")
            self.MED_TOKEN = vocab.get_word_id("<MED>", "TYPE")


    def __len__(self):
        if self.evaluate:
            return sum([len(d) - 1 for d in self.records])
        else:
            return sum([len(d) for d in self.records])

    def __iter__(self):
        data_to_iter = self.records
        if self.shuffle:
            random.shuffle(data_to_iter)
        #if self.evaluate:
        #    data_to_iter = [data for data in data_to_iter if len(data) > 1]
        for patient in data_to_iter:
            for idx, admission in enumerate(patient):

                if self.evaluate and idx == 0:
                    continue
                if self.evaluate:
                    y_target = np.zeros(self.med_vocab_size)
                    meds = admission[2]
                    if self.model_name in ["MedicalBert"]:
                        meds = [self.vocab.intra_type_index["MED-ATC"][m] for m in meds]
                    y_target[meds] = 1

                if self.model_name in ["GAMENet", "SafeDrug"]:
                    seq_inputs = patient[:idx+1]
                    loss1_target = np.zeros((1, self.med_vocab_size))
                    loss1_target[:, admission[2]] = 1
                    loss3_target = np.full((1, self.med_vocab_size), -1)
                    for i, item in enumerate(admission[2]):
                        loss3_target[0][i] = item
                    if self.evaluate:
                        yield seq_inputs, y_target
                    else:
                        yield seq_inputs, loss1_target, loss3_target
                elif self.model_name in ["Leap", "DMNC"]:
                    if self.evaluate:
                        yield admission, y_target
                    else:
                        yield admission, admission[2] + [self.END_TOKEN]
                elif self.model_name == "Nearest":
                    if self.evaluate:
                        yield patient[idx-1][2], y_target
                elif self.model_name in ["MLP", "Transformer"]:
                    # Single input with <cls> <diag> <proc> (<adm> if self.history)
                    union_inputs = [self.CLS_TOKEN]

                    if self.history:
                        patient_history = patient[:idx]
                        for history_adm in patient_history:
                            adm_diags = history_adm[0]
                            adm_procs = [p + self.diag_vocab_size for p in history_adm[1]]
                            adm_meds = [m + self.diag_vocab_size + self.proc_vocab_size for m in history_adm[2]]
                            union_inputs += [self.ADM_TOKEN, self.DIAG_TOKEN] + adm_diags + [self.PROC_TOKEN] + adm_procs + [self.MED_TOKEN] + adm_meds

                    diags = admission[0]
                    procs = [p + self.diag_vocab_size for p in admission[1]]
                    meds = admission[2]

                    if self.history:
                        union_inputs += [self.CUR_TOKEN]

                    union_inputs += [self.DIAG_TOKEN] + diags + [self.PROC_TOKEN] + procs
                    bce_loss_target = np.zeros((1, self.med_vocab_size))
                    bce_loss_target[:, meds] = 1
                    margin_loss_target = np.full((1, self.med_vocab_size), -1)
                    for i, item in enumerate(meds):
                        margin_loss_target[0][i] = item
                    if self.evaluate:
                        yield union_inputs, y_target
                    else:
                        yield union_inputs, bce_loss_target, margin_loss_target
                elif self.model_name in ["DualMLP", "DualTransformer"]:
                    # dual input
                    diags = admission[0]
                    procs = [p + self.diag_vocab_size for p in admission[1]]
                    meds = admission[2]

                    if self.history:
                        # do not use history medication
                        patient_history = patient[:idx]
                        history_diags = []
                        history_procs = []
                        for history_adm in patient_history:
                            adm_diags = history_adm[0]
                            adm_procs = [p + self.diag_vocab_size for p in history_adm[1]]
                            history_diags += [self.ADM_TOKEN] + adm_diags
                            history_procs += [self.ADM_TOKEN] + adm_procs

                        diags = history_diags + [self.CUR_TOKEN] + diags
                        procs = history_procs + [self.CUR_TOKEN] + procs

                    bce_loss_target = np.zeros((1, self.med_vocab_size))
                    bce_loss_target[:, meds] = 1
                    margin_loss_target = np.full((1, self.med_vocab_size), -1)
                    for i, item in enumerate(meds):
                        margin_loss_target[0][i] = item
                    if self.evaluate:
                        yield (diags, procs), y_target
                    else:
                        yield (diags, procs), bce_loss_target, margin_loss_target
                elif self.model_name in ["MedicalBert"]:
                    # Finetune task for pretrain

                    segment_id = 0
                    union_inputs = [self.CLS_TOKEN]
                    segment_ids = [0]

                    if self.history:
                        patient_history = patient[:idx]
                        for history_adm in patient_history:
                            adm_diags = history_adm[0]
                            adm_procs = history_adm[1]
                            adm_meds_original =  history_adm[3]
                            union_inputs += [self.ADM_TOKEN, self.DIAG_TOKEN] + adm_diags + [self.PROC_TOKEN] + adm_procs + [self.MED_TOKEN] + adm_meds_original
                            segment_ids += [segment_id] * (4 + len(adm_diags) + len(adm_procs) + len(adm_meds_original))
                            #segment_id += 1

                    diags = admission[0]
                    procs = admission[1]
                    meds_to_pred = admission[2]
                    meds_to_pred = [self.vocab.intra_type_index["MED-ATC"][m] for m in meds_to_pred]

                    if self.history:
                        union_inputs += [self.CUR_TOKEN]
                        segment_ids += [segment_id]

                    union_inputs += [self.DIAG_TOKEN] + diags + [self.PROC_TOKEN] + procs
                    segment_ids += [segment_id] * (2 + len(diags) + len(procs))
                    bce_loss_target = np.zeros((1, self.med_vocab_size))
                    bce_loss_target[:, meds_to_pred] = 1
                    margin_loss_target = np.full((1, self.med_vocab_size), -1)
                    for i, item in enumerate(meds_to_pred):
                        margin_loss_target[0][i] = item
                    if self.evaluate:
                        yield union_inputs, segment_ids, y_target
                    else:
                        yield union_inputs, segment_ids, bce_loss_target, margin_loss_target
                    pass



