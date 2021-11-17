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
    def __init__(self, data_path, data_prefix, train_set_ratio=2/3):

        self.data_path = data_path
        self.data = dill.load(open(os.path.join(data_path, "{}_records.pkl".format(data_prefix)), "rb"))
        self.vocab = dill.load(open(os.path.join(data_path, "{}_vocab.pkl".format(data_prefix)), "rb"))
        # get vocab_size
        self.diag_vocab = self.vocab['diag_vocab']
        self.proc_vocab = self.vocab['proc_vocab']
        self.med_vocab = self.vocab['med_vocab']
        self.diag_vocab_size = len(self.diag_vocab.idx2word)
        self.proc_vocab_size = len(self.proc_vocab.idx2word)
        self.med_vocab_size = len(self.med_vocab.idx2word)
        self.vocab_size = (self.diag_vocab_size, self.proc_vocab_size, self.med_vocab_size)
        # split data
        split_point = int(len(self.data) * train_set_ratio)
        self.data_train = self.data[:split_point]
        eval_len = int(len(self.data[split_point:]) / 2)
        self.data_test = self.data[split_point:split_point + eval_len]
        self.data_eval = self.data[split_point+eval_len:]

    def get_dataloader(self, model_name, shuffle=False, permute=False, history=False):
        train_loader = MedicalRecommendationDataloader(self.data_train, model_name, self.vocab_size, 
                                                       shuffle=shuffle, permute=permute, history=history)
        eval_loader = MedicalRecommendationDataloader(self.data_eval, model_name, self.vocab_size, 
                                                      evaluate=True, history=history)
        test_loader = MedicalRecommendationDataloader(self.data_test, model_name, self.vocab_size, 
                                                      evaluate=True, history=history)
        return train_loader, eval_loader, test_loader

    def get_train_eval_loader(self, model_name, history=False):
        return MedicalRecommendationDataloader(self.data_train, model_name, self.vocab_size, 
                                               evaluate=True, history=history)


    def get_extra_data(self, model_name):
        if model_name == "GAMENet":
            ehr_adj = dill.load(open(os.path.join(self.data_path, "ehr_adj_final.pkl"), "rb"))
            ddi_adj = dill.load(open(os.path.join(self.data_path, "ddi_A_final.pkl"), "rb"))
            return ehr_adj, ddi_adj


class MedicalRecommendationDataloader(object):
    def __init__(self, data, model_name, vocab_size, shuffle=False, 
                 permute=False, batch_size=1, evaluate=False, history=False):
        self.data = data
        self.shuffle = shuffle
        self.permute = permute
        self.batch_size = batch_size
        self.model_name = model_name
        self.diag_vocab_size = vocab_size[0]
        self.proc_vocab_size = vocab_size[1]
        self.med_vocab_size = vocab_size[2]
        self.evaluate = evaluate
        self.history = history
        if evaluate:
            self.shuffle = False
            self.permute = False

        if model_name == "Leap":
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

    def __len__(self):
        if self.evaluate:
            return sum([len(d) - 1 for d in self.data])
        else:
            return sum([len(d) for d in self.data])

    def __iter__(self):
        data_to_iter = self.data
        if self.shuffle:
            random.shuffle(data_to_iter)
        #if self.evaluate:
        #    data_to_iter = [data for data in data_to_iter if len(data) > 1]
        for patient in data_to_iter:
            for idx, admission in enumerate(patient):

                if self.permute:
                    random.shuffle(admission[0])
                    random.shuffle(admission[1])
                    random.shuffle(admission[2])
                if self.evaluate and idx == 0:
                    continue
                if self.evaluate:
                    y_target = np.zeros(self.med_vocab_size)
                    y_target[admission[2]] = 1

                if self.model_name == "GAMENet":
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
                elif self.model_name == "Leap":
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



