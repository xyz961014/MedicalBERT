import numpy as np
import pandas as pd
import os
import sys
import warnings
import dill
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score, roc_auc_score, precision_score, f1_score, average_precision_score
warnings.filterwarnings('ignore')

class MedicalRecommendationDataset(object):
    def __init__(self, data_path, train_set_ratio=2/3):

        self.data_path = data_path
        self.data = dill.load(open(os.path.join(data_path, "records_final.pkl"), "rb"))
        self.vocab = dill.load(open(os.path.join(data_path, "voc_final.pkl"), "rb"))
        # get vocab_size
        self.diag_vocab = self.vocab['diag_voc']
        self.proc_vocab = self.vocab['pro_voc']
        self.med_vocab = self.vocab['med_voc']
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

    def get_dataloader(self, model_name):
        train_loader = MedicalRecommendationDataloader(self.data_train, model_name, self.med_vocab_size)
        eval_loader = MedicalRecommendationDataloader(self.data_eval, model_name, self.med_vocab_size, evaluate=True)
        test_loader = MedicalRecommendationDataloader(self.data_test, model_name, self.med_vocab_size, evaluate=True)
        return train_loader, eval_loader, test_loader


    def get_extra_data(self, model_name):
        if model_name == "GAMENet":
            ehr_adj = dill.load(open(os.path.join(self.data_path, "ehr_adj_final.pkl"), "rb"))
            ddi_adj = dill.load(open(os.path.join(self.data_path, "ddi_A_final.pkl"), "rb"))
            return ehr_adj, ddi_adj

class MedicalRecommendationDataloader(object):
    def __init__(self, data, model_name, med_vocab_size, shuffle=False, permute=False, batch_size=1, evaluate=False):
        self.data = data
        self.shuffle = shuffle
        self.permute = permute
        self.batch_size = batch_size
        self.model_name = model_name
        self.med_vocab_size = med_vocab_size
        self.evaluate = evaluate
        if evaluate:
            self.shuffle = False

    def __len__(self):
        if self.evaluate:
            return sum([len(d) - 1 for d in self.data])
        else:
            return sum([len(d) for d in self.data])

    def __iter__(self):
        data_to_iter = self.data
        if self.evaluate:
            # TODO: this is a temp solution, data preprocessing should be rewritten
            data_to_iter = [data for data in data_to_iter if len(data) > 1]
        for patient in data_to_iter:
            for idx, admission in enumerate(patient):
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
                    END_TOKEN = self.med_vocab_size + 1
                    if self.evaluate:
                        yield admission, y_target
                    else:
                        yield admission, admission[2] + [END_TOKEN]
                elif self.model_name == "Nearest":
                    if self.evaluate:
                        yield patient[idx-1][2], y_target


# use the same metric from DMNC
def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()

def transform_split(X, Y):
    x_train, x_eval, y_train, y_eval = train_test_split(X, Y, train_size=2/3, random_state=1109)
    x_eval, x_test, y_eval, y_test = train_test_split(x_eval, y_eval, test_size=0.5, random_state=1109)
    return x_train, x_eval, x_test, y_train, y_eval, y_test

def sequence_output_process(output_logits, filter_token):
    pind = np.argsort(output_logits, axis=-1)[:, ::-1]
    out_list = []
    break_flag = False
    for i in range(len(pind)):
        if break_flag:
            break
        for j in range(pind.shape[1]):
            label = pind[i][j]
            if label in filter_token:
                break_flag = True
                break
            if label not in out_list:
                out_list.append(label)
                break
    y_pred_prob_tmp = []
    for idx, item in enumerate(out_list):
        y_pred_prob_tmp.append(output_logits[idx, item])
    sorted_predict = [x for _, x in sorted(zip(y_pred_prob_tmp, out_list), reverse=True)]
    return out_list, sorted_predict


def sequence_metric(y_target, y_pred, y_prob, y_label):
    def average_prc(y_target, y_label):
        score = []
        for b in range(y_target.shape[0]):
            target = np.where(y_target[b]==1)[0]
            out_list = y_label[b]
            inter = set(out_list) & set(target)
            prc_score = 0 if len(out_list) == 0 else len(inter) / len(out_list)
            score.append(prc_score)
        return score


    def average_recall(y_target, y_label):
        score = []
        for b in range(y_target.shape[0]):
            target = np.where(y_target[b] == 1)[0]
            out_list = y_label[b]
            inter = set(out_list) & set(target)
            recall_score = 0 if len(target) == 0 else len(inter) / len(target)
            score.append(recall_score)
        return score


    def average_f1(average_prc, average_recall):
        score = []
        for idx in range(len(average_prc)):
            if (average_prc[idx] + average_recall[idx]) == 0:
                score.append(0)
            else:
                score.append(2*average_prc[idx]*average_recall[idx] / (average_prc[idx] + average_recall[idx]))
        return score


    def jaccard(y_target, y_label):
        score = []
        for b in range(y_target.shape[0]):
            target = np.where(y_target[b] == 1)[0]
            out_list = y_label[b]
            inter = set(out_list) & set(target)
            union = set(out_list) | set(target)
            jaccard_score = 0 if union == 0 else len(inter) / len(union)
            score.append(jaccard_score)
        return np.mean(score)

    def f1(y_target, y_pred):
        all_micro = []
        for b in range(y_target.shape[0]):
            all_micro.append(f1_score(y_target[b], y_pred[b], average='macro'))
        return np.mean(all_micro)

    def roc_auc(y_target, y_pred_prob):
        all_micro = []
        for b in range(len(y_target)):
            all_micro.append(roc_auc_score(y_target[b], y_pred_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_auc(y_target, y_prob):
        all_micro = []
        for b in range(len(y_target)):
            all_micro.append(average_precision_score(y_target[b], y_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_at_k(y_target, y_prob_label, k):
        precision = 0
        for i in range(len(y_target)):
            TP = 0
            for j in y_prob_label[i][:k]:
                if y_target[i, j] == 1:
                    TP += 1
            precision += TP / k
        return precision / len(y_target)
    try:
        auc = roc_auc(y_target, y_prob)
    except ValueError:
        auc = 0
    p_1 = precision_at_k(y_target, y_label, k=1)
    p_3 = precision_at_k(y_target, y_label, k=3)
    p_5 = precision_at_k(y_target, y_label, k=5)
    f1 = f1(y_target, y_pred)
    prauc = precision_auc(y_target, y_prob)
    ja = jaccard(y_target, y_label)
    avg_prc = average_prc(y_target, y_label)
    avg_recall = average_recall(y_target, y_label)
    avg_f1 = average_f1(avg_prc, avg_recall)

    return ja, prauc, np.mean(avg_prc), np.mean(avg_recall), np.mean(avg_f1)


def multi_label_metric(y_target, y_pred, y_prob):

    def jaccard(y_target, y_pred):
        score = []
        for b in range(y_target.shape[0]):
            target = np.where(y_target[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            union = set(out_list) | set(target)
            jaccard_score = 0 if union == 0 else len(inter) / len(union)
            score.append(jaccard_score)
        return np.mean(score)

    def average_prc(y_target, y_pred):
        score = []
        for b in range(y_target.shape[0]):
            target = np.where(y_target[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            prc_score = 0 if len(out_list) == 0 else len(inter) / len(out_list)
            score.append(prc_score)
        return score

    def average_recall(y_target, y_pred):
        score = []
        for b in range(y_target.shape[0]):
            target = np.where(y_target[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            recall_score = 0 if len(target) == 0 else len(inter) / len(target)
            score.append(recall_score)
        return score

    def average_f1(average_prc, average_recall):
        score = []
        for idx in range(len(average_prc)):
            if average_prc[idx] + average_recall[idx] == 0:
                score.append(0)
            else:
                score.append(2*average_prc[idx]*average_recall[idx] / (average_prc[idx] + average_recall[idx]))
        return score

    def f1(y_target, y_pred):
        all_micro = []
        for b in range(y_target.shape[0]):
            all_micro.append(f1_score(y_target[b], y_pred[b], average='macro'))
        return np.mean(all_micro)

    def roc_auc(y_target, y_prob):
        all_micro = []
        for b in range(len(y_target)):
            all_micro.append(roc_auc_score(y_target[b], y_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_auc(y_target, y_prob):
        all_micro = []
        for b in range(len(y_target)):
            all_micro.append(average_precision_score(y_target[b], y_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_at_k(y_target, y_prob, k=3):
        precision = 0
        sort_index = np.argsort(y_prob, axis=-1)[:, ::-1][:, :k]
        for i in range(len(y_target)):
            TP = 0
            for j in range(len(sort_index[i])):
                if y_target[i, sort_index[i, j]] == 1:
                    TP += 1
            precision += TP / len(sort_index[i])
        return precision / len(y_target)

    auc = roc_auc(y_target, y_prob)
    p_1 = precision_at_k(y_target, y_prob, k=1)
    p_3 = precision_at_k(y_target, y_prob, k=3)
    p_5 = precision_at_k(y_target, y_prob, k=5)
    f1 = f1(y_target, y_pred)
    prauc = precision_auc(y_target, y_prob)
    ja = jaccard(y_target, y_pred)
    avg_prc = average_prc(y_target, y_pred)
    avg_recall = average_recall(y_target, y_pred)
    avg_f1 = average_f1(avg_prc, avg_recall)

    return ja, prauc, np.mean(avg_prc), np.mean(avg_recall), np.mean(avg_f1)

def ddi_rate_score(record, path='../data/ddi_A_final.pkl'):
    # ddi rate
    ddi_A = dill.load(open(path, 'rb'))
    all_cnt = 0
    dd_cnt = 0
    for adm in record:
        med_code_set = adm
        for i, med_i in enumerate(med_code_set):
            for j, med_j in enumerate(med_code_set):
                if j <= i:
                    continue
                all_cnt += 1
                if ddi_A[med_i, med_j] == 1 or ddi_A[med_j, med_i] == 1:
                    dd_cnt += 1
    if all_cnt == 0:
        return 0
    return dd_cnt / all_cnt
