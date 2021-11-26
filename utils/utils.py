import numpy as np
import os
import sys
import dill
import torch
import torch.distributed as dist
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score, roc_auc_score, precision_score, f1_score, average_precision_score

def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_main_process():
    return get_rank() == 0


def format_step(step):
    if isinstance(step, str):
        return step
    s = ""
    if len(step) > 0:
        s += "Epoch: {} ".format(step[0])
    if len(step) > 1:
        s += "Step: {} ".format(step[1])
    if len(step) > 2:
        s += "Validation Step: {} ".format(step[2])
    return s

def format_metric(metric, metadata, value):

    if metadata:
        unit = metadata["unit"] if "unit" in metadata.keys() else ""
        format_str = "{" + metadata["format"] + "}" if "format" in metadata.keys() else "{}"
    else:
        format_table = {
                "average_loss": "{:5.5f}",
                "step_loss": "{:5.5f}",
                "lr": "{:5.3e}",
                "predict_accuracy": "{:5.5f}"
                       }
        value_map = {"predict_accuracy": lambda x: x * 100}
        unit_table = {"predict_accuracy": "%\n"}

        metric_name = metric.split()[-1]
        value = value_map[metric_name](value) if metric_name in value_map.keys() else value
        unit = unit_table[metric_name] if metric_name in unit_table.keys() else ""
        format_str = format_table[metric_name] if metric_name in format_table.keys() else "{}"
    output_str = "{} : {} {}".format(metric, format_str.format(value) if value is not None else value, unit)
    return output_str


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
