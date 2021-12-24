import dill
import os
import re
import argparse
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import ipdb

id_columns = {
        "MED": "NDC",
        "DIAG": "ICD9_CODE",
        "PROC": "ICD9_CODE",
        "LAB": "ITEMID",
        "CHART": "ITEMID",
               }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="_data.pkl",
                        help="data path of pkl file")
    parser.add_argument("--data_dir", type=str, default="/home/xyz/Documents/Datasets/mimiciii-1.4",
                        help="data dir of MIMIC-III CSV file")
    parser.add_argument("--ddi_file", type=str, default="/home/xyz/Documents/Datasets/mimiciii-1.4/drug-DDI.csv",
                        help="DDI file")
    parser.add_argument("--cid_atc", type=str, default="/home/xyz/Documents/Datasets/mimiciii-1.4/drug-atc.csv",
                        help="Drug ATC file")
    parser.add_argument("--pretrain", action="store_true",
                        help="preprocess pretrain data")
    parser.add_argument("--buckets", type=int, default=10,
                        help="number of buckets per item")
    parser.add_argument("--save", type=str, default="final",
                        help="filename of saving vocabulary and records")
    parser.add_argument("--do_not_count_token", action="store_true",
                        help="skip counting tokens for vocab")
    parser.add_argument("--no_value_num", action="store_true",
                        help="only keep the value range of the value")
    parser.add_argument("--no_value", action="store_true",
                        help="do not keep the value")
    parser.add_argument("--day_level", action="store_true",
                        help="generate day level data rather than admission level")
    return parser.parse_args()

class Vocab(object):

    def __init__(self):
        super().__init__()
        self.idx2word = {}
        self.word2idx = {}

    def __len__(self):
        return len(self.idx2word.keys())

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx.keys():
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)


class PretrainVocab(object):

    def __init__(self, value_range=False):
        super().__init__()

        self.type_with_id = ["LAB", "CHART", "MED", "PROC", "DIAG"]
        self.type_with_value = ["LAB", "CHART"]

        self.idx2word = {}
        self.word2idx = {}
        self.word2count = {}
        self.vocab_size = 0

        self.itemid2valueids = {}
        self.idx2type = {}
        self.idx2detail = {}

        self.value_range = value_range

    def __len__(self):
        return self.vocab_size

    def __repr__(self):
        repr_str = "Medical Vocabulary for Pretrain"
        repr_str += "\nVocab size: {}".format(self.vocab_size)
        types = list(set(self.idx2type.values()))
        repr_str += "\nContaining types: " + ", ".join(types)
        for t in types:
            repr_str += "\n# {}: {}".format(t, self.get_type_vocab_size(t))
        return repr_str

    def get_type_vocab_size(self, typ):
        return len([v for v in self.idx2type.values() if v == typ])

    def build_intra_type_index(self, typ):
        if not hasattr(self, "intra_type_index"):
            self.intra_type_index = {}
        if not typ in list(set(self.idx2type.values())):
            raise ValueError("Type {} does not exist in Vocab".format(typ))
        intra_type_index = {}
        intra_idx = 0
        for i, t in self.idx2type.items():
            if typ == t:
                intra_type_index[i] = intra_idx
                intra_idx += 1
        self.intra_type_index[typ] = intra_type_index

        return intra_type_index

    def normalize_word(self, word, typ):
        if typ in self.type_with_id:
            try:
                word = float(word)
                word = int(word)
            except:
                pass
            finally:
                word = "{}-{}".format(typ, word)
            return word
        else:
            if typ == "FLAG" and word.upper() in ["NORMAL", "ABNORMAL", "DELTA"]:
                word = "<{}>".format(word.upper())
            return word

    def get_word_id(self, word, typ):
        word = self.normalize_word(word, typ)
        if word in self.word2idx.keys():
            return self.word2idx[word]
        else:
            return self.word2idx["<UNK>"]

    def get_value_word_id(self, value, unit, item_wordid):
        value_ids = self.itemid2valueids[item_wordid]
        values = [self.idx2word[v] for v in value_ids]
        value_nums = [v[:re.search(unit, v).start()].split("~") if not unit.strip() == "" else v.split("~") 
                      for v in values]
        value_boundaries = set()
        for bounds in value_nums:
            for bound in bounds:    
                bound = "inf" if bound == "max" else bound
                bound = "-inf" if bound == "min" else bound
                value_boundaries.add(float(bound))
        value_boundaries = sorted(list(value_boundaries))

        start, end = "", ""
        for i in range(len(value_boundaries) - 1):
            if value_boundaries[i] < value <= value_boundaries[i+1]:
                start, end = value_boundaries[i], value_boundaries[i+1]
                break
        start = "min" if start == float("-inf") else str(start)
        end = "max" if end == float("inf") else str(end)
        if self.value_range:
            value_token = "<RANGE{}>".format(i)
        else:
            value_token = "{}~{}{}".format(start, end, unit)

        return self.get_word_id(value_token, typ="VALUE")

    def add_word(self, word, typ):
        word = self.normalize_word(word, typ)
        if word not in self.word2idx.keys():
            self.idx2word[self.vocab_size] = word
            self.word2idx[word] = self.vocab_size
            self.idx2type[self.vocab_size] = typ
            self.word2count[word] = 0
            self.vocab_size += 1

    def add_words(self, words, typ):
        for word in words:
            self.add_word(word, typ)

    def count_word(self, word):
        self.word2count[word] += 1

    def add_item_values(self, word, typ, values):
        word = self.normalize_word(word, typ)
        idx = self.word2idx[word]
        self.itemid2valueids[idx] = [self.word2idx[v] for v in values]

    def get_detail(self, data_dir="/home/xyz/Documents/Datasets/mimiciii-1.4"):
        data_source = {
                "LAB": ["D_LABITEMS.csv", "ITEMID", "LABEL"],
                "MED": ["PRESCRIPTIONS.csv", "NDC", "DRUG_NAME_GENERIC"],
                "CHART": ["D_ITEMS.csv", "ITEMID", "LABEL"],
                "PROC": ["D_ICD_PROCEDURES.csv", "ICD9_CODE", "LONG_TITLE"],
                "DIAG": ["D_ICD_DIAGNOSES.csv", "ICD9_CODE", "LONG_TITLE"],
                      }

        for t, (filename, id_col, text_col) in tqdm(data_source.items(), desc="getting details"):
            filename = os.path.join(data_dir, filename)
            df = pd.read_csv(filename, low_memory=False)
            df[text_col].fillna("", inplace=True)
            idxs = list(df[id_col].dropna().unique())
            for idx in idxs:
                idx_df = df[df[id_col] == idx]
                self.add_word(idx, typ=t)
                self.idx2detail[self.word2idx[self.normalize_word(idx, t)]] = idx_df[text_col].values[0]

    def tokenize(self, sentence):
        words = sentence.strip().split()
        tokens = []
        for word in words:
            tokens.append(self.word2idx[word])

        return tokens

    def detokenize(self, ids):
        return " ".join([self.idx2word[i] for i in ids])


def build_vocab(args):
    print("build vocab ...")
    df = pd.read_pickle(args.data_path)

    diag_vocab = Vocab()
    proc_vocab = Vocab()
    med_vocab = Vocab()
    
    for index, row in df.iterrows():
        diag_vocab.add_sentence(row['ICD9_CODE'])
        proc_vocab.add_sentence(row['PROC_CODE'])
        med_vocab.add_sentence(row['NDC'])

    vocab = {
                "diag_vocab": diag_vocab,
                "proc_vocab": proc_vocab,
                "med_vocab": med_vocab
            }
    
    dill.dump(obj=vocab, file=open('{}_vocab.pkl'.format(args.save), 'wb'))
    print("build vocab complete!")
    print("#diag_vocab: {}".format(len(diag_vocab)))
    print("#proc_vocab: {}".format(len(proc_vocab)))
    print("#med_vocab: {}".format(len(med_vocab)))
    return vocab

def build_pretrain_vocab(args, df):

    special_tokens = ["<PAD>", "<CLS>", "<MASK>", "<ADMISSION>", "<DAY>", "<UNK>"]
    value_special_tokens = ["<NORMAL>", "<ABNORMAL>", "<DELTA>"]
    special_tokens += ["<SPECIAL{}>".format(i) for i in range(0, 64 - len(special_tokens))]
        
    print("build pretrain vocab ...")
    vocab = PretrainVocab(value_range=args.no_value_num)

    types = list(df["TYPE"].dropna().unique())
    type_tokens = ["<{}>".format(t) for t in types]
    
    # add special tokens
    vocab.add_words(special_tokens, typ="SPECIAL")
    vocab.add_words(value_special_tokens, typ="FLAG")
    # add type tokens
    vocab.add_words(type_tokens, typ="TYPE")
    # add bucket value tokens
    if "BUCKET_VALUE" in df.columns:
        vocab.add_words(list(df["BUCKET_VALUE"].dropna().unique()), typ="VALUE")
    if "BUCKET_VALUE_NUM" in df.columns:
        vocab.add_words(list(df["BUCKET_VALUE_NUM"].dropna().unique()), typ="VALUE")
    # add id tokens
    for t in types:
        t_df = df[df["TYPE"] == t]
        unique_tokens_in_type = list(t_df[id_columns[t]].dropna().unique())
        vocab.add_words(unique_tokens_in_type, typ=t)
    # add itemid2valueids
    if "ITEMID" in df.columns:
        for item_id in list(df["ITEMID"].dropna().unique()):
            item_df = df[df["ITEMID"] == item_id]
            typ = list(item_df["TYPE"].unique())[0]
            bucket_key= "BUCKET_VALUE_NUM" if args.no_value_num else "BUCKET_VALUE"
            if bucket_key in item_df.columns:
                bucket_values = list(item_df[bucket_key].dropna().unique())
                vocab.add_item_values(item_id, typ=typ, values=bucket_values)
    # get detail from original csv table
    vocab.get_detail(args.data_dir)

    dill.dump(obj=vocab, file=open('{}_vocab.pkl'.format(args.save), 'wb'))
    print("build pretrain vocab complete!")

    return vocab

def build_records(df, vocab, save=None):
    print("build records ...")
    diag_vocab = vocab["diag_vocab"]
    proc_vocab = vocab["proc_vocab"]
    med_vocab = vocab["med_vocab"]

    records = [] # (patient, code_type: 3, codes)  code_type:diag, proc, med
    num_admissions = 0
    for subject_id in df['SUBJECT_ID'].unique():
        item_df = df[df['SUBJECT_ID'] == subject_id]
        patient = []
        for index, row in item_df.iterrows():
            admission = []
            admission.append([diag_vocab.word2idx[i] for i in row['ICD9_CODE']])
            admission.append([proc_vocab.word2idx[i] for i in row['PROC_CODE']])
            admission.append([med_vocab.word2idx[i] for i in row['NDC']])
            patient.append(admission)
            num_admissions += 1
        records.append(patient) 
    if save is not None:
        dill.dump(obj=records, file=open('{}_records.pkl'.format(args.save), 'wb'))
    print("build records complete!")
    print("#patients: {}".format(len(records)))
    print("#admissions: {}".format(num_admissions))
    return records


def build_records_for_pretrain_vocab(df, vocab, save=None):
    print("build records ...")

    records = [] # (patient, code_type: 3, codes)  code_type:diag, proc, med
    num_admissions = 0
    atcs = []
    for subject_id in df['SUBJECT_ID'].unique():
        item_df = df[df['SUBJECT_ID'] == subject_id]
        patient = []
        for index, row in item_df.iterrows():
            admission = []
            admission.append([vocab.get_word_id(i, "DIAG") for i in row['ICD9_CODE']])
            admission.append([vocab.get_word_id(i, "PROC") for i in row['PROC_CODE']])
            for atc in row["NDC"]:
                if not atc in atcs:
                    atcs.append(atc)
                if not vocab.normalize_word(atc, "MED-ATC") in vocab.word2idx.keys():
                    vocab.add_word(atc, "MED-ATC")
            admission.append([vocab.get_word_id(i, "MED-ATC") for i in row['NDC']])
            admission.append([vocab.get_word_id(i, "MED") for i in row['NDC_ORIGINAL']])
            if "LAB_ID" in row.keys():
                if pd.notna(row["LAB_ID"]):
                    lab_ids = json.loads(row["LAB_ID"])
                    lab_values = json.loads(row["LAB_VALUE"])
                    lab_units = json.loads(row["LAB_UNIT"])
                    lab_flags = json.loads(row["LAB_FLAG"])
                    labs = list(zip(lab_ids, lab_values, lab_units, lab_flags))
                    lab_tokens = []
                    for lab_id, lab_value, lab_unit, lab_flag in labs:
                        lab_token_id = vocab.get_word_id(lab_id, typ="LAB")
                        lab_flag_id = vocab.get_word_id(lab_flag, typ="FLAG")
                        lab_value_id = vocab.get_value_word_id(lab_value, lab_unit, lab_token_id)
                        lab_tokens.append(lab_token_id)
                        lab_tokens.append(lab_value_id)
                        lab_tokens.append(lab_flag_id)
                    admission.append(lab_tokens)
            patient.append(admission)
            num_admissions += 1
        records.append(patient) 
    if save is not None:
        dill.dump(obj=records, file=open('{}_records.pkl'.format(args.save), 'wb'))
    print("build records for pretrain vocab complete!")
    print("#patients: {}".format(len(records)))
    print("#admissions: {}".format(num_admissions))
    print("#medications to predict: {}".format(len(atcs)))
    return records


def build_token(args, vocab, df):

    #def add_token(row):
    #    t = row["TYPE"]
    #    word_id = row[id_columns[t]]
    #    word = vocab.normalize_word(word_id, typ=t)
    #    row["TYPE_TOKEN"] = word
    #    row["TYPE_TOKEN_ID"] = vocab.word2idx[word]
    #    return row
    #chunksize = int(1e1)
    #chunks = [df[i: i+chunksize] for i in range(0, len(df), chunksize)]
    #for i, chunk in tqdm(enumerate(chunks), total=len(chunks), desc="build tokens"):
    #    df.iloc[i*chunksize: (i+1)*chunksize] = chunk.apply(add_token, axis=1)
    
    for t in vocab.type_with_id:
        t_df = df[df["TYPE"] == t]
        if id_columns[t] in t_df.columns:
            unique_tokens_in_type = list(t_df[id_columns[t]].dropna().unique())
            for idx in tqdm(unique_tokens_in_type, desc="build tokens for {}".format(t)):
                word = vocab.normalize_word(idx, typ=t)
                idx2build = t_df[t_df[id_columns[t]] == idx].index
                df.loc[idx2build, "TYPE_TOKEN"] = word
                df.loc[idx2build, "TYPE_TOKEN_ID"] = vocab.word2idx[word]

    df["TYPE_TOKEN_ID"] = df["TYPE_TOKEN_ID"].astype("int64")
    df.to_pickle("{}_data_with_token.pkl".format(args.save))
    return df 

def count_words(args, vocab, df):

    for idx, row in tqdm(df.iterrows(), desc="count tokens", total=len(df)):
        word = row["TYPE_TOKEN"]
        vocab.count_word(word)
        if "BUCKET_VALUE" in row.keys():
            if not pd.isna(row["BUCKET_VALUE"]):
                vocab.count_word(row["BUCKET_VALUE"])
        if "FLAG" in row.keys():
            if not pd.isna(row["FLAG"]):
                vocab.count_word("<{}>".format(row["FLAG"].upper()))
    dill.dump(obj=vocab, file=open('{}_vocab.pkl'.format(args.save), 'wb'))

    return vocab


def build_pretrain_data(args, vocab, df):

    token_file = open("{}_token_data.txt".format(args.save), "w")
    id_file = open("{}_id_data.txt".format(args.save), "w")

    def print_token(token):
        token_file.write("{} ".format(token))
        id_file.write("{} ".format(vocab.word2idx[token]))

    def print_tokens(tokens):
        for token in tokens:
            print_token(token)

    def print_df(dataframe):
        for idx, row in dataframe.iterrows():
            token_file.write("{} ".format(row["TYPE_TOKEN"]))
            id_file.write("{} ".format(int(row["TYPE_TOKEN_ID"])))
            if "BUCKET_VALUE" in row.keys():
                if pd.notna(row["BUCKET_VALUE"]):
                    print_token(row["BUCKET_VALUE"])
            if "FLAG" in row.keys():
                if pd.notna(row["FLAG"]):
                    flag_token = "<{}>".format(row["FLAG"].upper())
                    print_token(flag_token)

    subjects = list(df["SUBJECT_ID"].unique())
    for subject_id in tqdm(subjects, desc="build data"):
        subject_df = df[df["SUBJECT_ID"] == subject_id]
        # add potential static data here 
        admissions = list(subject_df["HADM_ID"].unique())
        for adm_id in admissions:
            print_token("<ADMISSION>")
            adm_df = subject_df[subject_df["HADM_ID"] == adm_id]
            # and add potential static data here 
            adm_df_wo_time = adm_df[pd.isna(adm_df["DATETIME"])]
            adm_df_w_time = adm_df[pd.notna(adm_df["DATETIME"])]
            # add tokens with no time
            for t in list(adm_df_wo_time["TYPE"].dropna().unique()):
                t_df = adm_df_wo_time[adm_df_wo_time["TYPE"] == t]
                print_token("<{}>".format(t))
                print_df(t_df)
            # add tokens in order of time
            last_date = None
            for datetime in list(adm_df_w_time["DATETIME"].dropna().unique()):
                datetime_df = adm_df_w_time[adm_df_w_time["DATETIME"] == datetime]
                df_types = list(datetime_df["TYPE"].unique())
                current_date = datetime.astype("str").split("T")[0]
                if not last_date == current_date:
                    if args.day_level:
                        if last_date is not None:
                            token_file.write("\n")
                            id_file.write("\n")
                            print_token("<ADMISSION>")
                            for t in list(adm_df_wo_time["TYPE"].dropna().unique()):
                                t_df = adm_df_wo_time[adm_df_wo_time["TYPE"] == t]
                                print_token("<{}>".format(t))
                                print_df(t_df)
                    else:
                        print_token("<DAY>")
                    last_date = current_date
                for df_type in df_types:
                    datetime_type_df = datetime_df[datetime_df["TYPE"] == df_type]
                    if df_type in vocab.type_with_value and "BUCKET_VALUE" in datetime_type_df.columns:
                        datetime_type_df = datetime_type_df[pd.notna(datetime_type_df["BUCKET_VALUE"])]
                    print_token("<{}>".format(df_type))
                    print_df(datetime_type_df)

            token_file.write("\n")
            id_file.write("\n")
        token_file.write("\n")
        id_file.write("\n")

    token_file.close()
    id_file.close()

def build_buckets(args):
    df = pd.read_pickle(args.data_path)
    if "ITEMID" in df.columns:
        item_ids = set(df["ITEMID"].dropna())
        for item_id in tqdm(item_ids, desc="build buckets"):
            item_df = df[df["ITEMID"] == item_id]
            unit = item_df["VALUEUOM"].unique()[0]
            item_df = item_df.sort_values(by=["VALUENUM"])

            item_num = len(item_df)
            min_value = min(item_df["VALUENUM"])
            max_value = max(item_df["VALUENUM"])
            bucket_len = item_num // args.buckets
            boundary_index = [i for i in range(bucket_len, item_num - bucket_len + 1, bucket_len)]
            boundary_values = [item_df.iloc[i]["VALUENUM"] for i in boundary_index]
            # rm duplicate
            boundary_values = sorted(list(set(boundary_values)))
            # recompute boundary_index 
            boundary_index = []
            iv = 0
            for i, value in enumerate(item_df["VALUENUM"].values):
                if value > boundary_values[iv]:
                    boundary_index.append(i)
                    iv += 1
                    if iv >= len(boundary_values):
                        break
            boundary_index = [0] + boundary_index + [item_num-1]
            boundary_values = ["min"] + boundary_values + ["max"]
            # fix boundary_index to guarantee max to occur, note that min is bound to occur
            # because out bucket contain the right boundary instead of the left
            for i in range(-1, -len(boundary_index)+1, -1):
                if boundary_index[i-1] == boundary_index[i]:
                    boundary_index[i-1] -= 1

            for i in range(len(boundary_index) - 1):
                bucket_index = item_df[boundary_index[i]:boundary_index[i+1]].index
                bucket_value = "{}~{}{}".format(boundary_values[i], boundary_values[i+1], unit)
                if args.no_value_num:
                    df.loc[bucket_index, "BUCKET_VALUE_NUM"] = bucket_value
                    bucket_value = "<RANGE{}>".format(i)
                df.loc[bucket_index, "BUCKET_VALUE"] = bucket_value

        print("build buckets complete")
    df.to_pickle("{}_data_with_bucket_value.pkl".format(args.save))
    return df


# get ddi matrix
def get_ddi_matrix(records, med_voc, ddi_file):

    TOPK = 40 # topk drug-drug interaction
    cid2atc_dic = defaultdict(set)
    med_voc_size = len(med_voc.idx2word)
    med_unique_word = [med_voc.idx2word[i] for i in range(med_voc_size)]
    atc3_atc4_dic = defaultdict(set)
    for item in med_unique_word:
        atc3_atc4_dic[item[:4]].add(item)
    
    with open(args.cid_atc, 'r') as f:
        for line in f:
            line_ls = line[:-1].split(',')
            cid = line_ls[0]
            atcs = line_ls[1:]
            for atc in atcs:
                if len(atc3_atc4_dic[atc[:4]]) != 0:
                    cid2atc_dic[cid].add(atc[:4])
            
    # ddi load
    ddi_df = pd.read_csv(ddi_file)
    # fliter sever side effect 
    ddi_most_pd = ddi_df.groupby(by=['Polypharmacy Side Effect', 'Side Effect Name']).size().reset_index().rename(columns={0:'count'}).sort_values(by=['count'],ascending=False).reset_index(drop=True)
    ddi_most_pd = ddi_most_pd.iloc[-TOPK:,:]
    # ddi_most_pd = pd.DataFrame(columns=['Side Effect Name'], data=['as','asd','as'])
    fliter_ddi_df = ddi_df.merge(ddi_most_pd[['Side Effect Name']], how='inner', on=['Side Effect Name'])
    ddi_df = fliter_ddi_df[['STITCH 1','STITCH 2']].drop_duplicates().reset_index(drop=True)

    # weighted ehr adj 
    ehr_adj = np.zeros((med_voc_size, med_voc_size))
    for patient in records:
        for adm in patient:
            med_set = adm[2]
            for i, med_i in enumerate(med_set):
                for j, med_j in enumerate(med_set):
                    if j<=i:
                        continue
                    ehr_adj[med_i, med_j] = 1
                    ehr_adj[med_j, med_i] = 1
    dill.dump(ehr_adj, open('{}_ehr_adj_final.pkl'.format(args.save), 'wb'))  

    # ddi adj
    ddi_adj = np.zeros((med_voc_size,med_voc_size))
    for index, row in ddi_df.iterrows():
        # ddi
        cid1 = row['STITCH 1']
        cid2 = row['STITCH 2']
        
        # cid -> atc_level3
        for atc_i in cid2atc_dic[cid1]:
            for atc_j in cid2atc_dic[cid2]:
                
                # atc_level3 -> atc_level4
                for i in atc3_atc4_dic[atc_i]:
                    for j in atc3_atc4_dic[atc_j]:
                        if med_voc.word2idx[i] != med_voc.word2idx[j]:
                            ddi_adj[med_voc.word2idx[i], med_voc.word2idx[j]] = 1
                            ddi_adj[med_voc.word2idx[j], med_voc.word2idx[i]] = 1
    dill.dump(ddi_adj, open('{}_ddi_A_final.pkl'.format(args.save), 'wb')) 

    return ddi_adj



if __name__ == "__main__":
    args = parse_args()
    if args.pretrain:
        # build buckets
        if args.no_value:
            bucket_df = pd.read_pickle(args.data_path)
            print("do not use value, skip building buckets")
        else:
            if os.path.exists("{}_data_with_bucket_value.pkl".format(args.save)):
                print("data with bucket value already exists, loading existing file")
                bucket_df = pd.read_pickle("{}_data_with_bucket_value.pkl".format(args.save))
            else:
                bucket_df = build_buckets(args)
        # build vocab
        if os.path.exists("{}_vocab.pkl".format(args.save)):
            print("vocab already exists, loading existing file")
            vocab = dill.load(open("{}_vocab.pkl".format(args.save), "rb"))
        else:
            vocab = build_pretrain_vocab(args, bucket_df)
        # build tokens
        if os.path.exists("{}_data_with_token.pkl".format(args.save)):
            print("data with token already exists, loading existing file")
            data_df = pd.read_pickle("{}_data_with_token.pkl".format(args.save))
        else:
            data_df = build_token(args, vocab, bucket_df)
        # count tokens
        if not args.do_not_count_token:
            vocab = count_words(args, vocab, data_df)

        build_pretrain_data(args, vocab, data_df)
    else:
        vocab = build_vocab(args)
        df = pd.read_pickle(args.data_path)
        records = build_records(df, vocab, save=args.save)
        ddi_adj = get_ddi_matrix(records, vocab["med_vocab"], args.ddi_file)


