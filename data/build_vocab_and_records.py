import dill
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
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
    parser.add_argument("--pretrain", action="store_true",
                        help="preprocess pretrain data")
    parser.add_argument("--buckets", type=int, default=10,
                        help="number of buckets per item")
    parser.add_argument("--save", type=str, default="final",
                        help="filename of saving vocabulary and records")
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

    def __init__(self):
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

    def __len__(self):
        return self.vocab_size

    def __repr__(self):
        repr_str = "Medical Vocabulary for Pretrain"
        repr_str += "\nVocab size: {}".format(self.vocab_size)
        types = list(set(self.idx2type.values()))
        repr_str += "\nContaining types: " + ", ".join(types)
        for t in types:
            repr_str += "\n# {}: {}".format(t, len([v for v in self.idx2type.values() if v == t]))
        return repr_str


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
            return word

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


def build_vocab(args):
    print("build vocab ...")
    df = pd.read_pickle(args.data_path)

    diag_vocab = Vocab()
    proc_vocab = Vocab()
    med_vocab = Vocab()
    
    for index, row in df.iterrows():
        diag_vocab.add_sentence(row['ICD9_CODE'])
        proc_vocab.add_sentence(row['PRO_CODE'])
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

    special_tokens = ["<CLS>", "<MASK>", "<NORMAL>", "<ABNORMAL>", "<DELTA>", "<ADMISSION>", "<DAY>"]
    special_tokens += ["<SPECIAL{}>".format(i) for i in range(0, 64 - len(special_tokens))]
        
    print("build pretrain vocab ...")
    vocab = PretrainVocab()

    types = list(df["TYPE"].dropna().unique())
    type_tokens = ["<{}>".format(t) for t in types]
    
    # add special tokens
    vocab.add_words(special_tokens, typ="SPECIAL")
    # add type tokens
    vocab.add_words(type_tokens, typ="TYPE")
    # add bucket value tokens
    vocab.add_words(list(df["BUCKET_VALUE"].dropna().unique()), typ="VALUE")
    # add id tokens
    for t in types:
        t_df = df[df["TYPE"] == t]
        unique_tokens_in_type = list(t_df[id_columns[t]].dropna().unique())
        vocab.add_words(unique_tokens_in_type, typ=t)
    # add itemid2valueids
    for item_id in list(df["ITEMID"].dropna().unique()):
        item_df = df[df["ITEMID"] == item_id]
        typ = list(item_df["TYPE"].unique())[0]
        bucket_values = list(item_df["BUCKET_VALUE"].dropna().unique())
        vocab.add_item_values(item_id, typ=typ, values=bucket_values)
    # get detail from original csv table
    vocab.get_detail(args.data_dir)

    dill.dump(obj=vocab, file=open('{}_vocab.pkl'.format(args.save), 'wb'))
    print("build pretrain vocab complete!")

    return vocab

def build_records(args, vocab):
    print("build records ...")
    df = pd.read_pickle(args.data_path)
    diag_vocab = vocab["diag_vocab"]
    proc_vocab = vocab["proc_vocab"]
    med_vocab = vocab["med_vocab"]

    records = [] # (patient, code_kind:3, codes)  code_kind:diag, proc, med
    num_admissions = 0
    for subject_id in df['SUBJECT_ID'].unique():
        item_df = df[df['SUBJECT_ID'] == subject_id]
        patient = []
        for index, row in item_df.iterrows():
            admission = []
            admission.append([diag_vocab.word2idx[i] for i in row['ICD9_CODE']])
            admission.append([proc_vocab.word2idx[i] for i in row['PRO_CODE']])
            admission.append([med_vocab.word2idx[i] for i in row['NDC']])
            patient.append(admission)
            num_admissions += 1
        records.append(patient) 
    dill.dump(obj=records, file=open('{}_records.pkl'.format(args.save), 'wb'))
    print("build records complete!")
    print("#patients: {}".format(len(records)))
    print("#admissions: {}".format(num_admissions))
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
        if not pd.isna(row["BUCKET_VALUE"]):
            vocab.count_word(row["BUCKET_VALUE"])
        if not pd.isna(row["FLAG"]):
            vocab.count_word("<{}>".format(row["FLAG"].upper()))
    dill.dump(obj=vocab, file=open('{}_vocab_with_count.pkl'.format(args.save), 'wb'))

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
            if pd.notna(row["BUCKET_VALUE"]):
                print_token(row["BUCKET_VALUE"])
            if pd.notna(row["FLAG"]):
                flag_token = "<{}>".format(row["FLAG"].upper())
                print_token(flag_token)

    subjects = list(df["SUBJECT_ID"].unique())
    for subject_id in tqdm(subjects, desc="build data"):
        subject_df = df[df["SUBJECT_ID"] == subject_id]
        # add potential static data here 
        admissions = list(subject_df["HADM_ID"].unique())
        print_tokens(["<CLS>", "<ADMISSION>"])
        for adm_id in admissions:
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
                    print_token("<DAY>")
                    last_date = current_date
                for df_type in df_types:
                    datetime_type_df = datetime_df[datetime_df["TYPE"] == df_type]
                    if df_type in vocab.type_with_value:
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
            df.loc[bucket_index, "BUCKET_VALUE"] = bucket_value

    print("build buckets complete")
    df.to_pickle("{}_data_with_bucket_value.pkl".format(args.save))
    return df


if __name__ == "__main__":
    args = parse_args()
    if args.pretrain:
        # build buckets
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
        if os.path.exists("{}_vocab_with_count.pkl".format(args.save)):
            print("vocab with count already exists, loading existing file")
            vocab = dill.load(open("{}_vocab_with_count.pkl".format(args.save), "rb"))
        else:
            vocab = count_words(args, vocab, data_df)

        build_pretrain_data(args, vocab, data_df)
    else:
        vocab = build_vocab(args)
        build_records(args, vocab)


