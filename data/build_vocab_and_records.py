import dill
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="_data.pkl",
                        help="data path of pkl file")
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


if __name__ == "__main__":
    args = parse_args()
    vocab = build_vocab(args)
    build_records(args, vocab)


