import argparse
import logging
import sys
import os
import random
import h5py
import numpy as np
import dill
from tqdm import tqdm
from copy import copy

import random
import collections

curr_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(curr_path, ".."))

from data.build_vocab_and_records import PretrainVocab

import ipdb

def parse_args():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--vocab_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The vocabulary the BERT model will train on.")
    parser.add_argument("--input_id_file",
                        default=None,
                        type=str,
                        help="The input train corpus with id. path to a single file")
    parser.add_argument("--input_file",
                        default=None,
                        type=str,
                        help="The input train corpus. path to a single file")
    parser.add_argument("--save",
                        default=None,
                        type=str,
                        required=True,
                        help="The output name prefix where the processed pretrain data will be written.")
    parser.add_argument("--seq_level_task",
                        default=None,
                        type=str,
                        help="The sequence level task to choose, choices: "
                             "NAP: next admission prediction "
                             "AOP: admission order prediction")
    ## Other parameters

    #int 
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--dupe_factor",
                        default=10,
                        type=int,
                        help="Number of times to duplicate the input data (with different masks).")
    parser.add_argument("--max_predictions_per_seq",
                        default=80,
                        type=int,
                        help="The maximum total of masked tokens in input sequence")

    # floats

    parser.add_argument("--masked_lm_prob",
                        default=0.15,
                        type=float,
                        help="Masked LM probability.")

    parser.add_argument("--short_seq_prob",
                        default=0.1,
                        type=float,
                        help="Probability to create a sequence shorter than maximum sequence length")

    parser.add_argument('--random_seed',
                        type=int,
                        default=12345,
                        help="random seed for initialization")

    return parser.parse_args()


class TrainingInstance(object):
  """A single training instance (sentence pair)."""

  def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels, seq_level_labels=None):
    self.tokens = tokens
    self.segment_ids = segment_ids
    self.masked_lm_positions = masked_lm_positions
    self.masked_lm_labels = masked_lm_labels
    self.seq_level_labels = seq_level_labels

  def __str__(self):
    s = ""
    s += "tokens: %s\n" % (" ".join([str(x) for x in self.tokens]))
    s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
    s += "seq_level_labels: %s\n" % self.seq_level_labels
    s += "masked_lm_positions: %s\n" % (" ".join([str(x) for x in self.masked_lm_positions]))
    s += "masked_lm_labels: %s\n" % (" ".join([str(x) for x in self.masked_lm_labels]))
    s += "\n"
    return s

  def __repr__(self):
    return self.__str__()


def create_pretrain_epochs(args, vocab, rng):

    assert args.input_file or args.input_id_file
    input_file = args.input_id_file or args.input_file
    print("creating instance from {}".format(input_file))

    all_subjects = [[]]
    with open(input_file, "rb") as f_input:
        for line in f_input:
            line = line.decode("utf-8").strip()

            # Empty lines are used as document delimiters
            if not line:
                all_subjects.append([])
            if args.input_id_file:
                tokens = [int(t) for t in line.split()]
            else:
                tokens = vocab.tokenize(line)
            if tokens:
                all_subjects[-1].append(tokens)

    all_subjects = [s for s in all_subjects if s]
    rng.shuffle(all_subjects)

    pretrain_epochs = []
    for i in range(args.dupe_factor):
        pretrain_epoch = []
        for idx in tqdm(range(len(all_subjects)), desc="duplication {}/{}".format(i + 1, args.dupe_factor)):
            subject_pretrain_epoch = convert_subject_pretrain_epoch(args, all_subjects, idx, vocab, rng)
            pretrain_epoch.extend(subject_pretrain_epoch)
        rng.shuffle(pretrain_epoch)
        pretrain_epochs.append(pretrain_epoch)
    return pretrain_epochs

def convert_subject_pretrain_epoch(args, all_subjects, subject_id, vocab, rng):
    
    subject_data = all_subjects[subject_id]

    target_seq_length = args.max_seq_length
    if rng.random() < args.short_seq_prob:
        target_seq_length = rng.randint(2, args.max_seq_length)

    subject_pretrain_epoch = []
    for adm_id, admission in enumerate(subject_data):
        # admission level pretrain data

        if args.seq_level_task is not None:
            seq_level_label = True
        else:
            seq_level_label = None

        tokens = [vocab.word2idx["<CLS>"]]
        segment_ids = [0]

        for token in admission:
            tokens.append(token)
            segment_ids.append(0)

        if args.seq_level_task == "NAP":
            if len(admission) == 1 or adm_id == len(subject_data) - 1 or rng.random() < 0.5:
                # random pick next admission from dataset
                seq_level_label = False

                for _ in range(10):
                    random_subject_id = rng.randint(0, len(all_subjects) - 1)
                    if random_subject_id != subject_id:
                        break

                # in case picked random admission is from the same subject
                if random_subject_id == subject_id:
                    seq_level_label = True

                random_subject_data = all_subjects[random_subject_id]
                random_adm_id = rng.randint(0, len(random_subject_data) - 1)
                random_admission = random_subject_data[random_adm_id]

                for token in random_admission:
                    tokens.append(token)
                    segment_ids.append(1)
            else:
                # exact next admission
                seq_level_label = True
                for token in subject_data[adm_id + 1]:
                    tokens.append(token)
                    segment_ids.append(1)
            pass
        elif args.seq_level_task == "AOP":
            pass

        truncate_tokens(args, tokens, segment_ids)
        assert len(tokens) <= args.max_seq_length
        assert len(segment_ids) <= args.max_seq_length

        tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(args, tokens, vocab, rng)

        instance = TrainingInstance(tokens, segment_ids, masked_lm_positions, masked_lm_labels, seq_level_label)
        subject_pretrain_epoch.append(instance)

    return subject_pretrain_epoch

def truncate_tokens(args, tokens, segment_ids):
    
    while True:
        total_len = len(tokens)
        if total_len <= args.max_seq_length:
            break
        tokens.pop()
        segment_ids.pop()

MaskedLMInstance = collections.namedtuple("MaskedLMInstance", ["index", "label"])

def create_masked_lm_predictions(args, tokens, vocab, rng):
    
    vocab_words = list(vocab.word2idx.keys())

    candidate_inds = []
    for ind, token in enumerate(tokens):
        if not vocab.idx2type[token] in ["SPECIAL", "TYPE"]:
            candidate_inds.append(ind)

    rng.shuffle(candidate_inds)

    num_to_predict = min(args.max_predictions_per_seq, 
                         max(1, int(round(len(tokens) * args.masked_lm_prob))))

    output_tokens = copy(tokens)
    masked_tokens = []
    covered_inds = set()
    for ind in candidate_inds:
        if len(masked_tokens) >= num_to_predict:
            break
        if ind in covered_inds:
            continue

        masked_token = None

        if rng.random() < 0.8:
            # 80% of the time, replace word with <MASK>
            masked_token = vocab.word2idx["<MASK>"]
        else:
            if rng.random() < 0.5:
                # 10% of the time, do not change
                masked_token = tokens[ind]
            else:
                # 10% of the time, replace word with random word in same type
                typ = vocab.idx2type[tokens[ind]]
                type_word_ids = [i for i in vocab.idx2type.keys() if vocab.idx2type[i] == typ]
                masked_token = type_word_ids[rng.randint(0, len(type_word_ids) - 1)]

        masked_tokens.append(MaskedLMInstance(index=ind, label=tokens[ind]))
        output_tokens[ind] = masked_token

    masked_tokens = sorted(masked_tokens, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for masked_token in masked_tokens:
        masked_lm_positions.append(masked_token.index)
        masked_lm_labels.append(masked_token.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)

def write_epochs_to_file(args, vocab, epochs):

    dir_name = "{}_seq_len_{}_max_pred_{}_mlm_prob_{}_random_seed_{}_dupe_{}".format(
            args.save, 
            args.max_seq_length, 
            args.max_predictions_per_seq, 
            args.masked_lm_prob,
            args.random_seed, 
            args.dupe_factor
            )

    if os.path.exists(dir_name) and (os.listdir(dir_name) and any([i for i in os.listdir(dir_name)])):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(dir_name))

    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    for epoch, instances in enumerate(epochs):
        total_written = 0
        features = collections.OrderedDict()
        pad_id = vocab.word2idx["<PAD>"]
 
        num_instances = len(instances)
        features["input_ids"] = np.zeros([num_instances, args.max_seq_length], dtype="int32")
        features["input_mask"] = np.zeros([num_instances, args.max_seq_length], dtype="int32")
        features["segment_ids"] = np.zeros([num_instances, args.max_seq_length], dtype="int32")
        features["masked_lm_positions"] =  np.zeros([num_instances, args.max_predictions_per_seq], dtype="int32")
        features["masked_lm_ids"] = np.zeros([num_instances, args.max_predictions_per_seq], dtype="int32")
        features["seq_level_labels"] = np.zeros(num_instances, dtype="int32")

        for ind, instance in enumerate(tqdm(instances, desc="writing to file {}/{}".format(epoch + 1, len(epochs)))):
            input_ids = instance.tokens
            input_mask = [1] * len(input_ids)
            segment_ids = instance.segment_ids
            assert len(input_ids) <= args.max_seq_length

            # add padding
            while len(input_ids) < args.max_seq_length:
                input_ids.append(pad_id)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == args.max_seq_length
            assert len(input_mask) == args.max_seq_length
            assert len(segment_ids) == args.max_seq_length

            masked_lm_positions = instance.masked_lm_positions
            masked_lm_ids = instance.masked_lm_labels
            masked_lm_weights = [1.0] * len(masked_lm_ids)

            while len(masked_lm_positions) < args.max_predictions_per_seq:
                masked_lm_positions.append(0)
                masked_lm_ids.append(0)
                masked_lm_weights.append(0.0)

            seq_level_labels = instance.seq_level_labels or 0

            features["input_ids"][ind] = input_ids
            features["input_mask"][ind] = input_mask
            features["segment_ids"][ind] = segment_ids
            features["masked_lm_positions"][ind] = masked_lm_positions
            features["masked_lm_ids"][ind] = masked_lm_ids
            features["seq_level_labels"][ind] = seq_level_labels

            total_written += 1

        filename = "{}_seq_len_{}_max_pred_{}_mlm_prob_{}_random_seed_{}_dupe_{}_{}.hdf5".format(
                os.path.split(args.save)[-1], 
                args.max_seq_length, 
                args.max_predictions_per_seq, 
                args.masked_lm_prob,
                args.random_seed, 
                args.dupe_factor,
                epoch
                )
        filename = os.path.join(dir_name, filename)
        print("saving data to {}".format(filename))
        f = h5py.File(filename, 'w')
        f.create_dataset("input_ids", data=features["input_ids"], dtype='i4', compression='gzip')
        f.create_dataset("input_mask", data=features["input_mask"], dtype='i1', compression='gzip')
        f.create_dataset("segment_ids", data=features["segment_ids"], dtype='i1', compression='gzip')
        f.create_dataset("masked_lm_positions", data=features["masked_lm_positions"], dtype='i4', compression='gzip')
        f.create_dataset("masked_lm_ids", data=features["masked_lm_ids"], dtype='i4', compression='gzip')
        f.create_dataset("seq_level_labels", data=features["seq_level_labels"], dtype='i1', compression='gzip')
        f.flush()
        f.close()


def main(args):
    
    vocab = dill.load(open(args.vocab_file, "rb"))

    rng = random.Random(args.random_seed)

    pretrain_epochs = create_pretrain_epochs(args, vocab, rng)

    write_epochs_to_file(args, vocab, pretrain_epochs)


if __name__ == "__main__":
    args = parse_args()
    main(args)


