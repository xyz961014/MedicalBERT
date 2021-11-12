import argparse
import json
import dill
import os
import sys
from pprint import pprint

curr_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(curr_path, ".."))
sys.path.append(os.path.join(curr_path, "../data"))

from data import PretrainVocab

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--vocab_file", type=str, required=True,
                        help="file contain vocab")
    parser.add_argument("--save", type=str, default="model_config",
                        help="filename to save model config")

    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.1,
                        help="dropout rate on attention probabilities")
    parser.add_argument("--hidden_act", type=str, default="gelu",
                        help="activation function in hidden layer")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1,
                        help="dropout rate on hidden layer")
    parser.add_argument("--hidden_size", type=int, default=768,
                        help="hidden size")
    parser.add_argument("--intermediate_size", type=int, default=3072,
                        help="intermediate size")
    parser.add_argument("--initializer_range", type=float, default=0.02,
                        help="range of initializer")
    parser.add_argument("--max_position_embeddings", type=int, default=512,
                        help="max value of position embeddings")
    parser.add_argument("--num_attention_heads", type=int, default=16,
                        help="number of attention heads")
    parser.add_argument("--num_hidden_layers", type=int, default=12,
                        help="number of hidden layers")
    parser.add_argument("--only_mlm", action="store_true",
                        help="disable Pooler in Bert when only train with MLM")

    return parser.parse_args()


def main(args):

    vocab = dill.load(open(args.vocab_file, "rb"))

    vocab_size = vocab.vocab_size
    type_vocab_size = len(set(vocab.idx2type.values()))

    config_json = {
              "attention_probs_dropout_prob": args.attention_probs_dropout_prob,
              "hidden_act": args.hidden_act,
              "hidden_dropout_prob": args.hidden_dropout_prob,
              "hidden_size": args.hidden_size,
              "initializer_range": args.initializer_range,
              "intermediate_size": args.intermediate_size,
              "max_position_embeddings": args.max_position_embeddings,
              "num_attention_heads": args.num_attention_heads,
              "num_hidden_layers": args.num_hidden_layers,
              "with_pooler": not args.only_mlm,
              "type_vocab_size": type_vocab_size,
              "vocab_size": vocab_size
                  }

    with open("{}.json".format(args.save), "w") as f:
        json.dump(config_json, f, indent=4)

    print("Model config created with")
    pprint(config_json)


if __name__ == "__main__":
    args = parse_args()
    main(args)
