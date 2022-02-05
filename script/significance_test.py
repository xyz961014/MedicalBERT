import json
import os
import argparse
from pprint import pprint
import numpy as np
from scipy.stats import ttest_ind
import ipdb

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--res_file_a", type=str, required=True,
                        help="dllogger json file a")
    parser.add_argument("--res_file_b", type=str, required=True,
                        help="dllogger json file b")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    res_a = json.load(open(args.res_file_a, "r"))
    res_b = json.load(open(args.res_file_b, "r"))

    # jaccard
    p_ja = ttest_ind(res_a["jaccard"], res_b["jaccard"]).pvalue
    print("JACCARD p-value: {}".format(p_ja))

    # prauc
    p_prauc = ttest_ind(res_a["prauc"], res_b["prauc"]).pvalue
    print("PRAUC p-value: {}".format(p_prauc))

    # f1
    p_f1 = ttest_ind(res_a["f1"], res_b["f1"]).pvalue
    print("F1 p-value: {}".format(p_f1))
