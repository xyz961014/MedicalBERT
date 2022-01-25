import dill
import os
import ipdb
import argparse
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from functools import partial


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datas", type=str, nargs="+",
                        help="pkl data to mix")
    parser.add_argument("--save", type=str, default="mixed",
                        help="filename of saving data")
    return parser.parse_args()
   

def main(args):
    dfs = []
    for data_file in args.datas:
        df = pd.read_pickle(data_file)
        dfs.append(df)
    mixed_df = pd.concat(dfs)
    mixed_df.sort_values(by=["SUBJECT_ID", "HADM_ID", "DATETIME"], inplace=True)
    mixed_df.reset_index(drop=True, inplace=True)

    mixed_df.to_pickle("{}_data.pkl".format(args.save))


if __name__ == "__main__":
    args = parse_args()
    main(args)
