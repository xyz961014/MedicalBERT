import dill
import os
import argparse
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import ipdb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="_data.pkl",
                        help="data path of pkl file")
    parser.add_argument("--static_data_path", type=str, default="./processed_data/static_data.pkl",
                        help="preprocessed data path of pkl file")
    parser.add_argument("--data_dir", type=str, default="/home/xyz/Documents/Datasets/mimiciii-1.4",
                        help="data dir of MIMIC-III CSV file")
    parser.add_argument("--save", type=str, default="final_with_static",
                        help="filename of saving vocabulary and records")
    return parser.parse_args()

def main(args):

    df = pd.read_pickle(args.data_path)
    data_df = pd.read_pickle(args.static_data_path)
    data_df.drop(data_df[data_df["TYPE"] == "DIAG"].index, inplace=True)
    data_df.drop(data_df[data_df["TYPE"] == "PROC"].index, inplace=True)
    data_df.drop(data_df[data_df["TYPE"] == "MED"].index, inplace=True)

    # remove DISCHARGE_LOCATION and DEATH
    data_df.drop(data_df[data_df["TYPE"] == "DISCHARGE_LOCATION"].index, inplace=True)
    data_df.drop(data_df[data_df["TYPE"] == "DEATH"].index, inplace=True)

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        hadm_id = row["HADM_ID"]
        hadm_static_df = data_df[data_df["HADM_ID"] == hadm_id]
        if len(hadm_static_df) == 0:
            continue
        hadm_static_df = hadm_static_df.reset_index()
        for static_id, static_row in hadm_static_df.iterrows():
            typ = static_row["TYPE"]
            val = static_row["STATIC_VALUE"]
            df.loc[idx, typ] = val

    df.to_pickle("{}_data.pkl".format(args.save))


if __name__ == "__main__":
    args = parse_args()
    main(args)
