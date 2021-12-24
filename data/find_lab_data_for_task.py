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
    parser.add_argument("--lab_data_path", type=str, default="./processed_data/pretrain_lab_data.pkl",
                        help="preprocessed data path of pkl file")
    parser.add_argument("--data_dir", type=str, default="/home/xyz/Documents/Datasets/mimiciii-1.4",
                        help="data dir of MIMIC-III CSV file")
    parser.add_argument("--save", type=str, default="final_with_lab",
                        help="filename of saving vocabulary and records")
    return parser.parse_args()

def filter_first24h_lab(lab_df):
    first_datetime = lab_df.loc[0, "DATETIME"]
    first_date = first_datetime.date().strftime("%Y-%m-%d")
    first_day_lab_dfs = []
    for datetime in list(lab_df["DATETIME"].dropna().unique()):
        date = datetime.astype("str").split("T")[0]
        if date == first_date:
            datetime_df = lab_df[lab_df["DATETIME"] == datetime]
            first_day_lab_dfs.append(datetime_df)
    first_day_lab_df = pd.concat(first_day_lab_dfs)
    return first_day_lab_df.reset_index()


def main(args):

    df = pd.read_pickle(args.data_path)
    data_pd = pd.read_pickle(args.lab_data_path)
    lab_pd = data_pd[data_pd["TYPE"] == "LAB"]

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        hadm_id = row["HADM_ID"]
        hadm_lab_df = lab_pd[lab_pd["HADM_ID"] == hadm_id]
        if len(hadm_lab_df) == 0:
            continue
        hadm_lab_df = hadm_lab_df.reset_index()
        hadm_lab_df = filter_first24h_lab(hadm_lab_df)

        lab_id_list = hadm_lab_df["ITEMID"].tolist()
        lab_id_list = [int(l) for l in lab_id_list]
        lab_value_list = hadm_lab_df["VALUENUM"].tolist()
        lab_unit_list = hadm_lab_df["VALUEUOM"].tolist()
        lab_flag_list = hadm_lab_df["FLAG"].tolist()

        df.loc[idx, "LAB_ID"] = json.dumps(lab_id_list)
        df.loc[idx, "LAB_VALUE"] = json.dumps(lab_value_list)
        df.loc[idx, "LAB_UNIT"] = json.dumps(lab_unit_list)
        df.loc[idx, "LAB_FLAG"] = json.dumps(lab_flag_list)

    df.to_pickle("{}_data.pkl".format(args.save))


if __name__ == "__main__":
    args = parse_args()
    main(args)
