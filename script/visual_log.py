import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import xlrd
import json
import argparse
import ipdb

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--log_file", type=str, required=True,
                        help="dllogger json file")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    ipdb.set_trace()
