import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import xlrd
import json
import os
import argparse
from pprint import pprint
import ipdb

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--log_file_a", type=str, required=True,
                        help="dllogger json file a")
    parser.add_argument("--log_file_b", type=str, required=True,
                        help="dllogger json file b")
    parser.add_argument("--field", type=str, required=True,
                        help="field to compare")
    parser.add_argument("--save_pdf", action="store_true",
                        help="save the picture as pdf")
    parser.add_argument("--top_k", type=int, default=10,
                        help="draw_top_k")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    def get_data_from_log(filename):
        data = []
        with open(filename, "rb") as f:
            for line in f:
                json_datum = json.loads(line.decode("utf-8")[5:-1])
                if "predict_accuracy" in " ".join(json_datum["data"].keys()):
                    if "eval_on_validation" in data[-1]["data"].keys():
                        if not  "valid_" in " ".join(json_datum["data"].keys()):
                            for key in list(json_datum["data"].keys()):
                                json_datum["data"]["valid_"+key] = json_datum["data"].pop(key)
                data.append(json_datum)
        return data

    def get_disease_data(data):
        disease_data = {}
        for datum in data:
            if "step" in datum.keys():
                if type(datum["step"]) == list and len(datum["step"]) > 2:
                    disease_id, disease_name = datum["step"][2]
                    disease_data[disease_name] = datum["data"]
        
        return disease_data


    def get_exp_name(filename):
        parse_filename = os.path.split(filename)
        parse_filename = os.path.split(parse_filename[0])
        exp_name = os.path.split(parse_filename[0])[-1]
        return exp_name

    name_a = get_exp_name(args.log_file_a)
    name_b = get_exp_name(args.log_file_b)

    data_a = get_data_from_log(args.log_file_a)
    data_b = get_data_from_log(args.log_file_b)

    disease_data_a = get_disease_data(data_a)
    disease_data_b = get_disease_data(data_b)


    compare_data = {}
    for key in disease_data_a.keys():
        if key in disease_data_b.keys():
            compare_data[key] = {}
            compare_data[key][name_a] = disease_data_a[key]
            compare_data[key][name_b] = disease_data_b[key]
            compare_data[key]["difference"] = disease_data_a[key][args.field] - disease_data_b[key][args.field]

    key_differences = sorted([(k, v["difference"]) for k, v in compare_data.items()], key=lambda x: x[1])

    for key, diff in key_differences:
        print("DISEASE: {}".format(key))
        print("{} DIFFERENCE: {:5.4f}".format(args.field, diff))
        print(name_a)
        for field, value in compare_data[key][name_a].items():
            print("{}: {:5.4f}".format(field, value), end=" ")
        print()
        print(name_b)
        for field, value in compare_data[key][name_b].items():
            print("{}: {:5.4f}".format(field, value), end=" ")
        print("\n" + "-" * 50)

    fig, ax = plt.subplots(figsize=(20, 10))

    # title
    title = "disease compare"
    plt.title(title, fontsize=20)

    x = list(range(args.top_k))
    total_width, n = 0.8, 2
    width = total_width / n
    list_name = [k[:20] for k, v in key_differences[:args.top_k]]
    list_value = [compare_data[k]["difference"] for k, v in key_differences[:args.top_k]]
    plt.bar(x, list_value, width=width, tick_label=list_name)

    plt.tight_layout()
    if args.save_pdf:
        plt.savefig("{}.pdf".format(title))
    
    plt.show()
