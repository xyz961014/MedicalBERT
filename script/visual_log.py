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
    parser.add_argument("--field", type=str, required=True,
                        help="field name to visualize")
    parser.add_argument("--title", type=str, default=None,
                        help="picture title")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    data = []
    with open(args.log_file, "rb") as f:
        for line in f:
            json_datum = json.loads(line.decode("utf-8")[5:-1])
            if "predict_accuracy" in " ".join(json_datum["data"].keys()):
                if "eval_on_validation" in data[-1]["data"].keys():
                    if not  "valid_" in " ".join(json_datum["data"].keys()):
                        for key in list(json_datum["data"].keys()):
                            json_datum["data"]["valid_"+key] = json_datum["data"].pop(key)
            data.append(json_datum)

    data_to_visualize = []
    for datum in data:
        for data_key in datum["data"].keys():
            if args.field == data_key:
                step = datum["step"][1]
                value = datum["data"][data_key]
                data_to_visualize.append({"step": step, "value": value})

    # prepare data
    x_name = "Step"
    y_name = args.field
    x_labels = np.array([item["step"] for item in data_to_visualize])
    y_labels = np.array([item["value"] for item in data_to_visualize])

    fig, ax = plt.subplots(figsize=(20, 10))
    
    # title
    title = (args.field) if args.title is None else args.title
    plt.title(title, fontsize=30)

    # draw

    plt.grid(axis='y' ,color='grey', linestyle='--', lw=0.5, alpha=0.5)
    plt.tick_params(axis='both', labelsize=24)

    plot = ax.plot(x_labels, y_labels, 'blue', markersize=10, label=y_name, linewidth=2)

    ax.set_xlabel(x_name, fontsize=24)
    ax.set_ylabel(y_name, fontsize=24)

    ax.legend(plot, [l.get_label() for l in plot], fontsize=24)

    plt.tight_layout()
    plt.savefig("{}.pdf".format(args.field))
    
    plt.show()
