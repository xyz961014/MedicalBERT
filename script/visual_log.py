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
    parser.add_argument("--fields", type=str, required=True, nargs="+",
                        help="field name to visualize")
    parser.add_argument("--title", type=str, default=None,
                        help="picture title")
    parser.add_argument("--save_pdf", action="store_true",
                        help="save the picture as pdf")

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


    fig, ax = plt.subplots(figsize=(20, 10))

    # title
    title = ("_VS_".join(args.fields)) if args.title is None else args.title
    plt.title(title, fontsize=30)

    # prepare data
    plots = []
    for field in args.fields:
        data_to_visualize = []
        for datum in data:
            for data_key in datum["data"].keys():
                if field == data_key:
                    step = datum["step"][1]
                    value = datum["data"][data_key]
                    data_to_visualize.append({"step": step, "value": value})
        x_name = "Step"
        y_name = field
        x_labels = np.array([item["step"] for item in data_to_visualize])
        y_labels = np.array([item["value"] for item in data_to_visualize])

        # draw

        plt.grid(axis='y' ,color='grey', linestyle='--', lw=0.5, alpha=0.5)
        plt.tick_params(axis='both', labelsize=24)

        plot = ax.plot(x_labels, y_labels, markersize=10, label=y_name, linewidth=2)
        plots.extend(plot)

        ax.set_xlabel(x_name, fontsize=24)
        ax.set_ylabel(y_name, fontsize=24)

    ax.legend(plots, [l.get_label() for l in plots], fontsize=20)

    plt.tight_layout()
    if args.save_pdf:
        plt.savefig("{}.pdf".format(title))
    
    plt.show()
