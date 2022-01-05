import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import xlrd
import json
import os
import argparse
import ipdb

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--log_files", type=str, required=True, nargs="+",
                        help="dllogger json files")
    parser.add_argument("--fields", type=str, required=True, nargs="+",
                        help="fields name to visualize")
    parser.add_argument("--title", type=str, default=None,
                        help="picture title")
    parser.add_argument("--save_pdf", action="store_true",
                        help="save the picture as pdf")
    parser.add_argument("--start_step", type=int, default=0,
                        help="start step")
    parser.add_argument("--end_step", type=int, default=1e9,
                        help="end step")

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

    if len(args.log_files) > 1:
        # compare training settings
        datas = {}
        for filename in args.log_files:
            parse_filename = os.path.split(filename)
            data_name = os.path.split(parse_filename[0])[-1]
            datas[data_name] = get_data_from_log(filename)
        title = args.fields[0] if args.title is None else args.title
    else:
        # compare fileds
        data = get_data_from_log(args.log_files[0])
        parse_filename = os.path.split(args.log_files[0])
        data_name = os.path.split(parse_filename[0])[-1]
        title = data_name if args.title is None else args.title


    fig, ax = plt.subplots(figsize=(20, 10))

    # title
    plt.title(title, fontsize=20)


    def plot_data(data, field, name=None):
        data_to_visualize = []
        for datum in data:
            for data_key in datum["data"].keys():
                if field == data_key:
                    step = datum["step"][1]
                    value = datum["data"][data_key]
                    if args.start_step <= step <= args.end_step:
                        data_to_visualize.append({"step": step, "value": value})
        x_name = "Step"
        y_name = name if name is not None else field
        x_labels = np.array([item["step"] for item in data_to_visualize])
        y_labels = np.array([item["value"] for item in data_to_visualize])

        # draw

        plt.grid(axis='y' ,color='grey', linestyle='--', lw=0.5, alpha=0.5)
        plt.tick_params(axis='both', labelsize=18)

        ax.set_xlabel(x_name, fontsize=18)
        ax.set_ylabel(field, fontsize=18)

        plot = ax.plot(x_labels, y_labels, markersize=10, label=y_name, linewidth=2)
        plots.extend(plot)


    
    plots = []
    if len(args.log_files) > 1:
        # compare training settings
        for data_name, data in datas.items():
            plot_data(data, args.fields[0], name=data_name)
    else:
        # compare fileds
        for field in args.fields:
            plot_data(data, field)
    ax.legend(plots, [l.get_label() for l in plots], fontsize=18)

    plt.tight_layout()
    if args.save_pdf:
        plt.savefig("{}.pdf".format(title))
    
    plt.show()
