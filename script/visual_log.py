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
    parser.add_argument("--names", type=str, nargs="+",
                        help="setting names")
    parser.add_argument("--fields", type=str, required=True, nargs="+",
                        help="fields name to visualize")
    parser.add_argument("--title", type=str, default=None,
                        help="picture title")
    parser.add_argument("--y_name", type=str, default=None,
                        help="picture y_name")
    parser.add_argument("--save_pdf", action="store_true",
                        help="save the picture as pdf")
    parser.add_argument("--average", action="store_true",
                        help="average on fields")
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
        for i, filename in enumerate(args.log_files):
            if len(args.names) > i:
                data_name = args.names[i]
            else:
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


    fig, ax = plt.subplots(figsize=(14, 10))

    # title
    plt.title(title, fontsize=40)


    def plot_data(data, field, name=None, y_name=None):
        data_to_visualize = []
        for datum in data:
            for data_key in datum["data"].keys():
                if field == data_key:
                    step = datum["step"][1]
                    value = datum["data"][data_key]
                    if args.start_step <= step <= args.end_step:
                        data_to_visualize.append({"step": step, "value": value})
        x_name = "Step"
        y_name = y_name if y_name is not None else field
        plot_name = name if name is not None else field
        x_labels = np.array([item["step"] for item in data_to_visualize])
        y_labels = np.array([item["value"] for item in data_to_visualize])

        # draw

        plt.grid(axis='y' ,color='grey', linestyle='--')
        plt.tick_params(axis='both', labelsize=36)

        ax.set_xlabel(x_name, fontsize=36)
        ax.set_ylabel(y_name, fontsize=36)

        plot = ax.plot(x_labels, y_labels, markersize=10, label=plot_name, linewidth=5)
        plots.extend(plot)


    
    plots = []
    if len(args.log_files) > 1:
        # compare training settings
        for data_name, data in datas.items():
            # compute average if needed
            if args.average and len(args.fields) > 1:
                for i, datum in enumerate(data):
                    data_to_average = []
                    for data_key in datum["data"].keys():
                        if data_key in args.fields:
                            data_to_average.append(datum["data"][data_key])
                    if len(data_to_average) == len(args.fields):
                        data[i]["data"]["AVERAGE"] = np.mean(data_to_average)
                plot_data(data, "AVERAGE", name=data_name, y_name=args.y_name)
            else:
                plot_data(data, args.fields[0], name=data_name, y_name=args.y_name)
    else:
        # compare fileds
        for field in args.fields:
            plot_data(data, field, y_name=args.y_name)
    ax.legend(plots, [l.get_label() for l in plots], fontsize=30)

    plt.tight_layout()
    if args.save_pdf:
        plt.savefig("{}.pdf".format(title))
    
    plt.show()
