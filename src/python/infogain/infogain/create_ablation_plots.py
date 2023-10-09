from enum import IntEnum
import re
import sys
import argparse
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from distinctipy import distinctipy
from os import listdir
from os.path import isfile, join

from latex import write_table

mpl.use("pdf")
#    port scienceplots
# plt.style.use(['science', 'ieee'])

# width as measured in inkscape
width = 8  # 3.487
height = width / 1.5

SHOW_BASELINES = True
HEADER_STR = "weight-test"
HEADER_SUBSTR = "Higgins"
# HEADER_SUBSTR = "Ours"
# HEADER_SUBSTR = "Andersen"
PREFIX_STR = HEADER_SUBSTR + "_Weight_Trials"


class Tags(IntEnum):
    METHOD = 0
    COST_FN = 1
    STEPS = 2
    HORIZON = 3
    SAMPLES = 4
    SEED = 5


def make_plot(
    x,
    y,
    policy,
    data,
    order,
    colours,
    x_label=None,
    y_label=None,
    y_limit=None,
    legend_location="best",
    plot_name="default.plt",
):
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.15, bottom=0.16, right=0.99, top=0.97)

    sb.color_palette("viridis", as_cmap=True)
    sb.lineplot(
        x=x,
        y=y,
        hue=policy,
        data=data,
        hue_order=order,
        # palette=colours,
        # linewidth=2.5,
    )

    ax.tick_params(axis="both", which="major", labelsize=16)
    handles, labels = ax.get_legend_handles_labels()
    # ax.set_yscale('log')

    if y_limit is not None:
        ax.set_ylim(y_limit)

    ax.legend(
        handles=handles,
        labels=labels,
        loc=legend_location,
        title=None,
        title_fontsize=18,
        fontsize=16,
    )

    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    fig.set_size_inches(width, height)
    fig.savefig(plot_name)


def plot_comparison(files, mode="baselines"):
    sb.set_theme(style="whitegrid")
    sb.set()

    plt.rc("font", family="serif", serif="Times")
    plt.rc("text", usetex=True)
    plt.rc("xtick", labelsize=16)
    plt.rc("ytick", labelsize=16)
    plt.rc("axes", labelsize=12)

    df_list = []
    for f in files:
        if HEADER_STR in f and HEADER_SUBSTR in f:
            try:
                df = pd.read_csv(f)
            except pd.errors.EmptyDataError:
                continue

            df.fillna("None", inplace=True)

            tags = f.split("-")  # get meta data
            method = tags[Tags.METHOD]
            cost_fn = tags[Tags.COST_FN]

            try:
                df["t"] = df["t"].round(2)
            except KeyError:
                runs = set(df["run"])
                for run in runs:
                    df_run = df.loc[df["run"] == run].reset_index()
                    df.loc[df["run"] == run, "t"] = df_run.index.values * 0.1

            df["y"] += 2
            df["weight-name"] = [str(w) for w in df["visibility-weight"]]

            df_list.append(df)

    df = pd.concat(df_list, ignore_index=True, sort=False)

    # df = df[(df["visibility-weight"] >= 0.05) & (df["visibility-weight"] <= 0.054)]
    WEIGHT_LIMIT = 6000
    weights = [str(w) for w in sorted(list(set(df["visibility-weight"]))) if w < WEIGHT_LIMIT]
    # weights = [0.02, 0.03, 0.04, 0.05, 0.1][::-1]
    str_weights = [str(w) for w in weights]

    colours = [
        # "darkorange",
        # "wheat",
        "lightsteelblue",
        "royalblue",
        "lavender",
        "slateblue",
        "dodgerblue",
        "bisque",
        "linen",
        "yellow",
        # "coral",
        # "orangered",
        "red",
        # "indianred",
        # "lightcoral",
        # "gold",
        "teal",
        # "darkcyan",
        "cyan",
        # "khaki",
        "darkkhaki",
        # "lightgray",
        # "lime",
    ]

    # hue_order = [
    #     "Ours",
    #     "Higgins",
    #     "None",
    # ]

    sb.set_style(style="whitegrid")

    write_table(
        df,
        policies=str_weights,
        policy_column="weight-name",
        columns=["x", "y", "v"],
        ranges=["last", "all", "all"],
        range_column="t",
        title="Weight Trials",
        caption="Results of varying the visibility weight while holding all other optimization factors constant",
        label="tbl:data",
    )

    df_slice = df[(df["t"] <= 25)]
    make_plot(
        "t",
        "x",
        policy="weight-name",
        data=df_slice,
        order=str_weights,
        colours=colours,
        x_label="Time (s)",
        y_label="X (m)",
        y_limit=None,
        legend_location="lower left",
        plot_name=f"{PREFIX_STR}_x_plot.pdf",
    )

    make_plot(
        "t",
        "y",
        policy="weight-name",
        data=df_slice,
        order=str_weights,
        colours=colours,
        x_label="Time (s)",
        y_label="Y (m)",
        y_limit=None,
        legend_location="lower left",
        plot_name=f"{PREFIX_STR}_y_plot.pdf",
    )

    make_plot(
        "t",
        "v",
        policy="weight-name",
        data=df_slice,
        order=str_weights,
        colours=colours,
        x_label="Time (s)",
        y_label="V (m/s)",
        y_limit=[5, 8],
        legend_location="lower left",
        plot_name=f"{PREFIX_STR}_v_plot.pdf",
    )


if __name__ == "__main__":
    path = "results/"

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "-p",
        "--path",
        default="results",
        type=str,
        help="path to data files",
    )
    argparser.add_argument(
        "-m",
        "--method",
        default="weight-test",
        type=str,
        help="weight-test group name",
    )
    argparser.add_argument(
        "-s",
        "--sub-group",
        default="oneside",
        type=str,
        help="type to graph: Higgins, Proposed, Andersen",
    )

    args = argparser.parse_args()

    HEADER_STR = args.method
    HEADER_SUBSTR = args.sub_group
    PREFIX_STR = HEADER_STR + "-" + HEADER_SUBSTR + "-Weight-Trials"

    files = []
    for f in listdir(args.path):
        file_path = join(args.path, f)
        if isfile(file_path):
            files.append(file_path)

    plot_comparison(files)
