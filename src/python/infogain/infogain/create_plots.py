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
# import scienceplots
# plt.style.use(['science', 'ieee'])

# width as measured in inkscape
width = 8  # 3.487
height = width / 1.5

SHOW_BASELINES = True
HEADER_STR = "mppi"
HEADER_SUBSTR = "optimized-twoside"
PREFIX_STR = "Optimized-Twoside-Trials"

# TABLE formatting
TITLE = "Two-Sided Occlusions"
CAPTION = "Cars parked on both sides"


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
    legend_location="best",
    plot_name="default.plt",
):
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.15, bottom=0.16, right=0.99, top=0.97)

    sb.lineplot(
        x=x,
        y=y,
        hue=policy,
        data=data,
        hue_order=order,
        palette=colours,
        # linewidth=2.5,
    )

    ax.tick_params(axis="both", which="major", labelsize=16)
    handles, labels = ax.get_legend_handles_labels()
    # ax.set_yscale('log')

    ax.legend(
        handles=handles,
        labels=labels,
        loc=legend_location,
        title="Method",
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
            df = pd.read_csv(f)
            df.fillna("None", inplace=True)

            tags = f.split("-")  # get meta data
            method = tags[Tags.METHOD]
            cost_fn = tags[Tags.COST_FN]

            nulls = [True if type(a) != float else False for a in df["x"]]
            nulls = df[nulls]
            if len(nulls):
                print(f"WARNING: Corrupt data in {f} starting on/near line {np.argmax(nulls)}")
                continue

            try:
                df["t"] = df["t"].round(2)
            except KeyError:
                runs = set(df["run"])
                for run in runs:
                    df_run = df.loc[df["run"] == run].reset_index()
                    df.loc[df["run"] == run, "t"] = df_run.index.values * 0.1

            df_list.append(df)

    df = pd.concat(df_list, ignore_index=True, sort=False)

    # df_trim = []
    # for hue in hue_order:
    #     df_hue = df.loc[df['Solver'] == hue]
    #     df_hue = df_hue.iloc[-100000:-1000, :]
    #     df_trim.append(df_hue)
    # df = pd.concat(df_trim, ignore_index=True, sort=False)

    # for each run, create a diff (x) from nominal
    nom_df = df.loc[df["policy"] == "Nominal"]
    mean_nom_df_x = nom_df.groupby("t").x.mean()
    mean_nom_df_y = nom_df.groupby("t").y.mean()
    mean_nom_df_v = nom_df.groupby("t").v.mean()

    runs = set(df["run"])
    seeds = set(df["seed"])
    policies = [
        "Ours",
        "Higgins",
        "Anderson",
        "None",
        "Nominal",
    ]

    df_means = []
    for seed in seeds:
        for run in runs:
            for policy in policies:
                loc = (df["run"] == run) & (df["seed"] == seed) & (df["policy"] == policy)

                df_slice = pd.DataFrame(df.loc[loc])
                if len(df_slice) != 200:
                    continue

                df_slice["dx"] = df_slice["x"].to_numpy() - mean_nom_df_x.to_numpy()
                df_slice["dy"] = df_slice["y"].to_numpy() - mean_nom_df_y.to_numpy()
                df_slice["dv"] = df_slice["v"].to_numpy() - mean_nom_df_v.to_numpy()

                df_means.append(df_slice)
    df_means = pd.concat(df_means, ignore_index=True, sort=False)

    #    .dropna()

    write_table(
        df,
        policies=policies,
        policy_column="policy",
        columns=["x", "y", "v"],
        ranges=["last", "all", "all"],
        range_column="t",
        title=TITLE,
        caption=CAPTION,
        label="tbl:data",
    )

    sb.set_style(style="whitegrid")
    colours = [
        "darkorange",
        # "wheat",
        "lightsteelblue",
        "royalblue",
        "salmon",
        "silver",
        # "dodgerblue",
        # # 'bisque',
        # "linen",
    ]
    hue_order = policies

    df_slice = df_means[(df_means["t"] <= 25)]
    make_plot(
        "t",
        "x",
        policy="policy",
        data=df_slice,
        order=hue_order,
        colours=colours,
        x_label="Time (s)",
        y_label="X (m)",
        legend_location="lower left",
        plot_name=f"{PREFIX_STR}_x_plot.pdf",
    )

    make_plot(
        "t",
        "dx",
        policy="policy",
        data=df_slice,
        order=hue_order,
        colours=colours,
        x_label="Time (s)",
        y_label="X Error (m)",
        legend_location="lower left",
        plot_name=f"{PREFIX_STR}_dx_plot.pdf",
    )

    make_plot(
        "t",
        "y",
        policy="policy",
        data=df_slice,
        order=hue_order,
        colours=colours,
        x_label="Time (s)",
        y_label="Y (m)",
        legend_location="lower left",
        plot_name=f"{PREFIX_STR}_y_plot.pdf",
    )

    make_plot(
        "t",
        "dy",
        policy="policy",
        data=df_slice,
        order=hue_order,
        colours=colours,
        x_label="Time (s)",
        y_label="Y Error (m)",
        legend_location="lower left",
        plot_name=f"{PREFIX_STR}_dy_plot.pdf",
    )

    make_plot(
        "t",
        "v",
        policy="policy",
        data=df_slice,
        order=hue_order,
        colours=colours,
        x_label="Time (s)",
        y_label="V (m/s)",
        legend_location="lower left",
        plot_name=f"{PREFIX_STR}_v_plot.pdf",
    )

    make_plot(
        "t",
        "dv",
        policy="policy",
        data=df_slice,
        order=hue_order,
        colours=colours,
        x_label="Time (s)",
        y_label="V Error (m/s)",
        legend_location="lower left",
        plot_name=f"{PREFIX_STR}_dv_plot.pdf",
    )

    # # export_table(df, hues=hue_order)
    # # export_table2(df, hues=hue_order)

    colours = [
        "darkorange",
        "wheat",
        "lightsteelblue",
        "royalblue",
        # "lavender",
        # "slateblue",
        # "dodgerblue",
        # # 'bisque',
        # "linen",
    ]

    hue_order = [
        "Ours",
        "Nominal",
        "Higgins",
        "None",
    ]

    # fig, ax = plt.subplots()
    # fig.subplots_adjust(left=0.15, bottom=0.16, right=0.99, top=0.97)

    # sb.lineplot(
    #     x="x",
    #     y="y",
    #     hue="policy",
    #     data=df_slice,
    #     palette=colours,
    #     # linewidth=2.5,
    # )

    # ax.tick_params(axis="both", which="major", labelsize=16)
    # handles, labels = ax.get_legend_handles_labels()
    # # ax.set_yscale('log')

    # ax.legend(
    #     handles=handles,
    #     labels=labels,
    #     loc="bottom left",
    #     title="Method",
    #     title_fontsize=18,
    #     fontsize=16,
    # )
    # fig.set_size_inches(width, height)
    # fig.savefig(f"{PREFIX_STR}_xy_plot.pdf")

    # # # export_table(df, hues=hue_order)
    # # export_table2(df, hues=hue_order)


if __name__ == "__main__":
    path = "results/"
    files = [path + "/" + f for f in listdir(path) if isfile(join(path, f))]
    plot_comparison(files)
