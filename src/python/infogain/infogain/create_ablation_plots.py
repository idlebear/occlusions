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

mpl.use("pdf")
# import scienceplots
# plt.style.use(['science', 'ieee'])

# width as measured in inkscape
width = 8  # 3.487
height = width / 1.5

SHOW_BASELINES = True
HEADER_STR = "weight_test"
# HEADER_SUBSTR = "Higgins"
HEADER_SUBSTR = "Ours"
PREFIX_STR = "Ours_Weight_Trials"

# def export_table2(df, hues):

#     print('%%%%%')
#     print('% Table Data for Uniform Distribution in metric space')
#     print('%')
#     print('\\begin{table*}')
#     print('\\caption{Mean, Median and Average Task Wait Times (m)}')
#     print('\\label{table:task-time-data}')
#     print('\\begin{center}')
#     print('\\begin{tabular}{@{} l  c c c  c c c  c c c  c c c  c c c @{}}')
#     print('\\toprule')
#     print(' & \\multicolumn{3}{c}{$\\rho=0.5$} & \\multicolumn{3}{c}{$\\rho=0.6$} & \\multicolumn{3}{c}{$\\rho=0.7$} & \\multicolumn{3}{c}{$\\rho=0.8$} & \\multicolumn{3}{c}{$\\rho=0.9$} \\\\')
#     print('Method & Mean  & $\\sigma$ & 95\% & Mean  & $\\sigma$ & 95\% & Mean  & $\\sigma$ & 95\% & Mean  & $\\sigma$ & 95\% & Mean & $\\sigma$ & 95\% \\\\')
#     print('\\midrule')

#     for index, hue in enumerate(hues):
#         s = hue
#         for rho in [0.5, 0.6, 0.7, 0.8, 0.9]:
#             df_slice = df[(df['Solver'] == hue) & (df['rho'] == rho)]
#             if USE_MONTREAL_DATA:
#                 s += ' & ' + \
#                     f"{(df_slice['wait_minutes'].mean()):5.1f} & {(df_slice['wait_minutes'].std()):5.1f} & {(df_slice['wait_minutes'].quantile(q=0.95)):5.1f}"
#             else:
#                 s += ' & ' + f"{(df_slice['Wait Time'].mean()):5.1f} & {(df_slice['Wait Time'].std()):5.1f} & {(df_slice['Wait Time'].quantile(q=0.95)):5.1f}"
#         s += "\\\\"
#         print(s)
#         if index == 1:
#             # hacky bit to insert a line after the second row
#             print('\\midrule')

#     print('\\bottomrule')
#     print('\\end{tabular}')
#     print('\\end{center}')
#     print('\\end{table*}')
#     print('%')
#     print('%%%%%')


# def export_table(df, hues):

#     print('%%%%%')
#     print('% Table Data for Uniform Distribution in metric space')
#     print('%')
#     print('\\begin{table}')
#     print('\\caption{Mean, Median and Average Task Wait Times (s)}')
#     print('\\label{table:task-time-data}')
#     print('\\begin{center}')
#     print('\\begin{tabular}{@{} l l c c c c @{}}')
#     print('\\toprule')
#     print('$\\rho$ & Method & Mean & Median & $\\sigma$ & 95\% \\\\')

#     for rho in [0.5, 0.6, 0.7, 0.8, 0.9]:
#         print('\\midrule')
#         rho_str = str(rho)
#         for hue in hues:
#             df_slice = df[(df['Solver'] == hue) & (df['rho'] == rho)]
#             if USE_MONTREAL_DATA:
#                 print(
#                     f"{rho_str} & {hue} & {(df_slice['wait_minutes'].mean()):5.1f} & {(df_slice['wait_minutes'].median()):5.1f} & {(df_slice['wait_minutes'].std()):5.1f} & {(df_slice['wait_minutes'].quantile(q=0.95)):5.1f} \\\\")
#             else:
#                 print(
#                     f"{rho_str} & {hue} & {(df_slice['Wait Time'].mean()):5.1f} & {(df_slice['Wait Time'].median()):5.1f} & {(df_slice['Wait Time'].std()):5.1f} & {(df_slice['Wait Time'].quantile(q=0.95)):5.1f} \\\\")
#             rho_str = ''

#     print('\\bottomrule')
#     print('\\end{tabular}')
#     print('\\end{center}')
#     print('\\end{table}')
#     print('%')
#     print('%%%%%')


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

    if y_limit is not None:
        ax.set_ylim(y_limit)

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

            df["weight-name"] = ["w:" + str(w) for w in df["visibility-weight"]]

            df_list.append(df)

    df = pd.concat(df_list, ignore_index=True, sort=False)

    WEIGHT_LIMIT = 1.5
    weights = [
        "w:" + str(w)
        for w in sorted(list(set(df["visibility-weight"])))
        if w < WEIGHT_LIMIT
    ]

    colours = [
        "darkorange",
        "wheat",
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
        # "teal",
        # "darkcyan",
        "cyan",
        # "khaki",
        # "darkkhaki",
        # "lightgray",
        # "lime",
    ]

    # hue_order = [
    #     "Ours",
    #     "Higgins",
    #     "None",
    # ]

    sb.set_style(style="whitegrid")

    for w in weights:
        df_slice = df[df["weight-name"] == w]
        print(
            f"{w}: x_max: {df_slice.x.max():6.3f}, y_min: {df_slice.y.min():5.3f}, y_max: {df_slice.y.max():5.3f}, mean v{df_slice.v.mean():5.3f}"
        )

    df_slice = df[(df["t"] <= 25) & (df["visibility-weight"] < WEIGHT_LIMIT)]
    make_plot(
        "t",
        "x",
        policy="weight-name",
        data=df_slice,
        order=weights,
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
        order=weights,
        colours=None,
        x_label="Time (s)",
        y_label="Y (m)",
        y_limit=[-3, 3],
        legend_location="lower left",
        plot_name=f"{PREFIX_STR}_y_plot.pdf",
    )

    make_plot(
        "t",
        "v",
        policy="weight-name",
        data=df_slice,
        order=weights,
        colours=None,
        x_label="Time (s)",
        y_label="V (m/s)",
        y_limit=[4, 8],
        legend_location="lower left",
        plot_name=f"{PREFIX_STR}_v_plot.pdf",
    )


if __name__ == "__main__":
    path = "results/"
    files = [path + "/" + f for f in listdir(path) if isfile(join(path, f))]
    plot_comparison(files)
