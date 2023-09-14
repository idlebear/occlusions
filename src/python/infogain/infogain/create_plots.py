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
HEADER_STR = "mppi"
HEADER_SUBSTR = ""
PREFIX_STR = "Trials"

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

            try:
                df["t"] = df["t"].round(2)
            except KeyError:
                runs = set(df["run"])
                for run in runs:
                    df_run = df.loc[df["run"] == run].reset_index()
                    df.loc[df["run"] == run, "t"] = df_run.index.values * 0.1

            df_list.append(df)

    df = pd.concat(df_list, ignore_index=True, sort=False)

    colours = [
        "darkorange",
        "wheat",
        "lightsteelblue",
        # "royalblue",
        # "lavender",
        # "slateblue",
        # "dodgerblue",
        # # 'bisque',
        # "linen",
    ]

    hue_order = [
        "Ours",
        "Higgins",
        "None",
    ]

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
    policies = ["None", "Higgins", "Ours"]

    df_means = []
    for seed in seeds:
        for run in runs:
            for policy in policies:
                loc = (
                    (df["run"] == run) & (df["seed"] == seed) & (df["policy"] == policy)
                )

                df_slice = pd.DataFrame(df.loc[loc])
                if len(df_slice) != 200:
                    continue

                df_slice["dx"] = df_slice["x"].to_numpy() - mean_nom_df_x.to_numpy()
                df_slice["dy"] = df_slice["y"].to_numpy() - mean_nom_df_y.to_numpy()
                df_slice["dv"] = df_slice["v"].to_numpy() - mean_nom_df_v.to_numpy()

                df_means.append(df_slice)
    df_means = pd.concat(df_means, ignore_index=True, sort=False)

    #    .dropna()

    sb.set_style(style="whitegrid")
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.15, bottom=0.16, right=0.99, top=0.97)

    df_slice = df_means[(df_means["t"] <= 25)]

    sb.lineplot(
        x="t",
        y="dx",
        hue="policy",
        data=df_slice,
        hue_order=hue_order,
        palette=colours,
        # linewidth=2.5,
    )

    ax.tick_params(axis="both", which="major", labelsize=16)
    handles, labels = ax.get_legend_handles_labels()
    # ax.set_yscale('log')

    ax.legend(
        handles=handles,
        labels=labels,
        loc="lower right",
        title="Method",
        title_fontsize=18,
        fontsize=16,
    )
    fig.set_size_inches(width, height)
    fig.savefig(f"{PREFIX_STR}_dx_plot.pdf")

    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.15, bottom=0.16, right=0.99, top=0.97)

    sb.lineplot(
        x="t",
        y="dy",
        hue="policy",
        data=df_slice,
        hue_order=hue_order,
        palette=colours,
        linewidth=2.5,
    )

    ax.tick_params(axis="both", which="major", labelsize=16)
    handles, labels = ax.get_legend_handles_labels()
    # ax.set_yscale('log')

    ax.legend(
        handles=handles,
        labels=labels,
        loc="upper left",
        title="Method",
        title_fontsize=18,
        fontsize=16,
    )
    fig.set_size_inches(width, height)
    fig.savefig(f"{PREFIX_STR}_dy_plot.pdf")

    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.15, bottom=0.16, right=0.99, top=0.97)

    sb.lineplot(
        x="t",
        y="dv",
        hue="policy",
        data=df_slice,
        hue_order=hue_order,
        palette=colours,
        linewidth=2.5,
    )

    ax.tick_params(axis="both", which="major", labelsize=16)
    handles, labels = ax.get_legend_handles_labels()
    # ax.set_yscale('log')

    ax.legend(
        handles=handles,
        labels=labels,
        loc="lower right",
        title="Method",
        title_fontsize=18,
        fontsize=16,
    )
    fig.set_size_inches(width, height)
    fig.savefig(f"{PREFIX_STR}_dv_plot.pdf")

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
