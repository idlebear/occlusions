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
HEADER_SUBSTR = "optimized"
PREFIX_STR = "Optimized-Trials"

# TABLE formatting
TITLE = "Optimized One-Sided Occlusions"
CAPTION = "Cars parked on one side only"

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
                print(
                    f"WARNING: Corrupt data in {f} starting on/near line {np.argmax(nulls)}"
                )
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
    policies = ["None", "Higgins", "Ours", "Nominal"]

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

    def write_table(
        df, policies, columns, ranges, range_column, title="", caption="", label=""
    ):
        num_columns = len(columns)
        column_format = "\\begin{tabular}{@{} l"
        title_str1 = "  "
        title_str2 = "Method  "
        for i in range(num_columns):
            column_format += " c c c c "
            title_str1 += f"& \\multicolumn{{4}}{{c}}{{ {columns[i]} }}"
            title_str2 += f" & $\\mu$ & $\\sigma$ & min & max "
        title_str1 += "\\\\"
        title_str2 += "\\\\"

        print("%%%%%")
        print(f"% Table Data ({title})")
        print("%")
        print("\\begin{table*}")
        print(f"\\caption{{ {caption} }}")
        print(f"\\label{{ {label} }}")
        print("\\begin{center}")
        column_format += " @{}}"
        print(column_format)
        print("\\toprule")

        print(title_str1)
        print(title_str2)
        print("\\midrule")

        for index, policy in enumerate(policies):
            s = policy
            for col, ran in zip(columns, ranges):
                if ran == "all":
                    df_slice = df[(df["policy"] == policy)]
                else:
                    max_row = df[range_column].max()
                    df_slice = df[
                        (df["policy"] == policy) & (df[range_column] == max_row)
                    ]

                s += (
                    " & "
                    + f"{(df_slice[col].mean()):5.3f} & {(df_slice[col].std()):5.3f} & {(df_slice[col].min()):5.3f} & {(df_slice[col].max()):5.3f} "
                )
            s += "\\\\"
            print(s)
        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\end{center}")
        print("\\end{table*}")
        print("%")
        print("%%%%%")

    write_table(
        df,
        policies=policies,
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
        "wheat",
        "lightsteelblue",
        "royalblue",
        # "lavender",
        # "slateblue",
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
