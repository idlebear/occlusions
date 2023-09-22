# Utility routines

import numpy as np
import pandas as pd


def write_table(
    df, policies, policy_column, columns, ranges, range_column, title="", caption="", label="", verbose=False
):
    num_columns = len(columns)
    if verbose:
        format_str = " c c c c "
        sub_columns = 4
        column_labels = " & $\\mu$ & $\\sigma$ & min & max "
    else:
        format_str = " c c "
        sub_columns = 2
        column_labels = " & $\\mu$ & $\\sigma$ "

    column_format = "\\begin{tabular}{@{} l"
    title_str1 = "  "
    title_str2 = "Method  "
    for i in range(num_columns):
        column_format += format_str
        title_str1 += f"& \\multicolumn{{{sub_columns}}}{{c}}{{ {columns[i]} }}"
        title_str2 += column_labels
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
                df_slice = df[(df[policy_column] == policy)]
            else:
                max_row = df[range_column].max()
                df_slice = df[(df[policy_column] == policy) & (df[range_column] == max_row)]

            if verbose:
                s += (
                    " & "
                    + f"{(df_slice[col].mean()):5.3f} & {(df_slice[col].std()):5.3f} & {(df_slice[col].min()):5.3f} & {(df_slice[col].max()):5.3f} "
                )
            else:
                s += " & " + f"{(df_slice[col].mean()):5.3f} & {(df_slice[col].std()):5.3f} "

        s += "\\\\"
        print(s)
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{center}")
    print("\\end{table*}")
    print("%")
    print("%%%%%")
