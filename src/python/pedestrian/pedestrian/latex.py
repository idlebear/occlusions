"""
This module provides functionality to generate LaTeX tables from pandas DataFrames.

Functions:
    write_table(df, categories, columns, column_properties, title="", caption="", label="", df2=None, columns2=None, column_properties2=None, output_file_path=None, columns_spec=None, columns_spec2=None):
        Generates a LaTeX table from the provided DataFrame(s) and prints it to the console.

        Parameters:
            df (pd.DataFrame): The primary DataFrame containing the data to be tabulated.
            categories (dict): A dictionary defining the hierarchical categories for the table rows.
                The dictionary should have keys "L1", "L2", etc. for the category levels.
                Each level should contain:
                    - 'name' (str): The name of the category level.
                    - 'column' (str): The column name in the DataFrame corresponding to this category level.
                    - 'labels' (list): The list of labels for this category level.
                    - 'proper_name' (dict, optional): A dictionary mapping labels to their proper names.
            columns (dict): A dictionary of column names from the DataFrame to be included in the table.
                Keys are DataFrame column names, values are display names. Deprecated when using columns_spec.
            column_properties (list): A list of dictionaries defining properties for each column.
                Each dictionary should contain:
                    - 'type' (str): Display type - "std" (alias "mean_std"), "sum", "ci", "rng", "single"
                    - 'highlight' (str): Highlighting rule - "min", "max", "none"
                    - 'decimals' (int, optional): Number of decimal places to display for both subcolumns.
                Deprecated when using columns_spec.
            title (str, optional): The title of the table. Default is an empty string.
            caption (str, optional): The caption for the table. Default is an empty string.
            label (str, optional): The label for the table. Default is an empty string.
            df2 (pd.DataFrame, optional): An optional secondary DataFrame for additional data columns. Default is None.
            columns2 (dict, optional): Column names from the secondary DataFrame to be included in the table. Default is None.
            column_properties2 (list, optional): Properties for columns2. Default is None.
            output_file_path (str, optional): File path to write the LaTeX output to.
            columns_spec (dict, optional): Combined column specification. Keys are true column names (e.g., "actual_visibility"),
                values are dicts with: { 'display': str, 'type': str, 'highlight': str, 'decimals': int }.
            columns_spec2 (dict, optional): Combined column specification for df2.

        Returns:
            None
"""

import numpy as np
from scipy import stats
from collections import OrderedDict

try:
    import pandas as pd
except ImportError:  # pragma: no cover - exercised in minimal test envs
    class _PandasFallback:
        @staticmethod
        def isna(value):
            try:
                return bool(np.isnan(value))
            except TypeError:
                return value is None

        @staticmethod
        def Series(data):
            return np.asarray(list(data), dtype=float)

    pd = _PandasFallback()


# BUGBUG - the data is fixed into 3 levels and exactly 2 tables (the second table is optional).  At some
#          point, this should be generalized.


# Consider using the tablularx package to make the table fit the page width instead of resizebox as the latter
# can make the font size inconsistent.

# Consider using the booktabs package for better looking tables.


# Helper function (can be defined inside write_table or globally)
def format_latex_num(value, fmt_str="5.2f", nan_rep="-"):
    if pd.isna(value):  # Using pandas.isna for robustness
        return nan_rep
    return f"{value:{fmt_str}}"


def calculate_confidence_interval(data, confidence=0.95):
    """
    Calculate the confidence interval for a dataset.
    Returns the half-width of the confidence interval.
    """
    data = np.asarray(list(pd.Series(data)), dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0 or len(data) < 2:
        return np.nan

    n = len(data)
    sem = stats.sem(data)  # Standard error of the mean
    h = sem * stats.t.ppf((1 + confidence) / 2.0, n - 1)  # Half-width of CI
    return h


# function to replace '_' with '\\_'
def r_u(x):
    """
    Replace underscores in a string with LaTeX-safe underscores.  A double underscore is
    replaced with a single underscore for the odd case where it is necessary.
    """
    if not isinstance(x, str):
        return x
    return x.replace("_", "\\_").replace("\\_\\_", "_")


def _write_line(f, line_content):
    if f:
        f.write(line_content + "\n")
    else:
        print(line_content)


def _generate_category_combinations(categories, category_levels):
    """Generate all combinations of category values for all levels."""
    import itertools

    combinations = []
    labels_by_level = []

    for level_key in category_levels:
        labels_by_level.append(categories[level_key]["labels"])

    # Generate all combinations using itertools.product
    for combo in itertools.product(*labels_by_level):
        combinations.append(combo)

    return combinations


def _build_dataframe_filter(df, categories, category_levels, combination):
    """Build a filter for the dataframe based on the category combination."""
    filter_conditions = []

    for i, level_key in enumerate(category_levels):
        category_value = combination[i]
        if category_value != "all":
            column_name = categories[level_key]["column"]
            filter_conditions.append(df[column_name] == category_value)

    if filter_conditions:
        combined_filter = filter_conditions[0]
        for condition in filter_conditions[1:]:
            combined_filter = combined_filter & condition
        return df[combined_filter]
    else:
        return df


def _get_category_label(categories, level_key, value):
    """Get the proper label for a category value."""
    if categories[level_key].get("proper_name") is not None:
        return categories[level_key]["proper_name"].get(value, str(value))
    else:
        return str(value)


def _format_row_labels(label_set, previous_label_set, num_levels):
    """Format row labels, showing only values that differ from the previous row."""
    if previous_label_set is None:
        # First row - show all labels
        return " & ".join(label_set)

    formatted_labels = []
    higher_level_changed = False

    for i in range(num_levels):
        if i < len(label_set):
            # Check if this level or any higher level changed
            current_level_changed = label_set[i] != previous_label_set[i]

            if current_level_changed or higher_level_changed:
                # Show label if this level changed OR if any higher-level category changed
                formatted_labels.append(label_set[i])
                # Once a level changes, all subsequent levels should be shown
                higher_level_changed = True
            else:
                # Same as previous row and no higher-level changes - show empty cell
                formatted_labels.append("")
        else:
            formatted_labels.append("")

    return " & ".join(formatted_labels)


def _get_default_decimals(col_type: str) -> int:
    if col_type == "sum":
        return 0
    return 2


def _normalize_col_type(col_type: str) -> str:
    if not col_type:
        return "std"
    if col_type == "mean_std":
        return "std"
    return col_type


def _get_subcolumn_count(col_type: str) -> int:
    return 1 if _normalize_col_type(col_type) == "single" else 2


def _get_subcolumn_labels(col_type: str):
    col_type = _normalize_col_type(col_type)
    if col_type == "single":
        return [""]
    if col_type == "rng":
        return ["min", "max"]
    if col_type == "sum":
        return ["num", "total"]
    if col_type == "ci":
        return ["$\\mu$", "CI"]
    return ["$\\mu$", "$\\sigma$"]


def _safe_argmin(values):
    if np.all(np.isnan(values)):
        return -1
    return int(np.nanargmin(values))


def _safe_argmax(values):
    if np.all(np.isnan(values)):
        return -1
    return int(np.nanargmax(values))


def write_table(
    df,
    categories,
    columns=None,
    column_properties=None,
    title="",
    caption="",
    label="",
    df2=None,
    columns2=None,
    column_properties2=None,
    output_file_path=None,
    columns_spec=None,
    columns_spec2=None,
):
    file_handle = None
    if output_file_path:
        try:
            file_handle = open(output_file_path, "w", encoding="utf-8")
        except IOError as e:
            print(f"Error: Could not open file {output_file_path} for writing: {e}")
            return

    # If a combined spec is provided, derive columns and column_properties from it (preserve insertion order)
    if columns_spec is not None and len(columns_spec) > 0:
        ordered = OrderedDict(columns_spec)
        # columns maps true column name -> display string
        columns = OrderedDict((k, v.get("display", k)) for k, v in ordered.items())
        # column_properties is a list aligned with columns order
        column_properties = []
        for k, v in ordered.items():
            col_type = _normalize_col_type(v.get("type", "std"))
            prop = {
                "type": col_type,
                "highlight": v.get("highlight", "none"),
                "decimals": v.get("decimals", _get_default_decimals(col_type)),
            }
            column_properties.append(prop)

    if columns_spec2 is not None and len(columns_spec2) > 0:
        ordered2 = OrderedDict(columns_spec2)
        columns2 = OrderedDict((k, v.get("display", k)) for k, v in ordered2.items())
        column_properties2 = []
        for k, v in ordered2.items():
            col_type = _normalize_col_type(v.get("type", "std"))
            prop = {
                "type": col_type,
                "highlight": v.get("highlight", "none"),
                "decimals": v.get("decimals", _get_default_decimals(col_type)),
            }
            column_properties2.append(prop)

    # Determine number of category levels dynamically
    category_levels = sorted([key for key in categories.keys() if key.startswith("L")])
    num_levels = len(category_levels)

    if num_levels == 0:
        raise ValueError("No category levels found in categories dictionary")

    num_columns = len(columns) if columns is not None else 0
    num_columns2 = 0
    column_meta = []
    total_sub_columns = 0
    # Removed unused header label templates in favor of explicit cell lists
    column_format = "\\begin{tabular}{@{} l "

    if column_properties is not None and columns is not None:
        # Build column format for category levels
        # Start with one 'l' already present in column_format initialization; add the remaining
        for _ in range(1, num_levels):
            column_format += " l "
        title_row1_cells = [""] * num_levels  # blank cells for category columns
        title_row2_cells = [
            r_u(categories[level_key]["name"]) for level_key in category_levels
        ]

        sep_str = ""
        column_num = num_levels + 1

        # Add primary df column groups
        for i, column in enumerate(columns.keys()):
            col_props = (
                column_properties[i]
                if i < len(column_properties)
                else {
                    "type": "std",
                    "highlight": "none",
                    "decimals": _get_default_decimals("std"),
                }
            )
            column_type = _normalize_col_type(col_props.get("type", "std"))
            sub_cols = _get_subcolumn_count(column_type)
            column_format += " c " * sub_cols
            column_name = r_u(columns[column])
            title_row1_cells.append(
                f"\\multicolumn{{{sub_cols}}}{{c}}{{ {column_name} }}"
            )

            title_row2_cells.extend(_get_subcolumn_labels(column_type))

            sep_str += (
                "\\cmidrule(lr){"
                + str(column_num)
                + "-"
                + str(column_num + sub_cols - 1)
                + "}"
            )
            column_num += sub_cols
            total_sub_columns += sub_cols
            column_meta.append(
                {
                    "name": column,
                    "type": column_type,
                    "highlight": col_props.get("highlight", "none"),
                    "decimals": col_props.get(
                        "decimals", _get_default_decimals(column_type)
                    ),
                    "subcols": sub_cols,
                    "is_secondary": False,
                }
            )

        # Add secondary df column groups (if any)
        if df2 is not None and columns2 is not None:
            num_columns2 = len(columns2)
            for i, column in enumerate(columns2.keys()):
                col_props = (
                    column_properties2[i]
                    if column_properties2 is not None and i < len(column_properties2)
                    else {
                        "type": "std",
                        "highlight": "none",
                        "decimals": _get_default_decimals("std"),
                    }
                )
                column_type = _normalize_col_type(col_props.get("type", "std"))
                sub_cols = _get_subcolumn_count(column_type)
                column_format += " c " * sub_cols
                column_name = r_u(columns2[column])
                title_row1_cells.append(
                    f"\\multicolumn{{{sub_cols}}}{{c}}{{ {column_name} }}"
                )

                title_row2_cells.extend(_get_subcolumn_labels(column_type))

                sep_str += (
                    "\\cmidrule(lr){"
                    + str(column_num)
                    + "-"
                    + str(column_num + sub_cols - 1)
                    + "}"
                )
                column_num += sub_cols
                total_sub_columns += sub_cols
                column_meta.append(
                    {
                        "name": column,
                        "type": column_type,
                        "highlight": col_props.get("highlight", "none"),
                        "decimals": col_props.get(
                            "decimals", _get_default_decimals(column_type)
                        ),
                        "subcols": sub_cols,
                        "is_secondary": True,
                    }
                )

        title_str1 = " & ".join(title_row1_cells) + "\\\\\n" + sep_str + "\n"
        title_str2 = " & ".join(title_row2_cells) + "\\\\"
        title_str = title_str1 + title_str2

    else:
        title_str1 = categories[category_levels[0]]["name"]
        for i, level_key in enumerate(category_levels):
            if i == 0:
                continue
            column_format += " l "
            title_str1 += f" & {r_u(categories[level_key]['name'])}"

        column_num = len(categories) + 1
        if columns is not None:
            for i, column in enumerate(columns.keys()):
                column_format += " c "
                column_name = r_u(columns[column])
                title_str1 += f" & {column_name} "
                column_num += 1

        if df2 is not None and columns2 is not None:
            num_columns2 = len(columns2)
            for i, column in enumerate(columns2):
                column_format += " c "
                column_name = r_u(columns2[column])
                title_str1 += f" & {column_name}"
                column_num += 1
        title_str = title_str1 + "\\\\"
        total_sub_columns = num_columns + num_columns2

    cmidrule = " \\cmidrule{2-" + str(num_levels + total_sub_columns) + "}"

    column_format += " @{}}"

    _write_line(file_handle, "%%%%%")
    _write_line(file_handle, f"% Table Data ({title})")
    _write_line(file_handle, "%")
    _write_line(file_handle, "\\begin{table*}")
    _write_line(file_handle, f"\\caption{{ {caption} }}")
    _write_line(file_handle, f"\\label{{ {label} }}")
    _write_line(file_handle, "\\begin{center}")
    _write_line(file_handle, "\\resizebox{\\linewidth}{!}{")
    _write_line(file_handle, column_format)
    _write_line(file_handle, "\\toprule")

    _write_line(file_handle, title_str)
    _write_line(file_handle, "\\midrule")

    output_columns = total_sub_columns

    output_col_cursor = 0
    for meta in column_meta:
        meta["start"] = output_col_cursor
        output_col_cursor += meta["subcols"]

    # Generate all category combinations
    combinations = _generate_category_combinations(categories, category_levels)

    # Group combinations by higher-level categories for proper table structure
    grouped_combinations = {}
    for combo in combinations:
        # Group by all levels except the last one
        if num_levels > 1:
            key = combo[:-1]  # All but the last level
        else:
            key = ()

        if key not in grouped_combinations:
            grouped_combinations[key] = []
        grouped_combinations[key].append(combo)

    for group_key, group_combinations in grouped_combinations.items():
        labels = []
        output = np.zeros([len(group_combinations), output_columns])

        for row_index, combination in enumerate(group_combinations):
            # Get labels for this combination
            combo_labels = []
            for i, level_key in enumerate(category_levels):
                label = _get_category_label(categories, level_key, combination[i])
                combo_labels.append(r_u(label))
            labels.append(combo_labels)

            # Filter dataframes based on this combination
            df_slice = _build_dataframe_filter(
                df, categories, category_levels, combination
            )
            df_slice2 = None
            if df2 is not None and columns2 is not None:
                df_slice2 = _build_dataframe_filter(
                    df2, categories, category_levels, combination
                )

            if column_properties is not None and columns is not None:
                for meta in column_meta:
                    col = meta["name"]
                    column_type = _normalize_col_type(meta["type"])
                    target_df = df_slice2 if meta["is_secondary"] else df_slice
                    start_idx = meta["start"]

                    if column_type == "single":
                        output[row_index, start_idx] = (
                            target_df[col].mean()
                            if target_df is not None and not target_df[col].empty
                            else np.nan
                        )
                    elif column_type == "rng":  # min/max
                        output[row_index, start_idx] = (
                            target_df[col].min()
                            if target_df is not None and not target_df[col].empty
                            else np.nan
                        )
                        output[row_index, start_idx + 1] = (
                            target_df[col].max()
                            if target_df is not None and not target_df[col].empty
                            else np.nan
                        )
                    elif column_type == "sum":  # count/total
                        output[row_index, start_idx] = (
                            len(target_df[col])
                            if target_df is not None and not target_df[col].empty
                            else 0
                        )
                        output[row_index, start_idx + 1] = (
                            target_df[col].sum()
                            if target_df is not None and not target_df[col].empty
                            else np.nan
                        )
                    elif column_type == "ci":  # mean/confidence interval
                        output[row_index, start_idx] = (
                            target_df[col].mean()
                            if target_df is not None and not target_df[col].empty
                            else np.nan
                        )
                        output[row_index, start_idx + 1] = (
                            calculate_confidence_interval(target_df[col])
                            if target_df is not None and not target_df[col].empty
                            else np.nan
                        )
                    else:  # std (default)
                        output[row_index, start_idx] = (
                            target_df[col].mean()
                            if target_df is not None and not target_df[col].empty
                            else np.nan
                        )
                        output[row_index, start_idx + 1] = (
                            target_df[col].std()
                            if target_df is not None and not target_df[col].empty
                            else np.nan
                        )
            else:
                col_index = 0
                if columns is not None:
                    for col in columns:
                        output[row_index, col_index] = float(df_slice[col].iloc[0])
                        col_index += 1

                if df2 is not None and columns2 is not None:
                    for col in columns2:
                        output[row_index, col_index] = float(df_slice2[col].iloc[0])
                        col_index += 1

        if column_properties is not None and columns is not None:
            # Calculate highlighting for each column
            max_rows = []
            for meta in column_meta:
                highlight_rule = meta.get("highlight", "none")
                start_idx = meta["start"]
                subcols = meta["subcols"]

                if highlight_rule == "min":
                    if subcols == 1:
                        max_rows.append(_safe_argmin(output[:, start_idx]))
                    else:
                        # Find row with minimum value (prefer minimum std as tiebreaker)
                        max_rows.append(
                            np.lexsort(
                                (output[:, start_idx + 1], output[:, start_idx])
                            )[0]
                        )
                elif highlight_rule == "max":
                    if subcols == 1:
                        max_rows.append(_safe_argmax(output[:, start_idx]))
                    else:
                        # Find row with maximum value (prefer minimum std as tiebreaker)
                        max_rows.append(
                            np.lexsort(
                                (
                                    output[:, start_idx + 1],
                                    output[:, start_idx] * (-1),
                                )
                            )[0]
                        )
                else:
                    max_rows.append(-1)  # No highlighting

            previous_label_set = None
            for row_index, label_set in enumerate(labels):
                s = _format_row_labels(label_set, previous_label_set, num_levels)

                for meta_idx, meta in enumerate(column_meta):
                    start_idx = meta["start"]
                    subcols = meta["subcols"]
                    decimals = meta.get(
                        "decimals", _get_default_decimals(meta.get("type", "std"))
                    )
                    fmt = f"5.{int(decimals)}f"

                    if subcols == 1:
                        val = output[row_index, start_idx]
                        fmt_val = format_latex_num(val, fmt_str=fmt)
                        if max_rows[meta_idx] == row_index:
                            s += " & " + f"\\textbf{{ {fmt_val} }} "
                        else:
                            s += " & " + f"{fmt_val}"
                    else:
                        val1 = output[row_index, start_idx]
                        val2 = output[row_index, start_idx + 1]
                        fmt_val1 = format_latex_num(val1, fmt_str=fmt)
                        fmt_val2 = format_latex_num(val2, fmt_str=fmt)

                        if max_rows[meta_idx] == row_index:
                            # bold this entry
                            s += (
                                " & "
                                + f"\\textbf{{ {fmt_val1} }} & \\textbf{{ {fmt_val2} }} "
                            )
                        else:
                            s += " & " + f"{fmt_val1} & {fmt_val2}"

                s += "\\\\"
                _write_line(file_handle, s)
                previous_label_set = label_set
        else:
            previous_label_set = None
            for row_index, label_set in enumerate(labels):
                s = _format_row_labels(label_set, previous_label_set, num_levels)

                for col_index in range(output.shape[1]):
                    s += (
                        " & "
                        + format_latex_num(output[row_index, col_index], fmt_str="5.1f")
                        + " "
                    )

                s += "\\\\"
                _write_line(file_handle, s)
                previous_label_set = label_set

        # Write separator if this is not the last group
        group_keys = list(grouped_combinations.keys())
        current_group_index = list(grouped_combinations.keys()).index(group_key)
        if current_group_index < len(group_keys) - 1:
            _write_line(file_handle, cmidrule)

    _write_line(file_handle, "\\midrule")

    _write_line(file_handle, "\\bottomrule")
    _write_line(file_handle, "\\end{tabular}")
    _write_line(file_handle, "} % end of resizebox")
    _write_line(file_handle, "\\end{center}")
    _write_line(file_handle, "\\end{table*}")
    _write_line(file_handle, "%")
    _write_line(file_handle, "%%%%%")

    if file_handle:
        file_handle.close()
        print(f"LaTeX table written to {output_file_path}")
