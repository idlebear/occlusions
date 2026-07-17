#!/usr/bin/env python3
import argparse
import csv
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PEDESTRIAN_DIR = Path(__file__).resolve().parents[1]
if str(PEDESTRIAN_DIR) not in sys.path:
    sys.path.insert(0, str(PEDESTRIAN_DIR))

from latex import write_table

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - exercised only on minimal envs
    sns = None


SUMMARY_SUFFIX = "uncertainty_summary"
TARGET_SUFFIX = "target_beliefs"
PHASE1_CASE_METHOD_SUFFIX = "phase1_case_methods"
PHASE1_CANDIDATE_SUFFIX = "phase1_candidates"
PHASE2_ROLLOUT_SUFFIX = "phase2_rollouts"
PHASE3_ROLLOUT_SUFFIX = "phase3_rollouts"
METHOD_ORDER = ["oce", "visibility", "none"]
BEST_REFERENCE_METHOD = "oce-gpu-exact-entropy_plus_information"
BEST_VS_BASELINE_METHODS = [
    BEST_REFERENCE_METHOD,
    "oce-gpu-approximate-entropy_plus_information",
    "vis-cpu-none-none",
    "none-cpu-none-none",
]
VISIBILITY_BASELINE_METHOD = "vis-cpu-none-none"
EXACT_VISIBILITY_WINNER_LABELS = [
    "exact+info wins",
    "visibility wins",
    "tie",
    "incomplete",
]
BEST_VS_BASELINE_ALIASES = {
    "visibility-cpu-none-none": "vis-cpu-none-none",
}


class TableMask:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=bool)

    def __and__(self, other):
        return TableMask(self.values & np.asarray(other.values, dtype=bool))


class TableSeries:
    def __init__(self, values):
        self.values = list(values)

    @property
    def empty(self):
        return len(self.values) == 0

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)

    def __eq__(self, other):
        return TableMask([value == other for value in self.values])

    def _finite_values(self):
        values = np.asarray([parse_float(value) for value in self.values], dtype=float)
        return values[np.isfinite(values)]

    def mean(self):
        values = self._finite_values()
        return float(np.mean(values)) if values.size else np.nan

    def min(self):
        values = self._finite_values()
        return float(np.min(values)) if values.size else np.nan

    def max(self):
        values = self._finite_values()
        return float(np.max(values)) if values.size else np.nan

    def sum(self):
        values = self._finite_values()
        return float(np.sum(values)) if values.size else np.nan

    def std(self):
        values = self._finite_values()
        return float(np.std(values, ddof=1)) if values.size > 1 else np.nan


class TableDataFrame:
    def __init__(self, records=None):
        self.records = [dict(record) for record in records or []]

    @property
    def empty(self):
        return len(self.records) == 0

    @property
    def columns(self):
        columns = []
        for record in self.records:
            for key in record:
                if key not in columns:
                    columns.append(key)
        return columns

    def __getitem__(self, key):
        if isinstance(key, str):
            return TableSeries([record.get(key, np.nan) for record in self.records])
        mask = np.asarray(key.values if isinstance(key, TableMask) else key, dtype=bool)
        return TableDataFrame(
            [record for record, keep in zip(self.records, mask) if bool(keep)]
        )

    def __setitem__(self, key, value):
        for record in self.records:
            record[key] = value


def parse_float(value):
    if value in (None, ""):
        return np.nan
    try:
        return float(value)
    except ValueError:
        return np.nan


def parse_int(value, default=0):
    parsed = parse_float(value)
    if not np.isfinite(parsed):
        return default
    return int(parsed)


def parse_bool(value, default=False):
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def read_csv_rows(path):
    with Path(path).open("r", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def latest_forward_run(rows):
    if not rows:
        return []

    start_index = 0
    previous_time = np.nan
    previous_tick = np.nan
    for index, row in enumerate(rows):
        time_s = parse_float(row.get("time_s"))
        tick = parse_float(row.get("tick"))
        time_reset = (
            np.isfinite(time_s)
            and np.isfinite(previous_time)
            and time_s < previous_time
        )
        tick_reset = (
            np.isfinite(tick)
            and np.isfinite(previous_tick)
            and tick < previous_tick
        )
        if time_reset or tick_reset:
            start_index = index
        previous_time = time_s
        previous_tick = tick
    return rows[start_index:]


def tracked_rows(rows):
    return [row for row in rows if parse_bool(row.get("tracked"), default=True)]


def grouped_by_target(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[str(row["track_id"])].append(row)
    for values in groups.values():
        values.sort(
            key=lambda row: (parse_float(row["time_s"]), parse_float(row.get("tick")))
        )
    return dict(sorted(groups.items(), key=lambda item: item[0]))


def configure_plot_style():
    if sns is not None:
        sns.set_theme(
            context="paper",
            style="whitegrid",
            palette="colorblind",
            font_scale=1.15,
            rc={
                "figure.dpi": 160,
                "savefig.dpi": 300,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.labelsize": 11,
                "axes.titlesize": 12,
                "legend.frameon": False,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            },
        )
    else:
        plt.style.use("seaborn-v0_8-whitegrid")


def file_metadata(path, prefix, suffix):
    name = Path(path).name
    escaped = re.escape(prefix)
    patterns = [
        rf"^{escaped}_experiment_(?P<experiment>-?\d+)_(?P<method>.+)_(?P<hw>cpu|gpu)_{suffix}\.csv$",
        rf"^{escaped}_(?P<experiment>-?\d+)_(?P<method>.+)_(?P<hw>cpu|gpu)_{suffix}\.csv$",
        rf"^{escaped}_experiment_(?P<experiment>-?\d+)_(?P<method>.+)_{suffix}\.csv$",
        rf"^{escaped}_(?P<experiment>-?\d+)_(?P<method>.+)_{suffix}\.csv$",
        rf"^{escaped}_(?P<method>.+)_{suffix}\.csv$",
    ]
    for pattern in patterns:
        match = re.match(pattern, name)
        if match:
            data = match.groupdict()
            return {
                "prefix": prefix,
                "experiment": parse_int(data.get("experiment"), default=0),
                "method": normalize_method(data.get("method", "")),
                "hw": normalize_hw(data.get("hw", "")),
            }
    return {"prefix": prefix, "experiment": 0, "method": "", "hw": ""}


def normalize_method(method):
    method = str(method or "").strip().lower()
    return "visibility" if method == "vis" else method


def normalize_hw(hw):
    hw = str(hw or "").strip().lower()
    return hw if hw in {"cpu", "gpu"} else ""


def series_label(row):
    method = normalize_method(row.get("method") or "run")
    hw = normalize_hw(row.get("hw"))
    if re.search(r"-(cpu|gpu)-", method):
        return method
    return f"{method}/{hw}" if hw else method


def load_rows_for_prefix(log_dir, prefix, suffix, explicit_paths=None):
    paths = [Path(path) for path in explicit_paths or []]
    if not paths:
        paths = sorted(Path(log_dir).glob(f"{prefix}*_{suffix}.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No {suffix} CSV files found in {log_dir} for prefix '{prefix}'."
        )

    rows = []
    for path in paths:
        metadata = file_metadata(path, prefix, suffix)
        file_rows = latest_forward_run(read_csv_rows(path))
        for row in file_rows:
            row = dict(row)
            row["prefix"] = row.get("prefix") or metadata["prefix"]
            row["experiment"] = parse_int(
                row.get("experiment"),
                default=metadata["experiment"],
            )
            row["method"] = normalize_method(row.get("method") or metadata["method"])
            row["hw"] = normalize_hw(row.get("hw") or metadata["hw"])
            row["_source_file"] = str(path)
            rows.append(row)
    return rows


def load_phase1_rows_for_prefix(log_dir, prefix, suffix, explicit_paths=None):
    paths = [Path(path) for path in explicit_paths or []]
    if not paths:
        paths = sorted(Path(log_dir).glob(f"{prefix}*_{suffix}.csv"))
    if not paths:
        return []

    rows = []
    for path in paths:
        metadata = file_metadata(path, prefix, suffix)
        for row in read_csv_rows(path):
            row = dict(row)
            row["prefix"] = row.get("prefix") or metadata["prefix"]
            row["experiment"] = parse_int(
                row.get("experiment"),
                default=metadata["experiment"],
            )
            row["method"] = normalize_method(row.get("method") or metadata["method"])
            row["hw"] = normalize_hw(row.get("hw") or row.get("backend") or metadata["hw"])
            row["_source_file"] = str(path)
            rows.append(row)
    return rows


def load_phase2_rows_for_prefix(log_dir, prefix, suffix, explicit_paths=None):
    rows = load_phase1_rows_for_prefix(
        log_dir,
        prefix,
        suffix,
        explicit_paths=explicit_paths,
    )
    for row in rows:
        if row.get("selector_method"):
            row["method"] = normalize_method(row["selector_method"])
        row["hw"] = normalize_hw(row.get("backend") or row.get("hw"))
    return rows


def load_phase1_rows_with_fallback(log_dir, prefix, explicit_paths=None):
    rows = load_phase1_rows_for_prefix(
        log_dir,
        prefix,
        PHASE1_CASE_METHOD_SUFFIX,
        explicit_paths=explicit_paths,
    )
    if rows or explicit_paths or "phase2" not in prefix:
        return rows
    phase1_prefix = prefix.replace("phase2", "phase1")
    return load_phase1_rows_for_prefix(
        log_dir,
        phase1_prefix,
        PHASE1_CASE_METHOD_SUFFIX,
    )


def parse_oce_selector(label):
    label = normalize_method(label)
    match = re.match(
        r"^oce-(?P<backend>cpu|gpu)-(?P<discrete>exact|approximate)-(?P<scoring>.+)$",
        label,
    )
    if not match:
        return None
    return match.groupdict()


def series_present(rows):
    labels = sorted({series_label(row) for row in rows if row.get("method")})
    ordered = []
    for method in METHOD_ORDER:
        ordered.extend(label for label in labels if label == method)
        ordered.extend(label for label in labels if label.startswith(f"{method}/"))
        ordered.extend(label for label in labels if label.startswith(f"{method}-"))
    return ordered + [
        label for label in labels if label not in set(ordered)
    ]


def phase1_series_present(rows):
    labels = sorted({series_label(row) for row in rows if row.get("method")})
    ordered = []
    for method in METHOD_ORDER:
        ordered.extend(label for label in labels if label == method)
        ordered.extend(label for label in labels if label.startswith(f"{method}/"))
        ordered.extend(label for label in labels if label.startswith(f"{method}-"))
    exact = [label for label in labels if "oce-gpu-exact-" in label]
    approximate = [label for label in labels if "oce-gpu-approximate-" in label]
    ordered.extend(label for label in exact if label not in ordered)
    ordered.extend(label for label in approximate if label not in ordered)
    return ordered + [label for label in labels if label not in set(ordered)]


def phase1_metric_values(rows, metric):
    values_by_method = defaultdict(list)
    for row in rows:
        label = series_label(row)
        value = parse_float(row.get(metric))
        if label and np.isfinite(value):
            values_by_method[label].append(float(value))
    return values_by_method


def best_baseline_label(row):
    label = series_label(row)
    return BEST_VS_BASELINE_ALIASES.get(label, label)


def best_vs_baseline_rows(rows):
    focused = []
    allowed = set(BEST_VS_BASELINE_METHODS)
    for row in rows or []:
        label = best_baseline_label(row)
        if label not in allowed:
            continue
        focused_row = dict(row)
        focused_row["method"] = label
        focused_row["selector_method"] = label
        focused.append(focused_row)
    return focused


def best_vs_baseline_series_present(rows):
    present = {best_baseline_label(row) for row in rows or []}
    return [label for label in BEST_VS_BASELINE_METHODS if label in present]


def vector_data(rows, metric):
    x = []
    y = []
    hue = []
    units = []
    for row in rows:
        value = parse_float(row.get(metric))
        time_s = parse_float(row.get("time_s"))
        if not np.isfinite(value) or not np.isfinite(time_s):
            continue
        x.append(time_s)
        y.append(value)
        hue.append(series_label(row))
        units.append(f"{row.get('experiment', '')}:{normalize_hw(row.get('hw'))}")
    return x, y, hue, units


def target_label(track_id):
    return f"target {track_id}"


def plot_metric_mean_sd(
    rows,
    metric,
    output_path,
    *,
    ylabel,
    title,
    xlabel="time (s)",
    ylim=None,
):
    x, y, hue, _units = vector_data(rows, metric)
    if not x:
        return

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    if sns is not None:
        sns.lineplot(
            x=x,
            y=y,
            hue=hue,
            hue_order=series_present(rows) or None,
            estimator="mean",
            errorbar="sd",
            linewidth=2.0,
            ax=ax,
        )
    else:
        plot_metric_mean_sd_matplotlib(ax, x, y, hue)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.25)
    ax.legend(title="method/hw")
    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def plot_metric_by_target_mean_sd(
    rows,
    metric,
    output_path,
    *,
    ylabel,
    title,
    xlabel="time (s)",
    ylim=None,
):
    target_groups = {
        track_id: group
        for track_id, group in grouped_by_target(rows).items()
        if vector_data(group, metric)[0]
    }
    if not target_groups:
        return

    target_count = len(target_groups)
    fig_height = max(3.8, 2.45 * target_count)
    fig, axes = plt.subplots(
        target_count,
        1,
        figsize=(7.2, fig_height),
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]
    hue_order = series_present(rows) or None
    legend_handles = None
    legend_labels = None

    for ax, (track_id, target_rows) in zip(axes, target_groups.items()):
        x, y, hue, _units = vector_data(target_rows, metric)
        if sns is not None:
            sns.lineplot(
                x=x,
                y=y,
                hue=hue,
                hue_order=hue_order,
                estimator="mean",
                errorbar="sd",
                linewidth=2.0,
                ax=ax,
            )
        else:
            plot_metric_mean_sd_matplotlib(ax, x, y, hue)

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

        ax.set_title(target_label(track_id), loc="left")
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel(xlabel)
    fig.suptitle(title)
    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            title="method/hw",
            loc="upper center",
            ncol=min(len(legend_labels), 4),
            bbox_to_anchor=(0.5, 1.0),
        )
        fig.tight_layout(rect=(0, 0, 1, 0.92))
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, output_path)
    plt.close(fig)


def plot_metric_mean_sd_matplotlib(ax, x, y, hue):
    grouped = defaultdict(lambda: defaultdict(list))
    for time_s, value, method in zip(x, y, hue):
        grouped[method][time_s].append(value)

    for method in sorted(grouped):
        times = np.asarray(sorted(grouped[method]), dtype=float)
        means = np.asarray([np.mean(grouped[method][t]) for t in times], dtype=float)
        stds = np.asarray([np.std(grouped[method][t]) for t in times], dtype=float)
        line = ax.plot(times, means, linewidth=2.0, label=method)[0]
        ax.fill_between(
            times,
            means - stds,
            means + stds,
            color=line.get_color(),
            alpha=0.2,
            linewidth=0,
        )


def save_figure(fig, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")


def _short_method_label(label):
    label = str(label)
    return (
        label.replace("oce-gpu-", "")
        .replace("vis-cpu-none-none", "visibility")
        .replace("none-cpu-none-none", "none")
    )


def plot_phase1_metric_bar(
    rows,
    metric,
    output_path,
    *,
    ylabel,
    title,
    ylim=None,
    methods=None,
):
    values_by_method = phase1_metric_values(rows, metric)
    if methods is None:
        methods = phase1_series_present(rows)
    methods = [method for method in methods if method in values_by_method]
    if not methods:
        return

    means = []
    ci = []
    for method in methods:
        values = np.asarray(values_by_method[method], dtype=float)
        means.append(float(np.mean(values)))
        if values.size > 1:
            ci.append(float(1.96 * np.std(values, ddof=1) / np.sqrt(values.size)))
        else:
            ci.append(0.0)

    fig, ax = plt.subplots(figsize=(max(7.2, 0.54 * len(methods)), 4.2))
    ax.bar(np.arange(len(methods)), means, yerr=ci, capsize=3.0)
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels([_short_method_label(method) for method in methods], rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def plot_phase1_metric_box(
    rows,
    metric,
    output_path,
    *,
    ylabel,
    title,
    methods=None,
):
    values_by_method = phase1_metric_values(rows, metric)
    if methods is None:
        methods = phase1_series_present(rows)
    methods = [method for method in methods if method in values_by_method]
    data = [values_by_method[method] for method in methods if values_by_method[method]]
    labels = [_short_method_label(method) for method in methods if values_by_method[method]]
    if not data:
        return

    fig, ax = plt.subplots(figsize=(max(7.2, 0.54 * len(labels)), 4.2))
    ax.boxplot(data, labels=labels, showfliers=True)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=45)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def plot_phase1_reference_rank_histogram(rows, output_path):
    rank_rows = [
        row
        for row in rows
        if np.isfinite(parse_float(row.get("reference_rank")))
        and series_label(row) != "oce-gpu-exact-entropy_plus_information"
    ]
    if not rank_rows:
        return
    ranks = sorted({parse_int(row.get("reference_rank")) for row in rank_rows})
    methods = [
        method
        for method in phase1_series_present(rank_rows)
        if any(series_label(row) == method for row in rank_rows)
    ]
    if not methods or not ranks:
        return

    counts = np.zeros((len(methods), len(ranks)), dtype=float)
    for method_index, method in enumerate(methods):
        method_rows = [row for row in rank_rows if series_label(row) == method]
        total = max(1, len(method_rows))
        for rank_index, rank in enumerate(ranks):
            counts[method_index, rank_index] = (
                sum(parse_int(row.get("reference_rank")) == rank for row in method_rows)
                / float(total)
            )

    fig, ax = plt.subplots(figsize=(max(7.2, 0.52 * len(methods)), 4.2))
    bottom = np.zeros((len(methods),), dtype=float)
    x = np.arange(len(methods))
    for rank_index, rank in enumerate(ranks):
        ax.bar(x, counts[:, rank_index], bottom=bottom, label=f"rank {rank}")
        bottom += counts[:, rank_index]
    ax.set_xticks(x)
    ax.set_xticklabels([_short_method_label(method) for method in methods], rotation=45, ha="right")
    ax.set_ylabel("fraction of cases")
    ax.set_title("Phase 1 Exact-Reference Rank Distribution")
    ax.set_ylim(0.0, 1.0)
    ax.legend(title="selected candidate")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def aggregate_final_summary(summary_rows):
    latest_by_run = {}
    for row in summary_rows:
        key = (
            row.get("prefix", ""),
            parse_int(row.get("experiment")),
            normalize_method(row.get("method")),
            normalize_hw(row.get("hw")),
        )
        current = latest_by_run.get(key)
        if current is None or parse_float(row.get("time_s")) >= parse_float(
            current.get("time_s")
        ):
            latest_by_run[key] = row

    metrics = [
        "total_uncertainty",
        "mean_mode_entropy",
        "mean_true_class_probability",
        "robot_distance_traveled",
    ]
    grouped = defaultdict(list)
    for row in latest_by_run.values():
        grouped[series_label(row)].append(row)

    summary = []
    for label in sorted(grouped):
        rows = grouped[label]
        first = rows[0]
        record = {
            "series": label,
            "method": normalize_method(first.get("method")),
            "hw": normalize_hw(first.get("hw")),
            "n_experiments": len(rows),
        }
        for metric in metrics:
            values = np.asarray([parse_float(row.get(metric)) for row in rows], dtype=float)
            values = values[np.isfinite(values)]
            record[f"{metric}_mean"] = float(np.mean(values)) if values.size else np.nan
            record[f"{metric}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        summary.append(record)
    return summary


def aggregate_phase1_summary(case_rows):
    metrics = [
        "selection_agreement",
        "top2_agreement",
        "top3_agreement",
        "agrees_with_majority",
        "reference_rank",
        "exact_reference_regret",
        "paired_exact_reference_regret",
    ]
    grouped = defaultdict(list)
    for row in case_rows:
        grouped[series_label(row)].append(row)

    summary = []
    for label in phase1_series_present(case_rows):
        rows = grouped.get(label, [])
        if not rows:
            continue
        first = rows[0]
        record = {
            "series": label,
            "method": normalize_method(first.get("method")),
            "backend": normalize_hw(first.get("backend") or first.get("hw")),
            "n_cases": len({row.get("case_id", "") for row in rows}),
            "n_rows": len(rows),
        }
        for metric in metrics:
            values = np.asarray([parse_float(row.get(metric)) for row in rows], dtype=float)
            values = values[np.isfinite(values)]
            record[f"{metric}_mean"] = float(np.mean(values)) if values.size else np.nan
            record[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            )
            record[f"{metric}_median"] = float(np.median(values)) if values.size else np.nan
            record[f"{metric}_iqr"] = (
                float(np.percentile(values, 75) - np.percentile(values, 25))
                if values.size
                else np.nan
            )
        summary.append(record)
    return summary


def write_phase1_aggregate_summary(case_rows, output_dir, prefix):
    rows = aggregate_phase1_summary(case_rows)
    if not rows:
        return
    output_path = Path(output_dir) / f"{prefix}_phase1_summary.csv"
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_aggregate_summary(summary_rows, output_dir, prefix):
    rows = aggregate_final_summary(summary_rows)
    if not rows:
        return
    output_path = Path(output_dir) / f"{prefix}_aggregate_final_summary.csv"
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def latest_rows_by_key(rows, key_fields):
    latest = {}
    for row in rows:
        key = tuple(row.get(field, "") for field in key_fields)
        current = latest.get(key)
        if current is None or parse_float(row.get("time_s")) >= parse_float(
            current.get("time_s")
        ):
            latest[key] = row
    return latest


def entropy_table_labels(target_rows, summary_rows):
    targets = sorted(
        {
            str(row.get("track_id"))
            for row in tracked_rows(target_rows)
            if row.get("track_id") not in (None, "")
        },
        key=lambda value: (parse_int(value, default=10**9), value),
    )
    series = series_present(target_rows + summary_rows)
    return targets, series


def build_entropy_table_dataframe(
    target_rows,
    summary_rows,
    *,
    target_metric,
    summary_metric,
):
    targets, series = entropy_table_labels(target_rows, summary_rows)
    if not targets or not series:
        return TableDataFrame(), [], []

    target_latest = latest_rows_by_key(
        tracked_rows(target_rows),
        ["prefix", "experiment", "method", "hw", "track_id"],
    )
    summary_latest = latest_rows_by_key(
        summary_rows,
        ["prefix", "experiment", "method", "hw"],
    )

    records = []
    for row in target_latest.values():
        label = series_label(row)
        value = parse_float(row.get(target_metric))
        track_id = str(row.get("track_id"))
        if label not in series or track_id not in targets or not np.isfinite(value):
            continue
        records.append({"target": target_label(track_id), label: value})

    for row in summary_latest.values():
        label = series_label(row)
        value = parse_float(row.get(summary_metric))
        if label not in series or not np.isfinite(value):
            continue
        records.append({"target": "total", label: value})

    target_order = [target_label(track_id) for track_id in targets] + ["total"]
    return TableDataFrame(records), target_order, series


def write_entropy_table(
    target_rows,
    summary_rows,
    output_dir,
    prefix,
    *,
    target_metric,
    summary_metric,
    filename,
    caption,
    label,
):
    df, target_order, series = build_entropy_table_dataframe(
        target_rows,
        summary_rows,
        target_metric=target_metric,
        summary_metric=summary_metric,
    )
    if df.empty:
        return

    for method in series:
        if method not in df.columns:
            df[method] = np.nan

    columns_spec = {
        method: {
            "display": method,
            "type": "ci",
            "highlight": "none",
            "decimals": 2,
        }
        for method in series
    }
    categories = {
        "L1": {
            "name": "target",
            "column": "target",
            "labels": target_order,
        }
    }
    write_table(
        df,
        categories,
        columns_spec=columns_spec,
        caption=caption,
        label=label,
        title=filename,
        output_file_path=Path(output_dir) / f"{prefix}_{filename}.tex",
    )


def write_entropy_tables(target_rows, summary_rows, output_dir, prefix):
    write_entropy_table(
        target_rows,
        summary_rows,
        output_dir,
        prefix,
        target_metric="state_entropy",
        summary_metric="sum_state_entropy",
        filename="state_entropy_table",
        caption=(
            "Final state entropy by tracked target and total, reported as mean "
            "and 95\\% confidence interval across experiment runs."
        ),
        label=f"tab:{prefix}-state-entropy",
    )
    write_entropy_table(
        target_rows,
        summary_rows,
        output_dir,
        prefix,
        target_metric="mode_entropy",
        summary_metric="sum_mode_entropy",
        filename="class_entropy_table",
        caption=(
            "Final destination-class entropy by tracked target and total, reported "
            "as mean and 95\\% confidence interval across experiment runs."
        ),
        label=f"tab:{prefix}-class-entropy",
    )


def build_phase1_metric_table_dataframe(case_rows, metrics, *, methods=None):
    methods = list(methods) if methods is not None else phase1_series_present(case_rows)
    if not methods:
        return TableDataFrame(), [], []

    records = []
    for row in case_rows:
        label = series_label(row)
        if label not in methods:
            continue
        for metric, display, _decimals in metrics:
            value = parse_float(row.get(metric))
            if np.isfinite(value):
                records.append({"metric": display, label: value})

    metric_order = [display for _metric, display, _decimals in metrics]
    return TableDataFrame(records), metric_order, methods


def write_phase1_metric_table(
    case_rows,
    output_dir,
    prefix,
    *,
    metrics,
    filename,
    caption,
    label,
    methods=None,
):
    df, metric_order, methods = build_phase1_metric_table_dataframe(
        case_rows,
        metrics,
        methods=methods,
    )
    if df.empty:
        return

    for method in methods:
        if method not in df.columns:
            df[method] = np.nan

    table_decimals = max(decimals for _metric, _display, decimals in metrics)
    columns_spec = {
        method: {
            "display": method,
            "type": "ci",
            "highlight": "none",
            "decimals": table_decimals,
        }
        for method in methods
    }
    categories = {
        "L1": {
            "name": "metric",
            "column": "metric",
            "labels": metric_order,
        }
    }
    write_table(
        df,
        categories,
        columns_spec=columns_spec,
        caption=caption,
        label=label,
        title=filename,
        output_file_path=Path(output_dir) / f"{prefix}_{filename}.tex",
    )


def write_phase1_tables(case_rows, output_dir, prefix):
    if not case_rows:
        return
    write_phase1_metric_table(
        case_rows,
        output_dir,
        prefix,
        metrics=[
            ("selection_agreement", "reference agreement", 2),
            ("top2_agreement", "top-2 agreement", 2),
            ("top3_agreement", "top-3 agreement", 2),
            ("agrees_with_majority", "majority agreement", 2),
            ("reference_rank", "reference rank", 2),
        ],
        filename="phase1_selection_table",
        caption=(
            "Frozen-candidate Phase 1 selection agreement, reference rank, and "
            "majority agreement, reported as mean and 95\\% confidence interval "
            "across decision cases."
        ),
        label=f"tab:{prefix}-phase1-selection",
    )
    write_phase1_metric_table(
        case_rows,
        output_dir,
        prefix,
        metrics=[
            ("exact_reference_regret", "primary regret", 3),
            ("paired_exact_reference_regret", "paired regret", 3),
        ],
        filename="phase1_regret_table",
        caption=(
            "Frozen-candidate Phase 1 exact-reference regret and paired "
            "exact-vs-approximate regret, reported as mean and 95\\% confidence "
            "interval across decision cases."
        ),
        label=f"tab:{prefix}-phase1-regret",
    )


def plot_total_uncertainty(summary_rows, output_dir):
    if not summary_rows:
        return
    output_dir = Path(output_dir)
    plot_metric_mean_sd(
        summary_rows,
        "total_uncertainty",
        output_dir / "total_uncertainty",
        ylabel="total mode entropy",
        title="Tracked Target Class Uncertainty",
    )
    plot_metric_mean_sd(
        summary_rows,
        "mean_mode_entropy",
        output_dir / "mean_mode_entropy",
        ylabel="mean mode entropy",
        title="Mean Destination-Class Entropy",
    )


def plot_robot_motion(summary_rows, output_dir):
    if not summary_rows:
        return
    output_dir = Path(output_dir)
    plot_metric_mean_sd(
        summary_rows,
        "robot_speed",
        output_dir / "robot_speed",
        ylabel="speed",
        title="Robot Speed",
    )
    plot_metric_mean_sd(
        summary_rows,
        "robot_distance_traveled",
        output_dir / "robot_distance_traveled",
        ylabel="distance",
        title="Robot Distance Traveled",
    )


def plot_per_target_entropy(target_rows, output_dir):
    target_rows = tracked_rows(target_rows)
    if not target_rows:
        return
    output_dir = Path(output_dir)
    plot_metric_by_target_mean_sd(
        target_rows,
        "state_entropy",
        output_dir / "per_target_state_entropy",
        ylabel="state entropy",
        title="Per-Target State Entropy",
    )
    plot_metric_by_target_mean_sd(
        target_rows,
        "mode_entropy",
        output_dir / "per_target_mode_entropy",
        ylabel="mode entropy",
        title="Per-Target Destination-Class Entropy",
    )
    legacy_png = output_dir / "per_target_entropy.png"
    source_png = output_dir / "per_target_state_entropy.png"
    if source_png.exists() and not legacy_png.exists():
        shutil.copyfile(source_png, legacy_png)


def plot_visibility(target_rows, output_dir):
    target_rows = tracked_rows(target_rows)
    if not target_rows:
        return
    output_dir = Path(output_dir)
    rows = []
    for row in target_rows:
        row = dict(row)
        row["visible_fraction"] = 1.0 if parse_bool(row.get("visible")) else 0.0
        rows.append(row)
    plot_metric_mean_sd(
        rows,
        "visible_fraction",
        output_dir / "visibility_timeline",
        ylabel="visible fraction",
        title="Tracked Target Visibility",
        ylim=(-0.02, 1.02),
    )


def plot_true_class_probability(target_rows, output_dir):
    target_rows = [
        row
        for row in tracked_rows(target_rows)
        if parse_int(row.get("true_class_id"), default=-1) >= 0
        and np.isfinite(parse_float(row.get("true_class_probability")))
    ]
    if not target_rows:
        return
    plot_metric_mean_sd(
        target_rows,
        "true_class_probability",
        Path(output_dir) / "true_class_probability",
        ylabel="P(true class)",
        title="True Destination-Class Probability",
        ylim=(-0.02, 1.02),
    )


def plot_phase1_results(case_rows, output_dir):
    if not case_rows:
        return
    output_dir = Path(output_dir)
    plot_phase1_metric_bar(
        case_rows,
        "selection_agreement",
        output_dir / "phase1_reference_agreement",
        ylabel="agreement fraction",
        title="Phase 1 Exact-Reference Selection Agreement",
        ylim=(0.0, 1.0),
    )
    plot_phase1_metric_bar(
        case_rows,
        "top2_agreement",
        output_dir / "phase1_top2_agreement",
        ylabel="fraction of cases",
        title="Phase 1 Top-2 Exact-Reference Agreement",
        ylim=(0.0, 1.0),
    )
    plot_phase1_metric_bar(
        case_rows,
        "agrees_with_majority",
        output_dir / "phase1_majority_agreement",
        ylabel="agreement fraction",
        title="Phase 1 Majority-Selection Agreement",
        ylim=(0.0, 1.0),
    )
    plot_phase1_metric_box(
        case_rows,
        "exact_reference_regret",
        output_dir / "phase1_exact_reference_regret",
        ylabel="reference score regret",
        title="Phase 1 Primary Exact-Reference Regret",
    )
    approximate_rows = [
        row
        for row in case_rows
        if normalize_method(row.get("discrete_oce_method")) == "approximate"
        and np.isfinite(parse_float(row.get("paired_exact_reference_regret")))
    ]
    plot_phase1_metric_box(
        approximate_rows,
        "paired_exact_reference_regret",
        output_dir / "phase1_paired_exact_reference_regret",
        ylabel="paired exact score regret",
        title="Phase 1 Paired Exact-vs-Approximate Regret",
    )
    plot_phase1_reference_rank_histogram(
        case_rows,
        output_dir / "phase1_reference_rank_distribution",
    )


def plot_phase2_results(rollout_rows, output_dir, *, phase="phase2", phase_title="Phase 2"):
    if not rollout_rows:
        return
    output_dir = Path(output_dir)
    plot_phase1_metric_bar(
        rollout_rows,
        "final_sum_state_entropy",
        output_dir / f"{phase}_final_state_entropy",
        ylabel="sum state entropy",
        title=f"{phase_title} Final State Entropy",
    )
    plot_phase1_metric_bar(
        rollout_rows,
        "final_sum_class_entropy",
        output_dir / f"{phase}_final_class_entropy",
        ylabel="sum class entropy",
        title=f"{phase_title} Final Destination-Class Entropy",
    )
    plot_phase1_metric_bar(
        rollout_rows,
        "final_true_class_probability",
        output_dir / f"{phase}_true_class_probability",
        ylabel="P(true class)",
        title=f"{phase_title} Final True-Class Probability",
        ylim=(0.0, 1.0),
    )
    plot_phase1_metric_bar(
        rollout_rows,
        "visibility_fraction",
        output_dir / f"{phase}_visibility_fraction",
        ylabel="visible fraction",
        title=f"{phase_title} Final Visibility Fraction",
        ylim=(0.0, 1.0),
    )
    plot_phase1_metric_bar(
        rollout_rows,
        "distance_traveled",
        output_dir / f"{phase}_distance_traveled",
        ylabel="distance traveled",
        title=f"{phase_title} Robot Distance Traveled",
    )
    plot_phase1_metric_bar(
        rollout_rows,
        "at_goal",
        output_dir / f"{phase}_goal_rate",
        ylabel="goal fraction",
        title=f"{phase_title} Goal Completion Rate",
        ylim=(0.0, 1.0),
    )


def write_phase2_summary(rollout_rows, output_dir, prefix, *, phase="phase2"):
    if not rollout_rows:
        return
    rows = []
    grouped = defaultdict(list)
    for row in rollout_rows:
        grouped[series_label(row)].append(row)
    metrics = [
        "final_sum_state_entropy",
        "final_sum_class_entropy",
        "final_true_class_probability",
        "visibility_fraction",
        "distance_traveled",
        "time_to_goal",
        "at_goal",
        "collision",
        "timeout",
    ]
    for label in phase1_series_present(rollout_rows):
        method_rows = grouped.get(label, [])
        if not method_rows:
            continue
        first = method_rows[0]
        record = {
            "series": label,
            "method": normalize_method(first.get("method")),
            "common_method": first.get("common_method", ""),
            "n_runs": len(method_rows),
        }
        for metric in metrics:
            values = np.asarray(
                [parse_float(row.get(metric)) for row in method_rows],
                dtype=float,
            )
            values = values[np.isfinite(values)]
            record[f"{metric}_mean"] = float(np.mean(values)) if values.size else np.nan
            record[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            )
        rows.append(record)

    if not rows:
        return
    output_path = Path(output_dir) / f"{prefix}_{phase}_summary.csv"
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_phase2_tables(
    rollout_rows,
    output_dir,
    prefix,
    *,
    phase="phase2",
    phase_title="Phase 2",
):
    if not rollout_rows:
        return
    write_phase1_metric_table(
        rollout_rows,
        output_dir,
        prefix,
        metrics=[
            ("final_sum_state_entropy", "state entropy", 2),
            ("final_sum_class_entropy", "class entropy", 2),
            ("final_true_class_probability", "true class probability", 2),
            ("visibility_fraction", "visibility fraction", 2),
            ("distance_traveled", "distance", 2),
        ],
        filename=f"{phase}_outcome_table",
        caption=(
            f"{phase_title} rollout outcomes by selector method, "
            "reported as mean and 95\\% confidence interval across runs."
        ),
        label=f"tab:{prefix}-{phase}-outcomes",
    )
    write_phase1_metric_table(
        rollout_rows,
        output_dir,
        prefix,
        metrics=[
            ("at_goal", "goal rate", 2),
            ("collision", "collision rate", 2),
            ("timeout", "timeout rate", 2),
        ],
        filename=f"{phase}_completion_table",
        caption=(
            f"{phase_title} completion, collision, and timeout rates by selector "
            "method, reported as mean and 95\\% confidence interval across runs."
        ),
        label=f"tab:{prefix}-{phase}-completion",
    )


def plot_best_vs_baselines(rollout_rows, output_dir, *, phase, phase_title):
    focused = best_vs_baseline_rows(rollout_rows)
    methods = best_vs_baseline_series_present(focused)
    if not focused or not methods:
        return
    output_dir = Path(output_dir)
    plot_phase1_metric_bar(
        focused,
        "final_sum_state_entropy",
        output_dir / f"{phase}_best_vs_baselines_state_entropy",
        ylabel="sum state entropy",
        title=f"{phase_title} Best OCE vs Baselines State Entropy",
        methods=methods,
    )
    plot_phase1_metric_bar(
        focused,
        "final_sum_class_entropy",
        output_dir / f"{phase}_best_vs_baselines_class_entropy",
        ylabel="sum class entropy",
        title=f"{phase_title} Best OCE vs Baselines Class Entropy",
        methods=methods,
    )
    plot_phase1_metric_bar(
        focused,
        "final_true_class_probability",
        output_dir / f"{phase}_best_vs_baselines_true_class_probability",
        ylabel="P(true class)",
        title=f"{phase_title} Best OCE vs Baselines True-Class Probability",
        ylim=(0.0, 1.0),
        methods=methods,
    )
    plot_phase1_metric_bar(
        focused,
        "visibility_fraction",
        output_dir / f"{phase}_best_vs_baselines_visibility_fraction",
        ylabel="visible fraction",
        title=f"{phase_title} Best OCE vs Baselines Visibility Fraction",
        ylim=(0.0, 1.0),
        methods=methods,
    )


def write_best_vs_baseline_summary(rollout_rows, output_dir, prefix, *, phase):
    focused = best_vs_baseline_rows(rollout_rows)
    if not focused:
        return
    write_phase2_summary(
        focused,
        output_dir,
        prefix,
        phase=f"{phase}_best_vs_baselines",
    )


def write_best_vs_baseline_tables(
    rollout_rows,
    output_dir,
    prefix,
    *,
    phase,
    phase_title,
):
    focused = best_vs_baseline_rows(rollout_rows)
    methods = best_vs_baseline_series_present(focused)
    if not focused or not methods:
        return
    write_phase1_metric_table(
        focused,
        output_dir,
        prefix,
        metrics=[
            ("final_sum_state_entropy", "state entropy", 2),
            ("final_sum_class_entropy", "class entropy", 2),
            ("final_true_class_probability", "true class probability", 2),
            ("visibility_fraction", "visibility fraction", 2),
            ("distance_traveled", "distance", 2),
        ],
        filename=f"{phase}_best_vs_baselines_outcome_table",
        caption=(
            f"{phase_title} focused comparison for exact+information OCE, "
            "approximate+information OCE, visibility, and no-information baselines, "
            "reported as mean and 95\\% confidence interval across runs."
        ),
        label=f"tab:{prefix}-{phase}-best-vs-baselines-outcomes",
        methods=methods,
    )
    write_phase1_metric_table(
        focused,
        output_dir,
        prefix,
        metrics=[
            ("at_goal", "goal rate", 2),
            ("collision", "collision rate", 2),
            ("timeout", "timeout rate", 2),
        ],
        filename=f"{phase}_best_vs_baselines_completion_table",
        caption=(
            f"{phase_title} focused completion, collision, and timeout comparison "
            "for exact+information OCE, approximate+information OCE, visibility, "
            "and no-information baselines."
        ),
        label=f"tab:{prefix}-{phase}-best-vs-baselines-completion",
        methods=methods,
    )


def best_vs_baseline_pair_key(row):
    return (
        str(row.get("prefix", "")),
        parse_int(row.get("experiment")),
        str(row.get("seed", "")),
        str(row.get("scenario", "")),
    )


def build_best_vs_baseline_advantage_rows(rollout_rows):
    focused = best_vs_baseline_rows(rollout_rows)
    grouped = defaultdict(dict)
    for row in focused:
        grouped[best_vs_baseline_pair_key(row)][best_baseline_label(row)] = row

    rows = []
    for key, rows_by_method in sorted(grouped.items()):
        reference = rows_by_method.get(BEST_REFERENCE_METHOD)
        if reference is None:
            continue
        for method in BEST_VS_BASELINE_METHODS:
            if method == BEST_REFERENCE_METHOD:
                continue
            comparator = rows_by_method.get(method)
            if comparator is None:
                continue
            state_reference = parse_float(reference.get("final_sum_state_entropy"))
            state_comparator = parse_float(comparator.get("final_sum_state_entropy"))
            class_reference = parse_float(reference.get("final_sum_class_entropy"))
            class_comparator = parse_float(comparator.get("final_sum_class_entropy"))
            true_reference = parse_float(reference.get("final_true_class_probability"))
            true_comparator = parse_float(comparator.get("final_true_class_probability"))
            visibility_reference = parse_float(reference.get("visibility_fraction"))
            visibility_comparator = parse_float(comparator.get("visibility_fraction"))
            goal_reference = parse_float(reference.get("at_goal"))
            goal_comparator = parse_float(comparator.get("at_goal"))
            collision_reference = parse_float(reference.get("collision"))
            collision_comparator = parse_float(comparator.get("collision"))
            timeout_reference = parse_float(reference.get("timeout"))
            timeout_comparator = parse_float(comparator.get("timeout"))
            distance_reference = parse_float(reference.get("distance_traveled"))
            distance_comparator = parse_float(comparator.get("distance_traveled"))

            rows.append(
                {
                    "prefix": key[0],
                    "experiment": key[1],
                    "seed": key[2],
                    "scenario": key[3],
                    "reference_method": BEST_REFERENCE_METHOD,
                    "comparator_method": method,
                    "state_entropy_advantage": state_comparator - state_reference,
                    "class_entropy_advantage": class_comparator - class_reference,
                    "true_class_probability_advantage": (
                        true_reference - true_comparator
                    ),
                    "visibility_fraction_advantage": (
                        visibility_reference - visibility_comparator
                    ),
                    "goal_rate_advantage": goal_reference - goal_comparator,
                    "collision_rate_advantage": (
                        collision_comparator - collision_reference
                    ),
                    "timeout_rate_advantage": timeout_comparator - timeout_reference,
                    "distance_delta": distance_reference - distance_comparator,
                }
            )
    return rows


def write_best_vs_baseline_advantage_csv(rows, output_dir, prefix, *, phase):
    if not rows:
        return
    output_path = Path(output_dir) / f"{prefix}_{phase}_best_vs_baselines_advantage.csv"
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_best_vs_baseline_advantage_table(rows, output_dir, prefix, *, phase, phase_title):
    if not rows:
        return
    metrics = [
        ("state_entropy_advantage", "state entropy advantage", 2),
        ("class_entropy_advantage", "class entropy advantage", 2),
        ("true_class_probability_advantage", "true probability advantage", 2),
        ("visibility_fraction_advantage", "visibility advantage", 2),
        ("goal_rate_advantage", "goal advantage", 2),
        ("collision_rate_advantage", "collision advantage", 2),
        ("timeout_rate_advantage", "timeout advantage", 2),
        ("distance_delta", "distance delta", 2),
    ]
    table_rows = []
    for row in rows:
        record = {"comparator": row["comparator_method"]}
        for metric, _display, _decimals in metrics:
            record[metric] = parse_float(row.get(metric))
        table_rows.append(record)
    df = TableDataFrame(table_rows)
    columns_spec = {
        metric: {
            "display": display,
            "type": "ci",
            "highlight": "none",
            "decimals": decimals,
        }
        for metric, display, decimals in metrics
    }
    categories = {
        "L1": {
            "name": "comparator",
            "column": "comparator",
            "labels": [
                method
                for method in BEST_VS_BASELINE_METHODS
                if method != BEST_REFERENCE_METHOD
            ],
        }
    }
    write_table(
        df,
        categories,
        columns_spec=columns_spec,
        caption=(
            f"Paired {phase_title} advantage of "
            f"{BEST_REFERENCE_METHOD} over each comparator. Positive values favor "
            "the exact+information reference except distance delta, where positive "
            "means the reference traveled farther."
        ),
        label=f"tab:{prefix}-{phase}-best-vs-baselines-advantage",
        title=f"{phase}_best_vs_baselines_advantage_table",
        output_file_path=Path(output_dir)
        / f"{prefix}_{phase}_best_vs_baselines_advantage_table.tex",
    )


def write_best_vs_baseline_outputs(rollout_rows, output_dir, prefix, *, phase, phase_title):
    focused = best_vs_baseline_rows(rollout_rows)
    if not focused:
        return
    plot_best_vs_baselines(focused, output_dir, phase=phase, phase_title=phase_title)
    write_best_vs_baseline_summary(focused, output_dir, prefix, phase=phase)
    write_best_vs_baseline_tables(
        focused,
        output_dir,
        prefix,
        phase=phase,
        phase_title=phase_title,
    )
    advantage_rows = build_best_vs_baseline_advantage_rows(focused)
    write_best_vs_baseline_advantage_csv(
        advantage_rows,
        output_dir,
        prefix,
        phase=phase,
    )
    write_best_vs_baseline_advantage_table(
        advantage_rows,
        output_dir,
        prefix,
        phase=phase,
        phase_title=phase_title,
    )


def _paired_delta(exact_value, comparator_value, *, higher_is_better=False):
    exact_value = parse_float(exact_value)
    comparator_value = parse_float(comparator_value)
    if not (np.isfinite(exact_value) and np.isfinite(comparator_value)):
        return np.nan
    if higher_is_better:
        return exact_value - comparator_value
    return comparator_value - exact_value


def build_exact_visibility_winner_rows(rollout_rows):
    focused = best_vs_baseline_rows(rollout_rows)
    grouped = defaultdict(dict)
    for row in focused:
        method = best_baseline_label(row)
        if method in {BEST_REFERENCE_METHOD, VISIBILITY_BASELINE_METHOD}:
            grouped[best_vs_baseline_pair_key(row)][method] = row

    rows = []
    for key, rows_by_method in sorted(grouped.items()):
        exact = rows_by_method.get(BEST_REFERENCE_METHOD)
        visibility = rows_by_method.get(VISIBILITY_BASELINE_METHOD)
        if exact is None or visibility is None:
            continue

        exact_class = parse_float(exact.get("final_sum_class_entropy"))
        visibility_class = parse_float(visibility.get("final_sum_class_entropy"))
        if not (np.isfinite(exact_class) and np.isfinite(visibility_class)):
            winner = "incomplete"
        elif exact_class < visibility_class:
            winner = "exact+info wins"
        elif visibility_class < exact_class:
            winner = "visibility wins"
        else:
            winner = "tie"

        rows.append(
            {
                "prefix": key[0],
                "experiment": key[1],
                "seed": key[2],
                "scenario": key[3],
                "winner_cluster": winner,
                "case_count": 1.0,
                "exact_final_sum_state_entropy": parse_float(
                    exact.get("final_sum_state_entropy")
                ),
                "visibility_final_sum_state_entropy": parse_float(
                    visibility.get("final_sum_state_entropy")
                ),
                "state_entropy_advantage": _paired_delta(
                    exact.get("final_sum_state_entropy"),
                    visibility.get("final_sum_state_entropy"),
                ),
                "exact_final_sum_class_entropy": exact_class,
                "visibility_final_sum_class_entropy": visibility_class,
                "class_entropy_advantage": _paired_delta(
                    exact.get("final_sum_class_entropy"),
                    visibility.get("final_sum_class_entropy"),
                ),
                "exact_true_class_probability": parse_float(
                    exact.get("final_true_class_probability")
                ),
                "visibility_true_class_probability": parse_float(
                    visibility.get("final_true_class_probability")
                ),
                "true_class_probability_advantage": _paired_delta(
                    exact.get("final_true_class_probability"),
                    visibility.get("final_true_class_probability"),
                    higher_is_better=True,
                ),
                "exact_visibility_fraction": parse_float(
                    exact.get("visibility_fraction")
                ),
                "visibility_visibility_fraction": parse_float(
                    visibility.get("visibility_fraction")
                ),
                "visibility_fraction_advantage": _paired_delta(
                    exact.get("visibility_fraction"),
                    visibility.get("visibility_fraction"),
                    higher_is_better=True,
                ),
                "exact_collision": parse_float(exact.get("collision")),
                "visibility_collision": parse_float(visibility.get("collision")),
                "collision_advantage": _paired_delta(
                    exact.get("collision"),
                    visibility.get("collision"),
                ),
                "exact_timeout": parse_float(exact.get("timeout")),
                "visibility_timeout": parse_float(visibility.get("timeout")),
                "timeout_advantage": _paired_delta(
                    exact.get("timeout"),
                    visibility.get("timeout"),
                ),
                "exact_distance_traveled": parse_float(
                    exact.get("distance_traveled")
                ),
                "visibility_distance_traveled": parse_float(
                    visibility.get("distance_traveled")
                ),
                "distance_delta": _paired_delta(
                    visibility.get("distance_traveled"),
                    exact.get("distance_traveled"),
                ),
                "exact_failure_reason": exact.get("failure_reason", ""),
                "visibility_failure_reason": visibility.get("failure_reason", ""),
            }
        )
    return rows


def write_exact_visibility_winner_cases(rows, output_dir, prefix, *, phase):
    if not rows:
        return
    output_path = Path(output_dir) / f"{prefix}_{phase}_exact_vs_visibility_winner_cases.csv"
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_values(rows, metric):
    values = np.asarray([parse_float(row.get(metric)) for row in rows], dtype=float)
    return values[np.isfinite(values)]


def summarize_exact_visibility_winner_clusters(rows):
    if not rows:
        return []
    metrics = [
        "class_entropy_advantage",
        "state_entropy_advantage",
        "true_class_probability_advantage",
        "visibility_fraction_advantage",
        "collision_advantage",
        "timeout_advantage",
        "distance_delta",
        "exact_final_sum_class_entropy",
        "visibility_final_sum_class_entropy",
        "exact_true_class_probability",
        "visibility_true_class_probability",
        "exact_collision",
        "visibility_collision",
    ]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("winner_cluster", "incomplete")].append(row)

    summary = []
    for cluster in EXACT_VISIBILITY_WINNER_LABELS:
        cluster_rows = grouped.get(cluster, [])
        if not cluster_rows:
            continue
        record = {
            "winner_cluster": cluster,
            "n_cases": len(cluster_rows),
            "n_finite_class_entropy_pairs": len(
                _aggregate_values(cluster_rows, "class_entropy_advantage")
            ),
        }
        for metric in metrics:
            values = _aggregate_values(cluster_rows, metric)
            record[f"{metric}_mean"] = (
                float(np.mean(values)) if values.size else np.nan
            )
            record[f"{metric}_median"] = (
                float(np.median(values)) if values.size else np.nan
            )
            record[f"{metric}_iqr"] = (
                float(np.percentile(values, 75) - np.percentile(values, 25))
                if values.size
                else np.nan
            )
        summary.append(record)
    return summary


def write_exact_visibility_winner_summary(rows, output_dir, prefix, *, phase):
    summary = summarize_exact_visibility_winner_clusters(rows)
    if not summary:
        return
    output_path = Path(output_dir) / f"{prefix}_{phase}_exact_vs_visibility_winner_summary.csv"
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)


def write_exact_visibility_winner_table(rows, output_dir, prefix, *, phase, phase_title):
    if not rows:
        return
    columns_spec = {
        "case_count": {
            "display": "cases",
            "type": "sum",
            "highlight": "none",
            "decimals": 0,
        },
        "class_entropy_advantage": {
            "display": "class advantage",
            "type": "ci",
            "highlight": "none",
            "decimals": 2,
        },
        "state_entropy_advantage": {
            "display": "state advantage",
            "type": "ci",
            "highlight": "none",
            "decimals": 2,
        },
        "true_class_probability_advantage": {
            "display": "true prob advantage",
            "type": "ci",
            "highlight": "none",
            "decimals": 2,
        },
        "visibility_fraction_advantage": {
            "display": "visibility advantage",
            "type": "ci",
            "highlight": "none",
            "decimals": 2,
        },
        "collision_advantage": {
            "display": "collision advantage",
            "type": "ci",
            "highlight": "none",
            "decimals": 2,
        },
        "exact_final_sum_class_entropy": {
            "display": "exact class",
            "type": "ci",
            "highlight": "none",
            "decimals": 2,
        },
        "visibility_final_sum_class_entropy": {
            "display": "vis class",
            "type": "ci",
            "highlight": "none",
            "decimals": 2,
        },
    }
    write_table(
        TableDataFrame(rows),
        {
            "L1": {
                "name": "winner",
                "column": "winner_cluster",
                "labels": EXACT_VISIBILITY_WINNER_LABELS,
            }
        },
        columns_spec=columns_spec,
        caption=(
            f"{phase_title} exact+information OCE versus visibility clustered by "
            "the lower final class-entropy winner. Positive advantage values favor "
            "exact+information OCE."
        ),
        label=f"tab:{prefix}-{phase}-exact-vs-visibility-winner-clusters",
        title=f"{phase}_exact_vs_visibility_winner_cluster_table",
        output_file_path=Path(output_dir)
        / f"{prefix}_{phase}_exact_vs_visibility_winner_cluster_table.tex",
    )


def write_exact_visibility_winner_outputs(
    rollout_rows,
    output_dir,
    prefix,
    *,
    phase,
    phase_title,
):
    rows = build_exact_visibility_winner_rows(rollout_rows)
    if not rows:
        return
    write_exact_visibility_winner_cases(rows, output_dir, prefix, phase=phase)
    write_exact_visibility_winner_summary(rows, output_dir, prefix, phase=phase)
    write_exact_visibility_winner_table(
        rows,
        output_dir,
        prefix,
        phase=phase,
        phase_title=phase_title,
    )


def _pair_key(row, scoring_mode):
    return (
        parse_int(row.get("experiment")),
        str(row.get("seed", "")),
        str(scoring_mode),
    )


def _phase1_regret_index(phase1_case_rows):
    index = {}
    for row in phase1_case_rows or []:
        parsed = parse_oce_selector(row.get("method"))
        if not parsed or parsed["discrete"] != "approximate":
            continue
        key = _pair_key(row, parsed["scoring"])
        index.setdefault(key, []).append(row)
    return index


def _best_phase1_regret_row(candidates, approximate_rollout):
    if not candidates:
        return None
    candidate_hash = approximate_rollout.get("candidate_set_hash", "")
    initial_tick = parse_int(approximate_rollout.get("initial_tick"), default=-1)
    for row in candidates:
        if (
            row.get("candidate_set_hash", "") == candidate_hash
            and parse_int(row.get("tick"), default=-2) == initial_tick
        ):
            return row
    for row in candidates:
        if row.get("candidate_set_hash", "") == candidate_hash:
            return row
    return candidates[0]


def build_phase2_exact_approx_pairs(rollout_rows, phase1_case_rows=None):
    grouped = defaultdict(dict)
    for row in rollout_rows or []:
        parsed = parse_oce_selector(row.get("selector_method") or row.get("method"))
        if not parsed:
            continue
        key = _pair_key(row, parsed["scoring"])
        grouped[key][parsed["discrete"]] = row

    phase1_regrets = _phase1_regret_index(phase1_case_rows)
    pairs = []
    for key, rows_by_discrete in sorted(grouped.items()):
        exact = rows_by_discrete.get("exact")
        approximate = rows_by_discrete.get("approximate")
        if exact is None or approximate is None:
            continue
        scoring_mode = key[2]
        phase1_row = _best_phase1_regret_row(phase1_regrets.get(key), approximate)
        exact_selected = parse_int(exact.get("selected_index"), default=-1)
        approximate_selected = parse_int(approximate.get("selected_index"), default=-1)

        record = {
            "prefix": approximate.get("prefix") or exact.get("prefix"),
            "experiment": key[0],
            "seed": key[1],
            "scoring_mode": scoring_mode,
            "exact_selector_method": exact.get("selector_method") or exact.get("method"),
            "approximate_selector_method": (
                approximate.get("selector_method") or approximate.get("method")
            ),
            "exact_candidate_set_hash": exact.get("candidate_set_hash", ""),
            "approximate_candidate_set_hash": approximate.get("candidate_set_hash", ""),
            "same_candidate_set": int(
                exact.get("candidate_set_hash", "")
                == approximate.get("candidate_set_hash", "")
            ),
            "exact_selected_index": exact_selected,
            "approximate_selected_index": approximate_selected,
            "same_selected_index": int(exact_selected == approximate_selected),
            "selected_index_delta": approximate_selected - exact_selected,
            "phase1_exact_reference_regret": (
                parse_float(phase1_row.get("exact_reference_regret"))
                if phase1_row is not None
                else np.nan
            ),
            "phase1_paired_exact_reference_regret": (
                parse_float(phase1_row.get("paired_exact_reference_regret"))
                if phase1_row is not None
                else np.nan
            ),
            "phase1_reference_rank": (
                parse_float(phase1_row.get("reference_rank"))
                if phase1_row is not None
                else np.nan
            ),
        }
        delta_metrics = {
            "final_sum_state_entropy": "state_entropy",
            "final_sum_class_entropy": "class_entropy",
            "final_true_class_probability": "true_class_probability",
            "visibility_fraction": "visibility_fraction",
            "distance_traveled": "distance",
            "at_goal": "at_goal",
            "collision": "collision",
            "timeout": "timeout",
        }
        for source_metric, output_metric in delta_metrics.items():
            exact_value = parse_float(exact.get(source_metric))
            approximate_value = parse_float(approximate.get(source_metric))
            record[f"exact_{output_metric}"] = exact_value
            record[f"approximate_{output_metric}"] = approximate_value
            record[f"approx_minus_exact_{output_metric}"] = (
                approximate_value - exact_value
                if np.isfinite(exact_value) and np.isfinite(approximate_value)
                else np.nan
            )
        pairs.append(record)
    return pairs


def write_phase2_exact_approx_pairs(pairs, output_dir, prefix):
    if not pairs:
        return
    output_path = Path(output_dir) / f"{prefix}_phase2_exact_approx_pairs.csv"
    fieldnames = list(pairs[0].keys())
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pairs)


def write_phase2_exact_approx_pair_table(pairs, output_dir, prefix):
    if not pairs:
        return
    metrics = [
        ("same_selected_index", "same selection", 2),
        ("phase1_exact_reference_regret", "phase 1 regret", 3),
        ("phase1_paired_exact_reference_regret", "paired regret", 3),
        ("approx_minus_exact_state_entropy", "state entropy delta", 2),
        ("approx_minus_exact_class_entropy", "class entropy delta", 2),
        ("approx_minus_exact_true_class_probability", "true prob delta", 2),
        ("approx_minus_exact_visibility_fraction", "visibility delta", 2),
        ("approx_minus_exact_distance", "distance delta", 2),
    ]
    rows = []
    for pair in pairs:
        row = {"scoring_mode": pair["scoring_mode"]}
        for metric, _display, _decimals in metrics:
            row[metric] = parse_float(pair.get(metric))
        rows.append(row)
    df = TableDataFrame(rows)
    columns_spec = {
        metric: {
            "display": display,
            "type": "ci",
            "highlight": "none",
            "decimals": decimals,
        }
        for metric, display, decimals in metrics
    }
    categories = {
        "L1": {
            "name": "scoring mode",
            "column": "scoring_mode",
            "labels": [
                "entropy",
                "oc_entropy",
                "entropy_plus_information",
                "oc_entropy_plus_information",
                "information_only",
            ],
        }
    }
    write_table(
        df,
        categories,
        columns_spec=columns_spec,
        caption=(
            "Paired Phase 2 approximate-minus-exact diagnostics by scoring mode. "
            "Negative entropy deltas favor approximate; positive true-probability "
            "and visibility deltas favor approximate."
        ),
        label=f"tab:{prefix}-phase2-exact-approx-pairs",
        title="phase2_exact_approx_pair_table",
        output_file_path=Path(output_dir)
        / f"{prefix}_phase2_exact_approx_pair_table.tex",
    )


def plot_phase2_exact_approx_pair_deltas(pairs, output_dir):
    if not pairs:
        return
    rows = []
    for pair in pairs:
        for metric in (
            "approx_minus_exact_state_entropy",
            "approx_minus_exact_class_entropy",
            "approx_minus_exact_true_class_probability",
            "approx_minus_exact_visibility_fraction",
        ):
            value = parse_float(pair.get(metric))
            if np.isfinite(value):
                rows.append(
                    {
                        "method": pair["scoring_mode"],
                        "metric": metric,
                        "value": value,
                    }
                )
    if not rows:
        return
    output_dir = Path(output_dir)
    metric_titles = {
        "approx_minus_exact_state_entropy": (
            "phase2_exact_approx_state_entropy_delta",
            "state entropy delta",
            "Approximate Minus Exact State Entropy",
        ),
        "approx_minus_exact_class_entropy": (
            "phase2_exact_approx_class_entropy_delta",
            "class entropy delta",
            "Approximate Minus Exact Class Entropy",
        ),
        "approx_minus_exact_true_class_probability": (
            "phase2_exact_approx_true_probability_delta",
            "true probability delta",
            "Approximate Minus Exact True-Class Probability",
        ),
        "approx_minus_exact_visibility_fraction": (
            "phase2_exact_approx_visibility_delta",
            "visibility delta",
            "Approximate Minus Exact Visibility Fraction",
        ),
    }
    for metric, (filename, ylabel, title) in metric_titles.items():
        metric_rows = [
            {
                "method": row["method"],
                metric: row["value"],
            }
            for row in rows
            if row["metric"] == metric
        ]
        plot_phase1_metric_box(
            metric_rows,
            metric,
            output_dir / filename,
            ylabel=ylabel,
            title=title,
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot target class-identification experiment CSV outputs, combining "
            "multiple experiment IDs for one required prefix."
        )
    )
    parser.add_argument("log_dir", nargs="?", help="Directory containing experiment CSVs.")
    parser.add_argument(
        "--prefix",
        required=True,
        help="Required log filename prefix to load, e.g. 'simple-robot'.",
    )
    parser.add_argument("--target-beliefs", action="append", default=None)
    parser.add_argument("--summary", action="append", default=None)
    parser.add_argument(
        "--phase1-case-methods",
        action="append",
        default=None,
        help="Phase 1 frozen-candidate case-method CSV path. Auto-detected by prefix when omitted.",
    )
    parser.add_argument(
        "--phase1-candidates",
        action="append",
        default=None,
        help="Phase 1 per-candidate CSV path. Auto-detected by prefix when omitted.",
    )
    parser.add_argument(
        "--phase2-rollouts",
        action="append",
        default=None,
        help="Phase 2 common-policy rollout CSV path. Auto-detected by prefix when omitted.",
    )
    parser.add_argument(
        "--phase3-rollouts",
        action="append",
        default=None,
        help="Phase 3 closed-loop rollout CSV path. Auto-detected by prefix when omitted.",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    configure_plot_style()
    log_dir = Path(args.log_dir or ".")
    output_dir = Path(args.out) if args.out else log_dir / f"{args.prefix}_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    phase1_case_rows = load_phase1_rows_with_fallback(
        log_dir,
        args.prefix,
        explicit_paths=args.phase1_case_methods,
    )
    phase1_candidate_rows = load_phase1_rows_for_prefix(
        log_dir,
        args.prefix,
        PHASE1_CANDIDATE_SUFFIX,
        explicit_paths=args.phase1_candidates,
    )
    phase2_rollout_rows = load_phase2_rows_for_prefix(
        log_dir,
        args.prefix,
        PHASE2_ROLLOUT_SUFFIX,
        explicit_paths=args.phase2_rollouts,
    )
    phase3_rollout_rows = load_phase2_rows_for_prefix(
        log_dir,
        args.prefix,
        PHASE3_ROLLOUT_SUFFIX,
        explicit_paths=args.phase3_rollouts,
    )

    try:
        target_rows = load_rows_for_prefix(
            log_dir,
            args.prefix,
            TARGET_SUFFIX,
            explicit_paths=args.target_beliefs,
        )
        summary_rows = load_rows_for_prefix(
            log_dir,
            args.prefix,
            SUMMARY_SUFFIX,
            explicit_paths=args.summary,
        )
    except FileNotFoundError:
        if not phase1_case_rows and not phase2_rollout_rows and not phase3_rollout_rows:
            raise
        target_rows = []
        summary_rows = []

    if summary_rows:
        plot_total_uncertainty(summary_rows, output_dir)
        plot_robot_motion(summary_rows, output_dir)
        write_aggregate_summary(summary_rows, output_dir, args.prefix)
    if target_rows:
        plot_per_target_entropy(target_rows, output_dir)
        plot_visibility(target_rows, output_dir)
        plot_true_class_probability(target_rows, output_dir)
    if target_rows and summary_rows:
        write_entropy_tables(target_rows, summary_rows, output_dir, args.prefix)
    if phase1_case_rows:
        plot_phase1_results(phase1_case_rows, output_dir)
        write_phase1_aggregate_summary(phase1_case_rows, output_dir, args.prefix)
        write_phase1_tables(phase1_case_rows, output_dir, args.prefix)
    if phase2_rollout_rows:
        plot_phase2_results(phase2_rollout_rows, output_dir)
        write_phase2_summary(phase2_rollout_rows, output_dir, args.prefix)
        write_phase2_tables(phase2_rollout_rows, output_dir, args.prefix)
        write_best_vs_baseline_outputs(
            phase2_rollout_rows,
            output_dir,
            args.prefix,
            phase="phase2",
            phase_title="Phase 2",
        )
        exact_approx_pairs = build_phase2_exact_approx_pairs(
            phase2_rollout_rows,
            phase1_case_rows,
        )
        write_phase2_exact_approx_pairs(exact_approx_pairs, output_dir, args.prefix)
        write_phase2_exact_approx_pair_table(
            exact_approx_pairs,
            output_dir,
            args.prefix,
        )
        plot_phase2_exact_approx_pair_deltas(exact_approx_pairs, output_dir)
    if phase3_rollout_rows:
        plot_phase2_results(
            phase3_rollout_rows,
            output_dir,
            phase="phase3",
            phase_title="Phase 3 Closed-Loop",
        )
        write_phase2_summary(
            phase3_rollout_rows,
            output_dir,
            args.prefix,
            phase="phase3",
        )
        write_phase2_tables(
            phase3_rollout_rows,
            output_dir,
            args.prefix,
            phase="phase3",
            phase_title="Phase 3 closed-loop",
        )
        write_best_vs_baseline_outputs(
            phase3_rollout_rows,
            output_dir,
            args.prefix,
            phase="phase3",
            phase_title="Phase 3 closed-loop",
        )
        write_exact_visibility_winner_outputs(
            phase3_rollout_rows,
            output_dir,
            args.prefix,
            phase="phase3",
            phase_title="Phase 3 closed-loop",
        )

    outputs = ["experiment plots", "aggregate summary", "LaTeX tables"]
    if phase1_case_rows:
        outputs.append("Phase 1 plots/tables")
    if phase1_candidate_rows:
        outputs.append("Phase 1 candidate diagnostics")
    if phase2_rollout_rows:
        outputs.append("Phase 2 plots/tables")
    if phase3_rollout_rows:
        outputs.append("Phase 3 plots/tables")
    print(f"Wrote {', '.join(outputs)} to {output_dir}")


if __name__ == "__main__":
    main()
