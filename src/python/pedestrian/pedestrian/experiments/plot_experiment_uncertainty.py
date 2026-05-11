#!/usr/bin/env python3
import argparse
import csv
import re
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - exercised only on minimal envs
    sns = None


SUMMARY_SUFFIX = "uncertainty_summary"
TARGET_SUFFIX = "target_beliefs"
METHOD_ORDER = ["oce", "visibility", "none"]


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
            }
    return {"prefix": prefix, "experiment": 0, "method": ""}


def normalize_method(method):
    method = str(method or "").strip().lower()
    return "visibility" if method == "vis" else method


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
            row["_source_file"] = str(path)
            rows.append(row)
    return rows


def methods_present(rows):
    methods = sorted({normalize_method(row.get("method")) for row in rows if row.get("method")})
    return [method for method in METHOD_ORDER if method in methods] + [
        method for method in methods if method not in METHOD_ORDER
    ]


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
        hue.append(normalize_method(row.get("method") or "run"))
        units.append(str(row.get("experiment", "")))
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
            hue_order=methods_present(rows) or None,
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
    ax.legend(title="method")
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
    hue_order = methods_present(rows) or None
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
            title="method",
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


def aggregate_final_summary(summary_rows):
    latest_by_run = {}
    for row in summary_rows:
        key = (
            row.get("prefix", ""),
            parse_int(row.get("experiment")),
            normalize_method(row.get("method")),
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
        grouped[normalize_method(row.get("method"))].append(row)

    summary = []
    for method in sorted(grouped):
        rows = grouped[method]
        record = {"method": method, "n_experiments": len(rows)}
        for metric in metrics:
            values = np.asarray([parse_float(row.get(metric)) for row in rows], dtype=float)
            values = values[np.isfinite(values)]
            record[f"{metric}_mean"] = float(np.mean(values)) if values.size else np.nan
            record[f"{metric}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        summary.append(record)
    return summary


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
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    configure_plot_style()
    log_dir = Path(args.log_dir or ".")
    output_dir = Path(args.out) if args.out else log_dir / f"{args.prefix}_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

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

    plot_total_uncertainty(summary_rows, output_dir)
    plot_robot_motion(summary_rows, output_dir)
    plot_per_target_entropy(target_rows, output_dir)
    plot_visibility(target_rows, output_dir)
    plot_true_class_probability(target_rows, output_dir)
    write_aggregate_summary(summary_rows, output_dir, args.prefix)
    print(f"Wrote experiment plots and aggregate summary to {output_dir}")


if __name__ == "__main__":
    main()
