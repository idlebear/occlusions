#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_float(value):
    if value in (None, ""):
        return np.nan
    try:
        return float(value)
    except ValueError:
        return np.nan


def read_csv_rows(path):
    with Path(path).open("r", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def grouped_by_target(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[str(row["track_id"])].append(row)
    for values in groups.values():
        values.sort(key=lambda row: parse_float(row["time_s"]))
    return dict(sorted(groups.items(), key=lambda item: item[0]))


def plot_total_uncertainty(summary_rows, output_dir):
    if not summary_rows:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    time_s = [parse_float(row["time_s"]) for row in summary_rows]
    total = [parse_float(row["total_uncertainty"]) for row in summary_rows]
    mode_mean = [parse_float(row["mean_mode_entropy"]) for row in summary_rows]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(time_s, total, label="total mode uncertainty", linewidth=2.0)
    ax.plot(time_s, mode_mean, label="mean mode entropy", linewidth=1.5)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("entropy")
    ax.set_title("Tracked Target Class Uncertainty")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "total_uncertainty.png", dpi=160)
    plt.close(fig)


def plot_per_target_entropy(target_rows, output_dir):
    groups = grouped_by_target(target_rows)
    if not groups:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for track_id, rows in groups.items():
        time_s = [parse_float(row["time_s"]) for row in rows]
        axes[0].plot(
            time_s,
            [parse_float(row["state_entropy"]) for row in rows],
            label=f"track {track_id}",
        )
        axes[1].plot(
            time_s,
            [parse_float(row["mode_entropy"]) for row in rows],
            label=f"track {track_id}",
        )
    axes[0].set_ylabel("state entropy")
    axes[1].set_ylabel("mode entropy")
    axes[1].set_xlabel("time (s)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize="small")
    axes[0].set_title("Per-Target Belief Entropy")
    fig.tight_layout()
    fig.savefig(output_dir / "per_target_entropy.png", dpi=160)
    plt.close(fig)


def plot_visibility(target_rows, output_dir):
    groups = grouped_by_target(target_rows)
    if not groups:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    yticks = []
    ylabels = []
    for index, (track_id, rows) in enumerate(groups.items()):
        time_s = np.asarray([parse_float(row["time_s"]) for row in rows], dtype=float)
        visible = np.asarray(
            [str(row.get("visible", "")).lower() == "true" for row in rows],
            dtype=bool,
        )
        ax.scatter(time_s[visible], np.full(np.count_nonzero(visible), index), s=14)
        ax.scatter(
            time_s[~visible],
            np.full(np.count_nonzero(~visible), index),
            s=14,
            marker="x",
        )
        yticks.append(index)
        ylabels.append(f"track {track_id}")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("time (s)")
    ax.set_title("Tracked Target Visibility")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "visibility_timeline.png", dpi=160)
    plt.close(fig)


def plot_true_class_probability(target_rows, output_dir):
    groups = grouped_by_target(
        [
            row
            for row in target_rows
            if int(float(row.get("true_class_id") or -1)) >= 0
            and np.isfinite(parse_float(row.get("true_class_probability")))
        ]
    )
    if not groups:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for track_id, rows in groups.items():
        ax.plot(
            [parse_float(row["time_s"]) for row in rows],
            [parse_float(row["true_class_probability"]) for row in rows],
            label=f"track {track_id}",
        )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("P(true class)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("True Destination-Class Probability")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize="small")
    fig.tight_layout()
    fig.savefig(output_dir / "true_class_probability.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot target class-identification experiment CSV outputs."
    )
    parser.add_argument("log_dir", nargs="?", help="Directory containing experiment CSVs.")
    parser.add_argument("--target-beliefs", default=None)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    log_dir = Path(args.log_dir or ".")
    target_path = Path(args.target_beliefs) if args.target_beliefs else log_dir / "target_beliefs.csv"
    summary_path = Path(args.summary) if args.summary else log_dir / "uncertainty_summary.csv"
    output_dir = Path(args.out) if args.out else log_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    target_rows = read_csv_rows(target_path)
    summary_rows = read_csv_rows(summary_path)
    plot_total_uncertainty(summary_rows, output_dir)
    plot_per_target_entropy(target_rows, output_dir)
    plot_visibility(target_rows, output_dir)
    plot_true_class_probability(target_rows, output_dir)
    print(f"Wrote experiment plots to {output_dir}")


if __name__ == "__main__":
    main()
