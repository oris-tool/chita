import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COMPACT_PLOT_DPI = 80
GRID_ROWS = 4
GRID_COLS = 2


DEFAULT_ROOT = "results/case_exports"


def numeric_subject_key(subject_id):
    try:
        return int(subject_id)
    except ValueError:
        return subject_id


def chunked(sequence, chunk_size):
    for start in range(0, len(sequence), chunk_size):
        yield sequence[start:start + chunk_size]


def hide_unused_axes(axes, used_count):
    for axis in axes[used_count:]:
        axis.set_visible(False)


def read_csv_rows(path):
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_name_for_case(case_dir):
    metadata_path = case_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        run_id = metadata.get("run_id")
        if run_id:
            return run_id
    return case_dir.name


def output_path(base_path, page_index, multiple_pages):
    if not multiple_pages:
        return base_path
    return base_path.with_name(f"{base_path.stem}_page_{page_index}{base_path.suffix}")


def read_probability_curve_csv(path):
    by_subject = {}
    for row in read_csv_rows(path):
        subject_id = str(row["subject_id"])
        by_subject.setdefault(subject_id, []).append(
            {
                "time_index": int(row["time_index"]),
                "time": float(row["time"]),
                "ground_truth_mean_probability": float(row["ground_truth_mean_probability"]),
                "ground_truth_lower_band": float(row["ground_truth_lower_band"]),
                "ground_truth_upper_band": float(row["ground_truth_upper_band"]),
                "analysis_probability": float(row["analysis_probability"]),
                "simulation_probability": float(row["simulation_probability"]),
            }
        )

    for rows in by_subject.values():
        rows.sort(key=lambda row: row["time_index"])
    return {
        subject_id: by_subject[subject_id]
        for subject_id in sorted(by_subject, key=numeric_subject_key)
    }


def plot_probability_curve_from_csv(csv_path, output_base_path, run_name):
    data = read_probability_curve_csv(csv_path)
    subject_ids = list(data.keys())
    subjects_per_page = GRID_ROWS * GRID_COLS
    multiple_pages = len(subject_ids) > subjects_per_page
    generated_paths = []

    for page_index, subject_page in enumerate(chunked(subject_ids, subjects_per_page), start=1):
        fig, axes = plt.subplots(
            GRID_ROWS,
            GRID_COLS,
            figsize=(14, 16),
            sharex=True,
            sharey=True,
        )
        flat_axes = list(axes.flat)
        legend_handles = None

        for axis, subject_id in zip(flat_axes, subject_page):
            rows = data[subject_id]
            time_axis = np.asarray([row["time"] for row in rows], dtype=float)
            ground_truth_values = np.asarray(
                [row["ground_truth_mean_probability"] for row in rows],
                dtype=float,
            )
            lower = np.asarray([row["ground_truth_lower_band"] for row in rows], dtype=float)
            upper = np.asarray([row["ground_truth_upper_band"] for row in rows], dtype=float)
            analysis_values = np.asarray(
                [row["analysis_probability"] for row in rows],
                dtype=float,
            )
            baseline_values = np.asarray(
                [row["simulation_probability"] for row in rows],
                dtype=float,
            )

            ground_truth_line, = axis.plot(
                time_axis,
                ground_truth_values,
                linewidth=1.6,
                color="tab:blue",
                label="Ground truth mean",
            )
            ground_truth_band = axis.fill_between(
                time_axis,
                lower,
                upper,
                alpha=0.2,
                color="tab:blue",
                label="Ground truth DKW band",
            )
            analysis_line, = axis.plot(
                time_axis,
                analysis_values,
                linewidth=1.6,
                color="tab:orange",
                label="Java analysis",
            )
            baseline_line, = axis.plot(
                time_axis,
                baseline_values,
                linewidth=1.6,
                color="tab:green",
                label="Simulation",
            )
            axis.set_title(f"Subject {subject_id}")
            axis.set_ylim(0.0, 1.0)
            axis.grid(True, alpha=0.3)

            if legend_handles is None:
                legend_handles = [
                    ground_truth_line,
                    ground_truth_band,
                    analysis_line,
                    baseline_line,
                ]

        hide_unused_axes(flat_axes, len(subject_page))
        fig.suptitle(f"{run_name} Subject Curves", fontsize=14)
        fig.supxlabel("Time")
        fig.supylabel("Probability")
        if legend_handles is not None:
            fig.legend(
                legend_handles,
                [
                    "Ground truth mean",
                    "Ground truth DKW band",
                    "Java analysis",
                    "Simulation",
                ],
                loc="upper center",
                ncol=4,
                frameon=False,
            )
        fig.tight_layout(rect=(0.03, 0.03, 1.0, 0.94))

        plot_path = output_path(output_base_path, page_index, multiple_pages)
        fig.savefig(plot_path, dpi=COMPACT_PLOT_DPI)
        plt.close(fig)
        generated_paths.append(plot_path)

    return generated_paths


def read_reliability_csv(path):
    by_subject = {}
    for row in read_csv_rows(path):
        subject_id = str(row["subject_id"])
        by_subject.setdefault(subject_id, []).append(
            {
                "time_index": int(row["time_index"]),
                "predicted_probability": float(row["predicted_probability"]),
                "ground_truth_probability": float(row["ground_truth_probability"]),
            }
        )

    for rows in by_subject.values():
        rows.sort(key=lambda row: row["time_index"])
    return {
        subject_id: by_subject[subject_id]
        for subject_id in sorted(by_subject, key=numeric_subject_key)
    }


def plot_reliability_from_csv(csv_path, output_base_path):
    data = read_reliability_csv(csv_path)
    subject_ids = list(data.keys())
    subjects_per_page = GRID_ROWS * GRID_COLS
    multiple_pages = len(subject_ids) > subjects_per_page
    generated_paths = []

    for page_index, subject_page in enumerate(chunked(subject_ids, subjects_per_page), start=1):
        fig, axes = plt.subplots(
            GRID_ROWS,
            GRID_COLS,
            figsize=(12, 16),
            sharex=True,
            sharey=True,
        )
        flat_axes = list(axes.flat)

        for axis, subject_id in zip(flat_axes, subject_page):
            rows = data[subject_id]
            p = [row["predicted_probability"] for row in rows]
            g = [row["ground_truth_probability"] for row in rows]
            axis.scatter(
                p,
                g,
                color="tab:blue",
                alpha=0.45,
                s=12,
                edgecolors="none",
            )
            axis.plot([0, 1], [0, 1], "r--", linewidth=1.0)
            axis.set_title(f"Subject {subject_id}")
            axis.set_xlim([-0.05, 1.05])
            axis.set_ylim([-0.05, 1.05])
            axis.grid(True, alpha=0.3)

        hide_unused_axes(flat_axes, len(subject_page))
        fig.suptitle("Reliability Diagrams", fontsize=14)
        fig.supxlabel("Predicted Probability p(t)")
        fig.supylabel("Ground Truth g(t)")
        fig.tight_layout(rect=(0.03, 0.03, 1.0, 0.96))

        plot_path = output_path(output_base_path, page_index, multiple_pages)
        fig.savefig(plot_path, dpi=COMPACT_PLOT_DPI)
        plt.close(fig)
        generated_paths.append(plot_path)

    return generated_paths


def remove_old_page_outputs(plots_dir, stem):
    for path in plots_dir.glob(f"{stem}_page_*.png"):
        path.unlink()


def plot_case(case_dir):
    case_dir = Path(case_dir)
    sources_dir = case_dir / "sources"
    plots_dir = case_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    run_name = run_name_for_case(case_dir)

    jobs = [
        (
            sources_dir / "probability_curve.csv",
            plots_dir / "probability_curve.png",
            lambda source, output: plot_probability_curve_from_csv(source, output, run_name),
        ),
        (
            sources_dir / "subject_reliability.csv",
            plots_dir / "subject_reliability.png",
            plot_reliability_from_csv,
        ),
        (
            sources_dir / "simulation_reliability.csv",
            plots_dir / "simulation_reliability.png",
            plot_reliability_from_csv,
        ),
    ]

    generated = []
    for source_path, output_base_path, plotter in jobs:
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        remove_old_page_outputs(plots_dir, output_base_path.stem)
        generated.extend(plotter(source_path, output_base_path))
    return generated


def iter_case_dirs(root):
    root = Path(root)
    if (root / "sources").is_dir():
        yield root
        return

    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "sources").is_dir():
            yield child


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate probability-curve and reliability PNGs from the simplified "
            "case CSV source files."
        )
    )
    parser.add_argument(
        "--root",
        default=DEFAULT_ROOT,
        help="Simplified export root, or one case folder containing sources/.",
    )
    args = parser.parse_args()

    total_cases = 0
    total_plots = 0
    for case_dir in iter_case_dirs(args.root):
        generated = plot_case(case_dir)
        total_cases += 1
        total_plots += len(generated)

    print(f"Generated {total_plots} plot file(s) for {total_cases} case folder(s).")


if __name__ == "__main__":
    main()
