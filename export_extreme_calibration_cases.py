import argparse
import csv
import json
import math
import os
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sweep_pipeline as sp
from compute_precision_metrics import calculate_brier_score, calculate_ece, process_and_save


DEFAULT_SWEEP_ROOT = "results/sweep_latest"
DEFAULT_OUTPUT_DIR = "results/sweep_latest/best_worst_brier_ece_case_plots"

METRIC_SPECS = [
    ("brier", "prediction_mean_brier"),
    ("ece", "prediction_mean_ece"),
]


def safe_label(value):
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return label or "case"


def finite_metric(row, key):
    value = row.get("precision_metrics", {}).get(key)
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def load_rows(summary_path):
    with open(summary_path, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"Expected list in {summary_path}")
    return rows


def select_extreme_rows(rows, limit):
    selected_by_run_id = {}
    bucket_rows = []

    for metric_name, metric_key in METRIC_SPECS:
        metric_rows = [row for row in rows if finite_metric(row, metric_key) is not None]
        for direction, reverse in (("best", False), ("worst", True)):
            sorted_rows = sorted(
                metric_rows,
                key=lambda row: (finite_metric(row, metric_key), row.get("run_id", "")),
                reverse=reverse,
            )
            for rank, row in enumerate(sorted_rows[:limit], start=1):
                run_id = row["run_id"]
                bucket = f"{direction}_{metric_name}"
                selected_by_run_id.setdefault(run_id, row)
                selected_by_run_id[run_id].setdefault("_selection_buckets", []).append(bucket)
                bucket_rows.append(
                    {
                        "bucket": bucket,
                        "rank": rank,
                        "run_id": run_id,
                        "dataset_stem": row.get("dataset_stem"),
                        "parameter_case_index": row.get("parameter_case_index"),
                        "parameter_case_id": row.get("parameter_case_id"),
                        "brier_score_analysis": finite_metric(row, "prediction_mean_brier"),
                        "ece_analysis": finite_metric(row, "prediction_mean_ece"),
                        "brier_score_simulation": finite_metric(row, "baseline_mean_brier"),
                        "ece_simulation": finite_metric(row, "baseline_mean_ece"),
                    }
                )

    unique_rows = list(selected_by_run_id.values())
    unique_rows.sort(
        key=lambda row: (
            row.get("dataset_stem", ""),
            int(row.get("parameter_case_index", 0)),
            row.get("run_id", ""),
        )
    )
    return unique_rows, bucket_rows


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_csv(path, rows, fieldnames=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_series(path):
    payload = sp.read_json(path)
    normalized = {}
    for subject_id, values in payload.items():
        normalized[str(subject_id)] = [float(value) for value in values]
    return {subject_id: normalized[subject_id] for subject_id in sorted(normalized, key=int)}


def validate_subjects(*series_maps):
    subjects = list(series_maps[0].keys())
    for series_map in series_maps[1:]:
        if list(series_map.keys()) != subjects:
            raise ValueError("All probability files must contain the same subjects.")
    return subjects


def reliability_bins(predicted, observed, bins):
    predicted = np.asarray(predicted, dtype=float)
    observed = np.asarray(observed, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    total_count = len(predicted)
    ece = 0.0

    for index in range(bins):
        left = bin_edges[index]
        right = bin_edges[index + 1]
        if index == 0:
            mask = (predicted >= left) & (predicted <= right)
        else:
            mask = (predicted > left) & (predicted <= right)
        count = int(mask.sum())
        if count:
            mean_predicted = float(predicted[mask].mean())
            mean_observed = float(observed[mask].mean())
            abs_error = abs(mean_observed - mean_predicted)
            ece_component = (count / total_count) * abs_error
            ece += ece_component
        else:
            mean_predicted = ""
            mean_observed = ""
            abs_error = ""
            ece_component = 0.0
        rows.append(
            {
                "bin_index": index,
                "bin_left": float(left),
                "bin_right": float(right),
                "count": count,
                "mean_predicted_probability": mean_predicted,
                "mean_ground_truth_probability": mean_observed,
                "absolute_calibration_error": abs_error,
                "ece_component": float(ece_component),
            }
        )
    return rows, float(ece)


def write_reliability_sources(case_dir, subjects, ground_truth, analysis, baseline, bins):
    source_dir = os.path.join(case_dir, "source_data", "reliability")
    os.makedirs(source_dir, exist_ok=True)

    methods = [
        ("analysis", analysis),
        ("simulation", baseline),
    ]
    point_rows = []
    bin_rows = []
    subject_metric_rows = []
    plot_payload = []

    for method_name, predictions in methods:
        all_predicted = []
        all_ground_truth = []
        for subject_id in subjects:
            predicted_values = predictions[subject_id]
            ground_truth_values = ground_truth[subject_id]
            if len(predicted_values) != len(ground_truth_values):
                raise ValueError(f"Inconsistent trajectory length for subject {subject_id}.")

            subject_metric_rows.append(
                {
                    "method": method_name,
                    "subject_id": subject_id,
                    "brier_score": calculate_brier_score(predicted_values, ground_truth_values),
                    "ece": calculate_ece(predicted_values, ground_truth_values, M=bins),
                }
            )

            for time_index, (predicted, observed) in enumerate(
                zip(predicted_values, ground_truth_values)
            ):
                point_rows.append(
                    {
                        "method": method_name,
                        "subject_id": subject_id,
                        "time_index": time_index,
                        "predicted_probability": predicted,
                        "ground_truth_probability": observed,
                    }
                )
                all_predicted.append(predicted)
                all_ground_truth.append(observed)

        rows, aggregate_ece = reliability_bins(all_predicted, all_ground_truth, bins)
        aggregate_brier = calculate_brier_score(all_predicted, all_ground_truth)
        for row in rows:
            row["method"] = method_name
            bin_rows.append(row)
        plot_payload.append(
            {
                "method": method_name,
                "bins": rows,
                "aggregate_brier_score": aggregate_brier,
                "aggregate_ece": aggregate_ece,
            }
        )

    write_csv(os.path.join(source_dir, "reliability_points.csv"), point_rows)
    write_csv(os.path.join(source_dir, "reliability_bins.csv"), bin_rows)
    write_csv(os.path.join(source_dir, "metrics_by_subject.csv"), subject_metric_rows)
    write_json(os.path.join(source_dir, "aggregate_metrics.json"), plot_payload)
    return source_dir, plot_payload


def plot_reliability(case_dir, run_id, plot_payload):
    output_dir = os.path.join(case_dir, "plots", "reliability")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "aggregate_reliability_diagram.png")

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"analysis": "tab:orange", "simulation": "tab:green"}
    for payload in plot_payload:
        xs = []
        ys = []
        for row in payload["bins"]:
            if row["count"] > 0:
                xs.append(row["mean_predicted_probability"])
                ys.append(row["mean_ground_truth_probability"])
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=1.8,
            color=colors[payload["method"]],
            label=(
                f"{payload['method'].title()} "
                f"(Brier {payload['aggregate_brier_score']:.5f}, "
                f"ECE {payload['aggregate_ece']:.5f})"
            ),
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, label="Perfect calibration")
    ax.set_title(f"{run_id} Aggregate Reliability")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Mean ground-truth probability")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def copy_existing_metrics(row, case_dir):
    source_dir = os.path.join(case_dir, "source_data", "existing_precision_metrics")
    os.makedirs(source_dir, exist_ok=True)
    copied = []
    metrics = row.get("precision_metrics", {})
    for key, filename in (
        ("prediction_metrics_path", "metrics_prediction.json"),
        ("baseline_metrics_path", "metrics_baseline.json"),
    ):
        source_path = metrics.get(key)
        if source_path and os.path.exists(source_path):
            destination_path = os.path.join(source_dir, filename)
            shutil.copy2(source_path, destination_path)
            copied.append(destination_path)
    return copied


def write_reference_style_precision_outputs(case_dir, analysis_path, baseline_path, ground_truth_path):
    precision_dir = os.path.join(case_dir, "precision_metrics")
    analysis_summary = process_and_save(
        analysis_path,
        ground_truth_path,
        M=10,
        metrics_output=os.path.join(precision_dir, "metrics_prediction.json"),
        plots_dir=os.path.join(precision_dir, "numericalAnalysis", "plots"),
        save_plots=True,
        verbose=False,
        include_scatter_coordinates=True,
    )
    baseline_summary = process_and_save(
        baseline_path,
        ground_truth_path,
        M=10,
        metrics_output=os.path.join(precision_dir, "metrics_baseline.json"),
        plots_dir=os.path.join(precision_dir, "simulatedBaseline", "plots"),
        save_plots=True,
        verbose=False,
        include_scatter_coordinates=True,
    )
    return {
        "prediction_metrics_path": analysis_summary["metrics_output"],
        "prediction_plots_dir": analysis_summary["plots_dir"],
        "baseline_metrics_path": baseline_summary["metrics_output"],
        "baseline_plots_dir": baseline_summary["plots_dir"],
    }


def case_folder_name(index, row):
    buckets = "__".join(safe_label(bucket) for bucket in row.get("_selection_buckets", []))
    brier = finite_metric(row, "prediction_mean_brier")
    ece = finite_metric(row, "prediction_mean_ece")
    return (
        f"{index:02d}__{buckets}"
        f"__brier_{brier:.6f}".replace(".", "p")
        + f"__ece_{ece:.6f}".replace(".", "p")
        + f"__{safe_label(row.get('dataset_stem'))}"
        + f"__case_{row.get('parameter_case_index')}"
    )


def build_case_metadata(
    row,
    case_dir,
    reliability_source_dir,
    reliability_plot_path,
    curve_result,
    copied_metrics,
    reference_style_precision_outputs,
):
    metadata = {
        "run_id": row.get("run_id"),
        "selection_buckets": row.get("_selection_buckets", []),
        "dataset_stem": row.get("dataset_stem"),
        "parameter_case_index": row.get("parameter_case_index"),
        "parameter_case_id": row.get("parameter_case_id"),
        "parameter_levels": row.get("parameter_levels"),
        "metrics": {
            "brier_score_analysis": finite_metric(row, "prediction_mean_brier"),
            "ece_analysis": finite_metric(row, "prediction_mean_ece"),
            "brier_score_simulation": finite_metric(row, "baseline_mean_brier"),
            "ece_simulation": finite_metric(row, "baseline_mean_ece"),
        },
        "source_paths": {
            "ground_truth_path": row.get("ground_truth_path"),
            "analysis_path": row.get("java_analysis", {}).get("analysis_path"),
            "baseline_path": row.get("python_analysis", {}).get("baseline_path"),
            "comparison_dir": row.get("comparison_dir"),
        },
        "generated_outputs": {
            "case_dir": case_dir,
            "aggregate_reliability_plot": reliability_plot_path,
            "reliability_source_dir": reliability_source_dir,
            "probability_curve_csv": curve_result.get("csv_path"),
            "probability_curve_plots": curve_result.get("plot_paths", []),
            "copied_existing_metrics": copied_metrics,
            "reference_style_precision_outputs": reference_style_precision_outputs,
        },
    }
    write_json(os.path.join(case_dir, "case_metadata.json"), metadata)
    return metadata


def export_case(index, row, output_root, time_step_hours, bins):
    case_dir = os.path.join(output_root, "cases", case_folder_name(index, row))
    os.makedirs(case_dir, exist_ok=True)

    ground_truth_path = row["ground_truth_path"]
    analysis_path = row.get("java_analysis", {}).get("analysis_path")
    baseline_path = row.get("python_analysis", {}).get("baseline_path")
    for path in (ground_truth_path, analysis_path, baseline_path):
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Required source file does not exist: {path}")

    curve_result = sp.create_analysis_subject_curve_plots(
        run_dir=case_dir,
        run_name=row["run_id"],
        analysis_path=analysis_path,
        ground_truth_path=ground_truth_path,
        baseline_path=baseline_path,
        granularity=time_step_hours,
        ground_truth_sample_size=row.get("ground_truth_iterations"),
        save_plots=True,
    )

    ground_truth = read_series(ground_truth_path)
    analysis = read_series(analysis_path)
    baseline = read_series(baseline_path)
    subjects = validate_subjects(ground_truth, analysis, baseline)

    reliability_source_dir, reliability_payload = write_reliability_sources(
        case_dir,
        subjects,
        ground_truth,
        analysis,
        baseline,
        bins,
    )
    reliability_plot_path = plot_reliability(case_dir, row["run_id"], reliability_payload)
    copied_metrics = copy_existing_metrics(row, case_dir)
    reference_style_precision_outputs = write_reference_style_precision_outputs(
        case_dir,
        analysis_path,
        baseline_path,
        ground_truth_path,
    )
    return build_case_metadata(
        row,
        case_dir,
        reliability_source_dir,
        reliability_plot_path,
        curve_result,
        copied_metrics,
        reference_style_precision_outputs,
    )


def write_readme(output_root, summary_path, selected_count):
    content = f"""# Extreme Calibration Case Export

Source summary: `{summary_path}`

This folder contains the unique cases selected from:

- 4 lowest Java-analysis mean Brier scores
- 4 highest Java-analysis mean Brier scores
- 4 lowest Java-analysis mean ECE scores
- 4 highest Java-analysis mean ECE scores

Overlapping run IDs are exported once and list all matching selection buckets in
`case_metadata.json`.

Each case folder contains:

- `plots/analysis_subject_curves/*.png`: probability curve plots
- `plots/analysis_subject_curves/*.csv`: source data for the probability curve plots
- `plots/reliability/aggregate_reliability_diagram.png`: aggregate reliability diagram
- `precision_metrics/numericalAnalysis/plots/*.png`: reference-style Java-analysis reliability diagrams
- `precision_metrics/simulatedBaseline/plots/*.png`: reference-style simulation reliability diagrams
- `source_data/reliability/reliability_bins.csv`: source data for the reliability diagram
- `source_data/reliability/reliability_points.csv`: raw reliability scatter points
- `source_data/reliability/metrics_by_subject.csv`: per-subject Brier/ECE
- `precision_metrics/metrics_prediction.json`: source metrics and scatter coordinates for Java analysis
- `precision_metrics/metrics_baseline.json`: source metrics and scatter coordinates for simulation
- `case_metadata.json`: selected metrics, parameters, and original source paths

Unique cases exported: {selected_count}
"""
    with open(os.path.join(output_root, "README.md"), "w", encoding="utf-8") as handle:
        handle.write(content)


def main():
    parser = argparse.ArgumentParser(
        description="Export best/worst Brier and ECE cases with reliability and probability curve plot sources."
    )
    parser.add_argument("--sweep-root", default=DEFAULT_SWEEP_ROOT)
    parser.add_argument("--summary-name", default="comparison_summary_q4.json")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument(
        "--dataset-stem",
        default=None,
        help="Optional dataset stem filter, for example dataset_200 or 200.",
    )
    parser.add_argument("--time-step-hours", type=float, default=1.0)
    parser.add_argument("--reliability-bins", type=int, default=10)
    args = parser.parse_args()

    summary_path = os.path.join(args.sweep_root, args.summary_name)
    rows = load_rows(summary_path)
    if args.dataset_stem:
        dataset_stem = str(args.dataset_stem)
        if dataset_stem.isdigit():
            dataset_stem = f"dataset_{dataset_stem}"
        rows = [row for row in rows if row.get("dataset_stem") == dataset_stem]
        if not rows:
            raise ValueError(f"No rows found for dataset_stem={dataset_stem!r}")
    selected_rows, bucket_rows = select_extreme_rows(rows, args.limit)

    output_root = os.path.abspath(args.output_dir)
    os.makedirs(output_root, exist_ok=True)

    exported_metadata = []
    for index, row in enumerate(selected_rows, start=1):
        exported_metadata.append(
            export_case(
                index=index,
                row=row,
                output_root=output_root,
                time_step_hours=args.time_step_hours,
                bins=args.reliability_bins,
            )
        )

    write_json(os.path.join(output_root, "selection_buckets.json"), bucket_rows)
    write_csv(os.path.join(output_root, "selection_buckets.csv"), bucket_rows)
    write_json(os.path.join(output_root, "export_manifest.json"), exported_metadata)
    write_readme(output_root, summary_path, len(exported_metadata))
    shutil.copy2(Path(__file__), os.path.join(output_root, "export_extreme_calibration_cases.py"))

    print(f"Exported {len(exported_metadata)} unique cases to {output_root}")


if __name__ == "__main__":
    main()
