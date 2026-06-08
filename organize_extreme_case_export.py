import argparse
import csv
import json
import os
import re
import shutil
from pathlib import Path


DEFAULT_DETAILED_ROOT = "results/sweep_latest/best_worst_brier_ece_case_plots_dataset_200"
DEFAULT_OUTPUT_ROOT = "results/sweep_latest/best_worst_brier_ece_case_plots_dataset_200_simplified"


BUCKET_LABELS = {
    "best_brier": "top_brier",
    "worst_brier": "worst_brier",
    "best_ece": "top_ece",
    "worst_ece": "worst_ece",
}


def safe_label(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_manifest_by_run(detailed_root):
    manifest = read_json(os.path.join(detailed_root, "export_manifest.json"))
    return {row["run_id"]: row for row in manifest}


def first_matching_file(root, pattern):
    matches = sorted(Path(root).glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one match for {pattern} in {root}, found {len(matches)}")
    return matches[0]


def reliability_csv_from_metrics(metrics_json_path, output_csv_path):
    payload = read_json(metrics_json_path)
    rows = []
    for subject_id in sorted(payload, key=int):
        subject_payload = payload[subject_id]
        for time_index, point in enumerate(subject_payload.get("scatter_coordinates", [])):
            rows.append(
                {
                    "subject_id": subject_id,
                    "time_index": time_index,
                    "predicted_probability": point[0],
                    "ground_truth_probability": point[1],
                }
            )
    write_csv(
        output_csv_path,
        rows,
        [
            "subject_id",
            "time_index",
            "predicted_probability",
            "ground_truth_probability",
        ],
    )


def copy_selection(selection, detailed_root, output_root, manifest_by_run):
    bucket = selection["bucket"]
    rank = int(selection["rank"])
    run_id = selection["run_id"]
    case_manifest = manifest_by_run[run_id]
    case_dir = Path(case_manifest["generated_outputs"]["case_dir"])
    bucket_label = BUCKET_LABELS.get(bucket, safe_label(bucket))
    folder_name = (
        f"{rank:02d}_{bucket_label}"
        f"__case_{selection['parameter_case_index']}"
        f"__{safe_label(run_id)}"
    )

    destination = Path(output_root) / folder_name
    plots_dir = destination / "plots"
    sources_dir = destination / "sources"
    plots_dir.mkdir(parents=True, exist_ok=True)
    sources_dir.mkdir(parents=True, exist_ok=True)

    probability_png = first_matching_file(
        case_dir / "plots" / "analysis_subject_curves",
        "*_analysis_subject_curves.png",
    )
    probability_csv = first_matching_file(
        case_dir / "plots" / "analysis_subject_curves",
        "*_analysis_subject_curves.csv",
    )
    reliability_png = (
        case_dir
        / "precision_metrics"
        / "numericalAnalysis"
        / "plots"
        / "all_subjects_reliability_diagrams.png"
    )
    reliability_metrics_json = (
        case_dir
        / "precision_metrics"
        / "metrics_prediction.json"
    )
    simulation_reliability_png = (
        case_dir
        / "precision_metrics"
        / "simulatedBaseline"
        / "plots"
        / "all_subjects_reliability_diagrams.png"
    )
    simulation_reliability_metrics_json = (
        case_dir
        / "precision_metrics"
        / "metrics_baseline.json"
    )

    if not reliability_png.exists():
        raise FileNotFoundError(reliability_png)
    if not reliability_metrics_json.exists():
        raise FileNotFoundError(reliability_metrics_json)
    if not simulation_reliability_png.exists():
        raise FileNotFoundError(simulation_reliability_png)
    if not simulation_reliability_metrics_json.exists():
        raise FileNotFoundError(simulation_reliability_metrics_json)

    shutil.copy2(probability_png, plots_dir / "probability_curve.png")
    shutil.copy2(reliability_png, plots_dir / "subject_reliability.png")
    shutil.copy2(simulation_reliability_png, plots_dir / "simulation_reliability.png")
    shutil.copy2(probability_csv, sources_dir / "probability_curve.csv")
    reliability_csv_from_metrics(
        reliability_metrics_json,
        sources_dir / "subject_reliability.csv",
    )
    reliability_csv_from_metrics(
        simulation_reliability_metrics_json,
        sources_dir / "simulation_reliability.csv",
    )

    metadata = {
        "bucket": bucket,
        "folder_bucket_label": bucket_label,
        "rank": rank,
        "run_id": run_id,
        "dataset_stem": selection["dataset_stem"],
        "parameter_case_index": selection["parameter_case_index"],
        "parameter_case_id": selection["parameter_case_id"],
        "brier_score_analysis": selection["brier_score_analysis"],
        "ece_analysis": selection["ece_analysis"],
        "source_case_dir": str(case_dir),
        "outputs": {
            "probability_curve_plot": str(plots_dir / "probability_curve.png"),
            "subject_reliability_plot": str(plots_dir / "subject_reliability.png"),
            "simulation_reliability_plot": str(plots_dir / "simulation_reliability.png"),
            "probability_curve_csv": str(sources_dir / "probability_curve.csv"),
            "subject_reliability_csv": str(sources_dir / "subject_reliability.csv"),
            "simulation_reliability_csv": str(sources_dir / "simulation_reliability.csv"),
        },
    }
    write_json(destination / "metadata.json", metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Copy the dataset_200 extreme-case export into a simplified plots/sources layout."
    )
    parser.add_argument("--detailed-root", default=DEFAULT_DETAILED_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    detailed_root = os.path.abspath(args.detailed_root)
    output_root = os.path.abspath(args.output_root)
    os.makedirs(output_root, exist_ok=True)

    selections = read_json(os.path.join(detailed_root, "selection_buckets.json"))
    manifest_by_run = build_manifest_by_run(detailed_root)
    output_rows = [
        copy_selection(selection, detailed_root, output_root, manifest_by_run)
        for selection in selections
    ]

    write_json(os.path.join(output_root, "manifest.json"), output_rows)
    shutil.copy2(Path(__file__), os.path.join(output_root, "organize_extreme_case_export.py"))
    print(f"Created {len(output_rows)} simplified folders in {output_root}")


if __name__ == "__main__":
    main()
