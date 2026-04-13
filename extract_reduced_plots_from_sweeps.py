import argparse
import csv
import json
import os
import shutil
from datetime import datetime
from json import JSONDecodeError

from tqdm import tqdm

import sweep_pipeline as sp
import sweep_pipeline_n_iteration as spi


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4)


def iter_sweep_roots(sweeps_root, output_root=None):
    normalized_output_root = None if output_root is None else os.path.normcase(os.path.abspath(output_root))
    for current_root, dirnames, filenames in os.walk(sweeps_root):
        abs_current_root = os.path.normcase(os.path.abspath(current_root))
        if normalized_output_root is not None and abs_current_root.startswith(normalized_output_root):
            dirnames[:] = []
            continue
        if "sweep_summary.json" in filenames:
            dirnames[:] = []
            yield current_root


def is_iteration_sweep(summary_rows):
    return any(isinstance(row, dict) and "iterations" in row for row in summary_rows)


def sweep_has_running_status(summary_rows):
    for row in summary_rows:
        status = row.get("status")
        if status == "running":
            return True
        for iteration in row.get("iterations", []):
            if iteration.get("status") == "running":
                return True
    return False


def reduced_selection_for_sweep(summary_rows):
    if is_iteration_sweep(summary_rows):
        selected_rows, manifest_rows = spi.select_bundles_for_reduced_plots(summary_rows)
        mode = "iteration"
    else:
        selected_rows, manifest_rows = sp.select_summaries_for_reduced_plots(summary_rows)
        mode = "standard"

    reasons_by_run = {
        row["run_name"]: {
            "selection_reasons": row["selection_reasons"],
            "analysis_spearman": row.get("analysis_spearman"),
            "analysis_spearman_mean": row.get("analysis_spearman_mean"),
        }
        for row in manifest_rows
    }
    return mode, selected_rows, manifest_rows, reasons_by_run


def collect_png_files(root_dir):
    png_paths = []
    for current_root, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(".png"):
                png_paths.append(os.path.join(current_root, filename))
    return sorted(png_paths)


def folder_name_for_selection(index, run_name, reasons, spearman_value):
    reasons_label = reasons.replace("|", "__")
    score_label = "na"
    score = sp.finite_float(spearman_value)
    if score is not None:
        score_label = f"{score:.6f}".replace("-", "m").replace(".", "p")
    return (
        f"{index:02d}__{sp.safe_path_label(reasons_label)}"
        f"__spearman_{score_label}"
        f"__{sp.safe_path_label(run_name)}"
    )


def copy_selected_run_plots(selected_rows, reasons_by_run, destination_root):
    copied_file_rows = []
    selection_rows = []
    total_copied_files = 0

    for index, row in enumerate(selected_rows, start=1):
        run_name = row["run_name"]
        run_dir = row["run_dir"]
        selection_info = reasons_by_run.get(run_name, {})
        reasons = selection_info.get("selection_reasons", "")
        spearman_value = selection_info.get("analysis_spearman")
        if spearman_value is None:
            spearman_value = selection_info.get("analysis_spearman_mean")

        selection_dir = os.path.join(
            destination_root,
            folder_name_for_selection(index, run_name, reasons, spearman_value),
        )
        sp.ensure_dir(selection_dir)

        png_paths = collect_png_files(run_dir)
        copied_count = 0
        for source_path in png_paths:
            relative_path = os.path.relpath(source_path, run_dir)
            destination_path = os.path.join(selection_dir, relative_path)
            sp.ensure_dir(os.path.dirname(destination_path))
            shutil.copy2(source_path, destination_path)
            copied_count += 1
            total_copied_files += 1
            copied_file_rows.append(
                {
                    "run_name": run_name,
                    "selection_reasons": reasons,
                    "analysis_spearman": spearman_value,
                    "source_path": source_path,
                    "destination_path": destination_path,
                    "relative_path": relative_path,
                }
            )

        selection_rows.append(
            {
                "run_name": run_name,
                "run_dir": run_dir,
                "selection_reasons": reasons,
                "analysis_spearman": spearman_value,
                "copied_plot_count": copied_count,
                "destination_dir": selection_dir,
            }
        )

    return {
        "selection_rows": selection_rows,
        "copied_file_rows": copied_file_rows,
        "total_copied_files": total_copied_files,
    }


def write_csv(path, rows):
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["run_name"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_single_sweep(sweep_root, export_root):
    sweep_summary_path = os.path.join(sweep_root, "sweep_summary.json")
    sweep_name = os.path.basename(os.path.abspath(sweep_root))
    sweep_export_root = os.path.join(export_root, sweep_name)
    sp.ensure_dir(sweep_export_root)

    try:
        summary_rows = load_json(sweep_summary_path)
    except (OSError, JSONDecodeError) as exc:
        result = {
            "sweep_name": sweep_name,
            "sweep_root": sweep_root,
            "status": "skipped",
            "reason": f"unreadable sweep_summary.json: {exc}",
        }
        write_json(os.path.join(sweep_export_root, "extraction_summary.json"), result)
        return result

    if not isinstance(summary_rows, list):
        result = {
            "sweep_name": sweep_name,
            "sweep_root": sweep_root,
            "status": "skipped",
            "reason": "sweep_summary.json is not a list",
        }
        write_json(os.path.join(sweep_export_root, "extraction_summary.json"), result)
        return result

    if sweep_has_running_status(summary_rows):
        result = {
            "sweep_name": sweep_name,
            "sweep_root": sweep_root,
            "status": "skipped",
            "reason": "sweep contains running entries",
        }
        write_json(os.path.join(sweep_export_root, "extraction_summary.json"), result)
        return result

    mode, selected_rows, manifest_rows, reasons_by_run = reduced_selection_for_sweep(summary_rows)
    copy_result = copy_selected_run_plots(
        selected_rows=selected_rows,
        reasons_by_run=reasons_by_run,
        destination_root=os.path.join(sweep_export_root, "selected_plots"),
    )

    manifest_json_path = os.path.join(sweep_export_root, "reduced_plot_selection.json")
    manifest_csv_path = os.path.join(sweep_export_root, "reduced_plot_selection.csv")
    write_json(manifest_json_path, manifest_rows)
    write_csv(manifest_csv_path, manifest_rows)

    copied_json_path = os.path.join(sweep_export_root, "copied_plot_manifest.json")
    copied_csv_path = os.path.join(sweep_export_root, "copied_plot_manifest.csv")
    write_json(copied_json_path, copy_result["copied_file_rows"])
    write_csv(copied_csv_path, copy_result["copied_file_rows"])

    summary = {
        "sweep_name": sweep_name,
        "sweep_root": sweep_root,
        "status": "processed",
        "mode": mode,
        "selected_run_count": len(selected_rows),
        "copied_plot_count": copy_result["total_copied_files"],
        "selection_manifest_json": manifest_json_path,
        "selection_manifest_csv": manifest_csv_path,
        "copied_plot_manifest_json": copied_json_path,
        "copied_plot_manifest_csv": copied_csv_path,
    }
    write_json(os.path.join(sweep_export_root, "extraction_summary.json"), summary)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Extract best/worst/median/mid-case plot sets from already completed sweeps."
    )
    parser.add_argument(
        "--sweeps-root",
        default="sweeps",
        help="Root directory containing concluded sweep folders.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Directory where the extracted plot collections will be written. "
            "Defaults to sweeps/_reduced_plot_exports_<timestamp>."
        ),
    )
    args = parser.parse_args()

    sweeps_root = os.path.abspath(args.sweeps_root)
    if args.output_root is None:
        output_root = os.path.join(
            sweeps_root,
            f"_reduced_plot_exports_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
    else:
        output_root = os.path.abspath(args.output_root)
    sp.ensure_dir(output_root)

    sweep_roots = list(iter_sweep_roots(sweeps_root, output_root=output_root))
    results = []
    for sweep_root in tqdm(sweep_roots, desc="Extracting sweep plots", unit="sweep"):
        results.append(process_single_sweep(sweep_root, output_root))

    write_json(os.path.join(output_root, "extraction_index.json"), results)
    write_csv(os.path.join(output_root, "extraction_index.csv"), results)


if __name__ == "__main__":
    main()
