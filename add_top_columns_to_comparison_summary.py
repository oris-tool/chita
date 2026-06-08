import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np


DEFAULT_ROOT = "results"
DEFAULT_DATASET_DIRS = ("dataset_1250", "dataset_2500", "dataset_5000")
SOURCE_FILENAME = "comparison_summary_q4.csv"
DEFAULT_OUTPUT_SUFFIX = "_with_top_columns"
TOP_THRESHOLDS = (10, 20, 30, 40, 50)
SHARED_ANALYSIS_FALLBACK_SWEEP_DIRS = ("sweep_20260502-0119",)
DEFAULT_MAX_WORKERS = max(1, min(8, (os.cpu_count() or 1) // 2))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Create copies of comparison_summary_q4.csv with true top-k accuracy columns "
            "for both analysis and simulation."
        )
    )
    parser.add_argument(
        "--root",
        default=DEFAULT_ROOT,
        help="Root directory containing dataset comparison output folders.",
    )
    parser.add_argument(
        "--dataset-dirs",
        nargs="+",
        default=list(DEFAULT_DATASET_DIRS),
        help="Dataset directories to process under --root.",
    )
    parser.add_argument(
        "--output-suffix",
        default=DEFAULT_OUTPUT_SUFFIX,
        help="Suffix inserted before the .csv extension for the copied files.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Number of worker threads used to recompute top-k accuracies.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the files that would be written without creating them.",
    )
    return parser.parse_args(argv)


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv_rows(csv_path):
    with open(csv_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames or [], list(reader)


def output_csv_path(source_csv_path, output_suffix):
    base, extension = os.path.splitext(source_csv_path)
    return f"{base}{output_suffix}{extension}"


def resolve_dataset_root(root, dataset_dir):
    dataset_root = os.path.join(root, dataset_dir)
    source_csv_path = os.path.join(dataset_root, SOURCE_FILENAME)
    if os.path.exists(source_csv_path):
        return dataset_root

    if os.path.isdir(dataset_root):
        nested_roots = [
            os.path.join(dataset_root, name)
            for name in os.listdir(dataset_root)
            if os.path.isdir(os.path.join(dataset_root, name))
            and os.path.exists(os.path.join(dataset_root, name, SOURCE_FILENAME))
        ]
        if len(nested_roots) == 1:
            return nested_roots[0]

    return dataset_root


def comparison_root(dataset_root):
    return os.path.join(dataset_root, "comparison")


def resolve_dataset_stem(dataset_root):
    dataset_root_name = os.path.basename(os.path.normpath(dataset_root))
    direct_dataset_path = os.path.join(comparison_root(dataset_root), dataset_root_name)
    if os.path.isdir(direct_dataset_path):
        return dataset_root_name

    nested_matches = [
        entry
        for entry in os.listdir(comparison_root(dataset_root))
        if os.path.isdir(os.path.join(comparison_root(dataset_root), entry, dataset_root_name))
    ]
    if nested_matches:
        return dataset_root_name

    nested_entries = []
    for entry in os.listdir(comparison_root(dataset_root)):
        entry_path = os.path.join(comparison_root(dataset_root), entry)
        if not os.path.isdir(entry_path):
            continue
        nested_entries.extend(
            name
            for name in os.listdir(entry_path)
            if os.path.isdir(os.path.join(entry_path, name))
        )
    nested_entries = sorted(set(nested_entries))
    if len(nested_entries) == 1:
        return nested_entries[0]

    entries = [
        name for name in os.listdir(comparison_root(dataset_root))
        if os.path.isdir(os.path.join(comparison_root(dataset_root), name))
    ]
    if len(entries) != 1:
        raise ValueError(
            f"Expected exactly one dataset stem under {comparison_root(dataset_root)}, found: {entries}"
        )
    return entries[0]


def comparison_summary_path(dataset_root, dataset_stem, parameter_case_code):
    candidates = [
        os.path.join(
            comparison_root(dataset_root),
            dataset_stem,
            str(parameter_case_code),
            "comparison_summary.json",
        )
    ]

    for entry in os.listdir(comparison_root(dataset_root)):
        nested_path = os.path.join(
            comparison_root(dataset_root),
            entry,
            dataset_stem,
            str(parameter_case_code),
            "comparison_summary.json",
        )
        if nested_path not in candidates:
            candidates.append(nested_path)

    return first_existing_path(candidates)


def first_existing_path(candidates):
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return candidates[0] if candidates else None


def candidate_roots(dataset_root):
    roots = []
    current = os.path.abspath(dataset_root)
    for _ in range(3):
        if current not in roots:
            roots.append(current)
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return roots


def resolve_saved_path(dataset_root, stored_path):
    if os.path.isabs(stored_path) and os.path.exists(stored_path):
        return stored_path

    candidate_paths = []
    roots = candidate_roots(dataset_root)

    for anchor in ("ground_truth", "java_analysis"):
        marker = f"{anchor}/"
        index = stored_path.find(marker)
        if index != -1:
            suffix = stored_path[index:]
            candidate_paths.extend(os.path.join(root, suffix) for root in roots)
            break

    if not candidate_paths:
        marker = "python_analysis/"
        index = stored_path.find(marker)
        if index != -1:
            suffix = stored_path[index:]
            candidate_paths.extend(os.path.join(root, suffix) for root in roots)
            candidate_paths.append(os.path.join(dataset_root, "python_analysis", suffix))

    for candidate_path in candidate_paths:
        if os.path.exists(candidate_path):
            return candidate_path

    if candidate_paths:
        return candidate_paths[0]
    raise ValueError(f"Could not resolve saved path relative to dataset root: {stored_path}")


def resolve_ground_truth_path(dataset_root, dataset_stem, stored_path):
    basename = os.path.basename(stored_path)
    candidates = [
        resolve_saved_path(dataset_root, stored_path),
    ]
    candidates.extend(
        os.path.join(root, "ground_truth", dataset_stem, basename)
        for root in candidate_roots(dataset_root)
    )
    candidates.extend(
        os.path.join(root, sweep_dir, "ground_truth", dataset_stem, basename)
        for root in candidate_roots(dataset_root)
        for sweep_dir in SHARED_ANALYSIS_FALLBACK_SWEEP_DIRS
    )
    return first_existing_path(candidates)


def resolve_analysis_path(dataset_root, dataset_stem, parameter_case_code, stored_path):
    basename = os.path.basename(stored_path)
    case_code = str(parameter_case_code)
    candidates = [
        resolve_saved_path(dataset_root, stored_path),
    ]
    candidates.extend(
        os.path.join(root, "java_analysis", dataset_stem, case_code, basename)
        for root in candidate_roots(dataset_root)
    )
    candidates.extend(
        os.path.join(root, sweep_dir, "java_analysis", dataset_stem, case_code, basename)
        for root in candidate_roots(dataset_root)
        for sweep_dir in SHARED_ANALYSIS_FALLBACK_SWEEP_DIRS
    )
    return first_existing_path(candidates)


def resolve_baseline_path(dataset_root, dataset_stem, parameter_case_code, stored_path):
    basename = os.path.basename(stored_path)
    shared_root = os.path.dirname(dataset_root)
    case_code = str(parameter_case_code)
    return first_existing_path(
        [
            resolve_saved_path(dataset_root, stored_path),
            os.path.join(dataset_root, "python_analysis", dataset_stem, case_code, basename),
            os.path.join(dataset_root, "python_analysis", "python_analysis", "q4", dataset_stem, case_code, basename),
            os.path.join(shared_root, "python_analysis", "q4", dataset_stem, case_code, basename),
        ]
    )


def column_names_for_threshold(threshold):
    return (
        f"top_{threshold}_accuracy_analysis",
        f"top_{threshold}_accuracy_simulation",
    )


def sorted_subject_ids(tracks):
    return sorted(tracks.keys(), key=int)


def subject_track_matrix(tracks, subject_ids):
    if set(tracks.keys()) != set(subject_ids):
        raise ValueError("Track files do not contain the same subject ids.")
    return np.asarray([tracks[subject_id] for subject_id in subject_ids], dtype=float).T


def ranking_order_rows_descending(matrix):
    return np.argsort(-matrix, axis=1, kind="mergesort")


def ground_truth_rank_positions(ground_truth_order):
    positions = np.empty_like(ground_truth_order)
    row_indices = np.arange(ground_truth_order.shape[0])[:, None]
    positions[row_indices, ground_truth_order] = np.arange(ground_truth_order.shape[1])
    return positions


def top_k_accuracy_means(ground_truth_positions, candidate_order, thresholds):
    row_indices = np.arange(candidate_order.shape[0])[:, None]
    subject_count = candidate_order.shape[1]
    results = {}

    for threshold in thresholds:
        effective_k = min(threshold, subject_count)
        if effective_k <= 0:
            results[threshold] = float("nan")
            continue

        candidate_top = candidate_order[:, :effective_k]
        candidate_gt_positions = ground_truth_positions[row_indices, candidate_top]
        hits = np.sum(candidate_gt_positions < effective_k, axis=1)
        results[threshold] = float(np.mean(hits / effective_k))

    return results


def compute_row_top_accuracies(dataset_root, dataset_stem, subject_ids, ground_truth_positions, row):
    summary = read_json(
        comparison_summary_path(dataset_root, dataset_stem, row["parameter_case_code"])
    )
    analysis_path = resolve_analysis_path(
        dataset_root,
        dataset_stem,
        row["parameter_case_code"],
        summary["java_analysis"]["analysis_path"],
    )
    baseline_path = resolve_baseline_path(
        dataset_root,
        dataset_stem,
        row["parameter_case_code"],
        summary["python_analysis"]["baseline_path"],
    )

    analysis_tracks = read_json(analysis_path)
    baseline_tracks = read_json(baseline_path)

    analysis_order = ranking_order_rows_descending(subject_track_matrix(analysis_tracks, subject_ids))
    baseline_order = ranking_order_rows_descending(subject_track_matrix(baseline_tracks, subject_ids))

    analysis_scores = top_k_accuracy_means(ground_truth_positions, analysis_order, TOP_THRESHOLDS)
    baseline_scores = top_k_accuracy_means(ground_truth_positions, baseline_order, TOP_THRESHOLDS)

    values = {}
    for threshold in TOP_THRESHOLDS:
        analysis_column, simulation_column = column_names_for_threshold(threshold)
        values[analysis_column] = analysis_scores[threshold]
        values[simulation_column] = baseline_scores[threshold]
    return row["run_id"], values


def dataset_top_accuracy_values(dataset_root, rows, max_workers):
    dataset_stem = resolve_dataset_stem(dataset_root)
    seed_summary = read_json(comparison_summary_path(dataset_root, dataset_stem, rows[0]["parameter_case_code"]))
    ground_truth_path = resolve_ground_truth_path(
        dataset_root,
        dataset_stem,
        seed_summary["ground_truth_path"],
    )
    ground_truth_tracks = read_json(ground_truth_path)
    subject_ids = sorted_subject_ids(ground_truth_tracks)
    ground_truth_order = ranking_order_rows_descending(subject_track_matrix(ground_truth_tracks, subject_ids))
    ground_truth_positions = ground_truth_rank_positions(ground_truth_order)

    values_by_run_id = {}
    worker_count = max(1, max_workers)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                compute_row_top_accuracies,
                dataset_root,
                dataset_stem,
                subject_ids,
                ground_truth_positions,
                row,
            )
            for row in rows
        ]
        for future in as_completed(futures):
            run_id, values = future.result()
            values_by_run_id[run_id] = values

    return values_by_run_id


def write_augmented_copy(dataset_root, output_suffix, max_workers, dry_run=False):
    source_csv_path = os.path.join(dataset_root, SOURCE_FILENAME)
    fieldnames, rows = read_csv_rows(source_csv_path)
    target_csv_path = output_csv_path(source_csv_path, output_suffix)

    if dry_run:
        return {
            "source": source_csv_path,
            "target": target_csv_path,
            "rows": len(rows),
        }

    values_by_run_id = dataset_top_accuracy_values(dataset_root, rows, max_workers)
    added_columns = []
    for threshold in TOP_THRESHOLDS:
        added_columns.extend(column_names_for_threshold(threshold))

    augmented_fieldnames = [name for name in fieldnames if name not in added_columns]
    augmented_fieldnames.extend(added_columns)

    with open(target_csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=augmented_fieldnames)
        writer.writeheader()
        for row in rows:
            augmented_row = dict(row)
            augmented_row.update(values_by_run_id[row["run_id"]])
            writer.writerow(augmented_row)

    return {
        "source": source_csv_path,
        "target": target_csv_path,
        "rows": len(rows),
    }


def main(argv=None):
    args = parse_args(argv)
    results = []
    for dataset_dir in args.dataset_dirs:
        dataset_root = resolve_dataset_root(args.root, dataset_dir)
        source_csv_path = os.path.join(dataset_root, SOURCE_FILENAME)
        if not os.path.exists(source_csv_path):
            raise FileNotFoundError(f"Comparison summary CSV not found: {source_csv_path}")
        result = write_augmented_copy(
            dataset_root=dataset_root,
            output_suffix=args.output_suffix,
            max_workers=args.max_workers,
            dry_run=args.dry_run,
        )
        results.append(result)

    for result in results:
        print(f"{result['source']} -> {result['target']} ({result['rows']} rows)")


if __name__ == "__main__":
    main()
