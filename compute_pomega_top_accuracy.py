#!/usr/bin/env python3
"""Compute top-10/20/30/40/50 accuracy columns for pomega results."""

import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from tqdm import tqdm

TOP_THRESHOLDS = (10, 20, 30, 40, 50)
DEFAULT_RESULTS_ROOT_ENV = "CHITA_POMEGA_RESULTS_ROOT"
DEFAULT_SOURCE_ROOT_ENV = "CHITA_POMEGA_SOURCE_ROOT"
OUTPUT_SUFFIX = "_top_10_20_30_40_50"
AGGREGATE_FILENAME = "pomega_top_10_20_30_40_50_accuracy_summary.csv"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        default=os.environ.get(DEFAULT_RESULTS_ROOT_ENV),
        help=f"Root containing pomega result folders. Defaults to ${DEFAULT_RESULTS_ROOT_ENV}.",
    )
    parser.add_argument(
        "--source-root",
        default=os.environ.get(DEFAULT_SOURCE_ROOT_ENV),
        help=f"Root containing source sweep artifacts. Defaults to ${DEFAULT_SOURCE_ROOT_ENV}.",
    )
    parser.add_argument("--max-workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.results_root:
        parser.error(f"--results-root is required unless ${DEFAULT_RESULTS_ROOT_ENV} is set.")
    if not args.source_root:
        parser.error(f"--source-root is required unless ${DEFAULT_SOURCE_ROOT_ENV} is set.")
    return args


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv_rows(path: str) -> Tuple[List[str], List[dict]]:
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames or [], list(reader)


def write_csv(path: str, fieldnames: Sequence[str], rows: Sequence[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def discover_comparison_jsons(results_root: str) -> List[str]:
    paths = []
    for current_root, _dirs, files in os.walk(results_root):
        if "comparison_summary_q4.json" in files:
            paths.append(os.path.join(current_root, "comparison_summary_q4.json"))
    return sorted(paths)


def source_roots(source_root: str, results_root: str) -> List[str]:
    roots = []

    def add(candidate: str) -> None:
        if candidate and os.path.isdir(candidate) and candidate not in roots:
            roots.append(candidate)

    if os.path.isdir(source_root):
        for name in sorted(os.listdir(source_root)):
            add(os.path.join(source_root, name))
    add(source_root)
    add(os.path.dirname(source_root))
    add(os.path.dirname(results_root))
    return roots


def suffix_after(path_value: str, marker: str) -> str | None:
    index = path_value.find(marker)
    if index == -1:
        return None
    return path_value[index + len(marker):].lstrip("/")


def first_existing(candidates: Iterable[str]) -> str:
    checked = []
    for candidate in candidates:
        if not candidate:
            continue
        checked.append(candidate)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("None of the candidate paths exists:\n" + "\n".join(checked[:25]))


def resolve_stored_path(pomega_dir: str, stored_path: str, source_candidates: Sequence[str]) -> str:
    if os.path.isabs(stored_path) and os.path.exists(stored_path):
        return stored_path

    candidates = []
    result_suffix = suffix_after(stored_path, "/result/")
    if result_suffix:
        candidates.append(os.path.join(pomega_dir, result_suffix))

    input_suffix = suffix_after(stored_path, "/_input/")
    if input_suffix:
        candidates.append(os.path.join(pomega_dir, "_input", input_suffix))
        candidates.append(os.path.join(pomega_dir, input_suffix))
        candidates.extend(os.path.join(root, input_suffix) for root in source_candidates)

    for anchor in ("ground_truth/", "python_analysis/", "java_analysis/", "comparison/"):
        index = stored_path.find(anchor)
        if index != -1:
            suffix = stored_path[index:]
            candidates.append(os.path.join(pomega_dir, suffix))
            candidates.extend(os.path.join(root, suffix) for root in source_candidates)
            break

    if not candidates and not os.path.isabs(stored_path):
        candidates.append(os.path.join(pomega_dir, stored_path))
        candidates.extend(os.path.join(root, stored_path) for root in source_candidates)

    return first_existing(candidates)


def sorted_subject_ids(tracks: Dict[str, list]) -> List[str]:
    return sorted(tracks.keys(), key=int)


def subject_track_matrix(tracks: Dict[str, list], subject_ids: Sequence[str]) -> np.ndarray:
    if set(tracks.keys()) != set(subject_ids):
        raise ValueError("Track files do not contain the same subject ids.")
    return np.asarray([tracks[subject_id] for subject_id in subject_ids], dtype=float).T


def ranking_order_rows_descending(matrix: np.ndarray) -> np.ndarray:
    return np.argsort(-matrix, axis=1, kind="mergesort")


def ground_truth_rank_positions(ground_truth_order: np.ndarray) -> np.ndarray:
    positions = np.empty_like(ground_truth_order)
    row_indices = np.arange(ground_truth_order.shape[0])[:, None]
    positions[row_indices, ground_truth_order] = np.arange(ground_truth_order.shape[1])
    return positions


def top_k_accuracy_means(ground_truth_positions: np.ndarray, candidate_order: np.ndarray) -> Dict[int, float]:
    row_indices = np.arange(candidate_order.shape[0])[:, None]
    subject_count = candidate_order.shape[1]
    values = {}
    for threshold in TOP_THRESHOLDS:
        k = min(threshold, subject_count)
        if k <= 0:
            values[threshold] = float("nan")
            continue
        candidate_top = candidate_order[:, :k]
        candidate_gt_positions = ground_truth_positions[row_indices, candidate_top]
        hits = np.sum(candidate_gt_positions < k, axis=1)
        values[threshold] = float(np.mean(hits / k))
    return values


def compute_row(summary: dict, pomega_dir: str, source_candidates: Sequence[str], subject_ids: Sequence[str], gt_positions: np.ndarray) -> Tuple[str, Dict[str, float]]:
    analysis_path = resolve_stored_path(pomega_dir, summary["java_analysis"]["analysis_path"], source_candidates)
    baseline_path = resolve_stored_path(pomega_dir, summary["python_analysis"]["baseline_path"], source_candidates)
    analysis = read_json(analysis_path)
    baseline = read_json(baseline_path)
    analysis_order = ranking_order_rows_descending(subject_track_matrix(analysis, subject_ids))
    baseline_order = ranking_order_rows_descending(subject_track_matrix(baseline, subject_ids))
    analysis_scores = top_k_accuracy_means(gt_positions, analysis_order)
    baseline_scores = top_k_accuracy_means(gt_positions, baseline_order)
    values = {}
    for threshold in TOP_THRESHOLDS:
        values[f"top_{threshold}_accuracy_analysis"] = analysis_scores[threshold]
        values[f"top_{threshold}_accuracy_simulation"] = baseline_scores[threshold]
    return summary["run_id"], values


def prior_label_from_dir(path: str) -> str:
    name = os.path.basename(path.rstrip(os.sep))
    if name.startswith("pomega_"):
        return name[len("pomega_"):].replace("p", ".")
    return name


def process_pomega_dir(comparison_json: str, source_candidates: Sequence[str], max_workers: int, dry_run: bool) -> dict:
    pomega_dir = os.path.dirname(comparison_json)
    comparison_csv = os.path.join(pomega_dir, "comparison_summary_q4.csv")
    if not os.path.exists(comparison_csv):
        raise FileNotFoundError(comparison_csv)

    summaries = read_json(comparison_json)
    if not summaries:
        raise ValueError(f"Empty comparison summary: {comparison_json}")
    fieldnames, rows = read_csv_rows(comparison_csv)
    row_by_run = {row["run_id"]: row for row in rows}
    output_csv = os.path.join(pomega_dir, f"comparison_summary_q4{OUTPUT_SUFFIX}.csv")

    if os.path.exists(output_csv) and not dry_run:
        _fields, output_rows = read_csv_rows(output_csv)
        return aggregate_from_rows(pomega_dir, summaries[0]["dataset_stem"], output_csv, output_rows)

    seed = summaries[0]
    dataset_stem = seed["dataset_stem"]
    ground_truth_path = resolve_stored_path(pomega_dir, seed["ground_truth_path"], source_candidates)
    ground_truth = read_json(ground_truth_path)
    subject_ids = sorted_subject_ids(ground_truth)
    ground_truth_order = ranking_order_rows_descending(subject_track_matrix(ground_truth, subject_ids))
    gt_positions = ground_truth_rank_positions(ground_truth_order)

    values_by_run = {}
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        futures = [
            executor.submit(compute_row, summary, pomega_dir, source_candidates, subject_ids, gt_positions)
            for summary in summaries
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc=os.path.relpath(pomega_dir, os.path.dirname(os.path.dirname(pomega_dir)))):
            run_id, values = future.result()
            values_by_run[run_id] = values

    added_columns = []
    for threshold in TOP_THRESHOLDS:
        added_columns.extend((f"top_{threshold}_accuracy_analysis", f"top_{threshold}_accuracy_simulation"))
    output_fieldnames = [field for field in fieldnames if field not in added_columns] + added_columns
    output_rows = []
    for row in rows:
        augmented = dict(row)
        augmented.update(values_by_run[row["run_id"]])
        output_rows.append(augmented)

    if not dry_run:
        write_csv(output_csv, output_fieldnames, output_rows)

    aggregate = aggregate_from_rows(pomega_dir, dataset_stem, output_csv, output_rows)
    return aggregate


def aggregate_from_rows(pomega_dir: str, dataset_stem: str, output_csv: str, output_rows: Sequence[dict]) -> dict:
    aggregate = {
        "pomega_dir": pomega_dir,
        "dataset_stem": dataset_stem,
        "prior": prior_label_from_dir(pomega_dir),
        "rows": len(output_rows),
        "output_csv": output_csv,
    }
    for threshold in TOP_THRESHOLDS:
        for side in ("analysis", "simulation"):
            column = f"top_{threshold}_accuracy_{side}"
            aggregate[f"mean_{column}"] = float(np.mean([float(row[column]) for row in output_rows]))
    return aggregate


def main(argv=None):
    args = parse_args(argv)
    comparisons = discover_comparison_jsons(args.results_root)
    if not comparisons:
        raise FileNotFoundError(f"No comparison_summary_q4.json files under {args.results_root}")
    sources = source_roots(args.source_root, args.results_root)
    aggregates = []
    for comparison_json in comparisons:
        aggregates.append(process_pomega_dir(comparison_json, sources, args.max_workers, args.dry_run))

    aggregate_path = os.path.join(args.results_root, AGGREGATE_FILENAME)
    aggregate_fields = ["pomega_dir", "dataset_stem", "prior", "rows", "output_csv"]
    for threshold in TOP_THRESHOLDS:
        aggregate_fields.extend((
            f"mean_top_{threshold}_accuracy_analysis",
            f"mean_top_{threshold}_accuracy_simulation",
        ))
    if not args.dry_run:
        write_csv(aggregate_path, aggregate_fields, aggregates)
    for aggregate in aggregates:
        print(f"{aggregate['output_csv']} ({aggregate['rows']} rows)")
    print(f"aggregate: {aggregate_path}")


if __name__ == "__main__":
    main()
