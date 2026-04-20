import contextlib
import csv
import io
import itertools
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from tqdm import tqdm

import dataset_graph as dg
import sweep_pipeline as sp
from compute_precision_metrics import process_and_save
from run_n_simulations import run_dataset_simulations
from utils import normalize_stpn_parameter_bundle, precompute_stpn_solution, run_stpn_analysis


# 0. Define paths and dataset parameters
TIME_STEP = 1  # hours
INTERNAL_STEPS = 2  # 0 means just external contacts, 1 means external + internal, 2 means one more propagation layer
QUANTILE = 4

EXTERNAL_CONTACTS = 100
TESTS = 100
SYMPTOMS = 100
INTERNAL_CONTACTS = [200, 400, 800]
SUBJECTS = 8

RUN_DIR_PREFIX = "sweep_"
RUN_COMPLETION_SENTINEL = "_run_completed.json"
PROGRESS_DIR_NAME = "_progress"
DEFAULT_MAX_WORKERS = max(1, (os.cpu_count() or 1) // 2)

LEVEL_LABELS_IT = {
    "lower": "basso",
    "mid": "medio",
    "upper": "alto",
}

REFERENCE_SWEEP_COLUMNS = [
    "run_id",
    "infectiousness_level",
    "healing_level",
    "symptoms_level",
    "isolating_level",
    "symptoms_onset_level",
    "notification_to_isolation_level",
    "symptomatic_period_level",
    "java_runtime_seconds",
    "simulation_runtime_seconds",
    "simulation_runs",
    "kendall_analysis",
    "kendall_simulation",
    "spearman_analysis",
    "spearman_simulation",
    "top_1_accuracy_analysis",
    "top_1_accuracy_simulation",
    "top_2_accuracy_analysis",
    "top_2_accuracy_simulation",
    "top_3_accuracy_analysis",
    "top_3_accuracy_simulation",
    "top_4_accuracy_analysis",
    "top_4_accuracy_simulation",
    "mrr_analysis",
    "mrr_simulation",
    "brier_score_analysis",
    "brier_score_simulation",
    "ece_analysis",
    "ece_simulation",
    "note",
]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def write_json(path, payload):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4)


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_text(path, content):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def resolve_worker_count(task_count):
    if task_count <= 0:
        return 1
    env_value = os.environ.get("CHITA_MAX_WORKERS")
    if env_value:
        try:
            requested = int(env_value)
        except ValueError:
            requested = DEFAULT_MAX_WORKERS
        return max(1, min(task_count, requested))
    return max(1, min(task_count, DEFAULT_MAX_WORKERS))


def to_italian_level(level):
    return LEVEL_LABELS_IT.get(level, level)


def progress_dir(save_path):
    return ensure_dir(os.path.join(save_path, PROGRESS_DIR_NAME))


def stage_checkpoint_path(save_path, stage_name):
    return os.path.join(progress_dir(save_path), f"{stage_name}.json")


def stage_complete_path(save_path, stage_name):
    return os.path.join(progress_dir(save_path), f"{stage_name}.complete")


def write_stage_checkpoint(save_path, stage_name, completed, total, status="running", extra=None):
    payload = {
        "stage": stage_name,
        "completed": int(completed),
        "total": int(total),
        "status": status,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        payload.update(extra)
    write_json(stage_checkpoint_path(save_path, stage_name), payload)


def mark_stage_complete(save_path, stage_name):
    write_text(
        stage_complete_path(save_path, stage_name),
        datetime.now().isoformat(timespec="seconds"),
    )


def resolve_run_directory(results_root="results"):
    ensure_dir(results_root)

    force_new = os.environ.get("CHITA_FORCE_NEW_RUN", "0") == "1"
    if not force_new:
        run_dirs = [
            os.path.join(results_root, name)
            for name in os.listdir(results_root)
            if name.startswith(RUN_DIR_PREFIX)
            and os.path.isdir(os.path.join(results_root, name))
        ]
        run_dirs.sort(key=os.path.getmtime, reverse=True)

        for run_dir in run_dirs:
            sentinel_path = os.path.join(run_dir, RUN_COMPLETION_SENTINEL)
            if not os.path.exists(sentinel_path):
                print(f"Resuming interrupted run at: {run_dir}")
                return run_dir, True

    save_path = os.path.join("results", RUN_DIR_PREFIX + time.strftime("%Y%m%d-%H%M"))
    ensure_dir(save_path)
    print(f"Starting new run at: {save_path}")
    return save_path, False


def percentile(values, probability):
    if not values:
        raise ValueError("Cannot compute a percentile from an empty sequence.")
    if probability <= 0:
        return float(min(values))
    if probability >= 1:
        return float(max(values))

    sorted_values = sorted(float(value) for value in values)
    position = (len(sorted_values) - 1) * probability
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    weight = position - lower_index
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    return lower_value + (upper_value - lower_value) * weight


def summarize_java_analysis_runtimes(java_analysis_summaries):
    runtimes = [summary["analysis_wall_runtime_seconds"] for summary in java_analysis_summaries]
    summary = {
        "n_runs": len(runtimes),
        "runtime_metric": "analysis_wall_runtime_seconds",
        "min_runtime_seconds": min(runtimes),
        "max_runtime_seconds": max(runtimes),
        "quartiles": {
            "q2_seconds": percentile(runtimes, 0.50),
            "q3_seconds": percentile(runtimes, 0.75),
            "q4_seconds": percentile(runtimes, 1.00),
        },
        "per_dataset": {},
        "per_parameter_case_id": {},
    }

    dataset_labels = sorted({item["dataset_stem"] for item in java_analysis_summaries})
    for dataset_label in dataset_labels:
        dataset_runtimes = [
            item["analysis_wall_runtime_seconds"]
            for item in java_analysis_summaries
            if item["dataset_stem"] == dataset_label
        ]
        summary["per_dataset"][dataset_label] = {
            "n_runs": len(dataset_runtimes),
            "min_runtime_seconds": min(dataset_runtimes),
            "max_runtime_seconds": max(dataset_runtimes),
            "quartiles": {
                "q2_seconds": percentile(dataset_runtimes, 0.50),
                "q3_seconds": percentile(dataset_runtimes, 0.75),
                "q4_seconds": percentile(dataset_runtimes, 1.00),
            },
        }

    parameter_case_ids = sorted({item["parameter_case_id"] for item in java_analysis_summaries})
    for parameter_case_id in parameter_case_ids:
        case_runtimes = [
            item["analysis_wall_runtime_seconds"]
            for item in java_analysis_summaries
            if item["parameter_case_id"] == parameter_case_id
        ]
        summary["per_parameter_case_id"][parameter_case_id] = {
            "n_runs": len(case_runtimes),
            "min_runtime_seconds": min(case_runtimes),
            "max_runtime_seconds": max(case_runtimes),
            "quartiles": {
                "q2_seconds": percentile(case_runtimes, 0.50),
                "q3_seconds": percentile(case_runtimes, 0.75),
                "q4_seconds": percentile(case_runtimes, 1.00),
            },
        }

    return summary


def build_parameter_combinations(parameter_path):
    with open(parameter_path, "r", encoding="utf-8") as handle:
        parameters = json.load(handle)

    transitions = list(parameters.keys())
    levels = [list(parameters[transition].keys()) for transition in transitions]
    combinations = list(itertools.product(*levels))

    all_cases = []
    ground_truth_case = None

    for combination in combinations:
        current_set = {}
        level_selection = {}
        is_ground_truth_combination = True

        for transition_name, level in zip(transitions, combination):
            transition_payload = dict(parameters[transition_name][level])
            transition_payload["level"] = level
            current_set[transition_name] = transition_payload
            level_selection[transition_name] = level
            if level != "mid":
                is_ground_truth_combination = False

        normalized = normalize_stpn_parameter_bundle(current_set)
        case_payload = {
            "parameter_bundle": current_set,
            "levels": level_selection,
            "case_id": normalized["case_id"],
        }

        if is_ground_truth_combination:
            ground_truth_case = case_payload
        all_cases.append(case_payload)

    if ground_truth_case is None:
        raise RuntimeError("Could not find the all-mid ground-truth parameter combination.")

    return all_cases, ground_truth_case


def ensure_dataset_analysis_input(analysis_dir, observed_simulated_path):
    ensure_dir(analysis_dir)
    destination = os.path.join(analysis_dir, os.path.basename(observed_simulated_path))
    if os.path.abspath(destination) != os.path.abspath(observed_simulated_path):
        shutil.copy2(observed_simulated_path, destination)
    return destination


def ensure_python_simulation_input(analysis_dir, dataset_path):
    ensure_dir(analysis_dir)
    destination = os.path.join(analysis_dir, os.path.basename(dataset_path))
    if os.path.abspath(destination) != os.path.abspath(dataset_path):
        shutil.copy2(dataset_path, destination)
    return destination


def find_generated_file(run_dir, suffix):
    matches = [
        os.path.join(run_dir, name)
        for name in os.listdir(run_dir)
        if name.endswith(suffix)
    ]
    if not matches:
        raise FileNotFoundError(
            f"Expected at least one file ending with '{suffix}' in {run_dir}, found none."
        )
    if len(matches) == 1:
        return matches[0]
    matches.sort(key=os.path.getmtime, reverse=True)
    return matches[0]


def _is_valid_path(path_value):
    return isinstance(path_value, str) and os.path.exists(path_value)


def run_precompute_task(save_path, parameter_case, time_step_hours):
    case_id = parameter_case["case_id"]
    parameter_bundle = parameter_case["parameter_bundle"]

    summary_path = os.path.join(save_path, "_precompute", case_id, "precompute_summary.json")
    if os.path.exists(summary_path):
        cached_summary = read_json(summary_path)
        if _is_valid_path(cached_summary.get("stpn_solution_path")) and _is_valid_path(
            cached_summary.get("observation_curve_path")
        ):
            return cached_summary

    precomputed = precompute_stpn_solution(
        parameter_bundle=parameter_bundle,
        cache_dir=None,
        repo_root=".",
        time_step_hours=time_step_hours,
    )
    summary = {
        "case_id": precomputed["parameter_bundle"]["case_id"],
        "cache_dir": precomputed["cache_dir"],
        "cache_hit": precomputed["cache_hit"],
        "stpn_solution_path": precomputed["stpn_solution_path"],
        "observation_curve_path": precomputed["observation_curve_path"],
    }
    write_json(summary_path, summary)
    return summary


def run_java_analysis_task(save_path, dataset_run, parameter_case, time_step_hours, iterations):
    case_id = parameter_case["case_id"]
    parameter_bundle = parameter_case["parameter_bundle"]

    analysis_dir = os.path.join(
        save_path,
        "java_analysis",
        dataset_run["dataset_stem"],
        case_id,
    )
    summary_path = os.path.join(analysis_dir, "java_analysis_summary.json")

    if os.path.exists(summary_path):
        cached_summary = read_json(summary_path)
        analysis_path = cached_summary.get("analysis_path")
        if not _is_valid_path(analysis_path):
            try:
                analysis_path = find_generated_file(analysis_dir, f"_tracks_it{iterations}.json")
                cached_summary["analysis_path"] = analysis_path
                write_json(summary_path, cached_summary)
            except FileNotFoundError:
                analysis_path = None

        if _is_valid_path(analysis_path):
            return cached_summary

    ensure_dataset_analysis_input(
        analysis_dir=analysis_dir,
        observed_simulated_path=dataset_run["observed_simulated_path"],
    )

    started_at = time.perf_counter()
    java_result = run_stpn_analysis(
        parameter_bundle=parameter_bundle,
        analysis_dir=analysis_dir,
        iterations=iterations,
        cache_dir=None,
        repo_root=".",
        time_step_hours=time_step_hours,
    )
    analysis_wall_runtime_seconds = time.perf_counter() - started_at
    analysis_path = find_generated_file(analysis_dir, f"_tracks_it{iterations}.json")

    summary = {
        "dataset_path": dataset_run["dataset_path"],
        "dataset_stem": dataset_run["dataset_stem"],
        "observed_simulated_path": dataset_run["observed_simulated_path"],
        "parameter_case_id": java_result["parameter_bundle"]["case_id"],
        "parameter_levels": java_result["parameter_bundle"].get("levels", {}),
        "parameter_bundle_path": java_result["parameter_bundle_path"],
        "stpn_solution_path": java_result["stpn_solution_path"],
        "observation_curve_path": java_result["observation_curve_path"],
        "analysis_dir": analysis_dir,
        "analysis_path": analysis_path,
        "time_step_hours": time_step_hours,
        "iterations": iterations,
        "analysis_wall_runtime_seconds": analysis_wall_runtime_seconds,
    }
    write_json(summary_path, summary)
    return summary


def run_python_analysis_task(
    save_path,
    dataset_run,
    parameter_case,
    quartile_label,
    runtime_budget_seconds,
    time_step_hours,
):
    case_id = parameter_case["case_id"]
    parameter_bundle = parameter_case["parameter_bundle"]

    analysis_dir = os.path.join(
        save_path,
        "python_analysis",
        quartile_label,
        dataset_run["dataset_stem"],
        case_id,
    )
    summary_path = os.path.join(analysis_dir, "python_analysis_summary.json")

    if os.path.exists(summary_path):
        cached_summary = read_json(summary_path)
        if _is_valid_path(cached_summary.get("averaged_results_path")):
            return cached_summary

    analysis_dataset_path = ensure_python_simulation_input(
        analysis_dir=analysis_dir,
        dataset_path=dataset_run["dataset_path"],
    )

    python_result = run_dataset_simulations(
        dataset_path=analysis_dataset_path,
        rep=None,
        run_until_convergence=False,
        iterations_cap=100_000,
        convergence_threshold=1e-6,
        fine_grained=False,
        dataset_label=f"{dataset_run['dataset_stem']}_{quartile_label}",
        seed=30,
        prune_after_positive_test=False,
        export_observed_simulation=False,
        pruning_seed=None,
        time_step_hours=time_step_hours,
        max_runtime_seconds=runtime_budget_seconds,
        parameter_bundle=parameter_bundle,
        save_plots=False,
    )

    summary = {
        "quartile_label": quartile_label,
        "runtime_budget_seconds": runtime_budget_seconds,
        "dataset_path": dataset_run["dataset_path"],
        "dataset_stem": dataset_run["dataset_stem"],
        "parameter_case_id": case_id,
        "parameter_levels": normalize_stpn_parameter_bundle(parameter_bundle).get("levels", {}),
        "analysis_dir": analysis_dir,
        "analysis_dataset_path": analysis_dataset_path,
        "time_step_hours": time_step_hours,
        "actual_runtime_seconds": python_result["actual_runtime_seconds"],
        "max_runtime_seconds": python_result["max_runtime_seconds"],
        "rep_done": python_result["rep_done"],
        "averaged_results_path": python_result["averaged_results_path"],
        "plots_dir": python_result["plots_dir"],
        "dkw_csv_path": python_result["dkw_csv_path"],
        "suppressed_stdout_log_path": python_result["suppressed_stdout_log_path"],
    }
    write_json(summary_path, summary)
    return summary


def run_comparison_task(
    save_path,
    quartile_label,
    runtime_budget_seconds,
    dataset_run,
    parameter_case,
    java_summary,
    python_summary,
    time_step_hours,
    iterations,
):
    case_id = parameter_case["case_id"]
    parameter_levels = parameter_case["levels"]

    comparison_dir = os.path.join(
        save_path,
        "comparison",
        quartile_label,
        dataset_run["dataset_stem"],
        case_id,
    )
    summary_path = os.path.join(comparison_dir, "comparison_summary.json")
    if os.path.exists(summary_path):
        cached_summary = read_json(summary_path)
        if _is_valid_path(cached_summary.get("ground_truth_path")):
            return cached_summary

    ensure_dir(comparison_dir)

    run_id = f"{dataset_run['dataset_stem']}__{case_id}__{quartile_label}"
    analysis_path = java_summary.get("analysis_path")
    if not _is_valid_path(analysis_path):
        analysis_path = find_generated_file(
            java_summary["analysis_dir"],
            f"_tracks_it{iterations}.json",
        )

    baseline_path = python_summary["averaged_results_path"]
    ground_truth_path = dataset_run["ground_truth_path"]

    analysis_curve_plots = sp.create_analysis_subject_curve_plots(
        run_dir=comparison_dir,
        run_name=run_id,
        analysis_path=analysis_path,
        ground_truth_path=ground_truth_path,
        baseline_path=baseline_path,
        granularity=time_step_hours,
        save_plots=True,
    )

    precision_dir = os.path.join(comparison_dir, "precision_metrics")
    ensure_dir(precision_dir)
    prediction_metrics_path = os.path.join(precision_dir, "metrics_prediction.json")
    baseline_metrics_path = os.path.join(precision_dir, "metrics_baseline.json")

    metrics_stdout = io.StringIO()
    with contextlib.redirect_stdout(metrics_stdout):
        process_and_save(
            analysis_path,
            ground_truth_path,
            M=10,
            metrics_output=prediction_metrics_path,
            plots_dir=os.path.join(precision_dir, "numericalAnalysis", "plots"),
            save_plots=True,
        )
        process_and_save(
            baseline_path,
            ground_truth_path,
            M=10,
            metrics_output=baseline_metrics_path,
            plots_dir=os.path.join(precision_dir, "simulatedBaseline", "plots"),
            save_plots=True,
        )

    write_text(os.path.join(precision_dir, "metrics_stdout.log"), metrics_stdout.getvalue())

    comparison_metrics = sp.create_analysis_vs_simulation_plots(
        run_dir=comparison_dir,
        run_name=run_id,
        ground_truth_path=ground_truth_path,
        analysis_path=analysis_path,
        baseline_path=baseline_path,
        include_moving_avg_metrics=True,
        save_plots=True,
    )

    summary = {
        "run_id": run_id,
        "quartile_label": quartile_label,
        "runtime_budget_seconds": runtime_budget_seconds,
        "dataset_path": dataset_run["dataset_path"],
        "dataset_stem": dataset_run["dataset_stem"],
        "ground_truth_path": ground_truth_path,
        "parameter_case_id": case_id,
        "parameter_levels": parameter_levels,
        "java_analysis": {
            "analysis_dir": java_summary["analysis_dir"],
            "analysis_path": analysis_path,
            "analysis_wall_runtime_seconds": java_summary["analysis_wall_runtime_seconds"],
            "parameter_bundle_path": java_summary["parameter_bundle_path"],
            "stpn_solution_path": java_summary["stpn_solution_path"],
            "observation_curve_path": java_summary["observation_curve_path"],
            "subject_curve_plots": analysis_curve_plots,
        },
        "python_analysis": {
            "analysis_dir": python_summary["analysis_dir"],
            "baseline_path": baseline_path,
            "actual_runtime_seconds": python_summary["actual_runtime_seconds"],
            "rep_done": python_summary["rep_done"],
            "dkw_csv_path": python_summary["dkw_csv_path"],
            "suppressed_stdout_log_path": python_summary["suppressed_stdout_log_path"],
        },
        "precision_metrics": {
            "prediction_metrics_path": prediction_metrics_path,
            "baseline_metrics_path": baseline_metrics_path,
            "prediction_mean_brier": sp.mean_metric_value(prediction_metrics_path, "Brier Score"),
            "prediction_mean_ece": sp.mean_metric_value(prediction_metrics_path, "ECE"),
            "baseline_mean_brier": sp.mean_metric_value(baseline_metrics_path, "Brier Score"),
            "baseline_mean_ece": sp.mean_metric_value(baseline_metrics_path, "ECE"),
        },
        "comparison_metrics": comparison_metrics,
        "comparison_dir": comparison_dir,
        "note": "",
    }

    write_json(summary_path, summary)
    return summary


def build_reference_row(comparison_summary):
    levels = comparison_summary.get("parameter_levels", {})
    analysis_metrics = comparison_summary.get("comparison_metrics", {}).get("analysis", {})
    simulation_metrics = comparison_summary.get("comparison_metrics", {}).get("simulation", {})
    precision_metrics = comparison_summary.get("precision_metrics", {})
    java_analysis = comparison_summary.get("java_analysis", {})
    python_analysis = comparison_summary.get("python_analysis", {})

    return {
        "run_id": comparison_summary.get("run_id"),
        "infectiousness_level": to_italian_level(levels.get("infectiousness")),
        "healing_level": to_italian_level(levels.get("healing")),
        "symptoms_level": to_italian_level(levels.get("symptoms")),
        "isolating_level": to_italian_level(levels.get("isolating")),
        "symptoms_onset_level": to_italian_level(levels.get("symptomsOnset")),
        "notification_to_isolation_level": to_italian_level(levels.get("notification_to_isolation")),
        "symptomatic_period_level": to_italian_level(levels.get("symptomatic_period")),
        "java_runtime_seconds": java_analysis.get("analysis_wall_runtime_seconds"),
        "simulation_runtime_seconds": python_analysis.get("actual_runtime_seconds"),
        "simulation_runs": python_analysis.get("rep_done"),
        "kendall_analysis": analysis_metrics.get("tau"),
        "kendall_simulation": simulation_metrics.get("tau"),
        "spearman_analysis": analysis_metrics.get("spearman"),
        "spearman_simulation": simulation_metrics.get("spearman"),
        "top_1_accuracy_analysis": analysis_metrics.get("top_1_precision_mean"),
        "top_1_accuracy_simulation": simulation_metrics.get("top_1_precision_mean"),
        "top_2_accuracy_analysis": analysis_metrics.get("top_2_precision_mean"),
        "top_2_accuracy_simulation": simulation_metrics.get("top_2_precision_mean"),
        "top_3_accuracy_analysis": analysis_metrics.get("top_3_precision_mean"),
        "top_3_accuracy_simulation": simulation_metrics.get("top_3_precision_mean"),
        "top_4_accuracy_analysis": analysis_metrics.get("top_4_precision_mean"),
        "top_4_accuracy_simulation": simulation_metrics.get("top_4_precision_mean"),
        "mrr_analysis": analysis_metrics.get("mrr"),
        "mrr_simulation": simulation_metrics.get("mrr"),
        "brier_score_analysis": precision_metrics.get("prediction_mean_brier"),
        "brier_score_simulation": precision_metrics.get("baseline_mean_brier"),
        "ece_analysis": precision_metrics.get("prediction_mean_ece"),
        "ece_simulation": precision_metrics.get("baseline_mean_ece"),
        "note": comparison_summary.get("note", ""),
    }


def write_reference_sweep_csv(csv_path, rows):
    ensure_dir(os.path.dirname(csv_path) or ".")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REFERENCE_SWEEP_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in REFERENCE_SWEEP_COLUMNS})


def main():
    save_path, resumed_run = resolve_run_directory(results_root="results")
    cache_path = ensure_dir(os.path.join("results", "cache"))

    write_json(
        os.path.join(save_path, "run_metadata.json"),
        {
            "save_path": save_path,
            "cache_path": cache_path,
            "time_step_hours": TIME_STEP,
            "internal_steps": INTERNAL_STEPS,
            "quantile": QUANTILE,
            "resumed_run": resumed_run,
            "started_at": datetime.now().isoformat(timespec="seconds"),
        },
    )

    # 1. Create datasets
    stage_name = "stage1_dataset_generation"
    write_stage_checkpoint(save_path, stage_name, 0, len(INTERNAL_CONTACTS), status="running")

    dataset_paths = []
    for index, internal_contacts in enumerate(INTERNAL_CONTACTS, start=1):
        dataset_path = os.path.join(save_path, f"dataset_{internal_contacts}.json")
        if not os.path.exists(dataset_path):
            dataset = dg.simulate_external_introduction(
                n_nodes=SUBJECTS,
                total_internal_contacts=internal_contacts,
                tmax_after_intro=2016,
                seed=30,
            )
            dg.save_dataset_event_sequence(dataset, dataset_path)
        dataset_paths.append(dataset_path)
        write_stage_checkpoint(save_path, stage_name, index, len(INTERNAL_CONTACTS), status="running")

    mark_stage_complete(save_path, stage_name)
    write_stage_checkpoint(save_path, stage_name, len(INTERNAL_CONTACTS), len(INTERNAL_CONTACTS), status="completed")

    parameter_cases, ground_truth_case = build_parameter_combinations(os.path.join(".", "parameters.json"))
    print(normalize_stpn_parameter_bundle(parameter_cases[0]["parameter_bundle"])["case_id"])
    print(f"\n\n\n{ground_truth_case['case_id']}\n\n\n")

    # 2. Run GT simulation on datasets (recoverable)
    stage_name = "stage2_ground_truth"
    write_stage_checkpoint(save_path, stage_name, 0, len(dataset_paths), status="running")

    dataset_runs = []
    for index, dataset_path in enumerate(tqdm(dataset_paths, desc="Running GT simulations"), start=1):
        gt_results_path = dataset_path.replace("dataset", "gt_results")
        gt_results = None

        if os.path.exists(gt_results_path):
            cached = read_json(gt_results_path)
            if _is_valid_path(cached.get("observed_simulated_path")) and _is_valid_path(
                cached.get("averaged_results_path")
            ):
                gt_results = cached

        if gt_results is None:
            gt_results = run_dataset_simulations(
                dataset_path=dataset_path,
                run_until_convergence=True,
                iterations_cap=100_000,
                convergence_threshold=1e-6,
                fine_grained=False,
                time_step_hours=TIME_STEP,
                seed=30,
                prune_after_positive_test=False,
                export_observed_simulation=True,
                pruning_seed=None,
                parameter_bundle=ground_truth_case["parameter_bundle"],
                save_plots=True,
            )
            write_json(gt_results_path, gt_results)

        dataset_runs.append(
            {
                "dataset_path": dataset_path,
                "dataset_stem": os.path.splitext(os.path.basename(dataset_path))[0],
                "gt_results_path": gt_results_path,
                "ground_truth_path": gt_results["averaged_results_path"],
                "observed_simulated_path": gt_results["observed_simulated_path"],
            }
        )
        write_stage_checkpoint(save_path, stage_name, index, len(dataset_paths), status="running")

    mark_stage_complete(save_path, stage_name)
    write_stage_checkpoint(save_path, stage_name, len(dataset_paths), len(dataset_paths), status="completed")
    print("GT simulations completed")

    # 3. Parallel precompute STPN curves (wait all)
    stage_name = "stage3_precompute"
    total_stage_tasks = len(parameter_cases)
    write_stage_checkpoint(save_path, stage_name, 0, total_stage_tasks, status="running")

    precompute_summaries = []
    worker_count = resolve_worker_count(total_stage_tasks)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(run_precompute_task, save_path, parameter_case, TIME_STEP)
            for parameter_case in parameter_cases
        ]
        completed = 0
        for future in tqdm(as_completed(futures), total=total_stage_tasks, desc="Precomputing STPN curves"):
            precompute_summaries.append(future.result())
            completed += 1
            write_stage_checkpoint(save_path, stage_name, completed, total_stage_tasks, status="running")

    precompute_summaries.sort(key=lambda item: item["case_id"])
    write_json(os.path.join(save_path, "java_precompute_summary.json"), precompute_summaries)
    mark_stage_complete(save_path, stage_name)
    write_stage_checkpoint(save_path, stage_name, total_stage_tasks, total_stage_tasks, status="completed")

    # 4. Parallel Java analysis (wait all)
    stage_name = "stage4_java_analysis"
    stage4_tasks = [
        (dataset_run, parameter_case)
        for dataset_run in dataset_runs
        for parameter_case in parameter_cases
    ]
    total_stage_tasks = len(stage4_tasks)
    write_stage_checkpoint(save_path, stage_name, 0, total_stage_tasks, status="running")

    java_analysis_summaries = []
    worker_count = resolve_worker_count(total_stage_tasks)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                run_java_analysis_task,
                save_path,
                dataset_run,
                parameter_case,
                TIME_STEP,
                INTERNAL_STEPS,
            )
            for dataset_run, parameter_case in stage4_tasks
        ]

        completed = 0
        for future in tqdm(as_completed(futures), total=total_stage_tasks, desc="Running Java analysis"):
            java_analysis_summaries.append(future.result())
            completed += 1
            write_stage_checkpoint(save_path, stage_name, completed, total_stage_tasks, status="running")

    java_analysis_summaries.sort(key=lambda item: (item["dataset_stem"], item["parameter_case_id"]))
    write_json(os.path.join(save_path, "java_analysis_summary.json"), java_analysis_summaries)
    mark_stage_complete(save_path, stage_name)
    write_stage_checkpoint(save_path, stage_name, total_stage_tasks, total_stage_tasks, status="completed")

    # 5. Runtime quartiles
    stage_name = "stage5_runtime_quartiles"
    write_stage_checkpoint(save_path, stage_name, 0, 1, status="running")

    java_analysis_runtime_quartiles = summarize_java_analysis_runtimes(java_analysis_summaries)
    write_json(
        os.path.join(save_path, "java_analysis_runtime_quartiles.json"),
        java_analysis_runtime_quartiles,
    )

    quartile_label = f"q{QUANTILE}"
    quartile_key = f"{quartile_label}_seconds"
    if quartile_key not in java_analysis_runtime_quartiles["quartiles"]:
        raise ValueError(f"Unsupported quantile: {QUANTILE}. Expected one of 2, 3, 4.")

    runtime_budget_seconds = java_analysis_runtime_quartiles["quartiles"][quartile_key]

    mark_stage_complete(save_path, stage_name)
    write_stage_checkpoint(save_path, stage_name, 1, 1, status="completed")

    # 6. Parallel Python analysis with runtime budget (wait all)
    stage_name = "stage6_python_analysis"
    stage6_tasks = [
        (dataset_run, parameter_case)
        for dataset_run in dataset_runs
        for parameter_case in parameter_cases
    ]
    total_stage_tasks = len(stage6_tasks)
    write_stage_checkpoint(save_path, stage_name, 0, total_stage_tasks, status="running")

    python_analysis_summaries = []
    worker_count = resolve_worker_count(total_stage_tasks)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                run_python_analysis_task,
                save_path,
                dataset_run,
                parameter_case,
                quartile_label,
                runtime_budget_seconds,
                TIME_STEP,
            )
            for dataset_run, parameter_case in stage6_tasks
        ]

        completed = 0
        for future in tqdm(as_completed(futures), total=total_stage_tasks, desc="Running Python analysis"):
            python_analysis_summaries.append(future.result())
            completed += 1
            write_stage_checkpoint(save_path, stage_name, completed, total_stage_tasks, status="running")

    python_analysis_summaries.sort(key=lambda item: (item["dataset_stem"], item["parameter_case_id"]))
    write_json(os.path.join(save_path, "python_analysis_summary.json"), python_analysis_summaries)
    mark_stage_complete(save_path, stage_name)
    write_stage_checkpoint(save_path, stage_name, total_stage_tasks, total_stage_tasks, status="completed")

    # 7. Parallel comparison + metrics + plots (wait all)
    stage_name = "stage7_comparison"
    java_summary_by_key = {
        (summary["dataset_stem"], summary["parameter_case_id"]): summary
        for summary in java_analysis_summaries
    }
    python_summary_by_key = {
        (summary["dataset_stem"], summary["parameter_case_id"]): summary
        for summary in python_analysis_summaries
    }

    stage7_tasks = []
    for dataset_run in dataset_runs:
        for parameter_case in parameter_cases:
            task_key = (dataset_run["dataset_stem"], parameter_case["case_id"])
            if task_key not in java_summary_by_key:
                raise KeyError(f"Missing Java analysis summary for key: {task_key}")
            if task_key not in python_summary_by_key:
                raise KeyError(f"Missing Python analysis summary for key: {task_key}")
            stage7_tasks.append(
                (
                    dataset_run,
                    parameter_case,
                    java_summary_by_key[task_key],
                    python_summary_by_key[task_key],
                )
            )

    total_stage_tasks = len(stage7_tasks)
    write_stage_checkpoint(save_path, stage_name, 0, total_stage_tasks, status="running")

    comparison_summaries = []
    worker_count = resolve_worker_count(total_stage_tasks)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                run_comparison_task,
                save_path,
                quartile_label,
                runtime_budget_seconds,
                dataset_run,
                parameter_case,
                java_summary,
                python_summary,
                TIME_STEP,
                INTERNAL_STEPS,
            )
            for dataset_run, parameter_case, java_summary, python_summary in stage7_tasks
        ]

        completed = 0
        for future in tqdm(as_completed(futures), total=total_stage_tasks, desc="Running comparisons"):
            comparison_summaries.append(future.result())
            completed += 1
            write_stage_checkpoint(save_path, stage_name, completed, total_stage_tasks, status="running")

    comparison_summaries.sort(key=lambda item: (item["dataset_stem"], item["parameter_case_id"]))

    comparison_summary_json_path = os.path.join(save_path, f"comparison_summary_{quartile_label}.json")
    write_json(comparison_summary_json_path, comparison_summaries)

    # Output CSV with the reference schema.
    sweep_summary_rows = [build_reference_row(summary) for summary in comparison_summaries]
    sweep_summary_csv_path = os.path.join(save_path, "sweep_summary.csv")
    write_reference_sweep_csv(sweep_summary_csv_path, sweep_summary_rows)

    # Keep the old filename as compatibility alias, but with the reference schema.
    comparison_summary_csv_path = os.path.join(save_path, f"comparison_summary_{quartile_label}.csv")
    write_reference_sweep_csv(comparison_summary_csv_path, sweep_summary_rows)

    mark_stage_complete(save_path, stage_name)
    write_stage_checkpoint(save_path, stage_name, total_stage_tasks, total_stage_tasks, status="completed")

    write_json(
        os.path.join(save_path, RUN_COMPLETION_SENTINEL),
        {
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "save_path": save_path,
            "quartile_label": quartile_label,
            "rows_written": len(sweep_summary_rows),
            "sweep_summary_csv_path": sweep_summary_csv_path,
            "comparison_summary_json_path": comparison_summary_json_path,
            "comparison_summary_csv_path": comparison_summary_csv_path,
        },
    )

    print(
        "Point 7 completed. "
        f"Saved {len(comparison_summaries)} comparison summaries to {comparison_summary_json_path}, "
        f"{comparison_summary_csv_path}, and {sweep_summary_csv_path}."
    )


if __name__ == "__main__":
    main()
