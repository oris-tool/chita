import argparse
import contextlib
import csv
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import io
import itertools
import os
import re
import shutil
import subprocess
import time
import traceback
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from compute_precision_metrics import process_and_save
from run_n_simulations import run_dataset_simulations
import sweep_pipeline as sp


PARAMETER_CASE_MODE_MID_ONLY = "mid-only"
PARAMETER_CASE_MODE_FOCUSED_GRID = "focused-grid"
PARAMETER_CASE_MODE_CHOICES = (
    PARAMETER_CASE_MODE_MID_ONLY,
    PARAMETER_CASE_MODE_FOCUSED_GRID,
    "aligned-scenarios",
    "all-combinations",
)
DEFAULT_FOCUSED_FAMILIES = (
    "infectiousness",
    "healing",
    "isolating",
)
DEFAULT_JAVA_ITERATIONS = tuple(range(1, 9))
PARALLEL_BACKEND_AUTO = "auto"
PARALLEL_BACKEND_THREAD = "thread"
PARALLEL_BACKEND_PROCESS = "process"
PARALLEL_BACKEND_CHOICES = (
    PARALLEL_BACKEND_AUTO,
    PARALLEL_BACKEND_THREAD,
    PARALLEL_BACKEND_PROCESS,
)

METRIC_SELECTION_BUCKET_SIZE = 10
LEVEL_LABELS_IT = {
    "lower": "basso",
    "mid": "medio",
    "upper": "alto",
}

COMPACT_RESULT_COLUMNS = [
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


def parse_java_iterations(values):
    if not values:
        raise ValueError("At least one Java iteration count is required.")

    parsed = sorted({int(value) for value in values})
    if any(value <= 0 for value in parsed):
        raise ValueError("Java iteration counts must all be greater than 0.")
    return parsed


def normalize_focus_families(families):
    normalized = []
    for family in families:
        family_name = str(family).strip()
        if family_name not in sp.PARAMETER_FAMILY_ORDER:
            raise ValueError(
                f"Unsupported focus family '{family_name}'. "
                f"Expected one of: {', '.join(sp.PARAMETER_FAMILY_ORDER)}"
            )
        if family_name not in normalized:
            normalized.append(family_name)
    if not normalized:
        raise ValueError("At least one focus family is required for focused-grid mode.")
    return tuple(normalized)


def enumerate_iteration_parameter_cases(parameter_space, mode, focused_families):
    if mode in ("all-combinations", "aligned-scenarios"):
        return sp.enumerate_parameter_cases(parameter_space, mode=mode)

    if mode == PARAMETER_CASE_MODE_MID_ONLY:
        return [
            sp.resolve_uniform_parameter_bundle(
                parameter_space,
                sp.GROUND_TRUTH_PARAMETER_LEVEL,
            )
        ]

    if mode != PARAMETER_CASE_MODE_FOCUSED_GRID:
        raise ValueError(f"Unsupported parameter case mode: {mode}")

    focus_tuple = normalize_focus_families(focused_families)
    base_levels = {
        family: sp.GROUND_TRUTH_PARAMETER_LEVEL
        for family in sp.PARAMETER_FAMILY_ORDER
    }
    cases = []
    for levels in itertools.product(sp.PARAMETER_LEVEL_ORDER, repeat=len(focus_tuple)):
        level_selection = dict(base_levels)
        for family, level in zip(focus_tuple, levels):
            level_selection[family] = level
        cases.append(sp.resolve_parameter_bundle(parameter_space, level_selection))
    return cases


def write_iteration_sweep_markdown(
    output_root,
    seed_base,
    time_step_hours,
    baseline_runtime_multiplier,
    include_moving_avg_metrics,
    dataset_source,
    tests_enabled,
    observed_test_ablation,
    parameter_space,
    ground_truth_parameter_bundle,
    parameter_case_mode,
    focused_families,
    parameter_case_count,
    parameter_manifest_paths,
    java_iterations,
    generate_plot_images,
    reduce_plots,
):
    content = f"""# Java Iteration Sweep Parameters

This file documents the parameters used by `sweep_pipeline_n_iteration.py`.

## Sweep Configuration

- `TIME_LIMITS` (days): {sp.TIME_LIMITS}
- `N_SUBJECTS`: {sp.N_SUBJECTS}
- `TOTAL_INTERNAL_CONTACTS`: {sp.TOTAL_INTERNAL_CONTACTS}
- `seed_base`: {seed_base}
- `time_step_hours`: {time_step_hours}
- `baseline_runtime_multiplier`: {baseline_runtime_multiplier}
- `moving_avg_metrics_enabled`: {include_moving_avg_metrics}
- `dataset_source`: {dataset_source}
- `tests_enabled_in_raw_dataset`: {tests_enabled}
- `observed_test_ablation`: {observed_test_ablation}
- `parameter_ods_path`: {parameter_space["source_path"]}
- `ground_truth_parameter_case_id`: {ground_truth_parameter_bundle["case_id"]}
- `parameter_case_mode`: {parameter_case_mode}
- `focused_families`: {list(focused_families)}
- `parameter_case_count`: {parameter_case_count}
- `java_iterations`: {java_iterations}
- `parameter_manifest_csv`: {parameter_manifest_paths["csv_path"]}
- `parameter_manifest_json`: {parameter_manifest_paths["json_path"]}
- `generate_plot_images`: {generate_plot_images}
- `reduce_plots`: {reduce_plots}

## Notes

- Shared ground truth is still generated once per dataset combination with the fixed mid bundle.
- Each parameter bundle is then evaluated repeatedly while only changing the Java `n_iterations` value.
- For every Java iteration count, the Python simulation baseline gets a runtime budget of `java_analysis_runtime_seconds * baseline_runtime_multiplier`.
- Java precompute cache reuse stays enabled because STPN solutions and observation curves depend on the parameter bundle and time step, not on `n_iterations`.
"""
    output_path = os.path.join(output_root, "iteration_sweep_parameters.md")
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(content)
    return output_path


def run_java_analysis_with_iterations(
    repo_root,
    run_dir,
    java_iterations,
    time_step_hours=sp.TIME_STEP_HOURS,
    parameter_bundle_path=None,
    parameter_bundle=None,
):
    class_root = os.path.join(repo_root, "out", "production", "chita-main-test")
    stpn_analysis_class_path = os.path.join(class_root, "com", "chita", "analysis", "STPNAnalysis.class")
    if not os.path.exists(stpn_analysis_class_path):
        raise FileNotFoundError("Compiled STPNAnalysis.class not found under out/production/chita-main-test.")

    java_class_fingerprint = sp.file_sha256(stpn_analysis_class_path)
    java_executable = sp.resolve_java_executable()
    gson_jar = sp.resolve_optional_jar(
        [
            os.path.join(repo_root, "lib__", "gson.jar"),
            os.path.join(repo_root, "lib__", "gson-2.13.1.jar"),
            os.path.join(repo_root, "lib__", "gson-2.11.0.jar"),
            os.path.join(
                os.path.expanduser("~"),
                ".m2",
                "repository",
                "com",
                "google",
                "code",
                "gson",
                "gson",
                "2.13.1",
                "gson-2.13.1.jar",
            ),
            os.path.join(
                os.path.expanduser("~"),
                ".m2",
                "repository",
                "com",
                "google",
                "code",
                "gson",
                "gson",
                "2.11.0",
                "gson-2.11.0.jar",
            ),
            os.path.join(
                os.path.expanduser("~"),
                ".gradle",
                "caches",
                "modules-2",
                "files-2.1",
                "com.google.code.gson",
                "gson",
                "2.10.1",
                "b3add478d4382b78ea20b1671390a858002feb6c",
                "gson-2.10.1.jar",
            ),
        ],
        "gson",
    )
    shared_precompute_cache = None
    seed_reuse = None
    observation_curve_cache_path = None
    local_observation_curve_path = None
    stpn_solution_filename = f"stpn_solution_ts{sp.sanitized_time_step_label(time_step_hours)}.csv"
    stpn_solution_path = os.path.join(run_dir, stpn_solution_filename)
    effective_parameter_bundle_path = parameter_bundle_path
    if parameter_bundle is not None:
        shared_precompute_cache = sp.prepare_java_precompute_cache(
            repo_root=repo_root,
            parameter_bundle=parameter_bundle,
            time_step_hours=time_step_hours,
            java_class_fingerprint=java_class_fingerprint,
        )
        stpn_solution_filename = shared_precompute_cache["stpn_solution_filename"]
        stpn_solution_path = shared_precompute_cache["stpn_solution_path"]
        observation_curve_cache_path = shared_precompute_cache["observation_curve_path"]
        local_observation_curve_path = os.path.join(
            run_dir,
            shared_precompute_cache["observation_curve_filename"],
        )
        effective_parameter_bundle_path = shared_precompute_cache["parameter_bundle_path"]

    classpath = os.pathsep.join(
        [
            class_root,
            os.path.join(repo_root, "lib__", "*"),
            gson_jar,
        ]
    )
    command = [
        java_executable,
        "-cp",
        classpath,
        "com.chita.analysis.STPNAnalysis",
        "--time-step",
        str(time_step_hours),
        "--iterations",
        str(java_iterations),
        "--stpn-solution-path",
        os.path.abspath(stpn_solution_path),
    ]
    if effective_parameter_bundle_path is not None:
        command.extend(
            [
                "--parameter-bundle",
                os.path.abspath(effective_parameter_bundle_path),
            ]
        )

    java_analysis_stdout_log_path = os.path.join(run_dir, "java_analysis_stdout.log")
    java_analysis_stderr_log_path = os.path.join(run_dir, "java_analysis_stderr.log")
    precompute_log_dir = shared_precompute_cache["cache_dir"] if shared_precompute_cache is not None else run_dir
    java_precompute_stdout_log_path = os.path.join(precompute_log_dir, "java_precompute_stdout.log")
    java_precompute_stderr_log_path = os.path.join(precompute_log_dir, "java_precompute_stderr.log")

    def execute_java_command(extra_args=None, stdout_log_path=None, stderr_log_path=None, working_dir=None):
        started_at = time.perf_counter()
        result = subprocess.run(
            command + ([] if extra_args is None else extra_args),
            cwd=run_dir if working_dir is None else working_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        elapsed = time.perf_counter() - started_at
        if result.returncode != 0:
            raise RuntimeError(
                "Java analysis failed.\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )
        if stdout_log_path is not None:
            with open(stdout_log_path, "w", encoding="utf-8") as handle:
                handle.write(result.stdout)
        if stderr_log_path is not None:
            with open(stderr_log_path, "w", encoding="utf-8") as handle:
                handle.write(result.stderr)
        return result, elapsed

    def parse_timing_metric(stdout, metric_name):
        match = re.search(rf"__TIMING__ {re.escape(metric_name)}=([0-9]+(?:\.[0-9]+)?)", stdout)
        if match is None:
            return None
        return float(match.group(1))

    precomputation_runtime_seconds = 0.0
    precomputation_performed = False
    if shared_precompute_cache is not None:
        java_precompute_cache_hit = (
            os.path.exists(stpn_solution_path)
            and os.path.exists(observation_curve_cache_path)
        )
        if not java_precompute_cache_hit:
            seed_reuse = sp.try_seed_java_precompute_cache(
                repo_root=repo_root,
                parameter_bundle=parameter_bundle,
                time_step_hours=time_step_hours,
                cache_entry=shared_precompute_cache,
            )
            java_precompute_cache_hit = (
                os.path.exists(stpn_solution_path)
                and os.path.exists(observation_curve_cache_path)
            )
        if not java_precompute_cache_hit:
            _, precomputation_runtime_seconds = execute_java_command(
                ["--precompute-only"],
                stdout_log_path=java_precompute_stdout_log_path,
                stderr_log_path=java_precompute_stderr_log_path,
                working_dir=shared_precompute_cache["cache_dir"],
            )
            precomputation_performed = True
        if not os.path.exists(observation_curve_cache_path):
            raise FileNotFoundError(
                f"Expected observation curve cache file after precompute: {observation_curve_cache_path}"
            )
        shutil.copy2(observation_curve_cache_path, local_observation_curve_path)
    else:
        java_precompute_cache_hit = False
        if not os.path.exists(stpn_solution_path):
            _, precomputation_runtime_seconds = execute_java_command(
                ["--precompute-only"],
                stdout_log_path=java_precompute_stdout_log_path,
                stderr_log_path=java_precompute_stderr_log_path,
            )
            precomputation_performed = True

    result, analysis_wall_runtime_seconds = execute_java_command(
        stdout_log_path=java_analysis_stdout_log_path,
        stderr_log_path=java_analysis_stderr_log_path,
    )
    analysis_core_runtime_seconds = parse_timing_metric(result.stdout, "core_analysis_runtime_seconds")
    analysis_runtime_seconds = (
        analysis_core_runtime_seconds
        if analysis_core_runtime_seconds is not None
        else analysis_wall_runtime_seconds
    )
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "java_executable": java_executable,
        "java_iterations": java_iterations,
        "stpn_solution_path": stpn_solution_path,
        "time_step_hours": time_step_hours,
        "analysis_runtime_seconds": analysis_runtime_seconds,
        "analysis_wall_runtime_seconds": analysis_wall_runtime_seconds,
        "analysis_runtime_excludes_overhead": analysis_core_runtime_seconds is not None,
        "stpn_precomputation_runtime_seconds": precomputation_runtime_seconds,
        "stpn_precomputation_performed": precomputation_performed,
        "java_precompute_cache_root": shared_precompute_cache["cache_root"] if shared_precompute_cache is not None else None,
        "java_precompute_cache_dir": shared_precompute_cache["cache_dir"] if shared_precompute_cache is not None else None,
        "java_precompute_cache_key": shared_precompute_cache["cache_key"] if shared_precompute_cache is not None else None,
        "java_precompute_cache_manifest_path": shared_precompute_cache["manifest_path"] if shared_precompute_cache is not None else None,
        "java_precompute_cache_hit": shared_precompute_cache is not None and not precomputation_performed,
        "java_precompute_seed_run_dir": None if seed_reuse is None else seed_reuse["seed_run_dir"],
        "observation_curve_cache_path": observation_curve_cache_path,
        "local_observation_curve_path": local_observation_curve_path,
        "java_analysis_stdout_log_path": java_analysis_stdout_log_path,
        "java_analysis_stderr_log_path": java_analysis_stderr_log_path,
        "java_precompute_stdout_log_path": java_precompute_stdout_log_path if precomputation_performed else None,
        "java_precompute_stderr_log_path": java_precompute_stderr_log_path if precomputation_performed else None,
    }


def iteration_run_dir_name(java_iterations):
    return f"java_it_{int(java_iterations):02d}"


def resolve_parallel_backend(parallel_backend):
    if parallel_backend == PARALLEL_BACKEND_AUTO:
        return PARALLEL_BACKEND_THREAD if os.name == "nt" else PARALLEL_BACKEND_PROCESS
    if parallel_backend not in PARALLEL_BACKEND_CHOICES:
        raise ValueError(
            f"Unsupported parallel backend '{parallel_backend}'. "
            f"Expected one of: {', '.join(PARALLEL_BACKEND_CHOICES)}"
        )
    return parallel_backend


def to_italian_level(level):
    return LEVEL_LABELS_IT.get(level, level)


def metric_selection_manifest_rows(rows, metric_key):
    candidates = [
        row for row in rows
        if sp.finite_float(row.get(metric_key)) is not None
    ]
    candidates.sort(key=lambda row: (float(row[metric_key]), row["run_id"]))
    if not candidates:
        return []

    bucket_size = min(METRIC_SELECTION_BUCKET_SIZE, len(candidates))
    median_center = len(candidates) // 2
    median_start = max(
        0,
        min(len(candidates) - bucket_size, median_center - bucket_size // 2),
    )

    selected = {}

    def add(entry, label, rank_index):
        info = selected.setdefault(
            entry["run_id"],
            {
                "row": entry,
                "labels": [],
            },
        )
        info["labels"].append(f"{label} {rank_index}")

    for idx, entry in enumerate(reversed(candidates[-bucket_size:]), start=1):
        add(entry, "Top", idx)
    for idx, entry in enumerate(candidates[:bucket_size], start=1):
        add(entry, "Worst", idx)
    for idx, entry in enumerate(candidates[median_start:median_start + bucket_size], start=1):
        add(entry, "Mid", idx)

    return [
        {
            "run_id": info["row"]["run_id"],
            "analysis_value": info["row"].get(metric_key),
            "labels": info["labels"],
        }
        for info in sorted(selected.values(), key=lambda item: item["row"]["run_id"])
    ]


def build_notes_map(rows):
    notes_by_run = {}
    metric_specs = [
        ("kendall_analysis", "kendall"),
        ("spearman_analysis", "spearman"),
    ]
    for metric_key, metric_label in metric_specs:
        for selected in metric_selection_manifest_rows(rows, metric_key):
            notes = notes_by_run.setdefault(selected["run_id"], [])
            for label in selected["labels"]:
                notes.append(f"{label} {metric_label.capitalize()}")
    return notes_by_run


def compact_row_from_iteration(bundle_summary, iteration_summary, note_text=""):
    parameter_levels = bundle_summary.get("parameter_levels", {})
    analysis_metrics = iteration_summary.get("comparison_metrics", {}).get("analysis", {})
    simulation_metrics = iteration_summary.get("comparison_metrics", {}).get("simulation", {})

    return {
        "run_id": iteration_summary.get("run_name"),
        "infectiousness_level": to_italian_level(parameter_levels.get("infectiousness")),
        "healing_level": to_italian_level(parameter_levels.get("healing")),
        "symptoms_level": to_italian_level(parameter_levels.get("symptoms")),
        "isolating_level": to_italian_level(parameter_levels.get("isolating")),
        "symptoms_onset_level": to_italian_level(parameter_levels.get("symptomsOnset")),
        "notification_to_isolation_level": to_italian_level(parameter_levels.get("notificationToIsolation")),
        "symptomatic_period_level": to_italian_level(parameter_levels.get("symptomaticPeriod")),
        "java_runtime_seconds": iteration_summary.get("analysis", {}).get("analysis_runtime_seconds"),
        "simulation_runtime_seconds": iteration_summary.get("baseline", {}).get("actual_runtime_seconds"),
        "simulation_runs": iteration_summary.get("baseline", {}).get("iterations"),
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
        "brier_score_analysis": iteration_summary.get("precision_metrics", {}).get("prediction_mean_brier"),
        "brier_score_simulation": iteration_summary.get("precision_metrics", {}).get("baseline_mean_brier"),
        "ece_analysis": iteration_summary.get("precision_metrics", {}).get("prediction_mean_ece"),
        "ece_simulation": iteration_summary.get("precision_metrics", {}).get("baseline_mean_ece"),
        "note": note_text,
    }


def build_compact_rows(bundle_summaries):
    rows = []
    for bundle_summary in bundle_summaries:
        for iteration_summary in bundle_summary.get("iterations", []):
            rows.append(compact_row_from_iteration(bundle_summary, iteration_summary))

    notes_by_run = build_notes_map(rows)
    for row in rows:
        notes = notes_by_run.get(row["run_id"], [])
        row["note"] = " - ".join(notes)
    return rows


def create_iteration_metric_artifacts(bundle_dir, iteration_records, compact_rows):
    summary_csv_path = os.path.join(bundle_dir, "iteration_sweep_summary.csv")
    rows = [
        row for row in compact_rows
        if row["run_id"] in {record.get("run_name") for record in iteration_records}
    ]
    notes_by_run = build_notes_map(rows)
    for row in rows:
        notes = notes_by_run.get(row["run_id"], [])
        row["note"] = " - ".join(notes)
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPACT_RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "summary_csv_path": summary_csv_path,
    }


def run_parameter_bundle_iteration_sweep(
    repo_root,
    output_root,
    time_limit_days,
    n_subjects,
    total_internal_contacts,
    baseline_seed,
    time_step_hours,
    baseline_runtime_multiplier,
    observed_test_ablation,
    include_moving_avg_metrics,
    java_iterations_values,
    parameter_bundle,
    ground_truth_parameter_bundle,
    shared_ground_truth,
    save_plots,
):
    if any(iteration > n_subjects for iteration in java_iterations_values):
        raise ValueError(
            f"Java iterations must be <= n_subjects ({n_subjects}). "
            f"Received: {java_iterations_values}"
        )

    run_name = sp.build_run_name(
        time_limit_days,
        n_subjects,
        total_internal_contacts,
        parameter_bundle,
    )
    bundle_dir = sp.ensure_dir(os.path.join(output_root, run_name))
    fine_grained = time_step_hours < 1.0
    parameter_bundle_path = os.path.join(bundle_dir, "parameter_bundle.json")
    sp.write_json(parameter_bundle_path, parameter_bundle)

    bundle_summary = {
        "run_name": run_name,
        "run_dir": bundle_dir,
        "status": "running",
        "time_limit": time_limit_days,
        "n_subjects": n_subjects,
        "total_internal_contacts": total_internal_contacts,
        "seed": shared_ground_truth["seed"],
        "baseline_seed": baseline_seed,
        "time_step_hours": time_step_hours,
        "baseline_runtime_multiplier": baseline_runtime_multiplier,
        "dataset_source": shared_ground_truth.get("dataset_source", sp.DEFAULT_DATASET_SOURCE),
        "dataset_generation_method": shared_ground_truth.get("dataset_generation_method"),
        "tests_enabled": shared_ground_truth.get("tests_enabled", True),
        "dataset_test_events": shared_ground_truth.get("dataset_test_events"),
        "observed_test_ablation": observed_test_ablation,
        "shared_ground_truth_run_dir": shared_ground_truth["run_dir"],
        "parameter_case_id": parameter_bundle["case_id"],
        "parameter_levels": parameter_bundle["levels"],
        "parameter_unit_measure": parameter_bundle["unit_measure"],
        "parameter_bundle_path": parameter_bundle_path,
        "ground_truth_parameter_case_id": ground_truth_parameter_bundle["case_id"],
        "ground_truth_parameter_levels": ground_truth_parameter_bundle["levels"],
        "ground_truth_parameter_bundle_path": shared_ground_truth["ground_truth_parameter_bundle_path"],
        "java_iterations": list(java_iterations_values),
        "iterations": [],
    }
    bundle_summary_path = os.path.join(bundle_dir, "bundle_summary.json")
    sp.write_json(bundle_summary_path, bundle_summary)

    for java_iterations in java_iterations_values:
        iteration_dir = sp.ensure_dir(os.path.join(bundle_dir, iteration_run_dir_name(java_iterations)))
        iteration_run_name = f"{run_name}__jit_{java_iterations}"
        run_inputs = sp.prepare_run_inputs_from_shared_ground_truth(
            iteration_dir,
            iteration_run_name,
            shared_ground_truth,
            observed_test_ablation=observed_test_ablation,
        )
        dataset_path = run_inputs["dataset_path"]
        iteration_summary = {
            "run_name": iteration_run_name,
            "run_dir": iteration_dir,
            "status": "running",
            "java_iterations": java_iterations,
            "time_limit": time_limit_days,
            "n_subjects": n_subjects,
            "total_internal_contacts": total_internal_contacts,
            "baseline_seed": baseline_seed,
            "time_step_hours": time_step_hours,
            "baseline_runtime_multiplier": baseline_runtime_multiplier,
            "dataset_source": shared_ground_truth.get("dataset_source", sp.DEFAULT_DATASET_SOURCE),
            "dataset_generation_method": shared_ground_truth.get("dataset_generation_method"),
            "tests_enabled": shared_ground_truth.get("tests_enabled", True),
            "dataset_test_events": shared_ground_truth.get("dataset_test_events"),
            "observed_test_ablation": run_inputs["observed_test_ablation"],
            "observed_test_ablation_stats": run_inputs["observed_test_ablation_stats"],
            "parameter_case_id": parameter_bundle["case_id"],
            "parameter_levels": parameter_bundle["levels"],
            "parameter_bundle_path": parameter_bundle_path,
            "ground_truth_parameter_case_id": ground_truth_parameter_bundle["case_id"],
            "ground_truth_parameter_bundle_path": shared_ground_truth["ground_truth_parameter_bundle_path"],
            "dataset_path": dataset_path,
            "shared_dataset_path": run_inputs["shared_dataset_path"],
            "shared_observed_simulated_path": run_inputs["shared_observed_simulated_path"],
            "convergence": {
                "ground_truth_path": shared_ground_truth["convergence"]["ground_truth_path"],
                "granularity": shared_ground_truth["convergence"].get("granularity", 1.0),
            },
        }
        sp.write_json(os.path.join(iteration_dir, "run_summary.json"), iteration_summary)

        try:
            java_result = run_java_analysis_with_iterations(
                repo_root=repo_root,
                run_dir=iteration_dir,
                java_iterations=java_iterations,
                time_step_hours=time_step_hours,
                parameter_bundle_path=parameter_bundle_path,
                parameter_bundle=parameter_bundle,
            )
            analysis_path = sp.find_generated_file(iteration_dir, f"_tracks_it{java_iterations}.json")
            baseline_runtime_budget_seconds = max(
                baseline_runtime_multiplier * java_result["analysis_runtime_seconds"],
                0.0,
            )
            baseline_result = run_dataset_simulations(
                dataset_path=dataset_path,
                rep=sp.BASELINE_ITERATIONS_CAP,
                run_until_convergence=False,
                fine_grained=fine_grained,
                time_step_hours=time_step_hours,
                dataset_label=iteration_run_name,
                seed=baseline_seed + 1,
                prune_after_positive_test=False,
                export_observed_simulation=False,
                pruning_seed=baseline_seed,
                max_runtime_seconds=baseline_runtime_budget_seconds,
                parameter_bundle=parameter_bundle,
                save_plots=False,
            )
            analysis_curve_plots = sp.create_analysis_subject_curve_plots(
                run_dir=iteration_dir,
                run_name=iteration_run_name,
                analysis_path=analysis_path,
                ground_truth_path=shared_ground_truth["convergence"]["ground_truth_path"],
                baseline_path=baseline_result["averaged_results_path"],
                granularity=shared_ground_truth["convergence"].get("granularity", 1.0),
                save_plots=save_plots,
            )

            precision_dir = sp.ensure_dir(os.path.join(iteration_dir, "precision_metrics"))
            prediction_metrics_path = os.path.join(precision_dir, "metrics_prediction.json")
            baseline_metrics_path = os.path.join(precision_dir, "metrics_baseline.json")
            metrics_stdout = io.StringIO()
            t0_metrics = time.time()
            with contextlib.redirect_stdout(metrics_stdout):
                process_and_save(
                    analysis_path,
                    shared_ground_truth["convergence"]["ground_truth_path"],
                    M=10,
                    metrics_output=prediction_metrics_path,
                    plots_dir=os.path.join(precision_dir, "numericalAnalysis", "plots"),
                    save_plots=save_plots,
                )
                process_and_save(
                    baseline_result["averaged_results_path"],
                    shared_ground_truth["convergence"]["ground_truth_path"],
                    M=10,
                    metrics_output=baseline_metrics_path,
                    plots_dir=os.path.join(precision_dir, "simulatedBaseline", "plots"),
                    save_plots=save_plots,
                )
            with open(os.path.join(precision_dir, "metrics_stdout.log"), "w", encoding="utf-8") as handle:
                handle.write(metrics_stdout.getvalue())

            comparison_summary = sp.create_analysis_vs_simulation_plots(
                iteration_dir,
                iteration_run_name,
                shared_ground_truth["convergence"]["ground_truth_path"],
                analysis_path,
                baseline_result["averaged_results_path"],
                include_moving_avg_metrics=include_moving_avg_metrics,
                save_plots=save_plots,
            )
            t1_metrics = time.time()
            metrics_runtime_seconds = t1_metrics - t0_metrics

            iteration_summary["analysis"] = {
                "analysis_path": analysis_path,
                "java_iterations": java_iterations,
                "stpn_solution_path": java_result["stpn_solution_path"],
                "analysis_runtime_seconds": java_result["analysis_runtime_seconds"],
                "analysis_wall_runtime_seconds": java_result["analysis_wall_runtime_seconds"],
                "analysis_runtime_excludes_overhead": java_result["analysis_runtime_excludes_overhead"],
                "stpn_precomputation_runtime_seconds": java_result["stpn_precomputation_runtime_seconds"],
                "stpn_precomputation_performed": java_result["stpn_precomputation_performed"],
                "java_precompute_cache_root": java_result["java_precompute_cache_root"],
                "java_precompute_cache_dir": java_result["java_precompute_cache_dir"],
                "java_precompute_cache_key": java_result["java_precompute_cache_key"],
                "java_precompute_cache_manifest_path": java_result["java_precompute_cache_manifest_path"],
                "java_precompute_cache_hit": java_result["java_precompute_cache_hit"],
                "java_precompute_seed_run_dir": java_result["java_precompute_seed_run_dir"],
                "observation_curve_cache_path": java_result["observation_curve_cache_path"],
                "local_observation_curve_path": java_result["local_observation_curve_path"],
                "java_analysis_stdout_log_path": java_result["java_analysis_stdout_log_path"],
                "java_analysis_stderr_log_path": java_result["java_analysis_stderr_log_path"],
                "java_precompute_stdout_log_path": java_result["java_precompute_stdout_log_path"],
                "java_precompute_stderr_log_path": java_result["java_precompute_stderr_log_path"],
                "subject_curve_plots": analysis_curve_plots,
            }
            iteration_summary["baseline"] = {
                "iterations": baseline_result["rep_done"],
                "baseline_path": baseline_result["averaged_results_path"],
                "dkw_csv_path": baseline_result["dkw_csv_path"],
                "effective_dataset_path": baseline_result["effective_dataset_path"],
                "pruned_dataset_path": baseline_result["pruned_dataset_path"],
                "time_step_hours": baseline_result.get("time_step_hours", time_step_hours),
                "runtime_multiplier": baseline_runtime_multiplier,
                "runtime_budget_seconds": baseline_runtime_budget_seconds,
                "actual_runtime_seconds": baseline_result.get("actual_runtime_seconds"),
                "suppressed_stdout_log_path": baseline_result.get("suppressed_stdout_log_path"),
            }
            iteration_summary["precision_metrics"] = {
                "prediction_metrics_path": prediction_metrics_path,
                "baseline_metrics_path": baseline_metrics_path,
                "prediction_mean_brier": sp.mean_metric_value(prediction_metrics_path, "Brier Score"),
                "baseline_mean_brier": sp.mean_metric_value(baseline_metrics_path, "Brier Score"),
                "prediction_mean_ece": sp.mean_metric_value(prediction_metrics_path, "ECE"),
                "baseline_mean_ece": sp.mean_metric_value(baseline_metrics_path, "ECE"),
                "metrics_runtime_seconds": metrics_runtime_seconds,
            }
            iteration_summary["comparison_metrics"] = comparison_summary
            iteration_summary["moving_avg_metrics_enabled"] = include_moving_avg_metrics
            iteration_summary["status"] = "completed"
            iteration_summary["completed_at"] = datetime.now().isoformat(timespec="seconds")
        except Exception as exc:
            iteration_summary["status"] = "failed"
            iteration_summary["error"] = str(exc)
            iteration_summary["traceback"] = traceback.format_exc()

        sp.write_json(os.path.join(iteration_dir, "run_summary.json"), iteration_summary)
        bundle_summary["iterations"].append(iteration_summary)
        sp.write_json(bundle_summary_path, bundle_summary)

    bundle_summary["aggregate_outputs"] = create_iteration_metric_artifacts(
        bundle_dir=bundle_dir,
        iteration_records=bundle_summary["iterations"],
        compact_rows=[
            compact_row_from_iteration(bundle_summary, iteration_summary)
            for iteration_summary in bundle_summary["iterations"]
        ],
    )
    completed_count = sum(1 for record in bundle_summary["iterations"] if record.get("status") == "completed")
    if completed_count == len(bundle_summary["iterations"]):
        bundle_summary["status"] = "completed"
    elif completed_count == 0:
        bundle_summary["status"] = "failed"
    else:
        bundle_summary["status"] = "partial-failure"
    bundle_summary["completed_at"] = datetime.now().isoformat(timespec="seconds")
    sp.write_json(bundle_summary_path, bundle_summary)
    return bundle_summary


def build_failed_bundle_summary(
    output_root,
    time_limit_days,
    n_subjects,
    total_internal_contacts,
    baseline_seed,
    time_step_hours,
    baseline_runtime_multiplier,
    dataset_source,
    observed_test_ablation,
    parameter_bundle,
    ground_truth_parameter_bundle,
    java_iterations_values,
    error_message,
    traceback_text,
):
    run_name = sp.build_run_name(
        time_limit_days,
        n_subjects,
        total_internal_contacts,
        parameter_bundle,
    )
    bundle_dir = sp.ensure_dir(os.path.join(output_root, run_name))
    summary = {
        "run_name": run_name,
        "run_dir": bundle_dir,
        "status": "failed",
        "time_limit": time_limit_days,
        "n_subjects": n_subjects,
        "total_internal_contacts": total_internal_contacts,
        "baseline_seed": baseline_seed,
        "time_step_hours": time_step_hours,
        "baseline_runtime_multiplier": baseline_runtime_multiplier,
        "dataset_source": dataset_source,
        "observed_test_ablation": observed_test_ablation,
        "parameter_case_id": parameter_bundle["case_id"],
        "parameter_levels": parameter_bundle["levels"],
        "ground_truth_parameter_case_id": ground_truth_parameter_bundle["case_id"],
        "java_iterations": list(java_iterations_values),
        "error": error_message,
        "traceback": traceback_text,
        "iterations": [
            {
                "java_iterations": java_iterations,
                "status": "failed",
                "error": error_message,
                "traceback": traceback_text,
            }
            for java_iterations in java_iterations_values
        ],
    }
    sp.write_json(os.path.join(bundle_dir, "bundle_summary.json"), summary)
    return summary


def write_aggregate_outputs(output_root, bundle_summaries):
    compact_rows = build_compact_rows(bundle_summaries)

    aggregate_json_path = os.path.join(output_root, "sweep_summary.json")
    sp.write_json(aggregate_json_path, compact_rows)

    csv_path = os.path.join(output_root, "sweep_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPACT_RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(compact_rows)

    return compact_rows


def select_rows_for_metric_plots(rows, metric_key):
    candidates = [
        row for row in rows
        if sp.finite_float(row.get(metric_key)) is not None
    ]
    candidates.sort(key=lambda row: (float(row[metric_key]), row["run_id"]))
    if not candidates:
        return []

    bucket_size = min(METRIC_SELECTION_BUCKET_SIZE, len(candidates))
    median_center = len(candidates) // 2
    median_start = max(
        0,
        min(len(candidates) - bucket_size, median_center - bucket_size // 2),
    )

    selected = {}
    for row in candidates[:bucket_size]:
        selected[row["run_id"]] = row
    for row in candidates[-bucket_size:]:
        selected[row["run_id"]] = row
    for row in candidates[median_start:median_start + bucket_size]:
        selected[row["run_id"]] = row
    return sorted(selected.values(), key=lambda row: (float(row[metric_key]), row["run_id"]))


def create_metric_selection_plot(output_root, rows, metric_label, metric_analysis_key, metric_simulation_key):
    selected_rows = select_rows_for_metric_plots(rows, metric_analysis_key)
    if not selected_rows:
        return None

    base_name = metric_label.lower()
    csv_path = os.path.join(output_root, f"selected_30_{base_name}.csv")
    sp.save_series_csv(
        csv_path,
        ["run_id", f"{base_name}_analysis", f"{base_name}_simulation", "note"],
        [
            [
                row["run_id"],
                row.get(metric_analysis_key),
                row.get(metric_simulation_key),
                row.get("note", ""),
            ]
            for row in selected_rows
        ],
    )

    figure_path = os.path.join(output_root, f"selected_30_{base_name}.png")
    x_values = list(range(1, len(selected_rows) + 1))
    analysis_values = [row.get(metric_analysis_key) for row in selected_rows]
    simulation_values = [row.get(metric_simulation_key) for row in selected_rows]

    fig, axis = plt.subplots(figsize=(14, 5.2))
    axis.plot(x_values, analysis_values, marker="o", linewidth=1.8, color="tab:orange", label="Java analysis")
    axis.plot(x_values, simulation_values, marker="x", linewidth=1.8, color="tab:green", label="Simulation")
    axis.set_title(f"Selected 30 Runs by {metric_label}")
    axis.set_xlabel("Selected run index")
    axis.set_ylabel(metric_label)
    axis.grid(True, alpha=0.3)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=sp.COMPACT_PLOT_DPI)
    plt.close(fig)
    return {
        "metric": metric_label,
        "csv_path": csv_path,
        "plot_path": figure_path,
    }


def select_iteration_rows_for_reduced_plots(rows):
    selected_by_run = {}
    metric_specs = [
        ("kendall_analysis", "kendall_analysis"),
        ("spearman_analysis", "spearman_analysis"),
    ]

    def add_selected(row, reason):
        run_id = row["run_id"]
        info = selected_by_run.setdefault(
            run_id,
            {
                "row": row,
                "reasons": [],
            },
        )
        if reason not in info["reasons"]:
            info["reasons"].append(reason)

    for metric_key, metric_label in metric_specs:
        candidates = [
            row for row in rows
            if sp.finite_float(row.get(metric_key)) is not None
        ]
        candidates.sort(key=lambda row: (float(row[metric_key]), row["run_id"]))
        if not candidates:
            continue

        bucket_size = min(METRIC_SELECTION_BUCKET_SIZE, len(candidates))
        median_center = len(candidates) // 2
        median_start = max(
            0,
            min(len(candidates) - bucket_size, median_center - bucket_size // 2),
        )

        for row in candidates[:bucket_size]:
            add_selected(row, f"worst_10_{metric_label}")
        for row in candidates[-bucket_size:]:
            add_selected(row, f"best_10_{metric_label}")
        for row in candidates[median_start:median_start + bucket_size]:
            add_selected(row, f"median_10_{metric_label}")

    for row in rows:
        if (
            row.get("infectiousness_level") == to_italian_level(sp.GROUND_TRUTH_PARAMETER_LEVEL)
            and row.get("healing_level") == to_italian_level(sp.GROUND_TRUTH_PARAMETER_LEVEL)
            and row.get("symptoms_level") == to_italian_level(sp.GROUND_TRUTH_PARAMETER_LEVEL)
            and row.get("isolating_level") == to_italian_level(sp.GROUND_TRUTH_PARAMETER_LEVEL)
            and row.get("symptoms_onset_level") == to_italian_level(sp.GROUND_TRUTH_PARAMETER_LEVEL)
            and row.get("notification_to_isolation_level") == to_italian_level(sp.GROUND_TRUTH_PARAMETER_LEVEL)
            and row.get("symptomatic_period_level") == to_italian_level(sp.GROUND_TRUTH_PARAMETER_LEVEL)
        ):
            add_selected(row, "mid_parameter_case")

    manifest_rows = [
        {
            "run_id": info["row"]["run_id"],
            "kendall_analysis": info["row"].get("kendall_analysis"),
            "spearman_analysis": info["row"].get("spearman_analysis"),
            "selection_reasons": "|".join(info["reasons"]),
        }
        for info in sorted(selected_by_run.values(), key=lambda item: item["row"]["run_id"])
    ]
    selected_run_ids = {row["run_id"] for row in manifest_rows}
    return selected_run_ids, manifest_rows


def write_reduced_plot_selection(output_root, manifest_rows):
    selection_json_path = os.path.join(output_root, "reduced_plot_selection.json")
    selection_csv_path = os.path.join(output_root, "reduced_plot_selection.csv")
    sp.write_json(selection_json_path, manifest_rows)
    with open(selection_csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(manifest_rows[0].keys()) if manifest_rows else ["run_id"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    return {
        "json_path": selection_json_path,
        "csv_path": selection_csv_path,
    }


def regenerate_iteration_run_plots(iteration_summary):
    if iteration_summary.get("status") != "completed":
        return iteration_summary

    run_dir = iteration_summary["run_dir"]
    run_name = iteration_summary["run_name"]
    ground_truth_path = iteration_summary["convergence"]["ground_truth_path"]
    analysis_path = iteration_summary["analysis"]["analysis_path"]
    baseline_path = iteration_summary["baseline"]["baseline_path"]
    granularity = iteration_summary["convergence"].get("granularity", 1.0)
    include_moving_avg_metrics = iteration_summary.get("moving_avg_metrics_enabled", True)

    analysis_curve_plots = sp.create_analysis_subject_curve_plots(
        run_dir=run_dir,
        run_name=run_name,
        analysis_path=analysis_path,
        ground_truth_path=ground_truth_path,
        baseline_path=baseline_path,
        granularity=granularity,
        save_plots=True,
    )
    iteration_summary["analysis"]["subject_curve_plots"] = analysis_curve_plots

    precision_dir = sp.ensure_dir(os.path.join(run_dir, "precision_metrics"))
    prediction_metrics_path = iteration_summary["precision_metrics"]["prediction_metrics_path"]
    baseline_metrics_path = iteration_summary["precision_metrics"]["baseline_metrics_path"]
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
    with open(os.path.join(precision_dir, "metrics_stdout.log"), "w", encoding="utf-8") as handle:
        handle.write(metrics_stdout.getvalue())

    iteration_summary["comparison_metrics"] = sp.create_analysis_vs_simulation_plots(
        run_dir,
        run_name,
        ground_truth_path,
        analysis_path,
        baseline_path,
        include_moving_avg_metrics=include_moving_avg_metrics,
        save_plots=True,
    )
    sp.write_json(os.path.join(run_dir, "run_summary.json"), iteration_summary)
    return iteration_summary


def main():
    parser = argparse.ArgumentParser(description="Run a CHITA sweep over Java analysis iterations.")
    parser.add_argument(
        "--output-root",
        default=os.path.join("sweeps", datetime.now().strftime("n_iteration_sweep_%Y%m%d_%H%M%S")),
        help="Directory where sweep outputs will be created.",
    )
    parser.add_argument("--seed-base", type=int, default=1000, help="Base seed used to derive per-run seeds.")
    parser.add_argument(
        "--time-step-hours",
        type=float,
        default=sp.TIME_STEP_HOURS,
        help="Shared time step in hours for both Python simulation and Java analysis.",
    )
    parser.add_argument(
        "--baseline-runtime-multiplier",
        type=float,
        default=sp.BASELINE_RUNTIME_MULTIPLIER,
        help="Multiplier applied to the measured Java runtime to set the Python baseline budget.",
    )
    parser.add_argument(
        "--parameter-ods-path",
        default=sp.DEFAULT_PARAMETER_ODS_PATH,
        help="ODS spreadsheet containing the lower/mid/upper transition parameters.",
    )
    parser.add_argument(
        "--parameter-case-mode",
        choices=PARAMETER_CASE_MODE_CHOICES,
        default=PARAMETER_CASE_MODE_MID_ONLY,
        help=(
            "How to choose parameter bundles. 'mid-only' keeps every family at mid. "
            "'focused-grid' sweeps only the selected focus families while keeping the rest at mid."
        ),
    )
    parser.add_argument(
        "--focus-families",
        nargs="*",
        default=list(DEFAULT_FOCUSED_FAMILIES),
        help=(
            "Families varied in focused-grid mode. Defaults to the three strongest "
            "latent-dynamics drivers: transmission, recovery, and isolation."
        ),
    )
    parser.add_argument(
        "--java-iterations",
        nargs="+",
        type=int,
        default=list(DEFAULT_JAVA_ITERATIONS),
        help="Java iteration counts to test. Default: 1 2 3 4 5 6 7 8",
    )
    parser.add_argument(
        "--d2",
        action="store_true",
        help="Generate the raw source dataset with the D2-style recipe instead of the classic graph generator.",
    )
    parser.add_argument(
        "--disable-tests",
        action="store_true",
        help="Remove all raw Test events from the generated dataset before running the sweep.",
    )
    parser.add_argument(
        "--observed-test-ablation",
        choices=sp.OBSERVED_TEST_ABLATION_CHOICES,
        default=sp.OBSERVED_TEST_ABLATION_NONE,
        help="Optional filtering applied only to the shared observed_simulated.json copied into each run.",
    )
    parser.add_argument(
        "--skip-plot-images",
        action="store_true",
        help="Skip PNG plot generation and keep only CSV/JSON outputs for a faster sweep.",
    )
    parser.add_argument(
        "--disable-moving-avg-metrics",
        action="store_true",
        help="Skip computing moving-average summary metrics in analysis outputs.",
    )
    parser.add_argument(
        "--reduce-plots",
        action="store_true",
        help=(
            "Defer plot generation and only render plots for the 10 worst cases, "
            "10 best cases, the mid-parameter case, and 10 cases around the median "
            "based on mean Java Spearman correlation across tested iteration counts."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of parameter-bundle workers to run in parallel.",
    )
    parser.add_argument(
        "--parallel-backend",
        choices=PARALLEL_BACKEND_CHOICES,
        default=PARALLEL_BACKEND_AUTO,
        help=(
            "Parallel execution backend. 'auto' uses threads on Windows for stability "
            "and processes elsewhere."
        ),
    )
    args = parser.parse_args()

    if args.time_step_hours <= 0:
        raise ValueError("--time-step-hours must be greater than 0.")
    if args.baseline_runtime_multiplier < 0:
        raise ValueError("--baseline-runtime-multiplier must be greater than or equal to 0.")

    repo_root = os.path.abspath(os.path.dirname(__file__))
    output_root = sp.ensure_dir(os.path.abspath(args.output_root))
    save_plots_during_run = not args.skip_plot_images
    include_moving_avg_metrics = not args.disable_moving_avg_metrics
    dataset_source = sp.DATASET_SOURCE_D2 if args.d2 else sp.DATASET_SOURCE_GENERATED
    parameter_space = sp.load_parameter_space_from_ods(os.path.abspath(args.parameter_ods_path))
    ground_truth_parameter_bundle = sp.resolve_uniform_parameter_bundle(
        parameter_space,
        sp.GROUND_TRUTH_PARAMETER_LEVEL,
    )
    java_iterations_values = parse_java_iterations(args.java_iterations)
    focused_families = normalize_focus_families(args.focus_families)
    parameter_cases = enumerate_iteration_parameter_cases(
        parameter_space,
        mode=args.parameter_case_mode,
        focused_families=focused_families,
    )
    parameter_manifest_paths = sp.write_parameter_case_manifest(output_root, parameter_cases)
    write_iteration_sweep_markdown(
        output_root=output_root,
        seed_base=args.seed_base,
        time_step_hours=args.time_step_hours,
        baseline_runtime_multiplier=args.baseline_runtime_multiplier,
        include_moving_avg_metrics=include_moving_avg_metrics,
        dataset_source=dataset_source,
        tests_enabled=not args.disable_tests,
        observed_test_ablation=args.observed_test_ablation,
        parameter_space=parameter_space,
        ground_truth_parameter_bundle=ground_truth_parameter_bundle,
        parameter_case_mode=args.parameter_case_mode,
        focused_families=focused_families,
        parameter_case_count=len(parameter_cases),
        parameter_manifest_paths=parameter_manifest_paths,
        java_iterations=java_iterations_values,
        generate_plot_images=not args.skip_plot_images,
        reduce_plots=args.reduce_plots,
    )

    summaries = []
    dataset_combinations = list(itertools.product(
        sp.TIME_LIMITS,
        sp.N_SUBJECTS,
        sp.TOTAL_INTERNAL_CONTACTS,
    ))
    parameter_case_count = len(parameter_cases)
    worker_count = sp.resolve_worker_count(args.max_workers, parameter_case_count)
    parallel_backend = resolve_parallel_backend(args.parallel_backend)
    total_progress_steps = len(dataset_combinations) + (len(dataset_combinations) * parameter_case_count)

    with tqdm(total=total_progress_steps, desc="Iteration sweep progress", unit="step") as progress_bar:
        for dataset_index, (time_limit_days, n_subjects, total_internal_contacts) in enumerate(dataset_combinations):
            if any(iteration > n_subjects for iteration in java_iterations_values):
                raise ValueError(
                    f"--java-iterations contains a value larger than n_subjects={n_subjects}: "
                    f"{java_iterations_values}"
                )

            dataset_label = f"t{time_limit_days}/s{n_subjects}/c{total_internal_contacts}"
            shared_ground_truth_seed = args.seed_base + dataset_index
            progress_bar.set_postfix_str(f"{dataset_label} | shared ground truth")
            try:
                shared_ground_truth = sp.compute_shared_ground_truth(
                    repo_root=repo_root,
                    output_root=output_root,
                    time_limit_days=time_limit_days,
                    n_subjects=n_subjects,
                    total_internal_contacts=total_internal_contacts,
                    seed=shared_ground_truth_seed,
                    time_step_hours=args.time_step_hours,
                    dataset_source=dataset_source,
                    ground_truth_parameter_bundle=ground_truth_parameter_bundle,
                    include_moving_avg_metrics=include_moving_avg_metrics,
                    save_plots=save_plots_during_run,
                    disable_tests=args.disable_tests,
                )
            except Exception as exc:
                shared_traceback = traceback.format_exc()
                progress_bar.update(1)
                for parameter_index, parameter_bundle in enumerate(parameter_cases):
                    baseline_seed = (
                        args.seed_base
                        + dataset_index * parameter_case_count
                        + parameter_index
                        + 1
                    )
                    summary = build_failed_bundle_summary(
                        output_root=output_root,
                        time_limit_days=time_limit_days,
                        n_subjects=n_subjects,
                        total_internal_contacts=total_internal_contacts,
                        baseline_seed=baseline_seed,
                        time_step_hours=args.time_step_hours,
                        baseline_runtime_multiplier=args.baseline_runtime_multiplier,
                        dataset_source=dataset_source,
                        observed_test_ablation=args.observed_test_ablation,
                        parameter_bundle=parameter_bundle,
                        ground_truth_parameter_bundle=ground_truth_parameter_bundle,
                        java_iterations_values=java_iterations_values,
                        error_message=f"Shared ground truth failed: {exc}",
                        traceback_text=shared_traceback,
                    )
                    summaries.append(summary)
                    write_aggregate_outputs(output_root, summaries)
                    progress_bar.set_postfix_str(
                        f"{dataset_label} | {parameter_bundle['case_id']} failed (shared GT)"
                    )
                    progress_bar.update(1)
                continue

            progress_bar.update(1)
            task_specs = []
            for parameter_index, parameter_bundle in enumerate(parameter_cases):
                baseline_seed = (
                    args.seed_base
                    + dataset_index * parameter_case_count
                    + parameter_index
                    + 1
                )
                task_specs.append(
                    {
                        "parameter_bundle": parameter_bundle,
                        "baseline_seed": baseline_seed,
                    }
                )

            completed_cases = 0
            executor = None
            if worker_count == 1:
                task_iterable = [(task_spec, None) for task_spec in task_specs]
            else:
                if parallel_backend == PARALLEL_BACKEND_PROCESS:
                    try:
                        executor = ProcessPoolExecutor(max_workers=worker_count)
                    except PermissionError:
                        executor = ThreadPoolExecutor(max_workers=worker_count)
                else:
                    executor = ThreadPoolExecutor(max_workers=worker_count)
                future_to_task = {
                    executor.submit(
                        run_parameter_bundle_iteration_sweep,
                        repo_root=repo_root,
                        output_root=output_root,
                        time_limit_days=time_limit_days,
                        n_subjects=n_subjects,
                        total_internal_contacts=total_internal_contacts,
                        baseline_seed=task_spec["baseline_seed"],
                        time_step_hours=args.time_step_hours,
                        baseline_runtime_multiplier=args.baseline_runtime_multiplier,
                        observed_test_ablation=args.observed_test_ablation,
                        include_moving_avg_metrics=include_moving_avg_metrics,
                        java_iterations_values=java_iterations_values,
                        parameter_bundle=task_spec["parameter_bundle"],
                        ground_truth_parameter_bundle=ground_truth_parameter_bundle,
                        shared_ground_truth=shared_ground_truth,
                        save_plots=save_plots_during_run,
                    ): task_spec
                    for task_spec in task_specs
                }
                task_iterable = ((future_to_task[future], future) for future in as_completed(future_to_task))

            try:
                for task_spec, future in task_iterable:
                    parameter_bundle = task_spec["parameter_bundle"]
                    progress_bar.set_postfix_str(
                        f"{dataset_label} | {completed_cases}/{parameter_case_count} bundles done"
                    )
                    try:
                        if future is None:
                            summary = run_parameter_bundle_iteration_sweep(
                                repo_root=repo_root,
                                output_root=output_root,
                                time_limit_days=time_limit_days,
                                n_subjects=n_subjects,
                                total_internal_contacts=total_internal_contacts,
                                baseline_seed=task_spec["baseline_seed"],
                                time_step_hours=args.time_step_hours,
                                baseline_runtime_multiplier=args.baseline_runtime_multiplier,
                                observed_test_ablation=args.observed_test_ablation,
                                include_moving_avg_metrics=include_moving_avg_metrics,
                                java_iterations_values=java_iterations_values,
                                parameter_bundle=parameter_bundle,
                                ground_truth_parameter_bundle=ground_truth_parameter_bundle,
                                shared_ground_truth=shared_ground_truth,
                                save_plots=save_plots_during_run,
                            )
                        else:
                            summary = future.result()
                    except Exception as exc:
                        summary = build_failed_bundle_summary(
                            output_root=output_root,
                            time_limit_days=time_limit_days,
                            n_subjects=n_subjects,
                            total_internal_contacts=total_internal_contacts,
                            baseline_seed=task_spec["baseline_seed"],
                            time_step_hours=args.time_step_hours,
                            baseline_runtime_multiplier=args.baseline_runtime_multiplier,
                            dataset_source=dataset_source,
                            observed_test_ablation=args.observed_test_ablation,
                            parameter_bundle=parameter_bundle,
                            ground_truth_parameter_bundle=ground_truth_parameter_bundle,
                            java_iterations_values=java_iterations_values,
                            error_message=str(exc),
                            traceback_text=traceback.format_exc(),
                        )

                    summaries.append(summary)
                    completed_cases += 1
                    write_aggregate_outputs(output_root, summaries)
                    progress_bar.set_postfix_str(
                        f"{dataset_label} | {parameter_bundle['case_id']} {summary['status']}"
                    )
                    progress_bar.update(1)
            finally:
                if executor is not None:
                    executor.shutdown(wait=True)

    compact_rows = write_aggregate_outputs(output_root, summaries)
    if args.reduce_plots:
        selected_run_ids, manifest_rows = select_iteration_rows_for_reduced_plots(compact_rows)
        write_reduced_plot_selection(output_root, manifest_rows)
        if not args.skip_plot_images:
            for bundle_summary in summaries:
                changed = False
                for index, iteration_summary in enumerate(bundle_summary.get("iterations", [])):
                    if iteration_summary.get("run_name") not in selected_run_ids:
                        continue
                    bundle_summary["iterations"][index] = regenerate_iteration_run_plots(iteration_summary)
                    changed = True
                if changed:
                    bundle_summary["aggregate_outputs"] = create_iteration_metric_artifacts(
                        bundle_dir=bundle_summary["run_dir"],
                        iteration_records=bundle_summary["iterations"],
                        compact_rows=[
                            compact_row_from_iteration(bundle_summary, iteration_summary)
                            for iteration_summary in bundle_summary["iterations"]
                        ],
                    )
                    sp.write_json(
                        os.path.join(bundle_summary["run_dir"], "bundle_summary.json"),
                        bundle_summary,
                    )
            compact_rows = write_aggregate_outputs(output_root, summaries)

    if not args.skip_plot_images:
        kendall_plot = create_metric_selection_plot(
            output_root=output_root,
            rows=compact_rows,
            metric_label="Kendall",
            metric_analysis_key="kendall_analysis",
            metric_simulation_key="kendall_simulation",
        )
        spearman_plot = create_metric_selection_plot(
            output_root=output_root,
            rows=compact_rows,
            metric_label="Spearman",
            metric_analysis_key="spearman_analysis",
            metric_simulation_key="spearman_simulation",
        )
        selection_manifest = {
            "kendall": kendall_plot,
            "spearman": spearman_plot,
        }
        sp.write_json(os.path.join(output_root, "selected_30_metric_plots.json"), selection_manifest)


if __name__ == "__main__":
    main()
