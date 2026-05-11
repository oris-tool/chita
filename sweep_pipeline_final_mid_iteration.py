import argparse
import contextlib
import csv
import io
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from tqdm import tqdm

import sweep_pipeline as sp
import sweep_pipeline_final as final
import sweep_pipeline_n_iteration as iteration_sweep
from compute_precision_metrics import process_and_save
from run_n_simulations import run_dataset_simulations
from utils import run_stpn_analysis


TIME_STEP = final.TIME_STEP
QUANTILE = final.QUANTILE
DEFAULT_JAVA_ITERATIONS = iteration_sweep.DEFAULT_JAVA_ITERATIONS
RUN_COMPLETION_SENTINEL = final.RUN_COMPLETION_SENTINEL

DATASET_PROFILES = {
    final.DATASET_FAMILY_BUBBLE: final.DatasetProfile(
        family=final.DATASET_FAMILY_BUBBLE,
        generator_module_name="dataset_graph",
        n_subjects=8,
        time_limit_hours=final.TIME_LIMIT_HOURS,
        internal_contacts=(400,),
        effective_external_contacts=3,
    ),
    final.DATASET_FAMILY_SCALE_FREE: final.DatasetProfile(
        family=final.DATASET_FAMILY_SCALE_FREE,
        generator_module_name="scale_free_dataset_graph",
        n_subjects=100,
        time_limit_hours=final.TIME_LIMIT_HOURS,
        internal_contacts=(2500,),
        effective_external_contacts=15,
        total_external_contacts=1000,
        total_symptom_observations=1000,
        total_test_observations=1000,
        barabasi_m=3,
    ),
    final.DATASET_FAMILY_SMALL_WORLD: final.DatasetProfile(
        family=final.DATASET_FAMILY_SMALL_WORLD,
        generator_module_name="small_world_dataset_graph",
        n_subjects=100,
        time_limit_hours=final.TIME_LIMIT_HOURS,
        internal_contacts=(3600,),
        effective_external_contacts=15,
        total_external_contacts=1000,
        total_symptom_observations=1000,
        total_test_observations=1000,
        watts_k=5,
        rewire_probability=0.1,
    ),
}
DATASET_PROFILE_ORDER = (
    final.DATASET_FAMILY_BUBBLE,
    final.DATASET_FAMILY_SCALE_FREE,
    final.DATASET_FAMILY_SMALL_WORLD,
)

ITERATION_SWEEP_COLUMNS = list(final.REFERENCE_SWEEP_COLUMNS)
ITERATION_SWEEP_COLUMNS.insert(
    ITERATION_SWEEP_COLUMNS.index("infectiousness_level"),
    "java_iterations",
)


def dataset_target_aliases(dataset_profile, internal_contacts):
    filename = final.dataset_filename(dataset_profile, internal_contacts)
    stem = os.path.splitext(filename)[0]
    return {
        dataset_profile.family,
        f"{dataset_profile.family}_{internal_contacts}",
        stem,
        filename,
    }


def fixed_dataset_targets(skip_datasets=None):
    targets = [
        (DATASET_PROFILES[family], internal_contacts)
        for family in DATASET_PROFILE_ORDER
        for internal_contacts in DATASET_PROFILES[family].internal_contacts
    ]
    skip_values = {
        str(value).strip()
        for value in (skip_datasets or [])
        if str(value).strip()
    }
    if not skip_values:
        return targets

    known_aliases = set()
    for dataset_profile, internal_contacts in targets:
        known_aliases.update(dataset_target_aliases(dataset_profile, internal_contacts))

    unknown_values = sorted(skip_values - known_aliases)
    if unknown_values:
        raise ValueError(
            "Unsupported --skip-datasets value(s): "
            f"{', '.join(unknown_values)}. "
            f"Expected one or more of: {', '.join(sorted(known_aliases))}"
        )

    filtered_targets = [
        (dataset_profile, internal_contacts)
        for dataset_profile, internal_contacts in targets
        if dataset_target_aliases(dataset_profile, internal_contacts).isdisjoint(skip_values)
    ]
    if not filtered_targets:
        raise ValueError("--skip-datasets removed every dataset target.")
    return filtered_targets


def iteration_label(java_iterations):
    return f"it_{int(java_iterations):02d}"


def effective_java_iterations(java_iterations):
    return max(1, int(java_iterations))


def validate_java_iterations(java_iterations_values, dataset_targets=None):
    targets = dataset_targets if dataset_targets is not None else fixed_dataset_targets()
    max_n_subjects_by_family = {}
    for dataset_profile, _internal_contacts in targets:
        max_n_subjects_by_family[dataset_profile.family] = dataset_profile.n_subjects
    for value in java_iterations_values:
        too_small_families = [
            family
            for family, max_subjects in max_n_subjects_by_family.items()
            if effective_java_iterations(value) > max_subjects
        ]
        if too_small_families:
            raise ValueError(
                f"Java iteration count {value} is larger than n_subjects for: "
                f"{', '.join(sorted(too_small_families))}"
            )


def load_mid_parameter_case(parameter_json_path):
    _parameter_cases, ground_truth_case = final.build_parameter_combinations(parameter_json_path)
    mid_case = dict(ground_truth_case)
    mid_case["case_run_code"] = "mid"
    mid_case["case_index"] = 0
    return mid_case


def gt_results_summary_path(save_path, dataset_stem):
    suffix = dataset_stem[len("dataset_") :] if dataset_stem.startswith("dataset_") else dataset_stem
    return os.path.join(save_path, f"gt_results_{suffix}.json")


def resolve_existing_path(path_value, candidate_roots):
    if not isinstance(path_value, str) or not path_value:
        return None

    candidates = [path_value] if os.path.isabs(path_value) else [os.path.abspath(path_value)]
    if not os.path.isabs(path_value):
        candidates.extend(os.path.join(root, path_value) for root in candidate_roots)

    for candidate in candidates:
        normalized = os.path.abspath(candidate)
        if os.path.exists(normalized):
            return normalized
    return None


def parse_gt_reps_count(dataset_stem, filename):
    prefix = f"{dataset_stem}_simulated_"
    suffix = "_reps.json"
    if not filename.startswith(prefix) or not filename.endswith(suffix):
        return None
    raw_count = filename[len(prefix) : -len(suffix)]
    try:
        return int(raw_count)
    except ValueError:
        return None


def choose_existing_gt_average(ground_truth_dir, dataset_stem):
    if not os.path.isdir(ground_truth_dir):
        return None

    candidates = []
    for filename in os.listdir(ground_truth_dir):
        reps_count = parse_gt_reps_count(dataset_stem, filename)
        if reps_count is not None:
            path = os.path.join(ground_truth_dir, filename)
            if os.path.isfile(path):
                candidates.append((reps_count, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[-1]


def choose_first_existing_file(directory, suffix):
    if not os.path.isdir(directory):
        return None
    matches = [
        os.path.join(directory, filename)
        for filename in os.listdir(directory)
        if filename.endswith(suffix) and os.path.isfile(os.path.join(directory, filename))
    ]
    if not matches:
        return None
    matches.sort(key=os.path.getmtime, reverse=True)
    return os.path.abspath(matches[0])


def normalize_cached_gt_summary(cached_summary, save_path, dataset_stem, summary_path):
    candidate_roots = [
        save_path,
        os.path.dirname(summary_path),
        os.path.join(save_path, "ground_truth", dataset_stem),
    ]
    averaged_results_path = resolve_existing_path(
        cached_summary.get("averaged_results_path"),
        candidate_roots,
    )
    if averaged_results_path is None:
        return None

    normalized = dict(cached_summary)
    normalized["averaged_results_path"] = averaged_results_path
    for key in (
        "observed_simulated_path",
        "dataset_path",
        "effective_dataset_path",
        "pruned_dataset_path",
        "dkw_csv_path",
        "suppressed_stdout_log_path",
    ):
        resolved = resolve_existing_path(cached_summary.get(key), candidate_roots)
        if resolved is not None:
            normalized[key] = resolved
    return normalized


def reconstruct_gt_summary_from_artifacts(save_path, dataset_path, dataset_stem, time_step_hours):
    ground_truth_dir = os.path.join(save_path, "ground_truth", dataset_stem)
    selected_average = choose_existing_gt_average(ground_truth_dir, dataset_stem)
    if selected_average is None:
        return None

    rep_done, averaged_results_path = selected_average
    ground_truth_dataset_path = os.path.join(ground_truth_dir, os.path.basename(dataset_path))
    observed_simulated_path = os.path.join(ground_truth_dir, f"{dataset_stem}_simulated.json")
    if not os.path.exists(observed_simulated_path):
        observed_simulated_path = None

    dkw_csv_path = choose_first_existing_file(os.path.join(ground_truth_dir, "plots"), "_dkw_band.csv")
    suppressed_stdout_log_path = choose_first_existing_file(ground_truth_dir, "_stdout.log")
    return {
        "dataset_path": os.path.abspath(ground_truth_dataset_path if os.path.exists(ground_truth_dataset_path) else dataset_path),
        "effective_dataset_path": os.path.abspath(ground_truth_dataset_path if os.path.exists(ground_truth_dataset_path) else dataset_path),
        "pruned_dataset_path": None,
        "dataset_dir": os.path.abspath(ground_truth_dir),
        "dataset_stem": os.path.splitext(os.path.abspath(ground_truth_dataset_path))[0],
        "rep_done": rep_done,
        "actual_runtime_seconds": None,
        "max_runtime_seconds": None,
        "run_until_convergence": True,
        "convergence_reached": None,
        "convergence_scores": [],
        "convergence_threshold": 1e-8,
        "observed_simulated_path": None if observed_simulated_path is None else os.path.abspath(observed_simulated_path),
        "averaged_results_path": os.path.abspath(averaged_results_path),
        "plots_dir": os.path.join(ground_truth_dir, "plots"),
        "dkw_csv_path": dkw_csv_path,
        "granularity": time_step_hours,
        "time_step_hours": time_step_hours,
        "positive_test_pruning": None,
        "suppressed_stdout_log_path": suppressed_stdout_log_path,
        "reconstructed_from_existing_gt_artifacts": True,
    }


def load_existing_gt_results(save_path, dataset_path, time_step_hours):
    dataset_stem = os.path.splitext(os.path.basename(dataset_path))[0]
    summary_path = gt_results_summary_path(save_path, dataset_stem)
    if os.path.exists(summary_path):
        cached_summary = final.read_cached_json(summary_path)
        if cached_summary:
            normalized = normalize_cached_gt_summary(
                cached_summary,
                save_path=save_path,
                dataset_stem=dataset_stem,
                summary_path=summary_path,
            )
            if normalized is not None:
                final.write_json(summary_path, normalized)
                return normalized, summary_path

    reconstructed = reconstruct_gt_summary_from_artifacts(
        save_path=save_path,
        dataset_path=dataset_path,
        dataset_stem=dataset_stem,
        time_step_hours=time_step_hours,
    )
    if reconstructed is not None:
        final.write_json(summary_path, reconstructed)
        return reconstructed, summary_path
    return None, summary_path


def iteration_run_id(dataset_stem, parameter_case, java_iterations, quartile_label):
    return (
        f"{dataset_stem}__{parameter_case['case_run_code']}"
        f"__{iteration_label(java_iterations)}__{quartile_label}"
    )


def iteration_case_dir(base_dir, java_iterations):
    return final.ensure_dir(os.path.join(base_dir, iteration_label(java_iterations)))


def write_run_readme(save_path, args, parameter_case, dataset_generation_summaries):
    output_path = os.path.join(save_path, "final_mid_iteration_sweep.md")
    lines = [
        "# Final Mid Java-Iteration Sweep",
        "",
        "This run follows `sweep_pipeline_final.py` but keeps only the all-mid parameter case.",
        "",
        "## Configuration",
        "",
        f"- `java_iterations`: {list(args.java_iterations)}",
        f"- `time_step_hours`: {TIME_STEP}",
        f"- `internal_steps`: swept by `java_iterations`",
        f"- `parameter_case_code`: {parameter_case['case_run_code']}",
        f"- `parameter_case_id`: {parameter_case['case_id']}",
        f"- `observed_test_ablation`: {args.observed_test_ablation}",
        f"- `generate_plot_images`: {not args.skip_plot_images}",
        "",
        "## Datasets",
        "",
    ]
    for summary in dataset_generation_summaries:
        lines.extend(
            [
                f"- `{summary['dataset_stem']}`: "
                f"{summary['requested_internal_contacts']} internal contacts",
            ]
        )
    final.write_text(output_path, "\n".join(lines) + "\n")
    return output_path


def run_java_analysis_iteration_task(
    repo_root,
    save_path,
    dataset_run,
    parameter_case,
    time_step_hours,
    java_iterations,
):
    logical_iterations = int(java_iterations)
    analysis_iterations = effective_java_iterations(logical_iterations)
    label = iteration_label(logical_iterations)
    parameter_bundle = parameter_case["parameter_bundle"]

    analysis_base_dir = os.path.join(save_path, "java_analysis", dataset_run["dataset_stem"])
    analysis_dir = iteration_case_dir(analysis_base_dir, logical_iterations)
    summary_path = os.path.join(analysis_dir, "java_analysis_summary.json")

    if os.path.exists(summary_path):
        cached_summary = final.read_cached_json(summary_path)
        if cached_summary:
            analysis_path = cached_summary.get("analysis_path")
            cached_observed_path = cached_summary.get("observed_simulated_path")
            observed_path_matches = final._is_valid_path(cached_observed_path) and (
                os.path.abspath(cached_observed_path)
                == os.path.abspath(dataset_run["observed_simulated_path"])
            )
            iterations_match = (
                cached_summary.get("java_iterations") == logical_iterations
                and cached_summary.get("effective_java_iterations") == analysis_iterations
            )
            if not final._is_valid_path(analysis_path):
                try:
                    analysis_path = final.find_generated_file(
                        analysis_dir,
                        f"_tracks_it{analysis_iterations}.json",
                    )
                    cached_summary["analysis_path"] = analysis_path
                    final.write_json(summary_path, cached_summary)
                except FileNotFoundError:
                    analysis_path = None

            if final._is_valid_path(analysis_path) and observed_path_matches and iterations_match:
                return cached_summary

    final.ensure_dataset_analysis_input(
        analysis_dir=analysis_dir,
        observed_simulated_path=dataset_run["observed_simulated_path"],
    )

    started_at = time.perf_counter()
    java_result = run_stpn_analysis(
        parameter_bundle=parameter_bundle,
        analysis_dir=analysis_dir,
        iterations=analysis_iterations,
        cache_dir=None,
        repo_root=repo_root,
        time_step_hours=time_step_hours,
    )
    analysis_wall_runtime_seconds = time.perf_counter() - started_at
    analysis_path = final.find_generated_file(analysis_dir, f"_tracks_it{analysis_iterations}.json")

    summary = {
        "dataset_path": dataset_run["dataset_path"],
        "dataset_stem": dataset_run["dataset_stem"],
        "observed_simulated_path": dataset_run["observed_simulated_path"],
        "parameter_case_id": java_result["parameter_bundle"]["case_id"],
        "parameter_case_code": parameter_case["case_run_code"],
        "parameter_case_index": parameter_case["case_index"],
        "parameter_levels": parameter_case["levels"],
        "parameter_bundle_path": java_result["parameter_bundle_path"],
        "stpn_solution_path": java_result["stpn_solution_path"],
        "observation_curve_path": java_result["observation_curve_path"],
        "analysis_dir": analysis_dir,
        "analysis_path": analysis_path,
        "time_step_hours": time_step_hours,
        "java_iterations": logical_iterations,
        "effective_java_iterations": analysis_iterations,
        "java_iteration_label": label,
        "iterations": analysis_iterations,
        "analysis_wall_runtime_seconds": analysis_wall_runtime_seconds,
    }
    final.write_json(summary_path, summary)
    return summary


def run_python_analysis_iteration_task(
    save_path,
    dataset_run,
    parameter_case,
    quartile_label,
    runtime_budget_seconds,
    time_step_hours,
    java_iterations,
):
    logical_iterations = int(java_iterations)
    parameter_bundle = parameter_case["parameter_bundle"]

    analysis_base_dir = os.path.join(
        save_path,
        "python_analysis",
        quartile_label,
        dataset_run["dataset_stem"],
    )
    analysis_dir = iteration_case_dir(analysis_base_dir, logical_iterations)
    summary_path = os.path.join(analysis_dir, "python_analysis_summary.json")

    if os.path.exists(summary_path):
        cached_summary = final.read_cached_json(summary_path)
        if cached_summary:
            cached_runtime_budget = cached_summary.get("runtime_budget_seconds")
            runtime_budget_matches = isinstance(cached_runtime_budget, (int, float)) and (
                abs(float(cached_runtime_budget) - float(runtime_budget_seconds)) <= 1e-9
            )
            cached_dataset_path = cached_summary.get("dataset_path")
            dataset_path_matches = final._is_valid_path(cached_dataset_path) and (
                os.path.abspath(cached_dataset_path) == os.path.abspath(dataset_run["dataset_path"])
            )
            iterations_match = cached_summary.get("java_iterations") == logical_iterations
            if (
                final._is_valid_path(cached_summary.get("averaged_results_path"))
                and runtime_budget_matches
                and dataset_path_matches
                and iterations_match
            ):
                return cached_summary

    analysis_dataset_path = final.ensure_python_simulation_input(
        analysis_dir=analysis_dir,
        dataset_path=dataset_run["dataset_path"],
    )

    python_result = run_dataset_simulations(
        dataset_path=analysis_dataset_path,
        rep=None,
        run_until_convergence=False,
        iterations_cap=4,
        convergence_threshold=1e-6,
        fine_grained=False,
        dataset_label=f"{dataset_run['dataset_stem']}_{iteration_label(logical_iterations)}",
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
        "parameter_case_id": parameter_case["case_id"],
        "parameter_case_code": parameter_case["case_run_code"],
        "parameter_case_index": parameter_case["case_index"],
        "parameter_levels": parameter_case["levels"],
        "java_iterations": logical_iterations,
        "java_iteration_label": iteration_label(logical_iterations),
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
    final.write_json(summary_path, summary)
    return summary


def run_comparison_iteration_task(
    save_path,
    quartile_label,
    runtime_budget_seconds,
    dataset_run,
    parameter_case,
    java_summary,
    python_summary,
    time_step_hours,
    save_plots=True,
):
    logical_iterations = int(java_summary["java_iterations"])
    analysis_iterations = effective_java_iterations(logical_iterations)
    comparison_base_dir = os.path.join(
        save_path,
        "comparison",
        quartile_label,
        dataset_run["dataset_stem"],
    )
    comparison_dir = iteration_case_dir(comparison_base_dir, logical_iterations)
    summary_path = os.path.join(comparison_dir, "comparison_summary.json")
    run_id = iteration_run_id(
        dataset_run["dataset_stem"],
        parameter_case,
        logical_iterations,
        quartile_label,
    )

    analysis_path = final.ensure_analysis_tracks_path(
        java_summary["analysis_dir"],
        java_summary.get("analysis_path"),
        parameter_case["parameter_bundle"],
        analysis_iterations,
        time_step_hours,
        case_id=parameter_case["case_id"],
    )
    baseline_path = python_summary["averaged_results_path"]
    ground_truth_path = dataset_run["ground_truth_path"]

    if os.path.exists(summary_path):
        cached_summary = final.read_cached_json(summary_path)
        if cached_summary:
            cached_java_analysis = cached_summary.get("java_analysis", {})
            cached_python_analysis = cached_summary.get("python_analysis", {})
            cached_analysis_path = cached_java_analysis.get("analysis_path")
            cached_baseline_path = cached_python_analysis.get("baseline_path")
            cached_ground_truth_path = cached_summary.get("ground_truth_path")
            cache_matches_inputs = (
                final._is_valid_path(cached_ground_truth_path)
                and final._is_valid_path(cached_analysis_path)
                and final._is_valid_path(cached_baseline_path)
                and os.path.abspath(cached_ground_truth_path) == os.path.abspath(ground_truth_path)
                and os.path.abspath(cached_analysis_path) == os.path.abspath(analysis_path)
                and os.path.abspath(cached_baseline_path) == os.path.abspath(baseline_path)
                and cached_summary.get("java_iterations") == logical_iterations
            )
            if cache_matches_inputs:
                if cached_summary.get("run_id") != run_id:
                    cached_summary["run_id"] = run_id
                    final.write_json(summary_path, cached_summary)
                return cached_summary

    final.ensure_dir(comparison_dir)
    if save_plots:
        analysis_curve_plots = sp.create_analysis_subject_curve_plots(
            run_dir=comparison_dir,
            run_name=run_id,
            analysis_path=analysis_path,
            ground_truth_path=ground_truth_path,
            baseline_path=baseline_path,
            granularity=time_step_hours,
            ground_truth_sample_size=dataset_run.get("ground_truth_iterations"),
            save_plots=True,
        )
    else:
        analysis_curve_plots = {
            "output_dir": None,
            "csv_path": None,
            "epsilon": None,
            "plot_paths": [],
            "java_curve_plot_paths": [],
        }

    precision_dir = os.path.join(comparison_dir, "precision_metrics")
    final.ensure_dir(precision_dir)
    prediction_metrics_path = os.path.join(precision_dir, "metrics_prediction.json")
    baseline_metrics_path = os.path.join(precision_dir, "metrics_baseline.json")

    if final.LOG_PRECISION_METRICS:
        metrics_stdout = io.StringIO()
        with contextlib.redirect_stdout(metrics_stdout):
            prediction_metrics_summary = process_and_save(
                analysis_path,
                ground_truth_path,
                M=10,
                metrics_output=prediction_metrics_path,
                plots_dir=os.path.join(precision_dir, "numericalAnalysis", "plots"),
                save_plots=final.SAVE_PRECISION_PLOTS and save_plots,
                verbose=True,
                include_scatter_coordinates=final.SAVE_PRECISION_PLOTS and save_plots,
            )
            baseline_metrics_summary = process_and_save(
                baseline_path,
                ground_truth_path,
                M=10,
                metrics_output=baseline_metrics_path,
                plots_dir=os.path.join(precision_dir, "simulatedBaseline", "plots"),
                save_plots=final.SAVE_PRECISION_PLOTS and save_plots,
                verbose=True,
                include_scatter_coordinates=final.SAVE_PRECISION_PLOTS and save_plots,
            )
        final.write_text(os.path.join(precision_dir, "metrics_stdout.log"), metrics_stdout.getvalue())
    else:
        prediction_metrics_summary = process_and_save(
            analysis_path,
            ground_truth_path,
            M=10,
            metrics_output=prediction_metrics_path,
            plots_dir=os.path.join(precision_dir, "numericalAnalysis", "plots"),
            save_plots=final.SAVE_PRECISION_PLOTS and save_plots,
            verbose=False,
            include_scatter_coordinates=final.SAVE_PRECISION_PLOTS and save_plots,
        )
        baseline_metrics_summary = process_and_save(
            baseline_path,
            ground_truth_path,
            M=10,
            metrics_output=baseline_metrics_path,
            plots_dir=os.path.join(precision_dir, "simulatedBaseline", "plots"),
            save_plots=final.SAVE_PRECISION_PLOTS and save_plots,
            verbose=False,
            include_scatter_coordinates=final.SAVE_PRECISION_PLOTS and save_plots,
        )

    if save_plots:
        comparison_metrics = sp.create_analysis_vs_simulation_plots(
            run_dir=comparison_dir,
            run_name=run_id,
            ground_truth_path=ground_truth_path,
            analysis_path=analysis_path,
            baseline_path=baseline_path,
            include_moving_avg_metrics=True,
            save_plots=True,
        )
    else:
        comparison_metrics = final.compute_analysis_vs_simulation_metrics_only(
            ground_truth_path=ground_truth_path,
            analysis_path=analysis_path,
            baseline_path=baseline_path,
            run_dir=comparison_dir,
            write_csvs=final.SAVE_COMPARISON_CSVS,
        )

    summary = {
        "run_id": run_id,
        "quartile_label": quartile_label,
        "runtime_budget_seconds": runtime_budget_seconds,
        "dataset_path": dataset_run["dataset_path"],
        "dataset_stem": dataset_run["dataset_stem"],
        "ground_truth_iterations": dataset_run.get("ground_truth_iterations"),
        "ground_truth_path": ground_truth_path,
        "parameter_case_id": parameter_case["case_id"],
        "parameter_case_code": parameter_case["case_run_code"],
        "parameter_case_index": parameter_case["case_index"],
        "parameter_levels": parameter_case["levels"],
        "java_iterations": logical_iterations,
        "effective_java_iterations": analysis_iterations,
        "java_iteration_label": iteration_label(logical_iterations),
        "java_analysis": {
            "analysis_dir": java_summary["analysis_dir"],
            "analysis_path": analysis_path,
            "analysis_wall_runtime_seconds": java_summary["analysis_wall_runtime_seconds"],
            "parameter_bundle_path": java_summary["parameter_bundle_path"],
            "stpn_solution_path": java_summary["stpn_solution_path"],
            "observation_curve_path": java_summary["observation_curve_path"],
            "java_iterations": logical_iterations,
            "effective_java_iterations": analysis_iterations,
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
            "prediction_mean_brier": prediction_metrics_summary["mean_brier_score"],
            "prediction_mean_ece": prediction_metrics_summary["mean_ece"],
            "baseline_mean_brier": baseline_metrics_summary["mean_brier_score"],
            "baseline_mean_ece": baseline_metrics_summary["mean_ece"],
        },
        "comparison_metrics": comparison_metrics,
        "comparison_dir": comparison_dir,
        "note": "",
    }
    final.write_json(summary_path, summary)
    return summary


def build_iteration_reference_row(comparison_summary):
    row = final.build_reference_row(comparison_summary)
    row["java_iterations"] = comparison_summary.get("java_iterations")
    return row


def write_iteration_reference_sweep_csv(csv_path, rows):
    final.ensure_dir(os.path.dirname(csv_path) or ".")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ITERATION_SWEEP_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in ITERATION_SWEEP_COLUMNS})


def persist_iteration_comparison_outputs(save_path, quartile_label, comparison_summaries):
    for comparison_summary in comparison_summaries:
        summary_path = os.path.join(comparison_summary["comparison_dir"], "comparison_summary.json")
        final.write_json(summary_path, comparison_summary)

    comparison_summary_json_path = os.path.join(save_path, f"comparison_summary_{quartile_label}.json")
    final.write_json(comparison_summary_json_path, comparison_summaries)

    sweep_summary_rows = [build_iteration_reference_row(summary) for summary in comparison_summaries]
    sweep_summary_json_path = os.path.join(save_path, "sweep_summary.json")
    final.write_json(sweep_summary_json_path, sweep_summary_rows)

    sweep_summary_csv_path = os.path.join(save_path, "sweep_summary.csv")
    write_iteration_reference_sweep_csv(sweep_summary_csv_path, sweep_summary_rows)

    comparison_summary_csv_path = os.path.join(save_path, f"comparison_summary_{quartile_label}.csv")
    write_iteration_reference_sweep_csv(comparison_summary_csv_path, sweep_summary_rows)

    iteration_summary_json_path = os.path.join(save_path, "iteration_sweep_summary.json")
    final.write_json(iteration_summary_json_path, sweep_summary_rows)
    iteration_summary_csv_path = os.path.join(save_path, "iteration_sweep_summary.csv")
    write_iteration_reference_sweep_csv(iteration_summary_csv_path, sweep_summary_rows)

    return {
        "comparison_summary_json_path": comparison_summary_json_path,
        "comparison_summary_csv_path": comparison_summary_csv_path,
        "sweep_summary_csv_path": sweep_summary_csv_path,
        "sweep_summary_json_path": sweep_summary_json_path,
        "iteration_sweep_summary_json_path": iteration_summary_json_path,
        "iteration_sweep_summary_csv_path": iteration_summary_csv_path,
        "rows_written": len(sweep_summary_rows),
    }


def regenerate_selected_iteration_run_plots(comparison_summary, time_step_hours):
    return final.regenerate_selected_run_plots(
        comparison_summary,
        time_step_hours,
        effective_java_iterations(comparison_summary["java_iterations"]),
    )


def refresh_selected_run_outputs(save_path, quartile_label, time_step_hours):
    comparison_summary_json_path = os.path.join(save_path, f"comparison_summary_{quartile_label}.json")
    if not os.path.exists(comparison_summary_json_path):
        raise FileNotFoundError(
            f"Comparison summary not found for quartile {quartile_label}: {comparison_summary_json_path}"
        )

    comparison_summaries = final.read_json(comparison_summary_json_path)
    comparison_summaries.sort(key=lambda item: (item["dataset_stem"], item.get("java_iterations", 0)))

    notes_by_run, selection_manifest = final.build_notes_map_and_selection_manifest(comparison_summaries)
    final.apply_notes_to_comparison_summaries(comparison_summaries, notes_by_run)
    selection_manifest_path = os.path.join(save_path, "selected_run_manifest.json")
    final.write_json(selection_manifest_path, selection_manifest)

    stage_name = "stage8_selected_plots"
    selected_summaries = [summary for summary in comparison_summaries if summary.get("note")]
    total_stage_tasks = len(selected_summaries)
    final.write_stage_checkpoint(
        save_path,
        stage_name,
        0,
        total_stage_tasks,
        status="running",
        extra={"mode": "refresh_selected_plots", "quartile_label": quartile_label},
    )

    if total_stage_tasks > 0:
        updated_by_run_id = {}
        worker_count = final.resolve_worker_count(total_stage_tasks)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(regenerate_selected_iteration_run_plots, summary, time_step_hours)
                for summary in selected_summaries
            ]
            completed = 0
            for future in tqdm(as_completed(futures), total=total_stage_tasks, desc="Generating selected plots"):
                updated_summary = future.result()
                updated_by_run_id[updated_summary["run_id"]] = updated_summary
                completed += 1
                final.write_stage_checkpoint(save_path, stage_name, completed, total_stage_tasks, status="running")

        comparison_summaries = [
            updated_by_run_id.get(summary["run_id"], summary)
            for summary in comparison_summaries
        ]

    final.mark_stage_complete(save_path, stage_name)
    final.write_stage_checkpoint(save_path, stage_name, total_stage_tasks, total_stage_tasks, status="completed")

    output_paths = persist_iteration_comparison_outputs(save_path, quartile_label, comparison_summaries)
    final.update_run_completion_metadata(
        save_path=save_path,
        quartile_label=quartile_label,
        comparison_summaries=comparison_summaries,
        output_paths=output_paths,
        selection_manifest_path=selection_manifest_path,
    )
    return {
        "save_path": save_path,
        "quartile_label": quartile_label,
        "selected_runs_for_plots": total_stage_tasks,
        "selected_run_manifest_path": selection_manifest_path,
        **output_paths,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the final-style all-mid Java-iteration sweep.")
    parser.add_argument(
        "--output-root",
        default=os.path.join("results", datetime.now().strftime("final_mid_iteration_sweep_%Y%m%d-%H%M")),
        help="Directory where sweep outputs will be created.",
    )
    parser.add_argument(
        "--java-iterations",
        nargs="+",
        type=int,
        default=list(DEFAULT_JAVA_ITERATIONS),
        help=f"Java iteration counts to test. Default: {' '.join(map(str, DEFAULT_JAVA_ITERATIONS))}",
    )
    parser.add_argument(
        "--parameter-json-path",
        default="parameters.json",
        help="Legacy parameters JSON containing lower/mid/upper transition parameters.",
    )
    parser.add_argument(
        "--skip-plot-images",
        action="store_true",
        help="Skip PNG plot generation except for no-plot CSV/JSON outputs.",
    )
    parser.add_argument(
        "--disable-moving-avg-metrics",
        action="store_true",
        help="Record moving-average metrics as disabled in run metadata.",
    )
    parser.add_argument(
        "--observed-test-ablation",
        choices=sp.OBSERVED_TEST_ABLATION_CHOICES,
        default=sp.OBSERVED_TEST_ABLATION_NONE,
        help="Optional filtering applied only to the observed_simulated.json copied into each run.",
    )
    parser.add_argument(
        "--skip-datasets",
        nargs="*",
        default=[],
        help=(
            "Fixed dataset targets to skip. Accepts family names such as bubble, scale_free, "
            "small_world, stems such as dataset_scale_free_2500, or filenames."
        ),
    )
    parser.add_argument(
        "--reuse-run",
        default=None,
        help="Existing results/final_mid_iteration_sweep_* directory to resume or refresh.",
    )
    parser.add_argument(
        "--only-selected-plots",
        action="store_true",
        help="Reuse an existing sweep directory and regenerate plots only for selected runs.",
    )
    parser.add_argument(
        "--quartile-label",
        default=None,
        help="Optional quartile label to use with --only-selected-plots, for example q4.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.quartile_label and not args.only_selected_plots:
        raise ValueError("--quartile-label can only be used together with --only-selected-plots.")
    if args.only_selected_plots:
        save_path = final.resolve_existing_run_path(args.reuse_run)
        quartile_label = final.resolve_quartile_label_for_existing_run(save_path, args.quartile_label)
        refresh_summary = refresh_selected_run_outputs(
            save_path=save_path,
            quartile_label=quartile_label,
            time_step_hours=TIME_STEP,
        )
        print(
            "Selected-run refresh completed. "
            f"Updated {refresh_summary['selected_runs_for_plots']} selected runs in {save_path}. "
            f"Manifest: {refresh_summary['selected_run_manifest_path']}"
        )
        return

    java_iterations_values = iteration_sweep.parse_java_iterations(args.java_iterations)
    dataset_targets = fixed_dataset_targets(skip_datasets=args.skip_datasets)
    validate_java_iterations(java_iterations_values, dataset_targets=dataset_targets)
    parameter_case = load_mid_parameter_case(args.parameter_json_path)
    if args.reuse_run:
        save_path = final.resolve_existing_run_path(args.reuse_run)
    else:
        save_path = final.ensure_dir(os.path.abspath(args.output_root))
    cache_path = final.ensure_dir(os.path.join("results", "cache"))
    repo_root = os.path.abspath(os.path.dirname(__file__))
    include_moving_avg_metrics = not args.disable_moving_avg_metrics
    metadata_path = os.path.join(save_path, "run_metadata.json")
    existing_metadata = {}
    if os.path.exists(metadata_path):
        existing_metadata = final.read_json(metadata_path)
    selected_dataset_targets = [
        {
            "dataset_family": dataset_profile.family,
            "internal_contacts": internal_contacts,
            "dataset_stem": os.path.splitext(final.dataset_filename(dataset_profile, internal_contacts))[0],
        }
        for dataset_profile, internal_contacts in dataset_targets
    ]

    final.write_json(
        metadata_path,
        {
            "save_path": save_path,
            "cache_path": cache_path,
            "dataset_family": "fixed_final_mid_iteration",
            "dataset_profiles": [
                final.dataset_profile_metadata(DATASET_PROFILES[family])
                for family in DATASET_PROFILE_ORDER
            ],
            "selected_dataset_targets": selected_dataset_targets,
            "skipped_datasets": list(args.skip_datasets),
            "time_step_hours": TIME_STEP,
            "java_iterations": java_iterations_values,
            "parameter_case_code": parameter_case["case_run_code"],
            "parameter_case_id": parameter_case["case_id"],
            "quantile": QUANTILE,
            "noise_fraction": final.NOISE_FRACTION,
            "noise_time_shift_hours": final.NOISE_TIME_SHIFT_HOURS,
            "save_comparison_csvs": final.SAVE_COMPARISON_CSVS,
            "contact_noise_event_types": list(final.CONTACT_NOISE_EVENT_TYPES),
            "observation_noise_event_types": list(final.OBSERVATION_NOISE_EVENT_TYPES),
            "observed_test_ablation": args.observed_test_ablation,
            "moving_avg_metrics_enabled": include_moving_avg_metrics,
            "started_at": existing_metadata.get(
                "started_at",
                datetime.now().isoformat(timespec="seconds"),
            ),
            "resumed_run": bool(args.reuse_run),
            "resumed_at": datetime.now().isoformat(timespec="seconds") if args.reuse_run else None,
        },
    )

    stage_name = "stage1_dataset_generation"
    final.write_stage_checkpoint(save_path, stage_name, 0, len(dataset_targets), status="running")

    dataset_paths = []
    dataset_generation_summaries = []
    dataset_profile_by_stem = {}
    for index, (dataset_profile, internal_contacts) in enumerate(dataset_targets, start=1):
        dataset_path = os.path.join(save_path, final.dataset_filename(dataset_profile, internal_contacts))
        if not os.path.exists(dataset_path):
            dataset_generation_summary = final.generate_dataset_for_profile(
                dataset_profile=dataset_profile,
                dataset_path=dataset_path,
                internal_contacts=internal_contacts,
                seed=30,
            )
        else:
            dataset_generation_summary = final.summarize_generated_dataset(
                dataset_profile=dataset_profile,
                dataset_path=dataset_path,
                payload=final.read_json(dataset_path),
                requested_internal_contacts=internal_contacts,
            )
        dataset_paths.append(dataset_path)
        dataset_generation_summaries.append(dataset_generation_summary)
        dataset_profile_by_stem[dataset_generation_summary["dataset_stem"]] = dataset_profile
        final.write_stage_checkpoint(save_path, stage_name, index, len(dataset_targets), status="running")

    final.write_json(os.path.join(save_path, "dataset_generation_summary.json"), dataset_generation_summaries)
    write_run_readme(save_path, args, parameter_case, dataset_generation_summaries)
    final.mark_stage_complete(save_path, stage_name)
    final.write_stage_checkpoint(save_path, stage_name, len(dataset_targets), len(dataset_targets), status="completed")

    stage_name = "stage2_ground_truth"
    final.write_stage_checkpoint(save_path, stage_name, 0, len(dataset_paths), status="running")

    ground_truth_results_by_stem = {}
    for index, dataset_path in enumerate(final.progress(dataset_paths, desc="Running GT simulations"), start=1):
        dataset_stem = os.path.splitext(os.path.basename(dataset_path))[0]
        gt_results, gt_results_path = load_existing_gt_results(
            save_path=save_path,
            dataset_path=dataset_path,
            time_step_hours=TIME_STEP,
        )

        if gt_results is None:
            ground_truth_dataset_path = final.ensure_ground_truth_simulation_input(
                save_path=save_path,
                dataset_path=dataset_path,
            )
            dataset_profile = dataset_profile_by_stem[dataset_stem]
            gt_results = run_dataset_simulations(
                dataset_path=ground_truth_dataset_path,
                run_until_convergence=True,
                iterations_cap=10_000,
                convergence_threshold=1e-6,
                fine_grained=False,
                time_step_hours=TIME_STEP,
                seed=30,
                prune_after_positive_test=False,
                export_observed_simulation=True,
                pruning_seed=None,
                parameter_bundle=parameter_case["parameter_bundle"],
                dataset_label=dataset_stem,
                save_plots=(not args.skip_plot_images and dataset_profile.family == final.DATASET_FAMILY_BUBBLE),
            )
            final.write_json(gt_results_path, gt_results)

        ground_truth_results_by_stem[dataset_stem] = {
            "clean_dataset_path": dataset_path,
            "dataset_stem": dataset_stem,
            "gt_results_path": gt_results_path,
            "gt_results": gt_results,
        }
        final.write_stage_checkpoint(save_path, stage_name, index, len(dataset_paths), status="running")

    final.mark_stage_complete(save_path, stage_name)
    final.write_stage_checkpoint(save_path, stage_name, len(dataset_paths), len(dataset_paths), status="completed")

    stage_name = "stage2b_contact_noise"
    final.write_stage_checkpoint(save_path, stage_name, 0, len(dataset_paths), status="running")

    contact_noisy_dataset_paths_by_stem = {}
    contact_noise_summaries_by_stem = {}
    contact_noise_summaries = []
    for index, dataset_path in enumerate(final.progress(dataset_paths, desc="Applying contact noise"), start=1):
        dataset_stem = os.path.splitext(os.path.basename(dataset_path))[0]
        contact_noisy_dataset_path = os.path.join(save_path, f"{dataset_stem}_contact_noisy.json")
        contact_noise_summary_path = os.path.join(
            save_path,
            "dataset_noise",
            f"{dataset_stem}_contact_noise_summary.json",
        )
        if os.path.exists(contact_noisy_dataset_path) and os.path.exists(contact_noise_summary_path):
            contact_noise_summary = final.read_json(contact_noise_summary_path)
        else:
            contact_noise_summary = final.apply_contact_noise_to_dataset(
                source_dataset_path=dataset_path,
                noisy_dataset_path=contact_noisy_dataset_path,
                seed_label=f"{dataset_stem}:contact",
            )
            final.write_json(contact_noise_summary_path, contact_noise_summary)

        contact_noisy_dataset_paths_by_stem[dataset_stem] = contact_noisy_dataset_path
        contact_noise_summaries_by_stem[dataset_stem] = contact_noise_summary
        contact_noise_summaries.append(contact_noise_summary)
        final.write_stage_checkpoint(save_path, stage_name, index, len(dataset_paths), status="running")

    final.write_json(os.path.join(save_path, "dataset_noise_contact_summary.json"), contact_noise_summaries)
    final.mark_stage_complete(save_path, stage_name)
    final.write_stage_checkpoint(save_path, stage_name, len(dataset_paths), len(dataset_paths), status="completed")

    stage_name = "stage2c_observed_one_run"
    final.write_stage_checkpoint(save_path, stage_name, 0, len(dataset_paths), status="running")

    observed_one_run_results_by_stem = {}
    observed_one_run_summaries = []
    for index, dataset_path in enumerate(final.progress(dataset_paths, desc="Running one observed simulation"), start=1):
        dataset_stem = os.path.splitext(os.path.basename(dataset_path))[0]
        contact_noisy_dataset_path = contact_noisy_dataset_paths_by_stem[dataset_stem]
        one_run_summary_path = os.path.join(
            save_path,
            "observed_one_run",
            f"{dataset_stem}_contact_noisy_one_run_summary.json",
        )
        one_run_result = None
        if os.path.exists(one_run_summary_path):
            cached = final.read_json(one_run_summary_path)
            cached_dataset_path = cached.get("dataset_path")
            dataset_path_matches = final._is_valid_path(cached_dataset_path) and (
                os.path.abspath(cached_dataset_path) == os.path.abspath(contact_noisy_dataset_path)
            )
            if (
                cached.get("rep_done") == 1
                and dataset_path_matches
                and final._is_valid_path(cached.get("observed_simulated_path"))
            ):
                one_run_result = cached

        if one_run_result is None:
            one_run_result = run_dataset_simulations(
                dataset_path=contact_noisy_dataset_path,
                rep=1,
                run_until_convergence=False,
                iterations_cap=1,
                convergence_threshold=1e-6,
                fine_grained=False,
                dataset_label=f"{dataset_stem}_contact_noisy_observed",
                seed=30,
                prune_after_positive_test=False,
                export_observed_simulation=True,
                pruning_seed=None,
                time_step_hours=TIME_STEP,
                parameter_bundle=parameter_case["parameter_bundle"],
                save_plots=False,
            )
            final.write_json(one_run_summary_path, one_run_result)

        observed_one_run_results_by_stem[dataset_stem] = one_run_result
        observed_one_run_summaries.append(one_run_result)
        final.write_stage_checkpoint(save_path, stage_name, index, len(dataset_paths), status="running")

    final.write_json(os.path.join(save_path, "observed_one_run_summary.json"), observed_one_run_summaries)
    final.mark_stage_complete(save_path, stage_name)
    final.write_stage_checkpoint(save_path, stage_name, len(dataset_paths), len(dataset_paths), status="completed")

    stage_name = "stage2d_observation_noise"
    final.write_stage_checkpoint(save_path, stage_name, 0, len(dataset_paths), status="running")

    dataset_runs = []
    observation_noise_summaries = []
    for index, dataset_path in enumerate(final.progress(dataset_paths, desc="Applying observation noise"), start=1):
        dataset_stem = os.path.splitext(os.path.basename(dataset_path))[0]
        ground_truth_entry = ground_truth_results_by_stem[dataset_stem]
        gt_results = ground_truth_entry["gt_results"]
        one_run_result = observed_one_run_results_by_stem[dataset_stem]
        clean_observed_one_run_path = one_run_result["observed_simulated_path"]
        final_observed_simulated_path = final.observation_noisy_simulated_path(clean_observed_one_run_path)
        observation_noise_summary_path = os.path.join(
            save_path,
            "dataset_noise",
            f"{dataset_stem}_observation_noise_summary.json",
        )

        if os.path.exists(final_observed_simulated_path) and os.path.exists(observation_noise_summary_path):
            observation_noise_summary = final.read_json(observation_noise_summary_path)
        else:
            observation_noise_summary = final.apply_observation_noise_to_dataset(
                source_dataset_path=clean_observed_one_run_path,
                noisy_dataset_path=final_observed_simulated_path,
                seed_label=f"{dataset_stem}:observation",
            )
            final.write_json(observation_noise_summary_path, observation_noise_summary)

        observation_noise_summaries.append(observation_noise_summary)
        dataset_runs.append(
            {
                "clean_dataset_path": dataset_path,
                "dataset_path": contact_noisy_dataset_paths_by_stem[dataset_stem],
                "dataset_stem": dataset_stem,
                "gt_results_path": ground_truth_entry["gt_results_path"],
                "ground_truth_path": gt_results["averaged_results_path"],
                "clean_ground_truth_observed_simulated_path": gt_results.get("observed_simulated_path"),
                "clean_observed_simulated_path": clean_observed_one_run_path,
                "observed_simulated_path": final_observed_simulated_path,
                "contact_noise_summary": contact_noise_summaries_by_stem[dataset_stem],
                "observation_noise_summary": observation_noise_summary,
                "observed_one_run_summary_path": os.path.join(
                    save_path,
                    "observed_one_run",
                    f"{dataset_stem}_contact_noisy_one_run_summary.json",
                ),
                "ground_truth_iterations": gt_results.get("rep_done"),
            }
        )
        final.write_stage_checkpoint(save_path, stage_name, index, len(dataset_paths), status="running")

    final.write_json(os.path.join(save_path, "dataset_noise_observation_summary.json"), observation_noise_summaries)
    final.write_json(os.path.join(save_path, "dataset_runs_summary.json"), dataset_runs)
    final.mark_stage_complete(save_path, stage_name)
    final.write_stage_checkpoint(save_path, stage_name, len(dataset_paths), len(dataset_paths), status="completed")

    stage_name = "stage3_precompute"
    final.write_stage_checkpoint(save_path, stage_name, 0, 1, status="running")
    precompute_summary = final.run_precompute_task(save_path, parameter_case, TIME_STEP)
    final.write_json(os.path.join(save_path, "java_precompute_summary.json"), [precompute_summary])
    final.mark_stage_complete(save_path, stage_name)
    final.write_stage_checkpoint(save_path, stage_name, 1, 1, status="completed")

    stage_name = "stage4_java_analysis"
    stage4_tasks = [
        (dataset_run, java_iterations)
        for dataset_run in dataset_runs
        for java_iterations in java_iterations_values
    ]
    final.write_stage_checkpoint(save_path, stage_name, 0, len(stage4_tasks), status="running")

    java_analysis_summaries = []
    worker_count = final.resolve_worker_count(len(stage4_tasks))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                run_java_analysis_iteration_task,
                repo_root,
                save_path,
                dataset_run,
                parameter_case,
                TIME_STEP,
                java_iterations,
            )
            for dataset_run, java_iterations in stage4_tasks
        ]
        completed = 0
        for future in final.progress(as_completed(futures), total=len(stage4_tasks), desc="Running Java analysis"):
            java_analysis_summaries.append(future.result())
            completed += 1
            final.write_stage_checkpoint(save_path, stage_name, completed, len(stage4_tasks), status="running")

    java_analysis_summaries.sort(key=lambda item: (item["dataset_stem"], item["java_iterations"]))
    final.write_json(os.path.join(save_path, "java_analysis_summary.json"), java_analysis_summaries)
    final.mark_stage_complete(save_path, stage_name)
    final.write_stage_checkpoint(save_path, stage_name, len(stage4_tasks), len(stage4_tasks), status="completed")

    stage_name = "stage5_runtime_quartiles"
    final.write_stage_checkpoint(save_path, stage_name, 0, 1, status="running")
    java_analysis_runtime_quartiles = final.summarize_java_analysis_runtimes(java_analysis_summaries)
    final.write_json(
        os.path.join(save_path, "java_analysis_runtime_quartiles.json"),
        java_analysis_runtime_quartiles,
    )
    quartile_label = f"q{QUANTILE}"
    java_summary_by_key = {
        (summary["dataset_stem"], summary["java_iterations"]): summary
        for summary in java_analysis_summaries
    }
    final.mark_stage_complete(save_path, stage_name)
    final.write_stage_checkpoint(save_path, stage_name, 1, 1, status="completed")

    stage_name = "stage6_python_analysis"
    stage6_tasks = [
        (
            dataset_run,
            java_iterations,
            java_summary_by_key[(dataset_run["dataset_stem"], java_iterations)],
        )
        for dataset_run in dataset_runs
        for java_iterations in java_iterations_values
    ]
    final.write_stage_checkpoint(save_path, stage_name, 0, len(stage6_tasks), status="running")

    python_analysis_summaries = []
    worker_count = final.resolve_worker_count(len(stage6_tasks))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                run_python_analysis_iteration_task,
                save_path,
                dataset_run,
                parameter_case,
                quartile_label,
                java_summary["analysis_wall_runtime_seconds"],
                TIME_STEP,
                java_iterations,
            )
            for dataset_run, java_iterations, java_summary in stage6_tasks
        ]
        completed = 0
        for future in final.progress(as_completed(futures), total=len(stage6_tasks), desc="Running Python analysis"):
            python_analysis_summaries.append(future.result())
            completed += 1
            final.write_stage_checkpoint(save_path, stage_name, completed, len(stage6_tasks), status="running")

    python_analysis_summaries.sort(key=lambda item: (item["dataset_stem"], item["java_iterations"]))
    final.write_json(os.path.join(save_path, "python_analysis_summary.json"), python_analysis_summaries)
    final.mark_stage_complete(save_path, stage_name)
    final.write_stage_checkpoint(save_path, stage_name, len(stage6_tasks), len(stage6_tasks), status="completed")

    stage_name = "stage7_comparison"
    python_summary_by_key = {
        (summary["dataset_stem"], summary["java_iterations"]): summary
        for summary in python_analysis_summaries
    }
    stage7_tasks = []
    for dataset_run in dataset_runs:
        for java_iterations in java_iterations_values:
            task_key = (dataset_run["dataset_stem"], java_iterations)
            if task_key not in java_summary_by_key:
                raise KeyError(f"Missing Java analysis summary for key: {task_key}")
            if task_key not in python_summary_by_key:
                raise KeyError(f"Missing Python analysis summary for key: {task_key}")
            stage7_tasks.append(
                (
                    dataset_run,
                    java_iterations,
                    java_summary_by_key[task_key],
                    python_summary_by_key[task_key],
                )
            )

    final.write_stage_checkpoint(save_path, stage_name, 0, len(stage7_tasks), status="running")
    comparison_summaries = []
    worker_count = final.resolve_worker_count(len(stage7_tasks))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                run_comparison_iteration_task,
                save_path,
                quartile_label,
                java_summary["analysis_wall_runtime_seconds"],
                dataset_run,
                parameter_case,
                java_summary,
                python_summary,
                TIME_STEP,
                False,
            )
            for dataset_run, _java_iterations, java_summary, python_summary in stage7_tasks
        ]
        completed = 0
        for future in final.progress(as_completed(futures), total=len(stage7_tasks), desc="Running comparisons"):
            comparison_summaries.append(future.result())
            completed += 1
            final.write_stage_checkpoint(save_path, stage_name, completed, len(stage7_tasks), status="running")

    comparison_summaries.sort(key=lambda item: (item["dataset_stem"], item["java_iterations"]))
    final.mark_stage_complete(save_path, stage_name)
    final.write_stage_checkpoint(save_path, stage_name, len(stage7_tasks), len(stage7_tasks), status="completed")

    notes_by_run, selection_manifest = final.build_notes_map_and_selection_manifest(comparison_summaries)
    final.apply_notes_to_comparison_summaries(comparison_summaries, notes_by_run)
    selection_manifest_path = os.path.join(save_path, "selected_run_manifest.json")
    final.write_json(selection_manifest_path, selection_manifest)

    stage_name = "stage8_selected_plots"
    selected_summaries = [summary for summary in comparison_summaries if summary.get("note")]
    final.write_stage_checkpoint(save_path, stage_name, 0, len(selected_summaries), status="running")

    if selected_summaries and not args.skip_plot_images:
        updated_by_run_id = {}
        worker_count = final.resolve_worker_count(len(selected_summaries))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(regenerate_selected_iteration_run_plots, summary, TIME_STEP)
                for summary in selected_summaries
            ]
            completed = 0
            for future in final.progress(
                as_completed(futures),
                total=len(selected_summaries),
                desc="Generating selected plots",
            ):
                updated_summary = future.result()
                updated_by_run_id[updated_summary["run_id"]] = updated_summary
                completed += 1
                final.write_stage_checkpoint(
                    save_path,
                    stage_name,
                    completed,
                    len(selected_summaries),
                    status="running",
                )
        comparison_summaries = [
            updated_by_run_id.get(summary["run_id"], summary)
            for summary in comparison_summaries
        ]

    final.mark_stage_complete(save_path, stage_name)
    final.write_stage_checkpoint(save_path, stage_name, len(selected_summaries), len(selected_summaries), status="completed")

    output_paths = persist_iteration_comparison_outputs(save_path, quartile_label, comparison_summaries)
    final.update_run_completion_metadata(
        save_path=save_path,
        quartile_label=quartile_label,
        comparison_summaries=comparison_summaries,
        output_paths=output_paths,
        selection_manifest_path=selection_manifest_path,
    )

    print(
        "Final mid Java-iteration sweep completed. "
        f"Saved {len(comparison_summaries)} comparison summaries to "
        f"{output_paths['comparison_summary_json_path']}, "
        f"{output_paths['comparison_summary_csv_path']}, and "
        f"{output_paths['sweep_summary_csv_path']}."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
