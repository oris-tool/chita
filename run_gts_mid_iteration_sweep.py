import argparse
import csv
import os
import re
import traceback
from dataclasses import dataclass
from datetime import datetime

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

from tqdm import tqdm

import sweep_pipeline as sp
import sweep_pipeline_n_iteration as iteration_sweep


DATASET_NOISE_DIR = "dataset_noise"
GROUND_TRUTH_DIR = "ground_truth"
OBSERVED_ONE_RUN_DIR = "observed_one_run"
LEGACY_PARAMETERS_JSON_PATH = "parameters.json"
DEFAULT_GTS_JAVA_ITERATIONS = tuple(range(0, 9))


@dataclass(frozen=True)
class GtsDatasetSpec:
    dataset_name: str
    dataset_dir: str
    dataset_path: str
    clean_dataset_path: str
    contact_noise_summary_path: str | None
    observed_one_run_summary_path: str | None
    observed_simulated_path: str
    ground_truth_path: str
    clean_ground_truth_observed_simulated_path: str | None
    n_subjects: int
    time_limit_days: int
    total_internal_contacts: int
    dataset_events: int
    dataset_test_events: int
    ground_truth_iterations: int | None
    ground_truth_summary_path: str | None
    ground_truth_summary: dict | None


def read_json(path):
    return sp.read_json(path)


def legacy_parameter_family_key(family):
    return {
        "infectiousness": "infectiousness",
        "healing": "healing",
        "symptoms": "symptoms",
        "isolating": "isolating",
        "symptomsOnset": "symptomsOnset",
        "notificationToIsolation": "notification_to_isolation",
        "symptomaticPeriod": "symptomatic_period",
    }[family]


def legacy_parameter_unit_measure(family):
    if family in {"notificationToIsolation", "symptomaticPeriod"}:
        return "days"
    return "hours"


def convert_legacy_parameter_transition(family, transition):
    if family == "symptoms":
        true_probability = float(transition["p"])
        return {
            "unit_measure": "probability",
            "true": true_probability,
            "false": 1.0 - true_probability,
        }

    if family == "notificationToIsolation":
        p1 = float(transition["p"])
        return {
            "unit_measure": legacy_parameter_unit_measure(family),
            "distribution": "hyperexponential",
            "p1": p1,
            "p2": 1.0 - p1,
            "lambda1": float(transition["lambda1"]),
            "lambda2": float(transition["lambda2"]),
        }

    return {
        "unit_measure": legacy_parameter_unit_measure(family),
        "erlang_stages": int(transition["n"]),
        "erlang_lambda": float(transition["lambdaErl"]),
        "exponential_lambda": float(transition["lambdaExp"]),
    }


def load_parameter_space_from_legacy_json(path):
    payload = read_json(path)
    transitions = {}
    for family in sp.PARAMETER_FAMILY_ORDER:
        legacy_key = legacy_parameter_family_key(family)
        transitions[family] = {}
        for level in sp.PARAMETER_LEVEL_ORDER:
            transitions[family][level] = convert_legacy_parameter_transition(
                family,
                payload[legacy_key][level],
            )

    return {
        "source_path": os.path.abspath(path),
        "source_format": "legacy_json",
        "unit_measure": "mixed",
        "levels": list(sp.PARAMETER_LEVEL_ORDER),
        "transitions": transitions,
    }


def load_mid_parameter_bundle(parameter_ods_path, repo_root):
    ods_path = os.path.abspath(parameter_ods_path)
    if os.path.exists(ods_path):
        parameter_space = sp.load_parameter_space_from_ods(ods_path)
    else:
        legacy_path = os.path.join(repo_root, LEGACY_PARAMETERS_JSON_PATH)
        if not os.path.exists(legacy_path):
            raise FileNotFoundError(
                f"Parameter spreadsheet not found: {ods_path}. "
                f"Fallback legacy parameter JSON not found: {legacy_path}"
            )
        parameter_space = load_parameter_space_from_legacy_json(legacy_path)
    return parameter_space, sp.resolve_uniform_parameter_bundle(
        parameter_space,
        sp.GROUND_TRUTH_PARAMETER_LEVEL,
    )


def _json_files(directory):
    if not os.path.isdir(directory):
        return []
    return [
        os.path.join(directory, name)
        for name in sorted(os.listdir(directory))
        if name.endswith(".json") and os.path.isfile(os.path.join(directory, name))
    ]


def _candidate_paths(raw_path, repo_root, gts_root, summary_dir):
    if raw_path is None:
        return []

    raw_path = str(raw_path)
    candidates = []
    if os.path.isabs(raw_path):
        candidates.append(raw_path)
    else:
        candidates.extend(
            [
                os.path.join(repo_root, raw_path),
                os.path.join(gts_root, raw_path),
                os.path.join(summary_dir, raw_path),
            ]
        )

    seen = set()
    unique_candidates = []
    for candidate in candidates:
        normalized = os.path.abspath(candidate)
        if normalized not in seen:
            seen.add(normalized)
            unique_candidates.append(normalized)
    return unique_candidates


def resolve_referenced_path(raw_path, repo_root, gts_root, summary_dir):
    candidates = _candidate_paths(raw_path, repo_root, gts_root, summary_dir)
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    basename = os.path.basename(str(raw_path))
    if basename:
        basename_matches = []
        for root, _dirs, files in os.walk(gts_root):
            if basename in files:
                basename_matches.append(os.path.abspath(os.path.join(root, basename)))
        if len(basename_matches) == 1:
            return basename_matches[0]

    searched = ", ".join(candidates) if candidates else "<no candidates>"
    raise FileNotFoundError(f"Could not resolve referenced path '{raw_path}'. Searched: {searched}")


def try_resolve_referenced_path(raw_path, repo_root, gts_root, summary_dir):
    try:
        return resolve_referenced_path(raw_path, repo_root, gts_root, summary_dir)
    except FileNotFoundError:
        return None


def validate_dataset_payload(dataset_path):
    payload = read_json(dataset_path)
    required = ("events", "n_subjects", "time_limit", "n_contacts")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Dataset JSON {dataset_path} is missing required keys: {', '.join(missing)}")
    if not isinstance(payload["events"], list):
        raise ValueError(f"Dataset JSON {dataset_path} has non-list 'events'.")
    return payload


def _simulated_reps_count(path):
    match = re.search(r"_simulated_(\d+)_reps\.json$", os.path.basename(path))
    if match is None:
        return None
    return int(match.group(1))


def select_ground_truth_path(dataset_name, dataset_dir):
    candidates = []
    for path in _json_files(dataset_dir):
        reps = _simulated_reps_count(path)
        if reps is not None:
            candidates.append((reps, path))
    if candidates:
        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates[-1][1], candidates[-1][0]

    simulated_candidates = [
        path
        for path in _json_files(dataset_dir)
        if path.endswith("_simulated.json")
    ]
    if simulated_candidates:
        return sorted(simulated_candidates)[0], None

    raise FileNotFoundError(
        f"Dataset '{dataset_name}' is missing a precomputed ground-truth file matching "
        f"'*_simulated_*_reps.json' or '*_simulated.json' under {dataset_dir}."
    )


def select_clean_observed_simulated_path(dataset_dir, ground_truth_path):
    candidates = [
        path
        for path in _json_files(dataset_dir)
        if path.endswith("_simulated.json") and os.path.abspath(path) != os.path.abspath(ground_truth_path)
    ]
    if not candidates:
        return None
    return sorted(candidates)[0]


def select_clean_dataset_path(dataset_name, dataset_dir):
    exact_path = os.path.join(dataset_dir, f"{dataset_name}.json")
    if os.path.exists(exact_path):
        return exact_path

    candidates = [
        path
        for path in _json_files(dataset_dir)
        if "_simulated" not in os.path.basename(path)
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"Dataset '{dataset_name}' has no raw dataset JSON under {dataset_dir}.")
    raise ValueError(
        f"Dataset '{dataset_name}' has multiple raw dataset JSON candidates under {dataset_dir}: "
        + ", ".join(os.path.basename(path) for path in candidates)
    )


def load_optional_summary(path):
    if path is None or not os.path.exists(path):
        return None
    return read_json(path)


def find_ground_truth_summary(repo_root, gts_root, dataset_name, dataset_dir):
    suffix = dataset_name.removeprefix("dataset_")
    candidate_paths = [
        os.path.join(gts_root, f"gt_results_{suffix}.json"),
        os.path.join(gts_root, f"{dataset_name}_gt_results.json"),
        os.path.join(dataset_dir, "gt_results.json"),
        os.path.join(dataset_dir, f"gt_results_{suffix}.json"),
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return os.path.abspath(path), read_json(path)
    return None, None


def discover_gts_datasets(gts_root, repo_root=None):
    repo_root = os.path.abspath(repo_root or os.path.dirname(__file__))
    gts_root = os.path.abspath(gts_root)
    ground_truth_root = os.path.join(gts_root, GROUND_TRUTH_DIR)
    dataset_noise_root = os.path.join(gts_root, DATASET_NOISE_DIR)
    observed_root = os.path.join(gts_root, OBSERVED_ONE_RUN_DIR)

    if not os.path.isdir(gts_root):
        raise FileNotFoundError(f"GTs root not found: {gts_root}")
    if not os.path.isdir(ground_truth_root):
        raise FileNotFoundError(
            "No GT datasets were found. Expected precomputed inputs under "
            f"{ground_truth_root}/<dataset_name>/ plus {DATASET_NOISE_DIR}/ and {OBSERVED_ONE_RUN_DIR}/ summaries."
        )

    dataset_dirs = [
        os.path.join(ground_truth_root, name)
        for name in sorted(os.listdir(ground_truth_root))
        if os.path.isdir(os.path.join(ground_truth_root, name))
    ]
    if not dataset_dirs:
        raise FileNotFoundError(
            "No GT datasets were found. Expected directories like "
            f"{ground_truth_root}/dataset_scale_free_2500 containing a raw dataset JSON, "
            "a '*_simulated.json' observed file, and a '*_simulated_*_reps.json' ground-truth file."
        )

    specs = []
    for dataset_dir in dataset_dirs:
        dataset_name = os.path.basename(dataset_dir)
        contact_noise_summary_path = os.path.join(
            dataset_noise_root,
            f"{dataset_name}_contact_noise_summary.json",
        )
        observed_one_run_summary_path = os.path.join(
            observed_root,
            f"{dataset_name}_contact_noisy_one_run_summary.json",
        )
        contact_noise_summary = load_optional_summary(contact_noise_summary_path)
        observed_one_run_summary = load_optional_summary(observed_one_run_summary_path)
        gt_summary_path, gt_summary = find_ground_truth_summary(repo_root, gts_root, dataset_name, dataset_dir)

        clean_dataset_path = select_clean_dataset_path(dataset_name, dataset_dir)
        ground_truth_path, reps_count = select_ground_truth_path(dataset_name, dataset_dir)
        clean_observed_path = select_clean_observed_simulated_path(dataset_dir, ground_truth_path)

        dataset_path = clean_dataset_path
        observed_simulated_path = clean_observed_path
        if contact_noise_summary is not None:
            resolved_dataset_path = try_resolve_referenced_path(
                contact_noise_summary["noisy_dataset_path"],
                repo_root,
                gts_root,
                os.path.dirname(contact_noise_summary_path),
            )
            if resolved_dataset_path is not None:
                dataset_path = resolved_dataset_path
        if observed_one_run_summary is not None:
            resolved_dataset_path = try_resolve_referenced_path(
                observed_one_run_summary["dataset_path"],
                repo_root,
                gts_root,
                os.path.dirname(observed_one_run_summary_path),
            )
            if resolved_dataset_path is not None:
                dataset_path = resolved_dataset_path
            resolved_observed_path = try_resolve_referenced_path(
                observed_one_run_summary["observed_simulated_path"],
                repo_root,
                gts_root,
                os.path.dirname(observed_one_run_summary_path),
            )
            if resolved_observed_path is not None:
                observed_simulated_path = resolved_observed_path
        if gt_summary is not None:
            resolved_ground_truth_path = try_resolve_referenced_path(
                gt_summary.get("averaged_results_path"),
                repo_root,
                gts_root,
                os.path.dirname(gt_summary_path),
            )
            if resolved_ground_truth_path is not None:
                ground_truth_path = resolved_ground_truth_path
            resolved_clean_observed_path = try_resolve_referenced_path(
                gt_summary.get("observed_simulated_path"),
                repo_root,
                gts_root,
                os.path.dirname(gt_summary_path),
            )
            if resolved_clean_observed_path is not None:
                clean_observed_path = resolved_clean_observed_path
                if observed_simulated_path is None:
                    observed_simulated_path = resolved_clean_observed_path
            reps_count = gt_summary.get("rep_done", reps_count)

        if observed_simulated_path is None:
            observed_simulated_path = ground_truth_path

        dataset_payload = validate_dataset_payload(dataset_path)
        tests_count = sum(1 for event in dataset_payload["events"] if event.get("type") == "Test")
        specs.append(
            GtsDatasetSpec(
                dataset_name=dataset_name,
                dataset_dir=os.path.abspath(dataset_dir),
                dataset_path=os.path.abspath(dataset_path),
                clean_dataset_path=os.path.abspath(clean_dataset_path),
                contact_noise_summary_path=(
                    os.path.abspath(contact_noise_summary_path)
                    if os.path.exists(contact_noise_summary_path)
                    else None
                ),
                observed_one_run_summary_path=(
                    os.path.abspath(observed_one_run_summary_path)
                    if os.path.exists(observed_one_run_summary_path)
                    else None
                ),
                observed_simulated_path=os.path.abspath(observed_simulated_path),
                ground_truth_path=os.path.abspath(ground_truth_path),
                clean_ground_truth_observed_simulated_path=(
                    os.path.abspath(clean_observed_path) if clean_observed_path is not None else None
                ),
                n_subjects=int(dataset_payload["n_subjects"]),
                time_limit_days=int(dataset_payload["time_limit"]),
                total_internal_contacts=int(dataset_payload["n_contacts"]),
                dataset_events=len(dataset_payload["events"]),
                dataset_test_events=tests_count,
                ground_truth_iterations=(int(reps_count) if reps_count is not None else None),
                ground_truth_summary_path=gt_summary_path,
                ground_truth_summary=gt_summary,
            )
        )
    return specs


def validate_java_iterations_for_specs(java_iterations_values, specs):
    for spec in specs:
        too_large = [value for value in java_iterations_values if value > spec.n_subjects]
        if too_large:
            raise ValueError(
                f"--java-iterations contains values larger than n_subjects={spec.n_subjects} "
                f"for dataset '{spec.dataset_name}': {too_large}"
            )


def build_shared_ground_truth(spec, output_root, seed, time_step_hours, ground_truth_parameter_bundle, include_moving_avg_metrics):
    shared_dir = sp.ensure_dir(os.path.join(output_root, "_shared_ground_truth", spec.dataset_name))
    ground_truth_parameter_bundle_path = os.path.join(shared_dir, "ground_truth_parameter_bundle.json")
    sp.write_json(ground_truth_parameter_bundle_path, ground_truth_parameter_bundle)

    convergence_threshold = None
    convergence_scores = []
    convergence_reached = True
    actual_runtime_seconds = None
    dkw_csv_path = None
    if spec.ground_truth_summary:
        convergence_threshold = spec.ground_truth_summary.get("convergence_threshold")
        convergence_scores = spec.ground_truth_summary.get("convergence_scores", [])
        convergence_reached = spec.ground_truth_summary.get("convergence_reached", True)
        actual_runtime_seconds = spec.ground_truth_summary.get("actual_runtime_seconds")
        dkw_csv_path = spec.ground_truth_summary.get("dkw_csv_path")

    return {
        "run_name": f"shared_ground_truth_{spec.dataset_name}__{ground_truth_parameter_bundle['case_id']}",
        "run_dir": shared_dir,
        "time_limit": spec.time_limit_days,
        "n_subjects": spec.n_subjects,
        "total_internal_contacts": spec.total_internal_contacts,
        "seed": seed,
        "time_step_hours": time_step_hours,
        "dataset_source": "gts",
        "dataset_generation_method": "precomputed_gts",
        "moving_avg_metrics_enabled": include_moving_avg_metrics,
        "dataset_path": spec.dataset_path,
        "dataset_events": spec.dataset_events,
        "tests_enabled": spec.dataset_test_events > 0,
        "dataset_test_events": spec.dataset_test_events,
        "effective_dataset_path": spec.dataset_path,
        "pruned_dataset_path": None,
        "positive_test_pruning": None,
        "ground_truth_parameter_case_id": ground_truth_parameter_bundle["case_id"],
        "ground_truth_parameter_levels": ground_truth_parameter_bundle["levels"],
        "ground_truth_parameter_unit_measure": ground_truth_parameter_bundle["unit_measure"],
        "ground_truth_parameter_bundle_path": ground_truth_parameter_bundle_path,
        "status": "completed",
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "gts_dataset_name": spec.dataset_name,
        "gts_clean_dataset_path": spec.clean_dataset_path,
        "gts_contact_noise_summary_path": spec.contact_noise_summary_path,
        "gts_observed_one_run_summary_path": spec.observed_one_run_summary_path,
        "gts_ground_truth_summary_path": spec.ground_truth_summary_path,
        "convergence": {
            "threshold": convergence_threshold,
            "iterations": spec.ground_truth_iterations,
            "reached": convergence_reached,
            "scores": convergence_scores,
            "ground_truth_path": spec.ground_truth_path,
            "observed_simulated_path": spec.observed_simulated_path,
            "effective_dataset_path": spec.dataset_path,
            "pruned_dataset_path": None,
            "granularity": time_step_hours,
            "time_step_hours": time_step_hours,
            "actual_runtime_seconds": actual_runtime_seconds,
            "suppressed_stdout_log_path": None,
            "dkw_csv_path": dkw_csv_path,
            "parameterization": "fixed_mid_bundle",
            "parameter_case_id": ground_truth_parameter_bundle["case_id"],
            "parameter_levels": ground_truth_parameter_bundle["levels"],
            "parameter_bundle_path": ground_truth_parameter_bundle_path,
        },
    }


def write_gts_run_readme(output_root, gts_root, specs, args, parameter_manifest_paths):
    lines = [
        "# GTs Mid Iteration Sweep",
        "",
        "This run evaluates only the all-mid parameter bundle against precomputed datasets under `GTs`.",
        "",
        "## Configuration",
        "",
        f"- `gts_root`: {os.path.abspath(gts_root)}",
        f"- `java_iterations`: {list(args.java_iterations)}",
        f"- `time_step_hours`: {args.time_step_hours}",
        f"- `baseline_runtime_multiplier`: {args.baseline_runtime_multiplier}",
        f"- `observed_test_ablation`: {args.observed_test_ablation}",
        f"- `moving_avg_metrics_enabled`: {not args.disable_moving_avg_metrics}",
        f"- `generate_plot_images`: {not args.skip_plot_images}",
        f"- `parameter_manifest_csv`: {parameter_manifest_paths['csv_path']}",
        f"- `parameter_manifest_json`: {parameter_manifest_paths['json_path']}",
        "",
        "## Datasets",
        "",
    ]
    for spec in specs:
        lines.extend(
            [
                f"### {spec.dataset_name}",
                "",
                f"- `dataset_path`: {spec.dataset_path}",
                f"- `ground_truth_path`: {spec.ground_truth_path}",
                f"- `observed_simulated_path`: {spec.observed_simulated_path}",
                f"- `contact_noise_summary_path`: {spec.contact_noise_summary_path}",
                f"- `observed_one_run_summary_path`: {spec.observed_one_run_summary_path}",
                "",
            ]
        )
    output_path = os.path.join(output_root, "gts_mid_iteration_sweep.md")
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return output_path


def dry_run_payload(specs):
    return [
        {
            "dataset_name": spec.dataset_name,
            "dataset_path": spec.dataset_path,
            "ground_truth_path": spec.ground_truth_path,
            "observed_simulated_path": spec.observed_simulated_path,
            "n_subjects": spec.n_subjects,
            "time_limit_days": spec.time_limit_days,
            "total_internal_contacts": spec.total_internal_contacts,
            "ground_truth_iterations": spec.ground_truth_iterations,
        }
        for spec in specs
    ]


def write_gts_aggregate_outputs(output_root, summaries):
    rows = iteration_sweep.write_aggregate_outputs(output_root, summaries)
    sp.write_json(os.path.join(output_root, "iteration_sweep_summary.json"), rows)
    with open(os.path.join(output_root, "iteration_sweep_summary.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=iteration_sweep.COMPACT_RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Run the Java-iteration sweep for the all-mid parameter case on precomputed GTs datasets."
    )
    parser.add_argument("--gts-root", default="GTs", help="Root directory containing precomputed GT datasets.")
    parser.add_argument(
        "--output-root",
        default=os.path.join("sweeps", datetime.now().strftime("gts_mid_iteration_sweep_%Y%m%d_%H%M%S")),
        help="Directory where sweep outputs will be created.",
    )
    parser.add_argument("--seed-base", type=int, default=1000, help="Base seed used to derive per-dataset seeds.")
    parser.add_argument(
        "--java-iterations",
        nargs="+",
        type=int,
        default=list(DEFAULT_GTS_JAVA_ITERATIONS),
        help="Logical Java iteration counts to test. Default: 0 1 2 3 4 5 6 7 8",
    )
    parser.add_argument(
        "--parameter-ods-path",
        default=sp.DEFAULT_PARAMETER_ODS_PATH,
        help="ODS spreadsheet containing the lower/mid/upper transition parameters.",
    )
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
        "--observed-test-ablation",
        choices=sp.OBSERVED_TEST_ABLATION_CHOICES,
        default=sp.OBSERVED_TEST_ABLATION_NONE,
        help="Optional filtering applied to the observed_simulated.json copied into each run.",
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
        "--dry-run",
        action="store_true",
        help="Only discover and validate GT datasets; do not create output files or run experiments.",
    )
    args = parser.parse_args()

    if args.time_step_hours <= 0:
        parser.exit(1, "Error: --time-step-hours must be greater than 0.\n")
    if args.baseline_runtime_multiplier < 0:
        parser.exit(1, "Error: --baseline-runtime-multiplier must be greater than or equal to 0.\n")

    repo_root = os.path.abspath(os.path.dirname(__file__))
    gts_root = os.path.abspath(args.gts_root)
    try:
        java_iterations_values = iteration_sweep.parse_java_iterations(args.java_iterations)
        specs = discover_gts_datasets(gts_root, repo_root=repo_root)
        validate_java_iterations_for_specs(java_iterations_values, specs)
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")

    if args.dry_run:
        print(f"Discovered {len(specs)} GT dataset(s).")
        for spec in specs:
            print(
                f"- {spec.dataset_name}: n_subjects={spec.n_subjects}, "
                f"time_limit={spec.time_limit_days}, contacts={spec.total_internal_contacts}"
            )
        return

    output_root = sp.ensure_dir(os.path.abspath(args.output_root))
    parameter_space, mid_bundle = load_mid_parameter_bundle(args.parameter_ods_path, repo_root)
    parameter_manifest_paths = sp.write_parameter_case_manifest(output_root, [mid_bundle])
    write_gts_run_readme(output_root, gts_root, specs, args, parameter_manifest_paths)

    summaries = []
    with tqdm(total=len(specs), desc="GTs mid iteration sweep", unit="dataset") as progress_bar:
        for dataset_index, spec in enumerate(specs):
            progress_bar.set_postfix_str(spec.dataset_name)
            baseline_seed = args.seed_base + dataset_index + 1
            try:
                dataset_output_root = sp.ensure_dir(os.path.join(output_root, spec.dataset_name))
                shared_ground_truth = build_shared_ground_truth(
                    spec=spec,
                    output_root=dataset_output_root,
                    seed=args.seed_base + dataset_index,
                    time_step_hours=args.time_step_hours,
                    ground_truth_parameter_bundle=mid_bundle,
                    include_moving_avg_metrics=not args.disable_moving_avg_metrics,
                )
                summary = iteration_sweep.run_parameter_bundle_iteration_sweep(
                    repo_root=repo_root,
                    output_root=dataset_output_root,
                    time_limit_days=spec.time_limit_days,
                    n_subjects=spec.n_subjects,
                    total_internal_contacts=spec.total_internal_contacts,
                    baseline_seed=baseline_seed,
                    time_step_hours=args.time_step_hours,
                    baseline_runtime_multiplier=args.baseline_runtime_multiplier,
                    observed_test_ablation=args.observed_test_ablation,
                    include_moving_avg_metrics=not args.disable_moving_avg_metrics,
                    java_iterations_values=java_iterations_values,
                    parameter_bundle=mid_bundle,
                    ground_truth_parameter_bundle=mid_bundle,
                    shared_ground_truth=shared_ground_truth,
                    save_plots=not args.skip_plot_images,
                )
                summary["gts_dataset_name"] = spec.dataset_name
            except Exception as exc:
                summary = iteration_sweep.build_failed_bundle_summary(
                    output_root=sp.ensure_dir(os.path.join(output_root, spec.dataset_name)),
                    time_limit_days=spec.time_limit_days,
                    n_subjects=spec.n_subjects,
                    total_internal_contacts=spec.total_internal_contacts,
                    baseline_seed=baseline_seed,
                    time_step_hours=args.time_step_hours,
                    baseline_runtime_multiplier=args.baseline_runtime_multiplier,
                    dataset_source="gts",
                    observed_test_ablation=args.observed_test_ablation,
                    parameter_bundle=mid_bundle,
                    ground_truth_parameter_bundle=mid_bundle,
                    java_iterations_values=java_iterations_values,
                    error_message=str(exc),
                    traceback_text=traceback.format_exc(),
                )
                summary["gts_dataset_name"] = spec.dataset_name

            summaries.append(summary)
            write_gts_aggregate_outputs(output_root, summaries)
            progress_bar.update(1)

    write_gts_aggregate_outputs(output_root, summaries)
    sp.write_json(
        os.path.join(output_root, "gts_dataset_manifest.json"),
        dry_run_payload(specs),
    )
    print(f"GTs mid iteration sweep completed. Outputs: {output_root}")


if __name__ == "__main__":
    main()
