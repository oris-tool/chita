import argparse
import contextlib
import copy
import csv
import io
import itertools
import json
import math
import os
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Optional, Tuple

from tqdm import tqdm

import dataset_graph as dg
import scale_free_dataset_graph as sfdg
import sweep_pipeline as sp
from compute_precision_metrics import process_and_save
from run_n_simulations import run_dataset_simulations
from utils import normalize_stpn_parameter_bundle, precompute_stpn_solution, run_stpn_analysis


# 0. Define paths and dataset parameters
TIME_STEP = 1  # hours
INTERNAL_STEPS = 2  # 0 means just external contacts, 1 means external + internal, 2 means one more propagation layer
QUANTILE = 4

TIME_LIMIT_HOURS = 2016.0
DATASET_FAMILY_BUBBLE = "bubble"
DATASET_FAMILY_SCALE_FREE = "scale_free"

RUN_DIR_PREFIX = "sweep_"
RUN_COMPLETION_SENTINEL = "_run_completed.json"
PROGRESS_DIR_NAME = "_progress"
DEFAULT_MAX_WORKERS = max(1, (os.cpu_count() or 1) // 2)
SAVE_PRECISION_PLOTS = os.environ.get("CHITA_SAVE_PRECISION_PLOTS", "0") == "1"
LOG_PRECISION_METRICS = os.environ.get("CHITA_LOG_PRECISION_METRICS", "0") == "1"
SAVE_COMPARISON_CSVS = os.environ.get("CHITA_SAVE_COMPARISON_CSVS", "1") == "1"
CASE_INDEX_BASE = 0
METRIC_SELECTION_BUCKET_SIZE = 10
NOISE_FRACTION = 0.05
NOISE_TIME_SHIFT_HOURS = 6.0
NOISE_SEED = 30
CONTACT_NOISE_EVENT_TYPES = ("Internal", "External")
OBSERVATION_NOISE_EVENT_TYPES = ("Test", "Symptoms")

LEVEL_LABELS_IT = {
    "lower": "basso",
    "mid": "medio",
    "upper": "alto",
}

REFERENCE_SWEEP_COLUMNS = [
    "run_id",
    "parameter_case_code",
    "parameter_case_id",
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


@dataclass(frozen=True)
class DatasetProfile:
    family: str
    generator_module_name: str
    n_subjects: int
    time_limit_hours: float
    internal_contacts: Tuple[int, ...]
    effective_external_contacts: int
    total_external_contacts: Optional[int] = None
    total_symptom_observations: Optional[int] = None
    total_test_observations: Optional[int] = None
    barabasi_m: Optional[int] = None


DATASET_PROFILES = {
    DATASET_FAMILY_BUBBLE: DatasetProfile(
        family=DATASET_FAMILY_BUBBLE,
        generator_module_name="dataset_graph",
        n_subjects=8,
        time_limit_hours=TIME_LIMIT_HOURS,
        internal_contacts=(32, 200, 400, 800),
        effective_external_contacts=3,
    ),
    DATASET_FAMILY_SCALE_FREE: DatasetProfile(
        family=DATASET_FAMILY_SCALE_FREE,
        generator_module_name="scale_free_dataset_graph",
        n_subjects=100,
        time_limit_hours=TIME_LIMIT_HOURS,
        internal_contacts=(32, 2500, 5000, 10000),
        effective_external_contacts=15,
        total_external_contacts=1000,
        total_symptom_observations=1000,
        total_test_observations=1000,
        barabasi_m=3,
    ),
}


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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the final CHITA sweep pipeline.")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_PROFILES.keys()),
        default=DATASET_FAMILY_BUBBLE,
        help="Dataset family to generate and run through the sweep pipeline.",
    )
    parser.add_argument(
        "--reuse-run",
        default=None,
        help="Existing results/sweep_* directory to reuse.",
    )
    parser.add_argument(
        "--only-selected-plots",
        action="store_true",
        help=(
            "Reuse an existing sweep directory, recompute the per-dataset top/worst/median "
            "selection, and regenerate plots only for the selected runs."
        ),
    )
    parser.add_argument(
        "--quartile-label",
        default=None,
        help="Optional quartile label to use with --only-selected-plots, for example q4.",
    )
    return parser.parse_args(argv)


def resolve_dataset_profile(dataset_family):
    try:
        return DATASET_PROFILES[dataset_family]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset family: {dataset_family}") from exc


def resolve_existing_run_path(run_path):
    if not run_path:
        raise ValueError("An existing run path is required.")
    resolved_path = os.path.abspath(run_path)
    if not os.path.isdir(resolved_path):
        raise FileNotFoundError(f"Existing run directory not found: {resolved_path}")
    return resolved_path


def dataset_profile_metadata(dataset_profile):
    payload = asdict(dataset_profile)
    payload["internal_contacts"] = list(dataset_profile.internal_contacts)
    return payload


def dataset_filename(dataset_profile, internal_contacts):
    return f"dataset_{dataset_profile.family}_{internal_contacts}.json"


def count_events_by_type(events):
    counts = {}
    for event in events:
        event_type = event.get("type")
        counts[event_type] = counts.get(event_type, 0) + 1
    return counts


def summarize_generated_dataset(dataset_profile, dataset_path, payload, requested_internal_contacts):
    event_counts = count_events_by_type(payload.get("events", []))
    return {
        "dataset_family": dataset_profile.family,
        "dataset_path": dataset_path,
        "dataset_stem": os.path.splitext(os.path.basename(dataset_path))[0],
        "generator_module_name": dataset_profile.generator_module_name,
        "requested_internal_contacts": requested_internal_contacts,
        "n_subjects": payload.get("n_subjects"),
        "time_limit_days": payload.get("time_limit"),
        "n_contacts": payload.get("n_contacts"),
        "event_counts": event_counts,
    }


def generate_dataset_for_profile(dataset_profile, dataset_path, internal_contacts, seed):
    if dataset_profile.family == DATASET_FAMILY_BUBBLE:
        dataset = dg.simulate_external_introduction(
            n_nodes=dataset_profile.n_subjects,
            total_internal_contacts=internal_contacts,
            tmax_after_intro=dataset_profile.time_limit_hours,
            effective_external_contacts=dataset_profile.effective_external_contacts,
            seed=seed,
        )
        payload = dg.save_dataset_event_sequence(dataset, dataset_path)
    elif dataset_profile.family == DATASET_FAMILY_SCALE_FREE:
        dataset = sfdg.simulate_scale_free_introduction(
            n_nodes=dataset_profile.n_subjects,
            total_internal_contacts=internal_contacts,
            total_external_contacts=dataset_profile.total_external_contacts,
            total_symptom_observations=dataset_profile.total_symptom_observations,
            total_test_observations=dataset_profile.total_test_observations,
            tmax_after_intro=dataset_profile.time_limit_hours,
            effective_external_contacts=dataset_profile.effective_external_contacts,
            barabasi_m=dataset_profile.barabasi_m,
            seed=seed,
        )
        payload = sfdg.save_dataset_event_sequence(dataset, dataset_path)
    else:
        raise ValueError(f"Unsupported dataset family: {dataset_profile.family}")

    return summarize_generated_dataset(
        dataset_profile=dataset_profile,
        dataset_path=dataset_path,
        payload=payload,
        requested_internal_contacts=internal_contacts,
    )


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


def format_case_run_code(case_index):
    return str(int(case_index))


def resolve_case_output_dir(base_dir, parameter_case):
    preferred_dir = os.path.join(base_dir, parameter_case["case_run_code"])
    legacy_dir = os.path.join(base_dir, parameter_case["case_id"])
    if os.path.exists(preferred_dir):
        return preferred_dir
    if os.path.exists(legacy_dir):
        return legacy_dir
    return preferred_dir


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


def run_directory_matches_dataset_family(run_dir, dataset_family):
    metadata_path = os.path.join(run_dir, "run_metadata.json")
    if not os.path.exists(metadata_path):
        return dataset_family == DATASET_FAMILY_BUBBLE
    try:
        metadata = read_json(metadata_path)
    except (OSError, json.JSONDecodeError):
        return False
    recorded_family = metadata.get("dataset_family")
    if recorded_family is None:
        return dataset_family == DATASET_FAMILY_BUBBLE
    return recorded_family == dataset_family


def resolve_run_directory(results_root="results", dataset_family=DATASET_FAMILY_BUBBLE):
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
            if not os.path.exists(sentinel_path) and run_directory_matches_dataset_family(
                run_dir,
                dataset_family,
            ):
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

    for case_index, combination in enumerate(combinations, start=CASE_INDEX_BASE):
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
            "case_index": case_index,
            "case_run_code": format_case_run_code(case_index),
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
    intended_path = os.path.abspath(destination)
    for root, _, filenames in os.walk(analysis_dir):
        for filename in filenames:
            if not filename.endswith("simulated.json"):
                continue
            candidate_path = os.path.abspath(os.path.join(root, filename))
            if candidate_path != intended_path:
                os.remove(candidate_path)
    return destination


def ensure_python_simulation_input(analysis_dir, dataset_path):
    ensure_dir(analysis_dir)
    destination = os.path.join(analysis_dir, os.path.basename(dataset_path))
    if os.path.abspath(destination) != os.path.abspath(dataset_path):
        shutil.copy2(dataset_path, destination)
    return destination


def ensure_ground_truth_simulation_input(save_path, dataset_path):
    dataset_stem = os.path.splitext(os.path.basename(dataset_path))[0]
    ground_truth_dir = os.path.join(save_path, "ground_truth", dataset_stem)
    ensure_dir(ground_truth_dir)
    destination = os.path.join(ground_truth_dir, os.path.basename(dataset_path))
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


def _flip_observation_result(value):
    if isinstance(value, bool):
        return not value

    if isinstance(value, (int, float)) and float(value).is_integer() and int(value) in (0, 1):
        return 1 - int(value)

    if isinstance(value, str):
        normalized = value.strip().lower()
        positive_aliases = {"positive", "pos", "true", "1"}
        negative_aliases = {"negative", "neg", "false", "0"}
        if normalized in positive_aliases:
            return "negative"
        if normalized in negative_aliases:
            return "positive"

    return None


def _shift_event_time(event, rng, time_limit_hours, time_shift_hours):
    original_time = float(event.get("time", 0.0))
    shift_delta = time_shift_hours if rng.random() < 0.5 else -time_shift_hours
    shifted_time = original_time + shift_delta
    if time_limit_hours > 0.0:
        shifted_time = min(max(0.0, shifted_time), time_limit_hours)
    else:
        shifted_time = max(0.0, shifted_time)
    event["time"] = float(shifted_time)


def _noise_operation_count(original_count, fraction):
    return max(0, int(original_count * fraction))


def _apply_event_noise_to_dataset(
    source_dataset_path,
    noisy_dataset_path,
    seed_label,
    event_types,
    flip_event_types=(),
    noise_stage="event",
    removal_fraction=NOISE_FRACTION,
    time_shift_hours=NOISE_TIME_SHIFT_HOURS,
):
    payload = read_json(source_dataset_path)
    events = copy.deepcopy(payload.get("events", []))
    rng = random.Random(f"{NOISE_SEED}:{seed_label}")
    time_limit_days = payload.get("time_limit", 0)
    time_limit_hours = max(0.0, float(time_limit_days) * 24.0)
    original_counts_by_type = {
        event_type: len([event for event in events if event.get("type") == event_type])
        for event_type in event_types
    }

    removed_indices = set()
    removed_by_type = {}
    for event_type in event_types:
        matching_indices = [index for index, event in enumerate(events) if event.get("type") == event_type]
        remove_count = min(
            len(matching_indices),
            _noise_operation_count(original_counts_by_type[event_type], removal_fraction),
        )
        if remove_count > 0:
            selected_indices = rng.sample(matching_indices, remove_count)
            removed_indices.update(selected_indices)
        else:
            selected_indices = []
        removed_by_type[event_type] = len(selected_indices)

    events = [event for index, event in enumerate(events) if index not in removed_indices]

    shifted_by_type = {}
    for event_type in event_types:
        matching_indices = [index for index, event in enumerate(events) if event.get("type") == event_type]
        shift_count = min(
            len(matching_indices),
            _noise_operation_count(original_counts_by_type[event_type], removal_fraction),
        )
        selected_indices = rng.sample(matching_indices, shift_count)
        for event_index in selected_indices:
            _shift_event_time(events[event_index], rng, time_limit_hours, time_shift_hours)
        shifted_by_type[event_type] = len(selected_indices)

    flipped_by_type = {}
    for event_type in flip_event_types:
        original_count = original_counts_by_type.get(
            event_type,
            len([event for event in payload.get("events", []) if event.get("type") == event_type]),
        )
        matching_indices = [index for index, event in enumerate(events) if event.get("type") == event_type]
        flip_count = min(len(matching_indices), _noise_operation_count(original_count, removal_fraction))
        flippable_indices = [
            index
            for index in matching_indices
            if _flip_observation_result(events[index].get("result")) is not None
        ]
        selected_indices = rng.sample(flippable_indices, min(flip_count, len(flippable_indices)))
        for event_index in selected_indices:
            events[event_index]["result"] = _flip_observation_result(events[event_index].get("result"))
        flipped_by_type[event_type] = len(selected_indices)

    events.sort(key=lambda event: float(event.get("time", 0.0)))
    payload["events"] = events
    if "n_contacts" in payload:
        payload["n_contacts"] = len([event for event in events if event.get("type") == "Internal"])

    write_json(noisy_dataset_path, payload)

    return {
        "source_dataset_path": source_dataset_path,
        "noisy_dataset_path": noisy_dataset_path,
        "noise_stage": noise_stage,
        "event_types": list(event_types),
        "flip_event_types": list(flip_event_types),
        "removal_fraction": removal_fraction,
        "time_shift_hours": time_shift_hours,
        "original_counts_by_type": original_counts_by_type,
        "removed_by_type": removed_by_type,
        "shifted_by_type": shifted_by_type,
        "flipped_by_type": flipped_by_type,
        "tests_flipped": flipped_by_type.get("Test", 0),
        "symptoms_flipped": flipped_by_type.get("Symptoms", 0),
        "seed_label": seed_label,
    }


def apply_contact_noise_to_dataset(
    source_dataset_path,
    noisy_dataset_path,
    seed_label,
    removal_fraction=NOISE_FRACTION,
    time_shift_hours=NOISE_TIME_SHIFT_HOURS,
):
    return _apply_event_noise_to_dataset(
        source_dataset_path=source_dataset_path,
        noisy_dataset_path=noisy_dataset_path,
        seed_label=seed_label,
        event_types=CONTACT_NOISE_EVENT_TYPES,
        flip_event_types=(),
        noise_stage="contact",
        removal_fraction=removal_fraction,
        time_shift_hours=time_shift_hours,
    )


def apply_observation_noise_to_dataset(
    source_dataset_path,
    noisy_dataset_path,
    seed_label,
    removal_fraction=NOISE_FRACTION,
    time_shift_hours=NOISE_TIME_SHIFT_HOURS,
):
    return _apply_event_noise_to_dataset(
        source_dataset_path=source_dataset_path,
        noisy_dataset_path=noisy_dataset_path,
        seed_label=seed_label,
        event_types=OBSERVATION_NOISE_EVENT_TYPES,
        flip_event_types=OBSERVATION_NOISE_EVENT_TYPES,
        noise_stage="observation",
        removal_fraction=removal_fraction,
        time_shift_hours=time_shift_hours,
    )


def observation_noisy_simulated_path(observed_simulated_path):
    directory = os.path.dirname(observed_simulated_path)
    basename = os.path.basename(observed_simulated_path)
    suffix = "_simulated.json"
    if basename.endswith(suffix):
        output_basename = basename[: -len(suffix)] + "_observation_noisy" + suffix
    else:
        output_basename = os.path.splitext(basename)[0] + "_observation_noisy" + suffix
    return os.path.join(directory, output_basename)


def finite_float(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def append_unique_note(notes_by_run, run_id, note_text):
    notes = notes_by_run.setdefault(run_id, [])
    if note_text not in notes:
        notes.append(note_text)


def selection_entry(item, rank, note_text):
    return {
        "rank": rank,
        "run_id": item["run_id"],
        "parameter_case_id": item.get("parameter_case_id"),
        "parameter_case_code": item.get("parameter_case_code"),
        "metric_value": item["metric_value"],
        "note": note_text,
    }


def add_metric_selection_notes(dataset_rows, metric_key, metric_label, notes_by_run):
    candidates = []
    for row in dataset_rows:
        metric_value = finite_float(row.get(metric_key))
        if metric_value is None:
            continue
        candidates.append(
            {
                "run_id": row["run_id"],
                "parameter_case_id": row.get("parameter_case_id"),
                "parameter_case_code": row.get("parameter_case_code"),
                "metric_value": metric_value,
            }
        )

    if not candidates:
        return {
            "metric_key": metric_key,
            "metric_label": metric_label,
            "total_candidates": 0,
            "bucket_size": 0,
            "top": [],
            "worst": [],
            "median": [],
        }

    bucket_size = min(METRIC_SELECTION_BUCKET_SIZE, len(candidates))
    candidates.sort(key=lambda item: (item["metric_value"], item["run_id"]))
    selection = {
        "metric_key": metric_key,
        "metric_label": metric_label,
        "total_candidates": len(candidates),
        "bucket_size": bucket_size,
        "top": [],
        "worst": [],
        "median": [],
    }

    for rank, item in enumerate(reversed(candidates[-bucket_size:]), start=1):
        note_text = f"Top {rank} {metric_label}"
        append_unique_note(notes_by_run, item["run_id"], note_text)
        selection["top"].append(selection_entry(item, rank, note_text))

    for rank, item in enumerate(candidates[:bucket_size], start=1):
        note_text = f"Worst {rank} {metric_label}"
        append_unique_note(notes_by_run, item["run_id"], note_text)
        selection["worst"].append(selection_entry(item, rank, note_text))

    median_value = percentile([item["metric_value"] for item in candidates], 0.50)
    nearest_to_median = sorted(
        candidates,
        key=lambda item: (abs(item["metric_value"] - median_value), item["run_id"]),
    )[:bucket_size]
    for rank, item in enumerate(nearest_to_median, start=1):
        note_text = f"Median {rank} {metric_label}"
        append_unique_note(notes_by_run, item["run_id"], note_text)
        selection["median"].append(selection_entry(item, rank, note_text))

    return selection


def build_notes_map_and_selection_manifest(comparison_summaries):
    rows = []
    for summary in comparison_summaries:
        analysis_metrics = summary.get("comparison_metrics", {}).get("analysis", {})
        rows.append(
            {
                "run_id": summary.get("run_id"),
                "dataset_stem": summary.get("dataset_stem"),
                "parameter_case_id": summary.get("parameter_case_id"),
                "parameter_case_code": summary.get("parameter_case_code"),
                "kendall_analysis": analysis_metrics.get("tau"),
                "spearman_analysis": analysis_metrics.get("spearman"),
            }
        )

    notes_by_run = {}
    manifest = {
        "selection_bucket_size": METRIC_SELECTION_BUCKET_SIZE,
        "datasets": {},
    }
    dataset_stems = sorted({row["dataset_stem"] for row in rows if row.get("dataset_stem") is not None})
    for dataset_stem in dataset_stems:
        dataset_rows = [row for row in rows if row.get("dataset_stem") == dataset_stem]
        manifest["datasets"][dataset_stem] = {
            "kendall": add_metric_selection_notes(
                dataset_rows,
                "kendall_analysis",
                "Kendall",
                notes_by_run,
            ),
            "spearman": add_metric_selection_notes(
                dataset_rows,
                "spearman_analysis",
                "Spearman",
                notes_by_run,
            ),
        }

    return notes_by_run, manifest


def build_notes_map_per_dataset(comparison_summaries):
    notes_by_run, _ = build_notes_map_and_selection_manifest(comparison_summaries)
    return notes_by_run


def apply_notes_to_comparison_summaries(comparison_summaries, notes_by_run):
    for summary in comparison_summaries:
        run_id = summary.get("run_id")
        notes = notes_by_run.get(run_id, [])
        summary["note"] = " - ".join(notes)


def regenerate_selected_run_plots(comparison_summary, time_step_hours, iterations):
    comparison_dir = comparison_summary["comparison_dir"]
    run_id = comparison_summary["run_id"]
    ground_truth_path = comparison_summary["ground_truth_path"]

    java_analysis = comparison_summary.get("java_analysis", {})
    python_analysis = comparison_summary.get("python_analysis", {})
    analysis_path = java_analysis.get("analysis_path")
    baseline_path = python_analysis.get("baseline_path")

    if not _is_valid_path(analysis_path):
        analysis_path = find_generated_file(comparison_dir, f"_tracks_it{iterations}.json")
    if not _is_valid_path(ground_truth_path):
        raise FileNotFoundError(f"Ground-truth path is invalid for run {run_id}: {ground_truth_path}")
    if not _is_valid_path(baseline_path):
        raise FileNotFoundError(f"Baseline path is invalid for run {run_id}: {baseline_path}")

    subject_curve_plots = sp.create_analysis_subject_curve_plots(
        run_dir=comparison_dir,
        run_name=run_id,
        analysis_path=analysis_path,
        ground_truth_path=ground_truth_path,
        baseline_path=baseline_path,
        granularity=time_step_hours,
        ground_truth_sample_size=comparison_summary.get("ground_truth_iterations"),
        save_plots=True,
    )
    comparison_metrics = sp.create_analysis_vs_simulation_plots(
        run_dir=comparison_dir,
        run_name=run_id,
        ground_truth_path=ground_truth_path,
        analysis_path=analysis_path,
        baseline_path=baseline_path,
        include_moving_avg_metrics=True,
        save_plots=True,
    )

    comparison_summary.setdefault("java_analysis", {})["analysis_path"] = analysis_path
    comparison_summary["java_analysis"]["subject_curve_plots"] = subject_curve_plots
    comparison_summary["comparison_metrics"] = comparison_metrics
    return comparison_summary


def run_precompute_task(save_path, parameter_case, time_step_hours):
    case_run_code = parameter_case["case_run_code"]
    parameter_bundle = parameter_case["parameter_bundle"]

    precompute_base_dir = os.path.join(save_path, "_precompute")
    precompute_dir = resolve_case_output_dir(precompute_base_dir, parameter_case)
    summary_path = os.path.join(precompute_dir, "precompute_summary.json")
    if os.path.exists(summary_path):
        cached_summary = read_json(summary_path)
        if _is_valid_path(cached_summary.get("stpn_solution_path")) and _is_valid_path(
            cached_summary.get("observation_curve_path")
        ):
            changed = False
            if cached_summary.get("parameter_case_code") != case_run_code:
                cached_summary["parameter_case_code"] = case_run_code
                changed = True
            if cached_summary.get("parameter_case_index") != parameter_case["case_index"]:
                cached_summary["parameter_case_index"] = parameter_case["case_index"]
                changed = True
            if changed:
                write_json(summary_path, cached_summary)
            return cached_summary

    precomputed = precompute_stpn_solution(
        parameter_bundle=parameter_bundle,
        cache_dir=None,
        repo_root=".",
        time_step_hours=time_step_hours,
    )
    summary = {
        "case_id": precomputed["parameter_bundle"]["case_id"],
        "parameter_case_code": case_run_code,
        "parameter_case_index": parameter_case["case_index"],
        "cache_dir": precomputed["cache_dir"],
        "cache_hit": precomputed["cache_hit"],
        "stpn_solution_path": precomputed["stpn_solution_path"],
        "observation_curve_path": precomputed["observation_curve_path"],
    }
    write_json(summary_path, summary)
    return summary


def run_java_analysis_task(save_path, dataset_run, parameter_case, time_step_hours, iterations):
    case_id = parameter_case["case_id"]
    case_run_code = parameter_case["case_run_code"]
    parameter_bundle = parameter_case["parameter_bundle"]

    analysis_base_dir = os.path.join(
        save_path,
        "java_analysis",
        dataset_run["dataset_stem"],
    )
    analysis_dir = resolve_case_output_dir(analysis_base_dir, parameter_case)
    summary_path = os.path.join(analysis_dir, "java_analysis_summary.json")

    if os.path.exists(summary_path):
        cached_summary = read_json(summary_path)
        analysis_path = cached_summary.get("analysis_path")
        cached_observed_path = cached_summary.get("observed_simulated_path")
        observed_path_matches = _is_valid_path(cached_observed_path) and (
            os.path.abspath(cached_observed_path) == os.path.abspath(dataset_run["observed_simulated_path"])
        )
        if not _is_valid_path(analysis_path):
            try:
                analysis_path = find_generated_file(analysis_dir, f"_tracks_it{iterations}.json")
                cached_summary["analysis_path"] = analysis_path
                write_json(summary_path, cached_summary)
            except FileNotFoundError:
                analysis_path = None

        if _is_valid_path(analysis_path) and observed_path_matches:
            changed = False
            if cached_summary.get("parameter_case_code") != case_run_code:
                cached_summary["parameter_case_code"] = case_run_code
                changed = True
            if cached_summary.get("parameter_case_index") != parameter_case["case_index"]:
                cached_summary["parameter_case_index"] = parameter_case["case_index"]
                changed = True
            if changed:
                write_json(summary_path, cached_summary)
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
        "parameter_case_code": case_run_code,
        "parameter_case_index": parameter_case["case_index"],
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
    case_run_code = parameter_case["case_run_code"]
    parameter_bundle = parameter_case["parameter_bundle"]

    analysis_base_dir = os.path.join(
        save_path,
        "python_analysis",
        quartile_label,
        dataset_run["dataset_stem"],
    )
    analysis_dir = resolve_case_output_dir(analysis_base_dir, parameter_case)
    summary_path = os.path.join(analysis_dir, "python_analysis_summary.json")

    if os.path.exists(summary_path):
        cached_summary = read_json(summary_path)
        cached_runtime_budget = cached_summary.get("runtime_budget_seconds")
        runtime_budget_matches = isinstance(cached_runtime_budget, (int, float)) and (
            abs(float(cached_runtime_budget) - float(runtime_budget_seconds)) <= 1e-9
        )
        cached_dataset_path = cached_summary.get("dataset_path")
        dataset_path_matches = _is_valid_path(cached_dataset_path) and (
            os.path.abspath(cached_dataset_path) == os.path.abspath(dataset_run["dataset_path"])
        )
        if (
            _is_valid_path(cached_summary.get("averaged_results_path"))
            and runtime_budget_matches
            and dataset_path_matches
        ):
            changed = False
            if cached_summary.get("parameter_case_code") != case_run_code:
                cached_summary["parameter_case_code"] = case_run_code
                changed = True
            if cached_summary.get("parameter_case_index") != parameter_case["case_index"]:
                cached_summary["parameter_case_index"] = parameter_case["case_index"]
                changed = True
            if cached_summary.get("runtime_budget_seconds") != runtime_budget_seconds:
                cached_summary["runtime_budget_seconds"] = runtime_budget_seconds
                changed = True
            if changed:
                write_json(summary_path, cached_summary)
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
        "parameter_case_code": case_run_code,
        "parameter_case_index": parameter_case["case_index"],
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


def compute_analysis_vs_simulation_metrics_only(
    ground_truth_path,
    analysis_path,
    baseline_path,
    run_dir=None,
    write_csvs=False,
):
    ground_truth = sort_subject_tracks(sp.read_json(ground_truth_path))
    analysis = sort_subject_tracks(sp.read_json(analysis_path))
    baseline = sort_subject_tracks(sp.read_json(baseline_path))
    output_dir = None
    csv_summaries = None
    analysis_summary = compute_candidate_summary_fast(
        ground_truth,
        analysis,
        include_moving_avg_metrics=True,
    )
    baseline_summary = compute_candidate_summary_fast(
        ground_truth,
        baseline,
        include_moving_avg_metrics=True,
    )
    if write_csvs:
        output_dir = ensure_dir(os.path.join(run_dir, "plots", "analysis_vs_simulation"))
        csv_summaries = write_analysis_vs_simulation_csvs_fast(
            output_dir=output_dir,
            ground_truth=ground_truth,
            analysis=analysis,
            baseline=baseline,
            analysis_summary=analysis_summary,
            baseline_summary=baseline_summary,
        )
    return {
        "analysis": analysis_summary,
        "simulation": baseline_summary,
        "output_dir": output_dir,
        "plot_paths": [],
        "csv_paths": [] if csv_summaries is None else csv_summaries["csv_paths"],
    }


def sort_subject_tracks(tracks):
    return {subject_id: tracks[subject_id] for subject_id in sorted(tracks, key=int)}


def subject_track_matrix(tracks):
    subjects = list(tracks.keys())
    if not subjects:
        return subjects, sp.np.empty((0, 0), dtype=float)
    matrix = sp.np.asarray([tracks[subject_id] for subject_id in subjects], dtype=float).T
    return subjects, matrix


def rank_rows_descending(matrix):
    ranks = sp.np.empty(matrix.shape, dtype=float)
    for row_index, row in enumerate(matrix):
        order = sp.np.argsort(-row, kind="mergesort")
        sorted_values = row[order]
        position = 0
        while position < len(order):
            end = position + 1
            while end < len(order) and sorted_values[end] == sorted_values[position]:
                end += 1
            average_rank = (position + 1 + end) / 2.0
            ranks[row_index, order[position:end]] = average_rank
            position = end
    return ranks


def spearman_per_timestep_fast(ground_truth_matrix, candidate_matrix):
    ground_truth_ranks = rank_rows_descending(ground_truth_matrix)
    candidate_ranks = rank_rows_descending(candidate_matrix)
    centered_ground_truth = ground_truth_ranks - ground_truth_ranks.mean(axis=1, keepdims=True)
    centered_candidate = candidate_ranks - candidate_ranks.mean(axis=1, keepdims=True)
    numerator = sp.np.sum(centered_ground_truth * centered_candidate, axis=1)
    denominator = sp.np.sqrt(
        sp.np.sum(centered_ground_truth * centered_ground_truth, axis=1)
        * sp.np.sum(centered_candidate * centered_candidate, axis=1)
    )
    return sp.np.divide(
        numerator,
        denominator,
        out=sp.np.full_like(numerator, sp.np.nan, dtype=float),
        where=denominator != 0,
    )


def kendall_per_timestep_fast(ground_truth_matrix, candidate_matrix):
    subject_count = ground_truth_matrix.shape[1]
    if subject_count < 2:
        return sp.np.full(ground_truth_matrix.shape[0], sp.np.nan, dtype=float)

    left_indices, right_indices = sp.np.triu_indices(subject_count, k=1)
    ground_truth_signs = sp.np.sign(ground_truth_matrix[:, left_indices] - ground_truth_matrix[:, right_indices])
    candidate_signs = sp.np.sign(candidate_matrix[:, left_indices] - candidate_matrix[:, right_indices])
    products = ground_truth_signs * candidate_signs

    concordant = sp.np.sum(products > 0, axis=1).astype(float)
    discordant = sp.np.sum(products < 0, axis=1).astype(float)
    ground_truth_ties = sp.np.sum((ground_truth_signs == 0) & (candidate_signs != 0), axis=1).astype(float)
    candidate_ties = sp.np.sum((candidate_signs == 0) & (ground_truth_signs != 0), axis=1).astype(float)
    numerator = concordant - discordant
    denominator = sp.np.sqrt(
        (concordant + discordant + ground_truth_ties)
        * (concordant + discordant + candidate_ties)
    )
    return sp.np.divide(
        numerator,
        denominator,
        out=sp.np.full_like(numerator, sp.np.nan, dtype=float),
        where=denominator != 0,
    )


def ranking_order_rows_descending(matrix):
    return sp.np.argsort(-matrix, axis=1, kind="mergesort")


def top_precision_series_fast(ground_truth_order, candidate_order, top_k):
    series = []
    for timestep in range(ground_truth_order.shape[0]):
        ground_truth_top = set(ground_truth_order[timestep, :top_k])
        candidate_top = set(candidate_order[timestep, :top_k])
        series.append(len(ground_truth_top & candidate_top) / top_k)
    return series


def write_analysis_vs_simulation_csvs_fast(
    output_dir,
    ground_truth,
    analysis,
    baseline,
    analysis_summary,
    baseline_summary,
):
    subjects, ground_truth_matrix = subject_track_matrix(ground_truth)
    analysis_subjects, analysis_matrix = subject_track_matrix(analysis)
    baseline_subjects, baseline_matrix = subject_track_matrix(baseline)
    if subjects != analysis_subjects or subjects != baseline_subjects:
        raise ValueError("Analysis, baseline, and ground-truth files do not contain the same subjects.")

    tau_analysis = kendall_per_timestep_fast(ground_truth_matrix, analysis_matrix)
    tau_baseline = kendall_per_timestep_fast(ground_truth_matrix, baseline_matrix)
    kendall_csv_path = os.path.join(output_dir, "kendall_correlation_data.csv")
    sp.save_series_csv(
        kendall_csv_path,
        ["timestep", "analysis_kendall", "simulation_kendall"],
        [[index, tau_analysis[index], tau_baseline[index]] for index in range(len(tau_analysis))],
    )

    spearman_analysis = spearman_per_timestep_fast(ground_truth_matrix, analysis_matrix)
    spearman_baseline = spearman_per_timestep_fast(ground_truth_matrix, baseline_matrix)
    spearman_csv_path = os.path.join(output_dir, "spearman_correlation_data.csv")
    sp.save_series_csv(
        spearman_csv_path,
        ["timestep", "analysis_spearman", "simulation_spearman"],
        [[index, spearman_analysis[index], spearman_baseline[index]] for index in range(len(spearman_analysis))],
    )

    ground_truth_order = ranking_order_rows_descending(ground_truth_matrix)
    analysis_order = ranking_order_rows_descending(analysis_matrix)
    baseline_order = ranking_order_rows_descending(baseline_matrix)
    csv_paths = [kendall_csv_path, spearman_csv_path]
    scalar_rows = []
    max_top_precision = min(sp.MAX_TOP_PRECISION, len(subjects))
    for top_k in range(1, max_top_precision + 1):
        analysis_precision = top_precision_series_fast(ground_truth_order, analysis_order, top_k)
        baseline_precision = top_precision_series_fast(ground_truth_order, baseline_order, top_k)
        top_precision_csv_path = os.path.join(output_dir, f"top_{top_k}_precision_data.csv")
        sp.save_series_csv(
            top_precision_csv_path,
            ["timestep", f"analysis_top_{top_k}_precision", f"simulation_top_{top_k}_precision"],
            [
                [index, analysis_precision[index], baseline_precision[index]]
                for index in range(len(analysis_precision))
            ],
        )
        csv_paths.append(top_precision_csv_path)
        scalar_rows.append(
            [
                f"top_{top_k}",
                float(sp.np.mean(analysis_precision)),
                float(sp.np.mean(baseline_precision)),
            ]
        )

    metrics_csv_path = os.path.join(output_dir, "metrics_results.csv")
    sp.save_series_csv(
        metrics_csv_path,
        ["metric", "analysis", "simulation"],
        [
            ["tau", analysis_summary["tau"], baseline_summary["tau"]],
            ["spearman", analysis_summary["spearman"], baseline_summary["spearman"]],
            ["mrr", analysis_summary["mrr"], baseline_summary["mrr"]],
        ]
        + scalar_rows,
    )
    csv_paths.append(metrics_csv_path)
    return {"output_dir": output_dir, "csv_paths": csv_paths}


def compute_candidate_summary_fast(ground_truth, candidate, include_moving_avg_metrics=True):
    summary = {}
    tau, p_value_tau = sp.ranking_metrics.compute_kendalls_tau_correlation(ground_truth, candidate)
    spearman, p_value_sp = sp.ranking_metrics.compute_spearmans_correlation(ground_truth, candidate)
    summary["tau"] = tau
    summary["tau_p_value"] = p_value_tau
    summary["spearman"] = spearman
    summary["spearman_p_value"] = p_value_sp

    subjects = list(ground_truth.keys())
    if subjects != list(candidate.keys()):
        raise ValueError("Ground-truth and candidate tracks do not contain the same subjects.")
    if not subjects:
        summary["mrr"] = 0.0
        return summary

    max_top_precision = min(sp.MAX_TOP_PRECISION, len(subjects))
    top_precision_totals = {top_k: 0.0 for top_k in range(1, max_top_precision + 1)}
    reciprocal_rank_total = 0.0
    timesteps = len(ground_truth[subjects[0]])

    for timestep in range(timesteps):
        ground_truth_rank = sorted(
            subjects,
            key=lambda subject_id: ground_truth[subject_id][timestep],
            reverse=True,
        )
        candidate_rank = sorted(
            subjects,
            key=lambda subject_id: candidate[subject_id][timestep],
            reverse=True,
        )

        ground_truth_top_subject = ground_truth_rank[0]
        reciprocal_rank_total += 1.0 / (candidate_rank.index(ground_truth_top_subject) + 1)

        ground_truth_top_set = set()
        candidate_top_set = set()
        for top_k in range(1, max_top_precision + 1):
            ground_truth_top_set.add(ground_truth_rank[top_k - 1])
            candidate_top_set.add(candidate_rank[top_k - 1])
            top_precision_totals[top_k] += len(ground_truth_top_set & candidate_top_set) / top_k

    summary["mrr"] = reciprocal_rank_total / timesteps if timesteps else 0.0
    for top_k in range(1, max_top_precision + 1):
        precision_mean = top_precision_totals[top_k] / timesteps if timesteps else 0.0
        summary[f"top_{top_k}_precision_mean"] = float(precision_mean)
        if include_moving_avg_metrics:
            summary[f"top_{top_k}_precision_moving_avg_mean"] = float(precision_mean)

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
    save_plots=True,
):
    case_id = parameter_case["case_id"]
    case_run_code = parameter_case["case_run_code"]
    parameter_levels = parameter_case["levels"]

    comparison_base_dir = os.path.join(
        save_path,
        "comparison",
        quartile_label,
        dataset_run["dataset_stem"],
    )
    comparison_dir = resolve_case_output_dir(comparison_base_dir, parameter_case)
    summary_path = os.path.join(comparison_dir, "comparison_summary.json")
    run_id = f"{dataset_run['dataset_stem']}__{case_run_code}__{quartile_label}"

    analysis_path = java_summary.get("analysis_path")
    if not _is_valid_path(analysis_path):
        analysis_path = find_generated_file(
            java_summary["analysis_dir"],
            f"_tracks_it{iterations}.json",
        )

    baseline_path = python_summary["averaged_results_path"]
    ground_truth_path = dataset_run["ground_truth_path"]

    if os.path.exists(summary_path):
        cached_summary = read_json(summary_path)
        cached_java_analysis = cached_summary.get("java_analysis", {})
        cached_python_analysis = cached_summary.get("python_analysis", {})
        cached_analysis_path = cached_java_analysis.get("analysis_path")
        cached_baseline_path = cached_python_analysis.get("baseline_path")
        cached_ground_truth_path = cached_summary.get("ground_truth_path")

        cache_matches_inputs = (
            _is_valid_path(cached_ground_truth_path)
            and _is_valid_path(cached_analysis_path)
            and _is_valid_path(cached_baseline_path)
            and os.path.abspath(cached_ground_truth_path) == os.path.abspath(ground_truth_path)
            and os.path.abspath(cached_analysis_path) == os.path.abspath(analysis_path)
            and os.path.abspath(cached_baseline_path) == os.path.abspath(baseline_path)
        )

        if cache_matches_inputs:
            changed = False
            if cached_summary.get("run_id") != run_id:
                cached_summary["run_id"] = run_id
                changed = True
            if cached_summary.get("parameter_case_code") != case_run_code:
                cached_summary["parameter_case_code"] = case_run_code
                changed = True
            if cached_summary.get("parameter_case_index") != parameter_case["case_index"]:
                cached_summary["parameter_case_index"] = parameter_case["case_index"]
                changed = True
            if cached_summary.get("runtime_budget_seconds") != runtime_budget_seconds:
                cached_summary["runtime_budget_seconds"] = runtime_budget_seconds
                changed = True
            if cached_summary.get("ground_truth_iterations") != dataset_run.get("ground_truth_iterations"):
                cached_summary["ground_truth_iterations"] = dataset_run.get("ground_truth_iterations")
                changed = True
            if changed:
                write_json(summary_path, cached_summary)
            return cached_summary

    ensure_dir(comparison_dir)

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
    ensure_dir(precision_dir)
    prediction_metrics_path = os.path.join(precision_dir, "metrics_prediction.json")
    baseline_metrics_path = os.path.join(precision_dir, "metrics_baseline.json")

    if LOG_PRECISION_METRICS:
        metrics_stdout = io.StringIO()
        with contextlib.redirect_stdout(metrics_stdout):
            prediction_metrics_summary = process_and_save(
                analysis_path,
                ground_truth_path,
                M=10,
                metrics_output=prediction_metrics_path,
                plots_dir=os.path.join(precision_dir, "numericalAnalysis", "plots"),
                save_plots=SAVE_PRECISION_PLOTS and save_plots,
                verbose=True,
                include_scatter_coordinates=SAVE_PRECISION_PLOTS and save_plots,
            )
            baseline_metrics_summary = process_and_save(
                baseline_path,
                ground_truth_path,
                M=10,
                metrics_output=baseline_metrics_path,
                plots_dir=os.path.join(precision_dir, "simulatedBaseline", "plots"),
                save_plots=SAVE_PRECISION_PLOTS and save_plots,
                verbose=True,
                include_scatter_coordinates=SAVE_PRECISION_PLOTS and save_plots,
            )
        write_text(os.path.join(precision_dir, "metrics_stdout.log"), metrics_stdout.getvalue())
    else:
        prediction_metrics_summary = process_and_save(
            analysis_path,
            ground_truth_path,
            M=10,
            metrics_output=prediction_metrics_path,
            plots_dir=os.path.join(precision_dir, "numericalAnalysis", "plots"),
            save_plots=SAVE_PRECISION_PLOTS and save_plots,
            verbose=False,
            include_scatter_coordinates=SAVE_PRECISION_PLOTS and save_plots,
        )
        baseline_metrics_summary = process_and_save(
            baseline_path,
            ground_truth_path,
            M=10,
            metrics_output=baseline_metrics_path,
            plots_dir=os.path.join(precision_dir, "simulatedBaseline", "plots"),
            save_plots=SAVE_PRECISION_PLOTS and save_plots,
            verbose=False,
            include_scatter_coordinates=SAVE_PRECISION_PLOTS and save_plots,
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
        comparison_metrics = compute_analysis_vs_simulation_metrics_only(
            ground_truth_path=ground_truth_path,
            analysis_path=analysis_path,
            baseline_path=baseline_path,
            run_dir=comparison_dir,
            write_csvs=SAVE_COMPARISON_CSVS,
        )

    summary = {
        "run_id": run_id,
        "quartile_label": quartile_label,
        "runtime_budget_seconds": runtime_budget_seconds,
        "dataset_path": dataset_run["dataset_path"],
        "dataset_stem": dataset_run["dataset_stem"],
        "ground_truth_iterations": dataset_run.get("ground_truth_iterations"),
        "ground_truth_path": ground_truth_path,
        "parameter_case_id": case_id,
        "parameter_case_code": case_run_code,
        "parameter_case_index": parameter_case["case_index"],
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
            "prediction_mean_brier": prediction_metrics_summary["mean_brier_score"],
            "prediction_mean_ece": prediction_metrics_summary["mean_ece"],
            "baseline_mean_brier": baseline_metrics_summary["mean_brier_score"],
            "baseline_mean_ece": baseline_metrics_summary["mean_ece"],
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
        "parameter_case_code": comparison_summary.get("parameter_case_code"),
        "parameter_case_id": comparison_summary.get("parameter_case_id"),
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


def resolve_quartile_label_for_existing_run(save_path, explicit_quartile_label=None):
    if explicit_quartile_label:
        return explicit_quartile_label

    sentinel_path = os.path.join(save_path, RUN_COMPLETION_SENTINEL)
    if os.path.exists(sentinel_path):
        completion_payload = read_json(sentinel_path)
        quartile_label = completion_payload.get("quartile_label")
        if isinstance(quartile_label, str) and quartile_label:
            return quartile_label

    comparison_summary_candidates = [
        filename[len("comparison_summary_") : -len(".json")]
        for filename in os.listdir(save_path)
        if filename.startswith("comparison_summary_")
        and filename.endswith(".json")
    ]
    comparison_summary_candidates = sorted(set(comparison_summary_candidates))
    if len(comparison_summary_candidates) == 1:
        return comparison_summary_candidates[0]
    if not comparison_summary_candidates:
        raise FileNotFoundError(
            f"Could not infer quartile label for {save_path}: no comparison_summary_*.json file found."
        )
    raise ValueError(
        "Could not infer quartile label automatically because multiple comparison summaries were found. "
        f"Use --quartile-label with one of: {', '.join(comparison_summary_candidates)}"
    )


def persist_comparison_outputs(save_path, quartile_label, comparison_summaries):
    for comparison_summary in comparison_summaries:
        summary_path = os.path.join(comparison_summary["comparison_dir"], "comparison_summary.json")
        write_json(summary_path, comparison_summary)

    comparison_summary_json_path = os.path.join(save_path, f"comparison_summary_{quartile_label}.json")
    write_json(comparison_summary_json_path, comparison_summaries)

    sweep_summary_rows = [build_reference_row(summary) for summary in comparison_summaries]
    sweep_summary_csv_path = os.path.join(save_path, "sweep_summary.csv")
    write_reference_sweep_csv(sweep_summary_csv_path, sweep_summary_rows)

    comparison_summary_csv_path = os.path.join(save_path, f"comparison_summary_{quartile_label}.csv")
    write_reference_sweep_csv(comparison_summary_csv_path, sweep_summary_rows)

    return {
        "comparison_summary_json_path": comparison_summary_json_path,
        "comparison_summary_csv_path": comparison_summary_csv_path,
        "sweep_summary_csv_path": sweep_summary_csv_path,
        "rows_written": len(sweep_summary_rows),
    }


def update_run_completion_metadata(save_path, quartile_label, comparison_summaries, output_paths, selection_manifest_path):
    completion_payload = {}
    sentinel_path = os.path.join(save_path, RUN_COMPLETION_SENTINEL)
    if os.path.exists(sentinel_path):
        completion_payload = read_json(sentinel_path)

    completion_payload.update(
        {
            "completed_at": completion_payload.get(
                "completed_at",
                datetime.now().isoformat(timespec="seconds"),
            ),
            "save_path": save_path,
            "quartile_label": quartile_label,
            "rows_written": output_paths["rows_written"],
            "selected_runs_for_plots": len([summary for summary in comparison_summaries if summary.get("note")]),
            "sweep_summary_csv_path": output_paths["sweep_summary_csv_path"],
            "comparison_summary_json_path": output_paths["comparison_summary_json_path"],
            "comparison_summary_csv_path": output_paths["comparison_summary_csv_path"],
            "selected_run_manifest_path": selection_manifest_path,
            "selection_refreshed_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    write_json(sentinel_path, completion_payload)


def refresh_selected_run_outputs(save_path, quartile_label, time_step_hours, iterations):
    comparison_summary_json_path = os.path.join(save_path, f"comparison_summary_{quartile_label}.json")
    if not os.path.exists(comparison_summary_json_path):
        raise FileNotFoundError(
            f"Comparison summary not found for quartile {quartile_label}: {comparison_summary_json_path}"
        )

    comparison_summaries = read_json(comparison_summary_json_path)
    comparison_summaries.sort(key=lambda item: (item["dataset_stem"], item["parameter_case_id"]))

    notes_by_run, selection_manifest = build_notes_map_and_selection_manifest(comparison_summaries)
    apply_notes_to_comparison_summaries(comparison_summaries, notes_by_run)
    selection_manifest_path = os.path.join(save_path, "selected_run_manifest.json")
    write_json(selection_manifest_path, selection_manifest)

    stage_name = "stage8_selected_plots"
    selected_summaries = [summary for summary in comparison_summaries if summary.get("note")]
    total_stage_tasks = len(selected_summaries)
    write_stage_checkpoint(
        save_path,
        stage_name,
        0,
        total_stage_tasks,
        status="running",
        extra={"mode": "refresh_selected_plots", "quartile_label": quartile_label},
    )

    if total_stage_tasks > 0:
        updated_by_run_id = {}
        worker_count = resolve_worker_count(total_stage_tasks)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    regenerate_selected_run_plots,
                    summary,
                    time_step_hours,
                    iterations,
                )
                for summary in selected_summaries
            ]

            completed = 0
            for future in tqdm(as_completed(futures), total=total_stage_tasks, desc="Generating selected plots"):
                updated_summary = future.result()
                updated_by_run_id[updated_summary["run_id"]] = updated_summary
                completed += 1
                write_stage_checkpoint(save_path, stage_name, completed, total_stage_tasks, status="running")

        comparison_summaries = [
            updated_by_run_id.get(summary["run_id"], summary)
            for summary in comparison_summaries
        ]

    mark_stage_complete(save_path, stage_name)
    write_stage_checkpoint(save_path, stage_name, total_stage_tasks, total_stage_tasks, status="completed")

    output_paths = persist_comparison_outputs(save_path, quartile_label, comparison_summaries)
    update_run_completion_metadata(
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


def main(argv=None):
    args = parse_args(argv)
    if args.reuse_run and not args.only_selected_plots:
        raise ValueError("--reuse-run currently requires --only-selected-plots.")
    if args.quartile_label and not args.only_selected_plots:
        raise ValueError("--quartile-label can only be used together with --only-selected-plots.")
    if args.only_selected_plots:
        save_path = resolve_existing_run_path(args.reuse_run)
        quartile_label = resolve_quartile_label_for_existing_run(save_path, args.quartile_label)
        refresh_summary = refresh_selected_run_outputs(
            save_path=save_path,
            quartile_label=quartile_label,
            time_step_hours=TIME_STEP,
            iterations=INTERNAL_STEPS,
        )
        print(
            "Selected-run refresh completed. "
            f"Updated {refresh_summary['selected_runs_for_plots']} selected runs in {save_path}. "
            f"Manifest: {refresh_summary['selected_run_manifest_path']}"
        )
        return

    dataset_profile = resolve_dataset_profile(args.dataset)

    save_path, resumed_run = resolve_run_directory(
        results_root="results",
        dataset_family=dataset_profile.family,
    )
    cache_path = ensure_dir(os.path.join("results", "cache"))

    write_json(
        os.path.join(save_path, "run_metadata.json"),
        {
            "save_path": save_path,
            "cache_path": cache_path,
            "dataset_family": dataset_profile.family,
            "dataset_profile": dataset_profile_metadata(dataset_profile),
            "time_step_hours": TIME_STEP,
            "internal_steps": INTERNAL_STEPS,
            "quantile": QUANTILE,
            "noise_fraction": NOISE_FRACTION,
            "noise_time_shift_hours": NOISE_TIME_SHIFT_HOURS,
            "save_comparison_csvs": SAVE_COMPARISON_CSVS,
            "contact_noise_event_types": list(CONTACT_NOISE_EVENT_TYPES),
            "observation_noise_event_types": list(OBSERVATION_NOISE_EVENT_TYPES),
            "resumed_run": resumed_run,
            "started_at": datetime.now().isoformat(timespec="seconds"),
        },
    )

    # 1. Create datasets
    stage_name = "stage1_dataset_generation"
    dataset_contact_targets = list(dataset_profile.internal_contacts)
    write_stage_checkpoint(save_path, stage_name, 0, len(dataset_contact_targets), status="running")

    dataset_paths = []
    dataset_generation_summaries = []
    for index, internal_contacts in enumerate(dataset_contact_targets, start=1):
        dataset_path = os.path.join(save_path, dataset_filename(dataset_profile, internal_contacts))
        if not os.path.exists(dataset_path):
            dataset_generation_summary = generate_dataset_for_profile(
                dataset_profile=dataset_profile,
                dataset_path=dataset_path,
                internal_contacts=internal_contacts,
                seed=30,
            )
        else:
            dataset_generation_summary = summarize_generated_dataset(
                dataset_profile=dataset_profile,
                dataset_path=dataset_path,
                payload=read_json(dataset_path),
                requested_internal_contacts=internal_contacts,
            )
        dataset_paths.append(dataset_path)
        dataset_generation_summaries.append(dataset_generation_summary)
        write_stage_checkpoint(save_path, stage_name, index, len(dataset_contact_targets), status="running")

    write_json(
        os.path.join(save_path, "dataset_generation_summary.json"),
        dataset_generation_summaries,
    )

    mark_stage_complete(save_path, stage_name)
    write_stage_checkpoint(
        save_path,
        stage_name,
        len(dataset_contact_targets),
        len(dataset_contact_targets),
        status="completed",
    )

    parameter_cases, ground_truth_case = build_parameter_combinations(os.path.join(".", "parameters.json"))
    print(normalize_stpn_parameter_bundle(parameter_cases[0]["parameter_bundle"])["case_id"])
    print(f"\n\n\n{ground_truth_case['case_id']}\n\n\n")

    # 2. Run clean GT simulation on clean datasets (recoverable)
    stage_name = "stage2_ground_truth"
    write_stage_checkpoint(save_path, stage_name, 0, len(dataset_paths), status="running")

    ground_truth_results_by_stem = {}
    for index, dataset_path in enumerate(tqdm(dataset_paths, desc="Running GT simulations"), start=1):
        dataset_stem = os.path.splitext(os.path.basename(dataset_path))[0]
        gt_results_path = dataset_path.replace("dataset", "gt_results")
        gt_results = None

        if os.path.exists(gt_results_path):
            cached = read_json(gt_results_path)
            if _is_valid_path(cached.get("observed_simulated_path")) and _is_valid_path(
                cached.get("averaged_results_path")
            ):
                gt_results = cached

        if gt_results is None:
            ground_truth_dataset_path = ensure_ground_truth_simulation_input(
                save_path=save_path,
                dataset_path=dataset_path,
            )
            gt_results = run_dataset_simulations(
                dataset_path=ground_truth_dataset_path,
                run_until_convergence=True,
                iterations_cap=100_000,
                convergence_threshold=1e-8,
                fine_grained=False,
                time_step_hours=TIME_STEP,
                seed=30,
                prune_after_positive_test=False,
                export_observed_simulation=True,
                pruning_seed=None,
                parameter_bundle=ground_truth_case["parameter_bundle"],
                dataset_label=os.path.splitext(os.path.basename(dataset_path))[0],
                save_plots=True,
            )
            write_json(gt_results_path, gt_results)

        ground_truth_results_by_stem[dataset_stem] = {
            "clean_dataset_path": dataset_path,
            "dataset_stem": dataset_stem,
            "gt_results_path": gt_results_path,
            "gt_results": gt_results,
        }
        write_stage_checkpoint(save_path, stage_name, index, len(dataset_paths), status="running")

    mark_stage_complete(save_path, stage_name)
    write_stage_checkpoint(save_path, stage_name, len(dataset_paths), len(dataset_paths), status="completed")
    print("GT simulations completed")

    # 2b. Inject contact noise on raw datasets used by the baseline simulation.
    stage_name = "stage2b_contact_noise"
    write_stage_checkpoint(save_path, stage_name, 0, len(dataset_paths), status="running")

    contact_noisy_dataset_paths_by_stem = {}
    contact_noise_summaries_by_stem = {}
    contact_noise_summaries = []
    for index, dataset_path in enumerate(tqdm(dataset_paths, desc="Applying contact noise"), start=1):
        dataset_stem = os.path.splitext(os.path.basename(dataset_path))[0]
        contact_noisy_dataset_path = os.path.join(save_path, f"{dataset_stem}_contact_noisy.json")
        contact_noise_summary_path = os.path.join(
            save_path,
            "dataset_noise",
            f"{dataset_stem}_contact_noise_summary.json",
        )

        if os.path.exists(contact_noisy_dataset_path) and os.path.exists(contact_noise_summary_path):
            contact_noise_summary = read_json(contact_noise_summary_path)
        else:
            contact_noise_summary = apply_contact_noise_to_dataset(
                source_dataset_path=dataset_path,
                noisy_dataset_path=contact_noisy_dataset_path,
                seed_label=f"{dataset_stem}:contact",
            )
            write_json(contact_noise_summary_path, contact_noise_summary)

        contact_noisy_dataset_paths_by_stem[dataset_stem] = contact_noisy_dataset_path
        contact_noise_summaries_by_stem[dataset_stem] = contact_noise_summary
        contact_noise_summaries.append(contact_noise_summary)
        write_stage_checkpoint(save_path, stage_name, index, len(dataset_paths), status="running")

    write_json(os.path.join(save_path, "dataset_noise_contact_summary.json"), contact_noise_summaries)
    mark_stage_complete(save_path, stage_name)
    write_stage_checkpoint(save_path, stage_name, len(dataset_paths), len(dataset_paths), status="completed")

    # 2c. Run one GT-parameter simulation on the contact-noisy dataset.
    stage_name = "stage2c_observed_one_run"
    write_stage_checkpoint(save_path, stage_name, 0, len(dataset_paths), status="running")

    observed_one_run_results_by_stem = {}
    observed_one_run_summaries = []
    for index, dataset_path in enumerate(tqdm(dataset_paths, desc="Running one observed simulation"), start=1):
        dataset_stem = os.path.splitext(os.path.basename(dataset_path))[0]
        contact_noisy_dataset_path = contact_noisy_dataset_paths_by_stem[dataset_stem]
        one_run_summary_path = os.path.join(
            save_path,
            "observed_one_run",
            f"{dataset_stem}_contact_noisy_one_run_summary.json",
        )
        one_run_result = None

        if os.path.exists(one_run_summary_path):
            cached = read_json(one_run_summary_path)
            cached_dataset_path = cached.get("dataset_path")
            dataset_path_matches = _is_valid_path(cached_dataset_path) and (
                os.path.abspath(cached_dataset_path) == os.path.abspath(contact_noisy_dataset_path)
            )
            if (
                cached.get("rep_done") == 1
                and dataset_path_matches
                and _is_valid_path(cached.get("observed_simulated_path"))
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
                parameter_bundle=ground_truth_case["parameter_bundle"],
                save_plots=False,
            )
            write_json(one_run_summary_path, one_run_result)

        observed_one_run_results_by_stem[dataset_stem] = one_run_result
        observed_one_run_summaries.append(one_run_result)
        write_stage_checkpoint(save_path, stage_name, index, len(dataset_paths), status="running")

    write_json(os.path.join(save_path, "observed_one_run_summary.json"), observed_one_run_summaries)
    mark_stage_complete(save_path, stage_name)
    write_stage_checkpoint(save_path, stage_name, len(dataset_paths), len(dataset_paths), status="completed")

    # 2d. Inject observation noise on the one-run observed simulation used by Java analysis.
    stage_name = "stage2d_observation_noise"
    write_stage_checkpoint(save_path, stage_name, 0, len(dataset_paths), status="running")

    dataset_runs = []
    observation_noise_summaries = []
    for index, dataset_path in enumerate(tqdm(dataset_paths, desc="Applying observation noise"), start=1):
        dataset_stem = os.path.splitext(os.path.basename(dataset_path))[0]
        ground_truth_entry = ground_truth_results_by_stem[dataset_stem]
        gt_results = ground_truth_entry["gt_results"]
        one_run_result = observed_one_run_results_by_stem[dataset_stem]
        clean_observed_one_run_path = one_run_result["observed_simulated_path"]
        final_observed_simulated_path = observation_noisy_simulated_path(clean_observed_one_run_path)
        observation_noise_summary_path = os.path.join(
            save_path,
            "dataset_noise",
            f"{dataset_stem}_observation_noise_summary.json",
        )

        if os.path.exists(final_observed_simulated_path) and os.path.exists(observation_noise_summary_path):
            observation_noise_summary = read_json(observation_noise_summary_path)
        else:
            observation_noise_summary = apply_observation_noise_to_dataset(
                source_dataset_path=clean_observed_one_run_path,
                noisy_dataset_path=final_observed_simulated_path,
                seed_label=f"{dataset_stem}:observation",
            )
            write_json(observation_noise_summary_path, observation_noise_summary)

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
        write_stage_checkpoint(save_path, stage_name, index, len(dataset_paths), status="running")

    write_json(os.path.join(save_path, "dataset_noise_observation_summary.json"), observation_noise_summaries)
    write_json(os.path.join(save_path, "dataset_runs_summary.json"), dataset_runs)
    mark_stage_complete(save_path, stage_name)
    write_stage_checkpoint(save_path, stage_name, len(dataset_paths), len(dataset_paths), status="completed")

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

    java_summary_by_key = {
        (summary["dataset_stem"], summary["parameter_case_id"]): summary
        for summary in java_analysis_summaries
    }

    mark_stage_complete(save_path, stage_name)
    write_stage_checkpoint(save_path, stage_name, 1, 1, status="completed")

    # 6. Parallel Python analysis with per-combination runtime budget (wait all)
    stage_name = "stage6_python_analysis"
    stage6_tasks = [
        (
            dataset_run,
            parameter_case,
            java_summary_by_key[(dataset_run["dataset_stem"], parameter_case["case_id"])],
        )
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
                java_summary["analysis_wall_runtime_seconds"],
                TIME_STEP,
            )
            for dataset_run, parameter_case, java_summary in stage6_tasks
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

    # 7. Parallel comparison + metrics (without plots)
    stage_name = "stage7_comparison"
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
                java_summary["analysis_wall_runtime_seconds"],
                dataset_run,
                parameter_case,
                java_summary,
                python_summary,
                TIME_STEP,
                INTERNAL_STEPS,
                False,
            )
            for dataset_run, parameter_case, java_summary, python_summary in stage7_tasks
        ]

        completed = 0
        for future in tqdm(as_completed(futures), total=total_stage_tasks, desc="Running comparisons"):
            comparison_summaries.append(future.result())
            completed += 1
            write_stage_checkpoint(save_path, stage_name, completed, total_stage_tasks, status="running")

    comparison_summaries.sort(key=lambda item: (item["dataset_stem"], item["parameter_case_id"]))
    mark_stage_complete(save_path, "stage7_comparison")
    write_stage_checkpoint(
        save_path,
        "stage7_comparison",
        len(stage7_tasks),
        len(stage7_tasks),
        status="completed",
    )

    notes_by_run, selection_manifest = build_notes_map_and_selection_manifest(comparison_summaries)
    apply_notes_to_comparison_summaries(comparison_summaries, notes_by_run)
    selection_manifest_path = os.path.join(save_path, "selected_run_manifest.json")
    write_json(selection_manifest_path, selection_manifest)

    # 8. Regenerate plots only for runs selected in dataset-wise top/worst/median buckets.
    stage_name = "stage8_selected_plots"
    selected_summaries = [summary for summary in comparison_summaries if summary.get("note")]
    total_stage_tasks = len(selected_summaries)
    write_stage_checkpoint(save_path, stage_name, 0, total_stage_tasks, status="running")

    if total_stage_tasks > 0:
        updated_by_run_id = {}
        worker_count = resolve_worker_count(total_stage_tasks)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    regenerate_selected_run_plots,
                    summary,
                    TIME_STEP,
                    INTERNAL_STEPS,
                )
                for summary in selected_summaries
            ]

            completed = 0
            for future in tqdm(as_completed(futures), total=total_stage_tasks, desc="Generating selected plots"):
                updated_summary = future.result()
                updated_by_run_id[updated_summary["run_id"]] = updated_summary
                completed += 1
                write_stage_checkpoint(save_path, stage_name, completed, total_stage_tasks, status="running")

        comparison_summaries = [
            updated_by_run_id.get(summary["run_id"], summary)
            for summary in comparison_summaries
        ]

    mark_stage_complete(save_path, stage_name)
    write_stage_checkpoint(save_path, stage_name, total_stage_tasks, total_stage_tasks, status="completed")

    output_paths = persist_comparison_outputs(save_path, quartile_label, comparison_summaries)
    update_run_completion_metadata(
        save_path=save_path,
        quartile_label=quartile_label,
        comparison_summaries=comparison_summaries,
        output_paths=output_paths,
        selection_manifest_path=selection_manifest_path,
    )

    print(
        "Point 7 completed. "
        f"Saved {len(comparison_summaries)} comparison summaries to {output_paths['comparison_summary_json_path']}, "
        f"{output_paths['comparison_summary_csv_path']}, and {output_paths['sweep_summary_csv_path']}."
    )


if __name__ == "__main__":
    main()
