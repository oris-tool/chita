#!/usr/bin/env python3
"""Run P(omega) sensitivity analyses while reusing GT and baseline artifacts.

This script is intended to run on a space-constrained compute machine.  It
stages one dataset/prior experiment at a time from an existing full sweep,
runs only the Java analysis with a different observation prior, recomputes
comparison metrics against the reused ground truth and baseline, generates the
selected best/worst/median plots, copies results back with rsync, and removes
local staged data.
"""

import argparse
import csv
import json
import os
import posixpath
import shlex
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

import sweep_pipeline_final as final


DEFAULT_PRIORS = (0.1, 0.3, 0.7, 0.9)
DEFAULT_DATASETS = ("bubble_400", "scale_free_2500", "small_world_3600")
QUARTILE_LABEL = f"q{final.QUANTILE}"
RUN_PREFIX = "pomega_sensitivity_"
COMPLETION_SENTINEL = "_run_completed.json"


@dataclass(frozen=True)
class Endpoint:
    raw: str
    is_remote: bool
    host: Optional[str]
    path: str


@dataclass(frozen=True)
class DatasetTarget:
    key: str
    dataset_stem: str
    aliases: Tuple[str, ...]


DATASET_TARGETS = (
    DatasetTarget(
        key="bubble_400",
        dataset_stem="dataset_bubble_400",
        aliases=("bubble_400", "household_bubble_400", "hb400", "dataset_bubble_400"),
    ),
    DatasetTarget(
        key="scale_free_2500",
        dataset_stem="dataset_scale_free_2500",
        aliases=("scale_free_2500", "sf2500", "dataset_scale_free_2500"),
    ),
    DatasetTarget(
        key="small_world_3600",
        dataset_stem="dataset_small_world_3600",
        aliases=("small_world_3600", "sw3600", "dataset_small_world_3600"),
    ),
)
TARGET_BY_ALIAS = {
    alias: target
    for target in DATASET_TARGETS
    for alias in target.aliases
}


def parse_endpoint(value: str) -> Endpoint:
    value = value.strip()
    if not value:
        raise ValueError("Endpoint cannot be empty.")
    if ":" in value and not value.startswith("/"):
        host, path = value.split(":", 1)
        if host and path.startswith("/"):
            return Endpoint(raw=value, is_remote=True, host=host, path=path.rstrip("/") or "/")
    return Endpoint(raw=value, is_remote=False, host=None, path=os.path.abspath(value))


def endpoint_path(endpoint: Endpoint, *parts: str) -> str:
    clean_parts = [str(part).strip("/") for part in parts if str(part).strip("/")]
    if endpoint.is_remote:
        path = endpoint.path
        for part in clean_parts:
            path = posixpath.join(path, part)
        return path
    return os.path.join(endpoint.path, *clean_parts)


def endpoint_spec(endpoint: Endpoint, *parts: str, trailing_slash: bool = False) -> str:
    path = endpoint_path(endpoint, *parts)
    if trailing_slash and not path.endswith("/"):
        path += "/"
    if endpoint.is_remote:
        return f"{endpoint.host}:{path}"
    return path


def endpoint_child(endpoint: Endpoint, *parts: str) -> Endpoint:
    path = endpoint_path(endpoint, *parts)
    if endpoint.is_remote:
        return Endpoint(raw=f"{endpoint.host}:{path}", is_remote=True, host=endpoint.host, path=path)
    return Endpoint(raw=path, is_remote=False, host=None, path=path)


def endpoint_basename(endpoint: Endpoint) -> str:
    if endpoint.is_remote:
        return posixpath.basename(endpoint.path.rstrip("/"))
    return os.path.basename(endpoint.path.rstrip(os.sep))


def ssh_command(endpoint: Endpoint, command: str, ssh_opts: Optional[str] = None) -> List[str]:
    if not endpoint.is_remote or not endpoint.host:
        raise ValueError("SSH command requested for a local endpoint.")
    args = ["ssh"]
    if ssh_opts:
        args.extend(shlex.split(ssh_opts))
    args.extend([endpoint.host, command])
    return args


def ensure_endpoint_dir(endpoint: Endpoint, *parts: str, ssh_opts: Optional[str] = None) -> None:
    path = endpoint_path(endpoint, *parts)
    if endpoint.is_remote:
        subprocess.run(
            ssh_command(endpoint, "mkdir -p " + shlex.quote(path), ssh_opts),
            check=True,
        )
    else:
        os.makedirs(path, exist_ok=True)


def endpoint_exists(endpoint: Endpoint, *parts: str, ssh_opts: Optional[str] = None) -> bool:
    path = endpoint_path(endpoint, *parts)
    if endpoint.is_remote:
        result = subprocess.run(
            ssh_command(endpoint, "test -e " + shlex.quote(path), ssh_opts),
            check=False,
        )
        return result.returncode == 0
    return os.path.exists(path)


def endpoint_find_files(
    endpoint: Endpoint,
    relative_dir: str,
    name_pattern: str,
    ssh_opts: Optional[str] = None,
) -> List[str]:
    if endpoint.is_remote:
        search_dir = endpoint_path(endpoint, relative_dir)
        command = (
            "if test -d " + shlex.quote(search_dir) + "; then "
            "find " + shlex.quote(search_dir) + " -maxdepth 1 -type f -name " + shlex.quote(name_pattern) + " -print; "
            "fi"
        )
        result = subprocess.run(
            ssh_command(endpoint, command, ssh_opts),
            check=True,
            capture_output=True,
            text=True,
        )
        prefix = endpoint.path.rstrip("/") + "/"
        relatives = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith(prefix):
                relatives.append(line[len(prefix):])
        return sorted(relatives)

    import glob

    search_dir = endpoint_path(endpoint, relative_dir)
    return sorted(
        os.path.relpath(match, endpoint.path).replace(os.sep, "/")
        for match in glob.glob(os.path.join(search_dir, name_pattern))
        if os.path.isfile(match)
    )


def run_rsync(
    source: str,
    destination: str,
    ssh_opts: Optional[str] = None,
    extra_args: Optional[Sequence[str]] = None,
) -> None:
    command = ["rsync", "-a"]
    if ssh_opts:
        command.extend(["-e", "ssh " + ssh_opts])
    if extra_args:
        command.extend(extra_args)
    command.extend([source, destination])
    subprocess.run(command, check=True)


def read_paths_file(path: str) -> Dict[str, str]:
    config: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise ValueError(f"Invalid paths-file line {line_number}: expected KEY=VALUE.")
            key, value = line.split("=", 1)
            config[key.strip().upper()] = value.strip()

    required = ("SOURCE_ROOT", "DEST_ROOT", "LOCAL_WORK")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing required paths-file key(s): {', '.join(missing)}")
    return config


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4)


def write_csv(path: str, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def run_relative_path(path_value: str, run_name: str) -> str:
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"Invalid artifact path: {path_value!r}")
    normalized = path_value.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    if run_name in parts:
        index = parts.index(run_name)
        tail = parts[index + 1 :]
        if not tail:
            raise ValueError(f"Artifact path points at run root, not a file: {path_value}")
        return posixpath.join(*tail)
    if normalized.startswith("/"):
        raise ValueError(
            f"Absolute artifact path does not contain source run name '{run_name}': {path_value}"
        )
    return posixpath.normpath(normalized)


def local_artifact_path(local_input_root: str, relative_path: str) -> str:
    return os.path.join(local_input_root, *relative_path.split("/"))


def copy_artifact(
    source_root: Endpoint,
    local_input_root: str,
    source_run_name: str,
    path_value: str,
    ssh_opts: Optional[str],
) -> str:
    relative_path = run_relative_path(path_value, source_run_name)
    local_path = local_artifact_path(local_input_root, relative_path)
    if os.path.exists(local_path):
        return local_path
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    run_rsync(
        endpoint_spec(source_root, relative_path),
        local_path,
        ssh_opts=ssh_opts,
    )
    return local_path


def copy_top_level_file(
    source_root: Endpoint,
    local_input_root: str,
    filename: str,
    ssh_opts: Optional[str],
) -> str:
    local_path = os.path.join(local_input_root, filename)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    run_rsync(endpoint_spec(source_root, filename), local_path, ssh_opts=ssh_opts)
    return local_path


def try_copy_top_level_file(
    source_root: Endpoint,
    local_input_root: str,
    filename: str,
    ssh_opts: Optional[str],
) -> Optional[str]:
    local_path = os.path.join(local_input_root, filename)
    if os.path.exists(local_path):
        return local_path
    if not endpoint_exists(source_root, filename, ssh_opts=ssh_opts):
        return None
    return copy_top_level_file(source_root, local_input_root, filename, ssh_opts)


def stage_source_artifacts(
    source_root: Endpoint,
    local_input_root: str,
    target: DatasetTarget,
    ssh_opts: Optional[str],
) -> None:
    stem = target.dataset_stem
    os.makedirs(local_input_root, exist_ok=True)
    include_patterns = [
        "/dataset_runs_summary.json",
        "/python_analysis_summary.json",
        f"/{stem}.json",
        f"/{stem}_contact_noisy.json",
        f"/{stem}_contact_noisy_simulated.json",
        f"/{stem}_contact_noisy_observation_noisy_simulated.json",
        f"/gt_results_{stem}.json",
        f"/gt_results_{stem[len('dataset_'):]}.json" if stem.startswith("dataset_") else f"/gt_results_{stem}.json",
        "/ground_truth/",
        f"/ground_truth/{stem}/***",
        "/dataset_noise/",
        f"/dataset_noise/{stem}_contact_noise_summary.json",
        f"/dataset_noise/{stem}_observation_noise_summary.json",
        "/observed_one_run/",
        f"/observed_one_run/{stem}_contact_noisy_one_run_summary.json",
        "/python_analysis/",
        f"/python_analysis/{QUARTILE_LABEL}/",
        f"/python_analysis/{QUARTILE_LABEL}/{stem}/***",
    ]
    extra_args = ["--prune-empty-dirs"]
    for pattern in include_patterns:
        extra_args.extend(["--include", pattern])
    extra_args.extend(["--exclude", "*"])
    run_rsync(
        endpoint_spec(source_root, trailing_slash=True),
        local_input_root.rstrip(os.sep) + os.sep,
        ssh_opts=ssh_opts,
        extra_args=extra_args,
    )


def resolve_targets(dataset_args: Sequence[str]) -> List[DatasetTarget]:
    targets: List[DatasetTarget] = []
    unknown = []
    for raw_value in dataset_args:
        value = raw_value.strip()
        target = TARGET_BY_ALIAS.get(value)
        if target is None:
            unknown.append(value)
        elif target not in targets:
            targets.append(target)
    if unknown:
        supported = ", ".join(sorted(TARGET_BY_ALIAS))
        raise ValueError(f"Unsupported dataset target(s): {', '.join(unknown)}. Supported: {supported}")
    return targets


def parse_priors(values: Sequence[str]) -> List[float]:
    priors = []
    for value in values:
        prior = float(value)
        if prior <= 0.0 or prior >= 1.0:
            raise ValueError(f"Invalid prior {value}: values must be greater than 0 and lower than 1.")
        priors.append(prior)
    return priors


def prior_label(prior: float) -> str:
    return (f"{prior:.12g}").replace(".", "p")


def load_source_indexes(
    source_root: Endpoint,
    local_input_root: str,
    ssh_opts: Optional[str],
) -> Tuple[List[dict], List[dict]]:
    dataset_runs_path = try_copy_top_level_file(source_root, local_input_root, "dataset_runs_summary.json", ssh_opts)
    python_summaries_path = try_copy_top_level_file(source_root, local_input_root, "python_analysis_summary.json", ssh_opts)
    dataset_runs = [] if dataset_runs_path is None else read_json(dataset_runs_path)
    python_summaries = [] if python_summaries_path is None else read_json(python_summaries_path)
    return dataset_runs, python_summaries


def localize_dataset_run(
    dataset_run: dict,
    source_root: Endpoint,
    local_input_root: str,
    source_run_name: str,
    ssh_opts: Optional[str],
) -> dict:
    localized = dict(dataset_run)
    required_paths = ("dataset_path", "observed_simulated_path", "ground_truth_path")
    for key in required_paths:
        localized[key] = copy_artifact(
            source_root,
            local_input_root,
            source_run_name,
            dataset_run[key],
            ssh_opts,
        )
    return localized


def localize_python_summaries(
    python_summaries: Sequence[dict],
    dataset_stem: str,
    source_root: Endpoint,
    local_input_root: str,
    source_run_name: str,
    ssh_opts: Optional[str],
) -> List[dict]:
    localized = []
    for summary in python_summaries:
        if summary.get("dataset_stem") != dataset_stem:
            continue
        copied = dict(summary)
        copied["averaged_results_path"] = copy_artifact(
            source_root,
            local_input_root,
            source_run_name,
            summary["averaged_results_path"],
            ssh_opts,
        )
        localized.append(copied)
    return localized


def build_parameter_case_maps() -> Tuple[List[dict], Dict[str, dict]]:
    parameter_cases, _ground_truth_case = final.build_parameter_combinations(os.path.join(".", "parameters.json"))
    cases_by_id = {case["case_id"]: case for case in parameter_cases}
    return parameter_cases, cases_by_id


def _require_source_file(
    source_root: Endpoint,
    relative_path: str,
    ssh_opts: Optional[str],
    local_input_root: Optional[str] = None,
) -> str:
    if local_input_root is not None and os.path.exists(local_artifact_path(local_input_root, relative_path)):
        return relative_path
    if not endpoint_exists(source_root, *relative_path.split("/"), ssh_opts=ssh_opts):
        raise FileNotFoundError(f"Required source artifact not found: {endpoint_spec(source_root, relative_path)}")
    return relative_path


def _parse_reps_from_ground_truth_path(dataset_stem: str, relative_path: str) -> Optional[int]:
    filename = posixpath.basename(relative_path)
    prefix = f"{dataset_stem}_simulated_"
    suffix = "_reps.json"
    if not filename.startswith(prefix) or not filename.endswith(suffix):
        return None
    try:
        return int(filename[len(prefix):-len(suffix)])
    except ValueError:
        return None


def reconstruct_dataset_run_from_artifacts(
    source_root: Endpoint,
    target: DatasetTarget,
    ssh_opts: Optional[str],
    local_input_root: Optional[str] = None,
) -> dict:
    stem = target.dataset_stem
    clean_dataset_path = _require_source_file(source_root, f"{stem}.json", ssh_opts, local_input_root)
    dataset_path = _require_source_file(source_root, f"{stem}_contact_noisy.json", ssh_opts, local_input_root)
    observed_simulated_path = _require_source_file(
        source_root,
        f"{stem}_contact_noisy_observation_noisy_simulated.json",
        ssh_opts,
        local_input_root,
    )
    if local_input_root is not None:
        local_gt_dir = local_artifact_path(local_input_root, f"ground_truth/{stem}")
        import glob
        ground_truth_candidates = sorted(
            os.path.relpath(match, local_input_root).replace(os.sep, "/")
            for match in glob.glob(os.path.join(local_gt_dir, f"{stem}_simulated_*_reps.json"))
            if os.path.isfile(match)
        )
    else:
        ground_truth_candidates = []
    if not ground_truth_candidates:
        ground_truth_candidates = endpoint_find_files(
            source_root,
            f"ground_truth/{stem}",
            f"{stem}_simulated_*_reps.json",
            ssh_opts=ssh_opts,
        )
    if not ground_truth_candidates:
        raise FileNotFoundError(
            f"Could not reconstruct {stem}: no ground_truth/{stem}/{stem}_simulated_*_reps.json found."
        )
    ground_truth_candidates.sort(
        key=lambda item: (_parse_reps_from_ground_truth_path(stem, item) or -1, item)
    )
    ground_truth_path = ground_truth_candidates[-1]
    ground_truth_iterations = _parse_reps_from_ground_truth_path(stem, ground_truth_path)
    gt_suffix = stem[len("dataset_"):] if stem.startswith("dataset_") else stem
    gt_results_path = f"gt_results_{gt_suffix}.json"

    return {
        "clean_dataset_path": clean_dataset_path,
        "dataset_path": dataset_path,
        "dataset_stem": stem,
        "gt_results_path": gt_results_path,
        "ground_truth_path": ground_truth_path,
        "clean_ground_truth_observed_simulated_path": f"ground_truth/{stem}/{stem}_simulated.json",
        "clean_observed_simulated_path": f"{stem}_contact_noisy_simulated.json",
        "observed_simulated_path": observed_simulated_path,
        "contact_noise_summary": {},
        "observation_noise_summary": {},
        "observed_one_run_summary_path": f"observed_one_run/{stem}_contact_noisy_one_run_summary.json",
        "ground_truth_iterations": ground_truth_iterations,
        "reconstructed_from_artifacts": True,
    }


def select_dataset_run(
    dataset_runs: Sequence[dict],
    target: DatasetTarget,
    source_root: Endpoint,
    ssh_opts: Optional[str],
    local_input_root: Optional[str] = None,
) -> dict:
    matches = [run for run in dataset_runs if run.get("dataset_stem") == target.dataset_stem]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Expected one dataset_run for {target.dataset_stem}, found {len(matches)}.")

    available = ", ".join(sorted(str(run.get("dataset_stem")) for run in dataset_runs))
    print(
        f"dataset_runs_summary.json has no entry for {target.dataset_stem}; "
        f"attempting artifact reconstruction. Available stems: {available or '<none>'}"
    )
    return reconstruct_dataset_run_from_artifacts(source_root, target, ssh_opts, local_input_root)


def reconstruct_python_summaries_from_case_files(
    source_root: Endpoint,
    local_input_root: str,
    source_run_name: str,
    target: DatasetTarget,
    parameter_cases: Sequence[dict],
    ssh_opts: Optional[str],
) -> List[dict]:
    summaries = []
    case_codes = {parameter_case["case_run_code"] for parameter_case in parameter_cases}
    local_dataset_dir = local_artifact_path(
        local_input_root,
        f"python_analysis/{QUARTILE_LABEL}/{target.dataset_stem}",
    )
    if os.path.isdir(local_dataset_dir):
        for case_code in sorted(case_codes, key=lambda value: int(value) if str(value).isdigit() else str(value)):
            local_summary_path = os.path.join(local_dataset_dir, case_code, "python_analysis_summary.json")
            if not os.path.exists(local_summary_path):
                continue
            summary = read_json(local_summary_path)
            summary["averaged_results_path"] = copy_artifact(
                source_root,
                local_input_root,
                source_run_name,
                summary["averaged_results_path"],
                ssh_opts,
            )
            summaries.append(summary)
        return summaries

    for parameter_case in parameter_cases:
        summary_relative_path = (
            f"python_analysis/{QUARTILE_LABEL}/{target.dataset_stem}/"
            f"{parameter_case['case_run_code']}/python_analysis_summary.json"
        )
        if not endpoint_exists(source_root, *summary_relative_path.split("/"), ssh_opts=ssh_opts):
            continue
        local_summary_path = copy_artifact(
            source_root,
            local_input_root,
            source_run_name,
            summary_relative_path,
            ssh_opts,
        )
        summary = read_json(local_summary_path)
        summary["averaged_results_path"] = copy_artifact(
            source_root,
            local_input_root,
            source_run_name,
            summary["averaged_results_path"],
            ssh_opts,
        )
        summaries.append(summary)
    return summaries


def validate_python_summaries(
    localized_python_summaries: Sequence[dict],
    target: DatasetTarget,
    cases_by_id: Dict[str, dict],
) -> Dict[Tuple[str, str], dict]:
    summary_by_key = {}
    for summary in localized_python_summaries:
        case_id = summary.get("parameter_case_id")
        if case_id in cases_by_id:
            summary_by_key[(target.dataset_stem, case_id)] = summary
    missing = [case_id for case_id in cases_by_id if (target.dataset_stem, case_id) not in summary_by_key]
    if missing:
        raise ValueError(
            f"Missing {len(missing)} Python baseline summaries for {target.dataset_stem}; "
            f"first missing case: {missing[0]}"
        )
    return summary_by_key


def worker_count(max_workers: Optional[int], task_count: int) -> int:
    if task_count <= 0:
        return 1
    if max_workers is not None:
        if max_workers <= 0:
            raise ValueError("--max-workers must be greater than 0")
        return min(max_workers, task_count)
    return final.resolve_worker_count(task_count)


def run_java_stage(
    save_path: str,
    dataset_run: dict,
    parameter_cases: Sequence[dict],
    observation_prior: float,
    max_workers: Optional[int],
) -> List[dict]:
    stage_name = "stage4_java_analysis"
    total = len(parameter_cases)
    final.write_stage_checkpoint(
        save_path,
        stage_name,
        0,
        total,
        status="running",
        extra={"observation_prior": observation_prior},
    )
    summaries = []
    completed = 0
    with ThreadPoolExecutor(max_workers=worker_count(max_workers, total)) as executor:
        futures = [
            executor.submit(
                final.run_java_analysis_task,
                save_path,
                dataset_run,
                parameter_case,
                final.TIME_STEP,
                final.INTERNAL_STEPS,
                observation_prior,
            )
            for parameter_case in parameter_cases
        ]
        for future in tqdm(as_completed(futures), total=total, desc="Running Java analysis"):
            summaries.append(future.result())
            completed += 1
            final.write_stage_checkpoint(
                save_path,
                stage_name,
                completed,
                total,
                status="running",
                extra={"observation_prior": observation_prior},
            )
    summaries.sort(key=lambda item: (item["dataset_stem"], item["parameter_case_id"]))
    final.write_json(os.path.join(save_path, "java_analysis_summary.json"), summaries)
    final.mark_stage_complete(save_path, stage_name)
    final.write_stage_checkpoint(
        save_path,
        stage_name,
        total,
        total,
        status="completed",
        extra={"observation_prior": observation_prior},
    )
    return summaries


_GROUND_TRUTH_CACHE = {}
_GROUND_TRUTH_CACHE_LOCK = threading.Lock()


def _track_cache_key(path: str) -> Tuple[str, int, int]:
    stat = os.stat(path)
    return (os.path.abspath(path), int(stat.st_mtime_ns), int(stat.st_size))


def read_ground_truth_tracks_cached(path: str):
    key = _track_cache_key(path)
    with _GROUND_TRUTH_CACHE_LOCK:
        cached = _GROUND_TRUTH_CACHE.get(key)
        if cached is None:
            cached = final.sort_subject_tracks(final.sp.read_json(path))
            _GROUND_TRUTH_CACHE[key] = cached
        return cached


def clear_ground_truth_tracks_cache() -> None:
    with _GROUND_TRUTH_CACHE_LOCK:
        _GROUND_TRUTH_CACHE.clear()


def compute_precision_metrics_from_tracks(candidate, ground_truth, metrics_output: str, m: int = 10) -> dict:
    os.makedirs(os.path.dirname(metrics_output), exist_ok=True)
    candidate_subjects = sorted(candidate.keys(), key=int)
    ground_truth_subjects = sorted(ground_truth.keys(), key=int)
    if candidate_subjects != ground_truth_subjects:
        raise ValueError("Prediction and ground-truth tracks do not contain the same subjects.")

    metrics_dict = {}
    total_brier_score = 0.0
    total_ece = 0.0
    bins = final.sp.np.linspace(0, 1, m + 1)
    for subject_id in ground_truth_subjects:
        predicted = final.sp.np.asarray(candidate[subject_id], dtype=float)
        observed = final.sp.np.asarray(ground_truth[subject_id], dtype=float)
        if predicted.shape != observed.shape:
            raise ValueError(
                f"Subject {subject_id} has a different T "
                f"(prediction:{predicted.shape[0]}, ground_truth:{observed.shape[0]})."
            )
        brier_score = float(final.sp.np.mean((predicted - observed) ** 2))
        ece = 0.0
        total = predicted.shape[0]
        for bin_index in range(m):
            if bin_index == 0:
                in_bin = (predicted >= bins[bin_index]) & (predicted <= bins[bin_index + 1])
            else:
                in_bin = (predicted > bins[bin_index]) & (predicted <= bins[bin_index + 1])
            count = int(final.sp.np.sum(in_bin))
            if count:
                confidence = float(final.sp.np.mean(predicted[in_bin]))
                accuracy = float(final.sp.np.mean(observed[in_bin]))
                ece += (count / total) * abs(accuracy - confidence)
        total_brier_score += brier_score
        total_ece += ece
        metrics_dict[subject_id] = {
            "Brier Score": brier_score,
            "ECE": float(ece),
        }

    with open(metrics_output, "w", encoding="utf-8") as handle:
        json.dump(metrics_dict, handle, indent=4)

    subject_count = len(ground_truth_subjects)
    return {
        "metrics_output": metrics_output,
        "plots_dir": None,
        "plots_generated": False,
        "subject_count": subject_count,
        "mean_brier_score": float(total_brier_score / subject_count) if subject_count else float("nan"),
        "mean_ece": float(total_ece / subject_count) if subject_count else float("nan"),
    }


def compute_analysis_vs_simulation_metrics_from_tracks(
    ground_truth,
    analysis,
    baseline,
    run_dir: str,
    write_csvs: bool,
) -> dict:
    output_dir = None
    csv_summaries = None
    analysis_summary = final.compute_candidate_summary_fast(
        ground_truth,
        analysis,
        include_moving_avg_metrics=True,
    )
    baseline_summary = final.compute_candidate_summary_fast(
        ground_truth,
        baseline,
        include_moving_avg_metrics=True,
    )
    if write_csvs:
        output_dir = final.ensure_dir(os.path.join(run_dir, "plots", "analysis_vs_simulation"))
        csv_summaries = final.write_analysis_vs_simulation_csvs_fast(
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


def run_comparison_task_fast(
    save_path: str,
    quartile_label: str,
    runtime_budget_seconds: float,
    dataset_run: dict,
    parameter_case: dict,
    java_summary: dict,
    python_summary: dict,
    time_step_hours: float,
    iterations: int,
) -> dict:
    case_id = parameter_case["case_id"]
    case_run_code = parameter_case["case_run_code"]
    parameter_levels = parameter_case["levels"]
    comparison_base_dir = os.path.join(
        save_path,
        "comparison",
        quartile_label,
        dataset_run["dataset_stem"],
    )
    comparison_dir = final.resolve_case_output_dir(comparison_base_dir, parameter_case)
    summary_path = os.path.join(comparison_dir, "comparison_summary.json")
    run_id = f"{dataset_run['dataset_stem']}__{case_run_code}__{quartile_label}"

    analysis_path = final.ensure_analysis_tracks_path(
        java_summary["analysis_dir"],
        java_summary.get("analysis_path"),
        parameter_case["parameter_bundle"],
        iterations,
        time_step_hours,
        case_id=case_id,
    )
    baseline_path = python_summary["averaged_results_path"]
    ground_truth_path = dataset_run["ground_truth_path"]

    if os.path.exists(summary_path):
        cached_summary = read_json(summary_path)
        cached_java_analysis = cached_summary.get("java_analysis", {})
        cached_python_analysis = cached_summary.get("python_analysis", {})
        if (
            os.path.abspath(cached_summary.get("ground_truth_path", "")) == os.path.abspath(ground_truth_path)
            and os.path.abspath(cached_java_analysis.get("analysis_path", "")) == os.path.abspath(analysis_path)
            and os.path.abspath(cached_python_analysis.get("baseline_path", "")) == os.path.abspath(baseline_path)
        ):
            return cached_summary

    final.ensure_dir(comparison_dir)
    precision_dir = os.path.join(comparison_dir, "precision_metrics")
    final.ensure_dir(precision_dir)
    prediction_metrics_path = os.path.join(precision_dir, "metrics_prediction.json")
    baseline_metrics_path = os.path.join(precision_dir, "metrics_baseline.json")

    ground_truth = read_ground_truth_tracks_cached(ground_truth_path)
    analysis = final.sort_subject_tracks(final.sp.read_json(analysis_path))
    baseline = final.sort_subject_tracks(final.sp.read_json(baseline_path))

    prediction_metrics_summary = compute_precision_metrics_from_tracks(
        analysis,
        ground_truth,
        prediction_metrics_path,
        m=10,
    )
    baseline_metrics_summary = compute_precision_metrics_from_tracks(
        baseline,
        ground_truth,
        baseline_metrics_path,
        m=10,
    )
    comparison_metrics = compute_analysis_vs_simulation_metrics_from_tracks(
        ground_truth=ground_truth,
        analysis=analysis,
        baseline=baseline,
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
        "parameter_case_id": case_id,
        "parameter_case_code": case_run_code,
        "parameter_case_index": parameter_case["case_index"],
        "parameter_levels": parameter_levels,
        "java_analysis": {
            "analysis_dir": java_summary["analysis_dir"],
            "analysis_path": analysis_path,
            "analysis_wall_runtime_seconds": java_summary["analysis_wall_runtime_seconds"],
            "observation_prior": java_summary.get("observation_prior"),
            "parameter_bundle_path": java_summary["parameter_bundle_path"],
            "stpn_solution_path": java_summary["stpn_solution_path"],
            "observation_curve_path": java_summary["observation_curve_path"],
            "subject_curve_plots": {
                "output_dir": None,
                "csv_path": None,
                "epsilon": None,
                "plot_paths": [],
                "java_curve_plot_paths": [],
            },
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


def run_comparison_stage(
    save_path: str,
    dataset_run: dict,
    parameter_cases: Sequence[dict],
    java_summaries: Sequence[dict],
    python_summary_by_key: Dict[Tuple[str, str], dict],
    observation_prior: float,
    max_workers: Optional[int],
) -> List[dict]:
    java_summary_by_key = {
        (summary["dataset_stem"], summary["parameter_case_id"]): summary
        for summary in java_summaries
    }
    tasks = []
    for parameter_case in parameter_cases:
        key = (dataset_run["dataset_stem"], parameter_case["case_id"])
        tasks.append((parameter_case, java_summary_by_key[key], python_summary_by_key[key]))

    stage_name = "stage7_comparison"
    total = len(tasks)
    final.write_stage_checkpoint(
        save_path,
        stage_name,
        0,
        total,
        status="running",
        extra={"observation_prior": observation_prior},
    )
    comparison_summaries = []
    completed = 0
    with ThreadPoolExecutor(max_workers=worker_count(max_workers, total)) as executor:
        futures = [
            executor.submit(
                run_comparison_task_fast,
                save_path,
                QUARTILE_LABEL,
                java_summary["analysis_wall_runtime_seconds"],
                dataset_run,
                parameter_case,
                java_summary,
                python_summary,
                final.TIME_STEP,
                final.INTERNAL_STEPS,
            )
            for parameter_case, java_summary, python_summary in tasks
        ]
        for future in tqdm(as_completed(futures), total=total, desc="Running comparisons"):
            summary = future.result()
            summary["observation_prior"] = observation_prior
            summary.setdefault("java_analysis", {})["observation_prior"] = observation_prior
            comparison_summaries.append(summary)
            completed += 1
            final.write_stage_checkpoint(
                save_path,
                stage_name,
                completed,
                total,
                status="running",
                extra={"observation_prior": observation_prior},
            )
    comparison_summaries.sort(key=lambda item: (item["dataset_stem"], item["parameter_case_id"]))
    final.mark_stage_complete(save_path, stage_name)
    final.write_stage_checkpoint(
        save_path,
        stage_name,
        total,
        total,
        status="completed",
        extra={"observation_prior": observation_prior},
    )
    return comparison_summaries


def regenerate_selected_plots(
    save_path: str,
    comparison_summaries: List[dict],
    observation_prior: float,
    max_workers: Optional[int],
) -> Tuple[List[dict], str]:
    notes_by_run, selection_manifest = final.build_notes_map_and_selection_manifest(comparison_summaries)
    selection_manifest["observation_prior"] = observation_prior
    final.apply_notes_to_comparison_summaries(comparison_summaries, notes_by_run)
    selection_manifest_path = os.path.join(save_path, "selected_run_manifest.json")
    final.write_json(selection_manifest_path, selection_manifest)

    stage_name = "stage8_selected_plots"
    selected_summaries = [summary for summary in comparison_summaries if summary.get("note")]
    total = len(selected_summaries)
    final.write_stage_checkpoint(
        save_path,
        stage_name,
        0,
        total,
        status="running",
        extra={"observation_prior": observation_prior},
    )
    if total:
        updated_by_run_id = {}
        with ThreadPoolExecutor(max_workers=worker_count(max_workers, total)) as executor:
            futures = [
                executor.submit(
                    final.regenerate_selected_run_plots,
                    summary,
                    final.TIME_STEP,
                    final.INTERNAL_STEPS,
                )
                for summary in selected_summaries
            ]
            completed = 0
            for future in tqdm(as_completed(futures), total=total, desc="Generating selected plots"):
                updated_summary = future.result()
                updated_summary["observation_prior"] = observation_prior
                updated_summary.setdefault("java_analysis", {})["observation_prior"] = observation_prior
                updated_by_run_id[updated_summary["run_id"]] = updated_summary
                completed += 1
                final.write_stage_checkpoint(
                    save_path,
                    stage_name,
                    completed,
                    total,
                    status="running",
                    extra={"observation_prior": observation_prior},
                )
        comparison_summaries = [
            updated_by_run_id.get(summary["run_id"], summary)
            for summary in comparison_summaries
        ]
    final.mark_stage_complete(save_path, stage_name)
    final.write_stage_checkpoint(
        save_path,
        stage_name,
        total,
        total,
        status="completed",
        extra={"observation_prior": observation_prior},
    )
    return comparison_summaries, selection_manifest_path


def write_experiment_metadata(
    save_path: str,
    target: DatasetTarget,
    observation_prior: float,
    source_root: Endpoint,
    started_at: str,
    rows_written: int,
    selected_runs: int,
) -> None:
    write_json(
        os.path.join(save_path, "pomega_experiment_metadata.json"),
        {
            "dataset_key": target.key,
            "dataset_stem": target.dataset_stem,
            "observation_prior": observation_prior,
            "source_root": source_root.raw,
            "time_step_hours": final.TIME_STEP,
            "internal_steps": final.INTERNAL_STEPS,
            "quartile_label": QUARTILE_LABEL,
            "rows_written": rows_written,
            "selected_runs_for_plots": selected_runs,
            "started_at": started_at,
            "completed_at": datetime.now().isoformat(timespec="seconds"),
        },
    )


def write_completion_sentinel(save_path: str, payload: dict) -> None:
    payload = dict(payload)
    payload.setdefault("completed_at", datetime.now().isoformat(timespec="seconds"))
    write_json(os.path.join(save_path, COMPLETION_SENTINEL), payload)


def copy_experiment_back(
    local_experiment_dir: str,
    dest_root: Endpoint,
    run_label: str,
    target: DatasetTarget,
    prior: float,
    ssh_opts: Optional[str],
) -> None:
    remote_parts = (run_label, target.dataset_stem, f"pomega_{prior_label(prior)}")
    ensure_endpoint_dir(dest_root, *remote_parts, ssh_opts=ssh_opts)
    run_rsync(
        local_experiment_dir.rstrip(os.sep) + os.sep,
        endpoint_spec(dest_root, *remote_parts, trailing_slash=True),
        ssh_opts=ssh_opts,
    )


def copy_top_level_summaries_back(
    local_run_root: str,
    dest_root: Endpoint,
    run_label: str,
    ssh_opts: Optional[str],
) -> None:
    ensure_endpoint_dir(dest_root, run_label, ssh_opts=ssh_opts)
    for filename in ("pomega_sensitivity_summary.json", "pomega_sensitivity_summary.csv"):
        path = os.path.join(local_run_root, filename)
        if os.path.exists(path):
            run_rsync(path, endpoint_spec(dest_root, run_label, filename), ssh_opts=ssh_opts)


def run_one_experiment(
    source_root: Endpoint,
    dest_root: Endpoint,
    run_label: str,
    local_run_root: str,
    target: DatasetTarget,
    prior: float,
    parameter_cases: Sequence[dict],
    cases_by_id: Dict[str, dict],
    max_workers: Optional[int],
    ssh_opts: Optional[str],
) -> dict:
    experiment_name = os.path.join(target.dataset_stem, f"pomega_{prior_label(prior)}")
    experiment_dir = os.path.join(local_run_root, experiment_name)
    local_input_root = os.path.join(experiment_dir, "_input")
    save_path = os.path.join(experiment_dir, "result")
    started_at = datetime.now().isoformat(timespec="seconds")
    os.makedirs(local_input_root, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    source_run_name = endpoint_basename(source_root)
    print(f"Staging source artifacts for {target.dataset_stem}...")
    stage_source_artifacts(source_root, local_input_root, target, ssh_opts)
    dataset_runs, python_summaries = load_source_indexes(source_root, local_input_root, ssh_opts)
    source_dataset_run = select_dataset_run(dataset_runs, target, source_root, ssh_opts, local_input_root)
    dataset_run = localize_dataset_run(
        source_dataset_run,
        source_root,
        local_input_root,
        source_run_name,
        ssh_opts,
    )
    localized_python_summaries = localize_python_summaries(
        python_summaries,
        target.dataset_stem,
        source_root,
        local_input_root,
        source_run_name,
        ssh_opts,
    )
    if not localized_python_summaries:
        print(
            f"python_analysis_summary.json has no entries for {target.dataset_stem}; "
            "attempting per-case summary reconstruction."
        )
        localized_python_summaries = reconstruct_python_summaries_from_case_files(
            source_root,
            local_input_root,
            source_run_name,
            target,
            parameter_cases,
            ssh_opts,
        )
    python_summary_by_key = validate_python_summaries(localized_python_summaries, target, cases_by_id)

    final.write_json(os.path.join(save_path, "dataset_runs_summary.json"), [dataset_run])
    final.write_json(os.path.join(save_path, "python_analysis_summary_reused.json"), localized_python_summaries)
    final.write_json(
        os.path.join(save_path, "run_metadata.json"),
        {
            "mode": "pomega_sensitivity_analysis_only",
            "dataset_key": target.key,
            "dataset_stem": target.dataset_stem,
            "observation_prior": prior,
            "source_root": source_root.raw,
            "started_at": started_at,
            "time_step_hours": final.TIME_STEP,
            "internal_steps": final.INTERNAL_STEPS,
            "quartile_label": QUARTILE_LABEL,
            "reused_ground_truth": True,
            "reused_python_baseline": True,
        },
    )

    java_summaries = run_java_stage(save_path, dataset_run, parameter_cases, prior, max_workers)
    comparison_summaries = run_comparison_stage(
        save_path,
        dataset_run,
        parameter_cases,
        java_summaries,
        python_summary_by_key,
        prior,
        max_workers,
    )
    clear_ground_truth_tracks_cache()
    comparison_summaries, selection_manifest_path = regenerate_selected_plots(
        save_path,
        comparison_summaries,
        prior,
        max_workers,
    )
    output_paths = final.persist_comparison_outputs(save_path, QUARTILE_LABEL, comparison_summaries)
    selected_count = len([summary for summary in comparison_summaries if summary.get("note")])
    final.update_run_completion_metadata(
        save_path=save_path,
        quartile_label=QUARTILE_LABEL,
        comparison_summaries=comparison_summaries,
        output_paths=output_paths,
        selection_manifest_path=selection_manifest_path,
    )
    write_experiment_metadata(
        save_path,
        target,
        prior,
        source_root,
        started_at,
        output_paths["rows_written"],
        selected_count,
    )
    write_completion_sentinel(
        save_path,
        {
            "dataset_key": target.key,
            "dataset_stem": target.dataset_stem,
            "observation_prior": prior,
            "rows_written": output_paths["rows_written"],
            "selected_runs_for_plots": selected_count,
        },
    )

    copy_experiment_back(save_path, dest_root, run_label, target, prior, ssh_opts)
    return {
        "dataset_key": target.key,
        "dataset_stem": target.dataset_stem,
        "observation_prior": prior,
        "local_result_path": save_path,
        "dest_result_path": posixpath.join(run_label, target.dataset_stem, f"pomega_{prior_label(prior)}"),
        "rows_written": output_paths["rows_written"],
        "selected_runs_for_plots": selected_count,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run analysis-only P(omega) sensitivity experiments.")
    parser.add_argument("--paths-file", default="pomega_paths.txt")
    parser.add_argument(
        "--source-run",
        default=None,
        help="Optional run directory name under SOURCE_ROOT when SOURCE_ROOT points to a parent directory.",
    )
    parser.add_argument("--priors", nargs="+", default=[str(value) for value in DEFAULT_PRIORS])
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Skip experiments whose destination sentinel already exists.")
    parser.add_argument("--keep-local", action="store_true", help="Do not delete each staged experiment after copying results back.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    priors = parse_priors(args.priors)
    targets = resolve_targets(args.datasets)

    config = read_paths_file(args.paths_file)
    source_root = parse_endpoint(config["SOURCE_ROOT"])
    if args.source_run:
        source_root = endpoint_child(source_root, args.source_run)
    dest_root = parse_endpoint(config["DEST_ROOT"])
    local_work = os.path.abspath(config["LOCAL_WORK"])
    ssh_opts = config.get("RSYNC_SSH_OPTS")
    run_label = args.run_label or RUN_PREFIX + datetime.now().strftime("%Y%m%d-%H%M%S")
    local_run_root = os.path.join(local_work, run_label)
    os.makedirs(local_run_root, exist_ok=True)

    parameter_cases, cases_by_id = build_parameter_case_maps()
    summary_rows: List[dict] = []
    summary_json_path = os.path.join(local_run_root, "pomega_sensitivity_summary.json")
    summary_csv_path = os.path.join(local_run_root, "pomega_sensitivity_summary.csv")

    total = len(targets) * len(priors)
    progress = tqdm(
        [(target, prior) for target in targets for prior in priors],
        total=total,
        desc="P(omega) experiments",
        unit="experiment",
    )
    for completed, (target, prior) in enumerate(progress, start=1):
        progress.set_postfix_str(f"{target.dataset_stem}, p={prior}")
        remote_sentinel_parts = (
            run_label,
            target.dataset_stem,
            f"pomega_{prior_label(prior)}",
            COMPLETION_SENTINEL,
        )
        if args.resume and endpoint_exists(dest_root, *remote_sentinel_parts, ssh_opts=ssh_opts):
            row = {
                "dataset_key": target.key,
                "dataset_stem": target.dataset_stem,
                "observation_prior": prior,
                "status": "skipped_existing_destination",
                "dest_result_path": posixpath.join(run_label, target.dataset_stem, f"pomega_{prior_label(prior)}"),
                "completed_at": datetime.now().isoformat(timespec="seconds"),
            }
            summary_rows.append(row)
            write_json(summary_json_path, summary_rows)
            write_csv(
                summary_csv_path,
                summary_rows,
                ("dataset_key", "dataset_stem", "observation_prior", "status", "dest_result_path", "rows_written", "selected_runs_for_plots", "completed_at"),
            )
            copy_top_level_summaries_back(local_run_root, dest_root, run_label, ssh_opts)
            tqdm.write(f"[{completed}/{total}] skipped existing {target.dataset_stem} p={prior}")
            continue

        experiment_dir = os.path.join(local_run_root, target.dataset_stem, f"pomega_{prior_label(prior)}")
        tqdm.write(f"[{completed}/{total}] running {target.dataset_stem} with P(omega)={prior}")
        try:
            row = run_one_experiment(
                source_root=source_root,
                dest_root=dest_root,
                run_label=run_label,
                local_run_root=local_run_root,
                target=target,
                prior=prior,
                parameter_cases=parameter_cases,
                cases_by_id=cases_by_id,
                max_workers=args.max_workers,
                ssh_opts=ssh_opts,
            )
            row["status"] = "completed"
        finally:
            if not args.keep_local and os.path.isdir(experiment_dir):
                shutil.rmtree(experiment_dir)

        summary_rows.append(row)
        write_json(summary_json_path, summary_rows)
        write_csv(
            summary_csv_path,
            summary_rows,
            ("dataset_key", "dataset_stem", "observation_prior", "status", "dest_result_path", "rows_written", "selected_runs_for_plots", "completed_at"),
        )
        copy_top_level_summaries_back(local_run_root, dest_root, run_label, ssh_opts)

    if not args.keep_local:
        with open(os.path.join(local_run_root, ".completed"), "w", encoding="utf-8") as handle:
            handle.write(datetime.now().isoformat(timespec="seconds"))


if __name__ == "__main__":
    main()
