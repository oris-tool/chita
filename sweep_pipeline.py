import argparse
import copy
import contextlib
import csv
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import hashlib
import io
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import time
import traceback
import zipfile
from datetime import datetime
from xml.etree import ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import dataset
import dataset_graph as dataset_generator
import metrics as ranking_metrics
from compute_precision_metrics import process_and_save
from run_n_simulations import compute_confidence_intervals, run_dataset_simulations


TIME_LIMITS = [84]
N_SUBJECTS = [8]
TOTAL_INTERNAL_CONTACTS = [84]
CONVERGENCE_THRESHOLD = 1e-6
CONVERGENCE_ITERATIONS_CAP = 100_000
BASELINE_ITERATIONS_CAP = 100_000
MAX_TOP_PRECISION = 7
TIME_STEP_HOURS = 1.0
BASELINE_RUNTIME_MULTIPLIER = 1.0
REDUCED_PLOT_BUCKET_SIZE = 10
DEFAULT_INCLUDE_MOVING_AVG_METRICS = True
DATASET_SOURCE_GENERATED = "generated"
DATASET_SOURCE_D2 = "d2"
DEFAULT_DATASET_SOURCE = DATASET_SOURCE_GENERATED
OBSERVED_TEST_ABLATION_NONE = "none"
OBSERVED_TEST_ABLATION_FIRST_POSITIVE_ONLY = "first-positive-only"
OBSERVED_TEST_ABLATION_THROUGH_FIRST_POSITIVE = "through-first-positive"
OBSERVED_TEST_ABLATION_ONE_PER_DAY = "one-per-day"
OBSERVED_TEST_ABLATION_CHOICES = (
    OBSERVED_TEST_ABLATION_NONE,
    OBSERVED_TEST_ABLATION_FIRST_POSITIVE_ONLY,
    OBSERVED_TEST_ABLATION_THROUGH_FIRST_POSITIVE,
    OBSERVED_TEST_ABLATION_ONE_PER_DAY,
)
JAVA_PRECOMPUTE_CACHE_ROOT = os.path.join("sweeps", "_java_precompute_cache")
JAVA_PRECOMPUTE_SEED_SWEEP_ROOT = os.path.join("sweeps", "run_84_8_84_parallel_1h")
JAVA_PRECOMPUTE_SEED_RUN_PREFIX = "run_t84_s8_c84__"
DEFAULT_PARAMETER_ODS_PATH = os.path.join(os.path.expanduser("~"), "Downloads", "new-parameters.ods")
COMPACT_PLOT_DPI = 80
GRID_ROWS = 4
GRID_COLS = 2

PARAMETER_FAMILY_ORDER = (
    "infectiousness",
    "healing",
    "symptoms",
    "isolating",
    "symptomsOnset",
    "notificationToIsolation",
    "symptomaticPeriod",
)
PARAMETER_LEVEL_ORDER = ("lower", "mid", "upper")
GROUND_TRUTH_PARAMETER_LEVEL = "mid"

FAMILY_ABBREVIATIONS = {
    "infectiousness": "inf",
    "healing": "heal",
    "symptoms": "sym",
    "isolating": "iso",
    "symptomsOnset": "onset",
    "notificationToIsolation": "notif",
    "symptomaticPeriod": "symdur",
}

LEVEL_ABBREVIATIONS = {
    "lower": "lo",
    "mid": "mid",
    "upper": "up",
}

LEVEL_ALIASES = {
    "lower": "lower",
    "lower bound": "lower",
    "mid": "mid",
    "middle": "mid",
    "baseline": "mid",
    "upper": "upper",
    "upper bound": "upper",
}

UNIT_ALIASES = {
    "hour": "hours",
    "hours": "hours",
    "day": "days",
    "days": "days",
}

ODS_NAMESPACES = {
    "office": "urn:oasis:names:tc:opendocument:xmlns:office:1.0",
    "table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
    "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
}

ODS_TABLE_NAME_ATTR = f"{{{ODS_NAMESPACES['table']}}}name"
ODS_ROW_REPEAT_ATTR = f"{{{ODS_NAMESPACES['table']}}}number-rows-repeated"
ODS_COLUMN_REPEAT_ATTR = f"{{{ODS_NAMESPACES['table']}}}number-columns-repeated"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4)


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def dataset_generation_method_for_source(dataset_source):
    if dataset_source == DATASET_SOURCE_D2:
        return "dataset.create_datasets"
    if dataset_source == DATASET_SOURCE_GENERATED:
        return "dataset_graph.simulate_external_introduction"
    return None


def payload_sha256(payload):
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean_metric_value(metrics_path, metric_name):
    metrics_payload = read_json(metrics_path)
    values = [subject_metrics[metric_name] for subject_metrics in metrics_payload.values()]
    return float(sum(values) / len(values)) if values else math.nan


def finite_float(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def plot_overlay(
    series_a,
    series_b,
    title,
    ylabel,
    output_path,
    label_a="Analysis",
    label_b="Simulation",
    save_plot=True,
):
    if not save_plot:
        return
    plt.figure(figsize=(10, 6))
    plt.plot(series_a, marker="o", label=label_a)
    plt.plot(series_b, marker="x", label=label_b)
    plt.title(title)
    plt.xlabel("Timestep")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=COMPACT_PLOT_DPI)
    plt.close()


def save_series_csv(output_path, columns, rows):
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def chunked(sequence, chunk_size):
    for start in range(0, len(sequence), chunk_size):
        yield sequence[start:start + chunk_size]


def hide_unused_axes(axes, used_count):
    for axis in axes[used_count:]:
        axis.set_visible(False)


def resolve_worker_count(requested_workers, task_count):
    if task_count <= 0:
        return 1
    if requested_workers is not None:
        if requested_workers <= 0:
            raise ValueError("--max-workers must be greater than 0.")
        return min(requested_workers, task_count)

    cpu_count = os.cpu_count() or 1
    default_workers = max(1, cpu_count // 2)
    return min(default_workers, task_count)


def sanitized_time_step_label(time_step_hours):
    return str(time_step_hours).replace(".", "p")


def java_time_step_label(time_step_hours):
    label = f"{float(time_step_hours):.6f}"
    while "." in label and label.endswith("0"):
        label = label[:-1]
    if label.endswith("."):
        label = label[:-1]
    return label.replace("-", "m").replace(".", "p")


def java_cache_label(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))


def safe_path_label(value):
    label = java_cache_label(value).strip("._-")
    return label or "value"


def _normalize_label(value):
    return " ".join(str(value).strip().lower().split())


def _extract_level_label(row):
    for cell in row:
        label = LEVEL_ALIASES.get(_normalize_label(cell))
        if label is not None:
            return label
    return None


def _extract_unit_label(row):
    for cell in row:
        label = UNIT_ALIASES.get(_normalize_label(cell))
        if label is not None:
            return label
    return None


def _extract_numeric_values(row):
    values = []
    for cell in row:
        for token in re.findall(r"[-+]?\d+(?:[.,]\d+)?", str(cell)):
            values.append(float(token.replace(",", ".")))
    return values


def _parse_decimal_cell(cell_value, grouped_decimal_scale=None):
    normalized = str(cell_value).strip()
    if not normalized:
        raise ValueError("Expected a numeric cell, found an empty value.")

    if "." in normalized:
        return float(normalized)

    if grouped_decimal_scale is not None:
        digits = normalized.replace(",", "")
        return float(digits) / (10 ** grouped_decimal_scale)

    comma_count = normalized.count(",")
    if comma_count == 0:
        return float(normalized)
    if comma_count == 1:
        return float(normalized.replace(",", "."))
    raise ValueError(f"Ambiguous localized numeric value: {cell_value}")


def _cell_text(cell):
    paragraphs = []
    for paragraph in cell.findall("text:p", ODS_NAMESPACES):
        text = "".join(paragraph.itertext()).strip()
        if text:
            paragraphs.append(text)
    return " ".join(paragraphs).strip()


def load_ods_tables(path):
    with zipfile.ZipFile(path) as archive:
        with archive.open("content.xml") as handle:
            tree = ET.parse(handle)

    spreadsheet = tree.getroot().find(".//office:spreadsheet", ODS_NAMESPACES)
    if spreadsheet is None:
        raise ValueError(f"Could not find spreadsheet content in {path}.")

    tables = {}
    for table in spreadsheet.findall("table:table", ODS_NAMESPACES):
        sheet_name = table.attrib.get(ODS_TABLE_NAME_ATTR)
        rows = []
        for row in table.findall("table:table-row", ODS_NAMESPACES):
            expanded_row = []
            for cell in row.findall("table:table-cell", ODS_NAMESPACES):
                repeat_count = int(cell.attrib.get(ODS_COLUMN_REPEAT_ATTR, "1"))
                cell_value = _cell_text(cell)
                expanded_row.extend([cell_value] * repeat_count)
            while expanded_row and not expanded_row[-1]:
                expanded_row.pop()
            if not any(expanded_row):
                continue
            repeat_rows = int(row.attrib.get(ODS_ROW_REPEAT_ATTR, "1"))
            for _ in range(repeat_rows):
                rows.append(list(expanded_row))
        if sheet_name is not None:
            tables[sheet_name] = rows
    return tables


def _parse_distribution_sheet(rows, sheet_name):
    parsed_levels = {}
    recent_labeled_levels = []
    in_hour_block = False
    hour_block_index = 0

    for row in rows:
        level = _extract_level_label(row)
        unit = _extract_unit_label(row)
        numeric_values = _extract_numeric_values(row)

        if level is not None and numeric_values:
            if level == PARAMETER_LEVEL_ORDER[0]:
                recent_labeled_levels = []
            recent_labeled_levels.append(level)
            in_hour_block = False
            continue

        if unit == "hours" and not in_hour_block:
            if not recent_labeled_levels:
                raise ValueError(
                    f"Found hour-based rows in {sheet_name} before any labeled scenario rows."
                )
            in_hour_block = True
            hour_block_index = 0
        elif (
            not in_hour_block
            and level is None
            and len(recent_labeled_levels) == len(PARAMETER_LEVEL_ORDER)
            and len(numeric_values) >= 3
            and float(numeric_values[0]).is_integer()
        ):
            in_hour_block = True
            hour_block_index = 0

        if not in_hour_block:
            continue

        if len(numeric_values) < 3 or hour_block_index >= len(recent_labeled_levels):
            in_hour_block = False
            continue

        level = recent_labeled_levels[hour_block_index]
        parsed_levels[level] = {
            "unit_measure": "hours",
            "erlang_stages": int(round(numeric_values[0])),
            "erlang_lambda": float(numeric_values[1]),
            "exponential_lambda": float(numeric_values[2]),
        }
        hour_block_index += 1

    missing_levels = [level for level in PARAMETER_LEVEL_ORDER if level not in parsed_levels]
    if missing_levels:
        raise ValueError(
            f"Missing hour-based rows for {sheet_name}: {', '.join(missing_levels)}"
        )
    return parsed_levels


def _parse_symptoms_sheet(rows):
    parsed_levels = {}
    for row in rows:
        level = _extract_level_label(row)
        if level is None:
            continue
        numeric_values = _extract_numeric_values(row)
        if not numeric_values:
            continue
        transition = {"unit_measure": "probability", "true": float(numeric_values[0])}
        if len(numeric_values) > 1:
            transition["false"] = float(numeric_values[1])
        parsed_levels[level] = transition

    missing_levels = [level for level in PARAMETER_LEVEL_ORDER if level not in parsed_levels]
    if missing_levels:
        raise ValueError(
            f"Missing symptomatic-probability rows: {', '.join(missing_levels)}"
        )
    return parsed_levels


def _parse_day_distribution_sheet(rows, sheet_name):
    parsed_levels = {}
    for row in rows:
        level = _extract_level_label(row)
        unit = _extract_unit_label(row)
        if level is None or unit != "days":
            continue
        parsed_levels[level] = {
            "unit_measure": "days",
            "erlang_stages": int(round(_parse_decimal_cell(row[5]))),
            "erlang_lambda": _parse_decimal_cell(row[6]),
            "exponential_lambda": _parse_decimal_cell(row[7]),
        }

    missing_levels = [level for level in PARAMETER_LEVEL_ORDER if level not in parsed_levels]
    if missing_levels:
        raise ValueError(
            f"Missing day-based rows for {sheet_name}: {', '.join(missing_levels)}"
        )
    return parsed_levels


def _parse_notification_to_isolation_sheet(rows):
    parsed_levels = {}
    for row in rows:
        level = _extract_level_label(row)
        unit = _extract_unit_label(row)
        if level is None or unit != "days":
            continue
        parsed_levels[level] = {
            "unit_measure": "days",
            "distribution": "hyperexponential",
            "p1": _parse_decimal_cell(row[7], grouped_decimal_scale=5),
            "p2": _parse_decimal_cell(row[8], grouped_decimal_scale=5),
            "lambda1": _parse_decimal_cell(row[9], grouped_decimal_scale=5),
            "lambda2": _parse_decimal_cell(row[10], grouped_decimal_scale=5),
        }

    missing_levels = [level for level in PARAMETER_LEVEL_ORDER if level not in parsed_levels]
    if missing_levels:
        raise ValueError(
            "Missing day-based rows for notification-to-isolation: "
            + ", ".join(missing_levels)
        )
    return parsed_levels


def load_parameter_space_from_ods(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Parameter spreadsheet not found: {path}")

    tables = load_ods_tables(path)
    transitions = {
        "infectiousness": _parse_distribution_sheet(tables["infectiousness"], "infectiousness"),
        "healing": _parse_distribution_sheet(tables["healing"], "healing"),
        "symptoms": _parse_symptoms_sheet(tables["symptoms"]),
        "isolating": _parse_distribution_sheet(tables["isolating"], "isolating"),
        "symptomsOnset": _parse_distribution_sheet(tables["symptomsOnset"], "symptomsOnset"),
        "notificationToIsolation": _parse_notification_to_isolation_sheet(tables["notification-to-isolation"]),
        "symptomaticPeriod": _parse_day_distribution_sheet(tables["symptomatic_period"], "symptomatic_period"),
    }
    return {
        "source_path": os.path.abspath(path),
        "source_format": "ods",
        "unit_measure": "mixed",
        "levels": list(PARAMETER_LEVEL_ORDER),
        "transitions": transitions,
    }


def build_parameter_case_id(level_selection):
    parts = []
    for family in PARAMETER_FAMILY_ORDER:
        level = level_selection[family]
        parts.append(f"{FAMILY_ABBREVIATIONS[family]}_{LEVEL_ABBREVIATIONS[level]}")
    return "__".join(parts)


def resolve_parameter_bundle(parameter_space, level_selection):
    return {
        "source_path": parameter_space["source_path"],
        "source_format": parameter_space["source_format"],
        "unit_measure": parameter_space["unit_measure"],
        "case_id": build_parameter_case_id(level_selection),
        "levels": {family: level_selection[family] for family in PARAMETER_FAMILY_ORDER},
        "transitions": {
            family: dict(parameter_space["transitions"][family][level_selection[family]])
            for family in PARAMETER_FAMILY_ORDER
        },
    }


def enumerate_parameter_cases(parameter_space, mode="all-combinations"):
    if mode == "all-combinations":
        level_combinations = itertools.product(
            PARAMETER_LEVEL_ORDER,
            repeat=len(PARAMETER_FAMILY_ORDER),
        )
    elif mode == "aligned-scenarios":
        level_combinations = (
            (level,) * len(PARAMETER_FAMILY_ORDER)
            for level in PARAMETER_LEVEL_ORDER
        )
    else:
        raise ValueError(f"Unsupported parameter case mode: {mode}")

    cases = []
    for levels in level_combinations:
        level_selection = {
            family: level
            for family, level in zip(PARAMETER_FAMILY_ORDER, levels)
        }
        cases.append(resolve_parameter_bundle(parameter_space, level_selection))
    return cases


def resolve_uniform_parameter_bundle(parameter_space, level):
    normalized_level = LEVEL_ALIASES.get(_normalize_label(level))
    if normalized_level is None:
        raise ValueError(f"Unsupported parameter level: {level}")

    level_selection = {
        family: normalized_level
        for family in PARAMETER_FAMILY_ORDER
    }
    return resolve_parameter_bundle(parameter_space, level_selection)


def manifest_row_for_bundle(bundle):
    row = {
        "case_id": bundle["case_id"],
        "infectiousness_level": bundle["levels"]["infectiousness"],
        "healing_level": bundle["levels"]["healing"],
        "symptoms_level": bundle["levels"]["symptoms"],
        "isolating_level": bundle["levels"]["isolating"],
        "symptoms_onset_level": bundle["levels"]["symptomsOnset"],
        "notification_to_isolation_level": bundle["levels"]["notificationToIsolation"],
        "symptomatic_period_level": bundle["levels"]["symptomaticPeriod"],
    }
    for family in PARAMETER_FAMILY_ORDER:
        transition = bundle["transitions"][family]
        for key, value in transition.items():
            row[f"{family}_{key}"] = value
    return row


def write_parameter_case_manifest(output_root, parameter_cases):
    manifest_json_path = os.path.join(output_root, "parameter_case_manifest.json")
    manifest_csv_path = os.path.join(output_root, "parameter_case_manifest.csv")
    write_json(manifest_json_path, parameter_cases)

    rows = [manifest_row_for_bundle(bundle) for bundle in parameter_cases]
    with open(manifest_csv_path, "w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0].keys()) if rows else ["case_id"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "json_path": manifest_json_path,
        "csv_path": manifest_csv_path,
    }


def write_dataset_generation_parameters_markdown(
    output_root,
    seed_base,
    time_step_hours,
    baseline_runtime_multiplier,
    include_moving_avg_metrics,
    dataset_source,
    tests_enabled,
    observed_test_ablation,
    java_precompute_cache_root,
    java_precompute_seed_sweep_root,
    parameter_space,
    ground_truth_parameter_bundle,
    parameter_case_mode,
    parameter_case_count,
    parameter_manifest_paths,
    generate_plot_images,
    reduce_plots,
):
    content = f"""# Dataset Generation Parameters

This file documents the parameters used by `sweep_pipeline.py` to create the datasets for this sweep.

## Sweep Configuration

- `TIME_LIMITS` (days): {TIME_LIMITS}
- `N_SUBJECTS`: {N_SUBJECTS}
- `TOTAL_INTERNAL_CONTACTS`: {TOTAL_INTERNAL_CONTACTS}
- `seed_base`: {seed_base}
- `TIME_STEP_HOURS`: {time_step_hours}
- `baseline_runtime_multiplier`: {baseline_runtime_multiplier}
- `moving_avg_metrics_enabled`: {include_moving_avg_metrics}
- `dataset_source`: {dataset_source}
- `tests_enabled_in_raw_dataset`: {tests_enabled}
- `observed_test_ablation`: {observed_test_ablation}
- `java_precompute_cache_root`: {java_precompute_cache_root}
- `java_precompute_seed_sweep_root`: {java_precompute_seed_sweep_root}
- `parameter_ods_path`: {parameter_space["source_path"]}
- `ground_truth_parameter_case_id`: {ground_truth_parameter_bundle["case_id"]}
- `parameter_case_mode`: {parameter_case_mode}
- `parameter_case_count`: {parameter_case_count}
- `parameter_manifest_csv`: {parameter_manifest_paths["csv_path"]}
- `parameter_manifest_json`: {parameter_manifest_paths["json_path"]}
- `generate_plot_images`: {generate_plot_images}
- `reduce_plots`: {reduce_plots}
- Shared ground-truth seed rule: `seed = seed_base + dataset_combination_index`
- Per-run baseline seed rule: `seed = seed_base + dataset_combination_index * parameter_case_count + parameter_case_index + 1`

## Parameter Cases

- The sweep loads the lower/mid/upper parameter values from the ODS spreadsheet.
- The hour-based rows are used for `infectiousness`, `healing`, `isolating`, and `symptomsOnset`.
- The day-based rows are used for `notification-to-isolation` and `symptomatic_period`, then converted to hours inside the simulator after sampling.
- The `symptomatic_period` rows are day-based because that sheet is only provided in days in the ODS file.
- The `symptoms` sheet is unitless and is used as a probability.

## Parameter Roles

- The dataset-generation step itself does not depend on the lower/mid/upper transition bundle.
- Dataset selection mode: `{dataset_source}`.
- When `dataset_source` is `d2`, the sweep generates the raw dataset with the same `dataset.create_datasets(...)` recipe used for D2 in `run_n_simulations.py`, then the usual convergence run computes tests, symptoms, and isolation behavior on top of it.
- Raw test events are {"kept" if tests_enabled else "removed"} before the sweep starts.
- The shared observed trace copied into each run uses the test-ablation mode `{observed_test_ablation}`. This only changes the evidence file consumed by Java; the raw dataset and the Python baseline simulation inputs stay unchanged.
- The converged Python ground-truth run is computed once per `(time_limit_days, n_subjects, total_internal_contacts)` tuple using the fixed median bundle `{ground_truth_parameter_bundle["case_id"]}`.
- Each swept run then reuses that shared ground truth while applying its own bundle to both the timed Python baseline simulation and the Java analysis.
- For each swept run, the Python baseline runtime budget is `java_analysis_runtime_seconds * baseline_runtime_multiplier`.
- Java precompute artifacts first try to reuse the known-good files from `java_precompute_seed_sweep_root`, then fall back to the shared cache under `java_precompute_cache_root`, and only recompute when neither source contains the matching bundle.

## Per-Run Dataset Parameters

For each run, `generate_dataset()` uses one of these source-dataset methods:

- Default mode: `dataset_graph.simulate_external_introduction()` with `n_nodes = n_subjects`, `tmax_after_intro = time_limit_days * 24`, `max_intro_time = min(48.0, time_limit_days * 24)`, `total_internal_contacts = selected TOTAL_INTERNAL_CONTACTS value`, and `seed = derived per-run seed`.
- `--d2` mode: `dataset.create_datasets()` with `fine_grained = True`, `internal_contacts = selected TOTAL_INTERNAL_CONTACTS value`, `max_contacts = 6`, and the same derived per-run seed. This matches the raw D2-generation path used in `run_n_simulations.py`.

Datasets are saved with filenames like:

- `dataset_s{{n_subjects}}_t{{time_limit_days}}_c{{total_internal_contacts}}.json`

## Dataset Sampler Defaults In Effect

These come from `dataset_graph.py`:

- Contact graph: `nx.complete_graph(n_nodes)`
- `transmission_rate = 0.6`
- `recovery_rate = 0.2`
- `max_external_contacts_per_node = 8`
- Introduced node: sampled uniformly at random from the graph nodes
- Introduction time: sampled uniformly in `[0.0, max_intro_time]`
- External-contact risk factor: sampled uniformly in `[0.0, 0.99]`
- Internal-contact risk factor: sampled uniformly in `[0.0, 0.99]`
- Internal group defaults:
  - `max_group_size = 4`
  - `group_event_probability = 0.35`
- Number of tests per subject:
  - `randint(0, 2 * time_limit_days // 7)`{" (disabled in this sweep)" if not tests_enabled else ""}
- Event times are stored as floating-point hours in the dataset JSON

## Dataset Event Construction Notes

- External events are generated before simulation output is converted into dataset events
- Internal transmission events come from the epidemic simulation and are supplemented with extra internal contacts when needed to reach the requested total
- Tests are {"added independently per subject using the sampler above" if tests_enabled else "disabled for this sweep"}
- The saved dataset metadata also records:
  - `run_name`
  - `dataset_path`
  - `n_subjects`
  - `time_limit`
  - `n_contacts`
  - `seed`
"""
    output_path = os.path.join(output_root, "dataset_generation_parameters.md")
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(content)
    return output_path


def normalize_subject_series(series_payload):
    if not isinstance(series_payload, dict) or not series_payload:
        raise ValueError("Expected a non-empty subject-to-series mapping.")

    normalized = {}
    for subject_id, values in series_payload.items():
        normalized[str(subject_id)] = list(values)

    return {
        subject_id: normalized[subject_id]
        for subject_id in sorted(normalized, key=int)
    }


def create_analysis_subject_curve_plots(
    run_dir,
    run_name,
    analysis_path,
    ground_truth_path,
    baseline_path,
    granularity=1.0,
    ground_truth_sample_size=None,
    save_plots=True,
):
    analysis = normalize_subject_series(read_json(analysis_path))
    ground_truth = normalize_subject_series(read_json(ground_truth_path))
    baseline = normalize_subject_series(read_json(baseline_path))

    analysis_subjects = list(analysis.keys())
    ground_truth_subjects = list(ground_truth.keys())
    baseline_subjects = list(baseline.keys())
    if analysis_subjects != ground_truth_subjects or baseline_subjects != ground_truth_subjects:
        raise ValueError(
            "Analysis, baseline, and ground-truth files do not contain the same subjects."
        )

    output_dir = ensure_dir(os.path.join(run_dir, "plots", "analysis_subject_curves"))
    dkw_result = compute_confidence_intervals(
        ground_truth,
        sample_size=ground_truth_sample_size,
    )
    csv_rows = []
    generated_plots = []

    for subject_id in ground_truth_subjects:
        ground_truth_values = ground_truth[subject_id]
        analysis_values = analysis[subject_id]
        baseline_values = baseline[subject_id]
        lower = dkw_result["band"][subject_id]["lower"]
        upper = dkw_result["band"][subject_id]["upper"]

        if not (
            len(ground_truth_values)
            == len(analysis_values)
            == len(baseline_values)
            == len(lower)
            == len(upper)
        ):
            raise ValueError(f"Inconsistent trajectory length for subject {subject_id}.")

        time_axis = np.arange(len(ground_truth_values)) * granularity

        for time_index, (
            ground_truth_value,
            lower_value,
            upper_value,
            analysis_value,
            baseline_value,
        ) in enumerate(
            zip(ground_truth_values, lower, upper, analysis_values, baseline_values)
        ):
            csv_rows.append(
                [
                    subject_id,
                    time_index,
                    time_index * granularity,
                    ground_truth_value,
                    lower_value,
                    upper_value,
                    analysis_value,
                    baseline_value,
                ]
            )

    if save_plots:
        subjects_per_page = GRID_ROWS * GRID_COLS
        for page_index, subject_page in enumerate(chunked(ground_truth_subjects, subjects_per_page), start=1):
            fig, axes = plt.subplots(
                GRID_ROWS,
                GRID_COLS,
                figsize=(14, 16),
                sharex=True,
                sharey=True,
            )
            flat_axes = list(axes.flat)
            legend_handles = None

            for axis, subject_id in zip(flat_axes, subject_page):
                ground_truth_values = ground_truth[subject_id]
                analysis_values = analysis[subject_id]
                baseline_values = baseline[subject_id]
                lower = dkw_result["band"][subject_id]["lower"]
                upper = dkw_result["band"][subject_id]["upper"]
                time_axis = np.arange(len(ground_truth_values)) * granularity

                ground_truth_line, = axis.plot(
                    time_axis,
                    ground_truth_values,
                    linewidth=1.6,
                    color="tab:blue",
                    label="Ground truth mean",
                )
                ground_truth_band = axis.fill_between(
                    time_axis,
                    lower,
                    upper,
                    alpha=0.2,
                    color="tab:blue",
                    label="Ground truth DKW band",
                )
                analysis_line, = axis.plot(
                    time_axis,
                    analysis_values,
                    linewidth=1.6,
                    color="tab:orange",
                    label="Java analysis",
                )
                baseline_line, = axis.plot(
                    time_axis,
                    baseline_values,
                    linewidth=1.6,
                    color="tab:green",
                    label="Simulation",
                )
                axis.set_title(f"Subject {subject_id}")
                axis.set_ylim(0.0, 1.0)
                axis.grid(True, alpha=0.3)

                if legend_handles is None:
                    legend_handles = [
                        ground_truth_line,
                        ground_truth_band,
                        analysis_line,
                        baseline_line,
                    ]

            hide_unused_axes(flat_axes, len(subject_page))
            fig.suptitle(f"{run_name} Subject Curves", fontsize=14)
            fig.supxlabel("Time")
            fig.supylabel("Probability")
            if legend_handles is not None:
                fig.legend(
                    legend_handles,
                    [
                        "Ground truth mean",
                        "Ground truth DKW band",
                        "Java analysis",
                        "Simulation",
                    ],
                    loc="upper center",
                    ncol=4,
                    frameon=False,
                )
            fig.tight_layout(rect=(0.03, 0.03, 1.0, 0.94))

            output_name = f"{run_name}_analysis_subject_curves.png"
            if len(ground_truth_subjects) > subjects_per_page:
                output_name = f"{run_name}_analysis_subject_curves_page_{page_index}.png"
            output_path = os.path.join(output_dir, output_name)
            fig.savefig(output_path, dpi=COMPACT_PLOT_DPI)
            plt.close(fig)
            generated_plots.append(output_path)

    csv_path = os.path.join(output_dir, f"{run_name}_analysis_subject_curves.csv")
    save_series_csv(
        csv_path,
        [
            "subject_id",
            "time_index",
            "time",
            "ground_truth_mean_probability",
            "ground_truth_lower_band",
            "ground_truth_upper_band",
            "analysis_probability",
            "simulation_probability",
        ],
        csv_rows,
    )

    return {
        "output_dir": output_dir,
        "csv_path": csv_path,
        "epsilon": dkw_result["epsilon"],
        "plot_paths": generated_plots,
        "java_curve_plot_paths": [],
    }


def compute_candidate_summary(ground_truth, candidate, include_moving_avg_metrics=True):
    summary = {}
    tau, p_value_tau = ranking_metrics.compute_kendalls_tau_correlation(ground_truth, candidate)
    spearman, p_value_sp = ranking_metrics.compute_spearmans_correlation(ground_truth, candidate)
    summary["tau"] = tau
    summary["tau_p_value"] = p_value_tau
    summary["spearman"] = spearman
    summary["spearman_p_value"] = p_value_sp
    summary["mrr"] = ranking_metrics.compute_mrr(ground_truth, candidate)
    for top_k in range(1, min(MAX_TOP_PRECISION, len(ground_truth)) + 1):
        precision = ranking_metrics.compute_top_n_precision(ground_truth, candidate, top_k)
        summary[f"top_{top_k}_precision_mean"] = float(np.mean(precision))
        if include_moving_avg_metrics:
            summary[f"top_{top_k}_precision_moving_avg_mean"] = summary[f"top_{top_k}_precision_mean"]
    return summary


def create_analysis_vs_simulation_plots(
    run_dir,
    run_name,
    ground_truth_path,
    analysis_path,
    baseline_path,
    include_moving_avg_metrics=True,
    save_plots=True,
):
    ground_truth = read_json(ground_truth_path)
    analysis = read_json(analysis_path)
    baseline = read_json(baseline_path)

    output_dir = ensure_dir(os.path.join(run_dir, "plots", "analysis_vs_simulation"))
    generated_plots = []

    tau_analysis, _ = ranking_metrics.compute_kendalls_tau_correlation_per_timestep(ground_truth, analysis)
    tau_baseline, _ = ranking_metrics.compute_kendalls_tau_correlation_per_timestep(ground_truth, baseline)
    save_series_csv(
        os.path.join(output_dir, "kendall_correlation_data.csv"),
        ["timestep", "analysis_kendall", "simulation_kendall"],
        [[index, tau_analysis[index], tau_baseline[index]] for index in range(len(tau_analysis))],
    )

    spearman_analysis, _ = ranking_metrics.compute_spearmans_correlation_per_timestep(ground_truth, analysis)
    spearman_baseline, _ = ranking_metrics.compute_spearmans_correlation_per_timestep(ground_truth, baseline)
    save_series_csv(
        os.path.join(output_dir, "spearman_correlation_data.csv"),
        ["timestep", "analysis_spearman", "simulation_spearman"],
        [[index, spearman_analysis[index], spearman_baseline[index]] for index in range(len(spearman_analysis))],
    )

    scalar_rows = []
    max_top_precision = min(MAX_TOP_PRECISION, len(ground_truth))
    top_precision_series = []
    for top_k in range(1, max_top_precision + 1):
        analysis_precision = ranking_metrics.compute_top_n_precision(ground_truth, analysis, top_k)
        baseline_precision = ranking_metrics.compute_top_n_precision(ground_truth, baseline, top_k)
        save_series_csv(
            os.path.join(output_dir, f"top_{top_k}_precision_data.csv"),
            ["timestep", f"analysis_top_{top_k}_precision", f"simulation_top_{top_k}_precision"],
            [[index, analysis_precision[index], baseline_precision[index]] for index in range(len(analysis_precision))],
        )
        top_precision_series.append((top_k, analysis_precision, baseline_precision))

        scalar_rows.append(
            {
                "comparison": f"top_{top_k}",
                "analysis_mean": float(np.mean(analysis_precision)),
                "simulation_mean": float(np.mean(baseline_precision)),
            }
        )
        if include_moving_avg_metrics:
            scalar_rows[-1]["analysis_moving_avg_mean"] = float(np.mean(analysis_precision))
            scalar_rows[-1]["simulation_moving_avg_mean"] = float(np.mean(baseline_precision))

    if save_plots:
        correlation_path = os.path.join(output_dir, "correlation_plots.png")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
        correlation_specs = [
            ("Kendall's Tau", tau_analysis, tau_baseline),
            ("Spearman's Rho", spearman_analysis, spearman_baseline),
        ]
        for axis, (title, analysis_series, baseline_series) in zip(axes, correlation_specs):
            axis.plot(analysis_series, linewidth=1.8, color="tab:orange", label="Analysis")
            axis.plot(baseline_series, linewidth=1.8, color="tab:green", label="Simulation")
            axis.set_title(title)
            axis.set_xlabel("Timestep")
            axis.grid(True, alpha=0.3)
        axes[0].set_ylabel("Correlation")
        axes[0].legend(frameon=False)
        fig.suptitle(f"{run_name} Correlation Comparison", fontsize=14)
        fig.tight_layout()
        fig.savefig(correlation_path, dpi=COMPACT_PLOT_DPI)
        plt.close(fig)
        generated_plots.append(correlation_path)

        precision_path = os.path.join(output_dir, "top_precision_grid.png")
        fig, axes = plt.subplots(
            GRID_ROWS,
            GRID_COLS,
            figsize=(14, 16),
            sharex=True,
            sharey=True,
        )
        flat_axes = list(axes.flat)
        for axis, (top_k, analysis_precision, baseline_precision) in zip(flat_axes, top_precision_series):
            axis.plot(analysis_precision, linewidth=1.8, color="tab:orange", label="Analysis")
            axis.plot(baseline_precision, linewidth=1.8, color="tab:green", label="Simulation")
            axis.set_title(f"Top-{top_k} Precision")
            axis.set_ylim(0.0, 1.0)
            axis.grid(True, alpha=0.3)
        hide_unused_axes(flat_axes, len(top_precision_series))
        fig.suptitle(f"{run_name} Top Precision", fontsize=14)
        fig.supxlabel("Timestep")
        fig.supylabel("Precision")
        if top_precision_series:
            fig.legend(
                [flat_axes[0].lines[0], flat_axes[0].lines[1]],
                ["Analysis", "Simulation"],
                loc="upper center",
                ncol=2,
                frameon=False,
            )
        fig.tight_layout(rect=(0.03, 0.03, 1.0, 0.95))
        fig.savefig(precision_path, dpi=COMPACT_PLOT_DPI)
        plt.close(fig)
        generated_plots.append(precision_path)

    analysis_summary = compute_candidate_summary(
        ground_truth,
        analysis,
        include_moving_avg_metrics=include_moving_avg_metrics,
    )
    baseline_summary = compute_candidate_summary(
        ground_truth,
        baseline,
        include_moving_avg_metrics=include_moving_avg_metrics,
    )
    save_series_csv(
        os.path.join(output_dir, "metrics_results.csv"),
        ["metric", "analysis", "simulation"],
        [
            ["tau", analysis_summary["tau"], baseline_summary["tau"]],
            ["spearman", analysis_summary["spearman"], baseline_summary["spearman"]],
            ["mrr", analysis_summary["mrr"], baseline_summary["mrr"]],
        ]
        + [
            [row["comparison"], row["analysis_mean"], row["simulation_mean"]]
            for row in scalar_rows
        ],
    )

    return {
        "analysis": analysis_summary,
        "simulation": baseline_summary,
        "output_dir": output_dir,
        "plot_paths": generated_plots,
    }


def find_generated_file(run_dir, suffix):
    matches = [name for name in os.listdir(run_dir) if name.endswith(suffix)]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one file ending with '{suffix}' in {run_dir}, found: {matches}")
    return os.path.join(run_dir, matches[0])


def parse_java_major_version(version_output):
    match = re.search(r'"(\d+)(?:\.(\d+))?', version_output)
    if not match:
        return None
    major = int(match.group(1))
    if major == 1 and match.group(2):
        return int(match.group(2))
    return major


def java_major_version(java_executable):
    result = subprocess.run(
        [java_executable, "-version"],
        capture_output=True,
        text=True,
        check=False,
    )
    output = (result.stderr or "") + "\n" + (result.stdout or "")
    return parse_java_major_version(output)


def resolve_java_executable():
    candidates = []
    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        candidates.append(os.path.join(java_home, "bin", "java.exe"))
        candidates.append(os.path.join(java_home, "bin", "java"))

    candidates.extend(
        [
            r"C:\Program Files\Eclipse Adoptium\jdk-21.0.3.9-hotspot\bin\java.exe",
            r"C:\Program Files\Eclipse Adoptium\jdk-21\bin\java.exe",
            "java",
        ]
    )

    seen = set()
    for candidate in candidates:
        normalized = os.path.normcase(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)

        executable = candidate
        if candidate != "java" and not os.path.exists(candidate):
            continue

        major_version = java_major_version(executable)
        if major_version is not None and major_version >= 21:
            return executable

    raise RuntimeError(
        "No compatible Java runtime found. A Java 21+ runtime is required for STPNAnalysis."
    )


def resolve_optional_jar(candidates, jar_name):
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise RuntimeError(f"Required Java dependency not found: {jar_name}")


def prepare_java_precompute_cache(repo_root, parameter_bundle, time_step_hours, java_class_fingerprint):
    cache_root = ensure_dir(os.path.join(repo_root, JAVA_PRECOMPUTE_CACHE_ROOT))
    case_id = parameter_bundle.get("case_id", "parameter_bundle")
    bundle_fingerprint = payload_sha256(parameter_bundle)
    cache_key = (
        f"{safe_path_label(case_id)}"
        f"__bundle_{bundle_fingerprint[:16]}"
        f"__ts{sanitized_time_step_label(time_step_hours)}"
        f"__java_{java_class_fingerprint[:12]}"
    )
    cache_dir = ensure_dir(os.path.join(cache_root, cache_key))
    cached_parameter_bundle_path = os.path.join(cache_dir, "parameter_bundle.json")
    if not os.path.exists(cached_parameter_bundle_path):
        write_json(cached_parameter_bundle_path, parameter_bundle)

    manifest_path = os.path.join(cache_dir, "cache_manifest.json")
    if not os.path.exists(manifest_path):
        write_json(
            manifest_path,
            {
                "case_id": case_id,
                "time_step_hours": time_step_hours,
                "java_precompute_cache_key": cache_key,
                "parameter_bundle_sha256": bundle_fingerprint,
                "java_class_sha256": java_class_fingerprint,
                "parameter_bundle_path": cached_parameter_bundle_path,
            },
        )

    observation_curve_filename = (
        f"observation_curves_{java_cache_label(case_id)}"
        f"_step{java_time_step_label(time_step_hours)}.csv"
    )
    stpn_solution_filename = f"stpn_solution_ts{sanitized_time_step_label(time_step_hours)}.csv"
    return {
        "cache_root": cache_root,
        "cache_dir": cache_dir,
        "cache_key": cache_key,
        "manifest_path": manifest_path,
        "parameter_bundle_path": cached_parameter_bundle_path,
        "stpn_solution_filename": stpn_solution_filename,
        "stpn_solution_path": os.path.join(cache_dir, stpn_solution_filename),
        "observation_curve_filename": observation_curve_filename,
        "observation_curve_path": os.path.join(cache_dir, observation_curve_filename),
    }


def try_seed_java_precompute_cache(repo_root, parameter_bundle, time_step_hours, cache_entry):
    seed_root = os.path.join(repo_root, JAVA_PRECOMPUTE_SEED_SWEEP_ROOT)
    seed_run_dir = os.path.join(
        seed_root,
        f"{JAVA_PRECOMPUTE_SEED_RUN_PREFIX}{parameter_bundle['case_id']}",
    )
    if not os.path.isdir(seed_run_dir):
        return None

    seed_parameter_bundle_path = os.path.join(seed_run_dir, "parameter_bundle.json")
    if not os.path.exists(seed_parameter_bundle_path):
        return None

    try:
        seed_parameter_bundle = read_json(seed_parameter_bundle_path)
    except (OSError, json.JSONDecodeError):
        return None
    if seed_parameter_bundle != parameter_bundle:
        return None

    seed_stpn_solution_path = os.path.join(seed_run_dir, cache_entry["stpn_solution_filename"])
    seed_observation_curve_path = os.path.join(seed_run_dir, cache_entry["observation_curve_filename"])
    if not (os.path.exists(seed_stpn_solution_path) and os.path.exists(seed_observation_curve_path)):
        return None

    shutil.copy2(seed_stpn_solution_path, cache_entry["stpn_solution_path"])
    shutil.copy2(seed_observation_curve_path, cache_entry["observation_curve_path"])
    return {
        "seed_run_dir": seed_run_dir,
        "seed_stpn_solution_path": seed_stpn_solution_path,
        "seed_observation_curve_path": seed_observation_curve_path,
    }


def run_java_analysis(
    repo_root,
    run_dir,
    time_step_hours=TIME_STEP_HOURS,
    parameter_bundle_path=None,
    parameter_bundle=None,
):
    class_root = os.path.join(repo_root, "out", "production", "chita-main-test")
    stpn_analysis_class_path = os.path.join(class_root, "com", "chita", "analysis", "STPNAnalysis.class")
    if not os.path.exists(stpn_analysis_class_path):
        raise FileNotFoundError("Compiled STPNAnalysis.class not found under out/production/chita-main-test.")

    java_class_fingerprint = file_sha256(stpn_analysis_class_path)
    java_executable = resolve_java_executable()
    gson_jar = resolve_optional_jar(
        [
            os.path.join(repo_root, "lib", "gson.jar"),
            os.path.join(repo_root, "lib", "gson-2.13.1.jar"),
            os.path.join(repo_root, "lib", "gson-2.11.0.jar"),
            os.path.join(os.path.expanduser("~"), ".m2", "repository", "com", "google", "code", "gson", "gson", "2.13.1", "gson-2.13.1.jar"),
            os.path.join(os.path.expanduser("~"), ".m2", "repository", "com", "google", "code", "gson", "gson", "2.11.0", "gson-2.11.0.jar"),
            os.path.join(os.path.expanduser("~"), ".gradle", "caches", "modules-2", "files-2.1", "com.google.code.gson", "gson", "2.10.1", "b3add478d4382b78ea20b1671390a858002feb6c", "gson-2.10.1.jar"),
        ],
        "gson",
    )
    shared_precompute_cache = None
    seed_reuse = None
    observation_curve_cache_path = None
    local_observation_curve_path = None
    stpn_solution_filename = f"stpn_solution_ts{sanitized_time_step_label(time_step_hours)}.csv"
    stpn_solution_path = os.path.join(run_dir, stpn_solution_filename)
    effective_parameter_bundle_path = parameter_bundle_path
    if parameter_bundle is not None:
        shared_precompute_cache = prepare_java_precompute_cache(
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
            os.path.join(repo_root, "lib", "*"),
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
    java_precompute_cache_hit = False
    if shared_precompute_cache is not None:
        java_precompute_cache_hit = (
            os.path.exists(stpn_solution_path)
            and os.path.exists(observation_curve_cache_path)
        )
        if not java_precompute_cache_hit:
            seed_reuse = try_seed_java_precompute_cache(
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
    elif not os.path.exists(stpn_solution_path):
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


def generate_d2_style_source_dataset(run_dir, time_limit_days, n_subjects, total_internal_contacts, seed):
    d2_source_dir = ensure_dir(os.path.join(run_dir, "_d2_style_source"))
    dataset_prefix = os.path.join(
        d2_source_dir,
        f"dataset_s{n_subjects}_t{time_limit_days}",
    )
    with contextlib.redirect_stdout(io.StringIO()):
        created_files = dataset.create_datasets(
            dataset_prefix,
            [n_subjects],
            [time_limit_days],
            seed=seed,
            fine_grained=True,
            internal_contacts={total_internal_contacts},
            max_contacts=6,
        )
    if len(created_files) != 1:
        raise RuntimeError(
            "Expected exactly one D2-style dataset to be generated, "
            f"but got {len(created_files)}: {created_files}"
        )

    source_dataset_path = created_files[0]
    if not os.path.exists(source_dataset_path):
        raise FileNotFoundError(
            f"D2-style dataset generation did not create the expected file: {source_dataset_path}"
        )
    return source_dataset_path, read_json(source_dataset_path)


def remove_test_events_from_payload(payload):
    filtered_events = [event for event in payload["events"] if event.get("type") != "Test"]
    removed_tests = len(payload["events"]) - len(filtered_events)
    filtered_payload = copy.deepcopy(payload)
    filtered_payload["events"] = filtered_events
    return filtered_payload, removed_tests


def _observed_test_subject_key(event):
    return tuple(event.get("involved_subjects", []))


def _sort_events_for_serialization(events):
    return sorted(events, key=lambda event: float(event.get("time", 0.0)))


def apply_observed_test_ablation(payload, ablation_mode):
    normalized_mode = ablation_mode or OBSERVED_TEST_ABLATION_NONE
    if normalized_mode not in OBSERVED_TEST_ABLATION_CHOICES:
        raise ValueError(
            f"Unsupported observed-test ablation mode: {normalized_mode}. "
            f"Expected one of {OBSERVED_TEST_ABLATION_CHOICES}."
        )

    filtered_payload = copy.deepcopy(payload)
    events = list(filtered_payload["events"])
    test_events = [
        event for event in events
        if event.get("type") == "Test"
    ]
    positive_before = sum(1 for event in test_events if event.get("result") is True)
    negative_before = sum(1 for event in test_events if event.get("result") is False)
    stats = {
        "mode": normalized_mode,
        "tests_before": len(test_events),
        "positive_tests_before": positive_before,
        "negative_tests_before": negative_before,
    }
    if normalized_mode == OBSERVED_TEST_ABLATION_NONE or not test_events:
        stats.update(
            {
                "tests_after": len(test_events),
                "positive_tests_after": positive_before,
                "negative_tests_after": negative_before,
                "tests_removed": 0,
            }
        )
        return filtered_payload, stats

    non_test_events = [
        event for event in events
        if event.get("type") != "Test"
    ]
    sorted_test_events = _sort_events_for_serialization(test_events)
    kept_tests = []

    if normalized_mode == OBSERVED_TEST_ABLATION_FIRST_POSITIVE_ONLY:
        first_positive_by_subject = {}
        for event in sorted_test_events:
            if event.get("result") is not True:
                continue
            subject_key = _observed_test_subject_key(event)
            if subject_key not in first_positive_by_subject:
                first_positive_by_subject[subject_key] = event
        kept_tests = list(first_positive_by_subject.values())
    elif normalized_mode == OBSERVED_TEST_ABLATION_THROUGH_FIRST_POSITIVE:
        locked_subjects = set()
        for event in sorted_test_events:
            subject_key = _observed_test_subject_key(event)
            if subject_key in locked_subjects:
                continue
            kept_tests.append(event)
            if event.get("result") is True:
                locked_subjects.add(subject_key)
    elif normalized_mode == OBSERVED_TEST_ABLATION_ONE_PER_DAY:
        kept_by_subject_day = {}
        for event in sorted_test_events:
            subject_key = _observed_test_subject_key(event)
            day_index = int(math.floor(float(event.get("time", 0.0)) / 24.0))
            key = (subject_key, day_index)
            existing_event = kept_by_subject_day.get(key)
            if existing_event is None:
                kept_by_subject_day[key] = event
                continue
            if existing_event.get("result") is True:
                continue
            if event.get("result") is True:
                kept_by_subject_day[key] = event
        kept_tests = list(kept_by_subject_day.values())
    else:
        raise ValueError(f"Unhandled observed-test ablation mode: {normalized_mode}")

    kept_tests = _sort_events_for_serialization(kept_tests)
    filtered_payload["events"] = _sort_events_for_serialization(non_test_events + kept_tests)
    positive_after = sum(1 for event in kept_tests if event.get("result") is True)
    negative_after = sum(1 for event in kept_tests if event.get("result") is False)
    stats.update(
        {
            "tests_after": len(kept_tests),
            "positive_tests_after": positive_after,
            "negative_tests_after": negative_after,
            "tests_removed": len(test_events) - len(kept_tests),
        }
    )
    return filtered_payload, stats


def generate_dataset(
    repo_root,
    run_dir,
    run_name,
    time_limit_days,
    n_subjects,
    total_internal_contacts,
    seed,
    dataset_source=DEFAULT_DATASET_SOURCE,
    disable_tests=False,
):
    dataset_path = os.path.join(
        run_dir,
        f"dataset_s{n_subjects}_t{time_limit_days}_c{total_internal_contacts}.json",
    )
    source_dataset_path = None
    dataset_generation_seed = seed
    dataset_generation_method = dataset_generation_method_for_source(dataset_source)
    if dataset_source == DATASET_SOURCE_D2:
        source_dataset_path, payload = generate_d2_style_source_dataset(
            run_dir,
            time_limit_days,
            n_subjects,
            total_internal_contacts,
            seed,
        )
        write_json(dataset_path, payload)
    elif dataset_source == DATASET_SOURCE_GENERATED:
        result = dataset_generator.simulate_external_introduction(
            n_nodes=n_subjects,
            tmax_after_intro=float(time_limit_days * 24),
            max_intro_time=min(48.0, float(time_limit_days * 24)),
            total_internal_contacts=total_internal_contacts,
            seed=seed,
        )
        payload = dataset_generator.save_dataset_event_sequence(result, dataset_path)
    else:
        raise ValueError(f"Unsupported dataset source: {dataset_source}")

    removed_test_events = 0
    if disable_tests:
        payload, removed_test_events = remove_test_events_from_payload(payload)
        write_json(dataset_path, payload)
        if source_dataset_path is not None and os.path.abspath(source_dataset_path) != os.path.abspath(dataset_path):
            write_json(source_dataset_path, payload)

    write_json(
        os.path.join(run_dir, "dataset_metadata.json"),
        {
            "run_name": run_name,
            "dataset_path": dataset_path,
            "dataset_source": dataset_source,
            "source_dataset_path": source_dataset_path,
            "dataset_generation_method": dataset_generation_method,
            "n_subjects": payload["n_subjects"],
            "time_limit": payload["time_limit"],
            "n_contacts": payload["n_contacts"],
            "seed": dataset_generation_seed,
            "tests_enabled": not disable_tests,
            "removed_test_events": removed_test_events,
        },
    )
    return dataset_path, payload


def build_run_name(time_limit_days, n_subjects, total_internal_contacts, parameter_bundle):
    return (
        f"run_t{time_limit_days}_s{n_subjects}_c{total_internal_contacts}"
        f"__{parameter_bundle['case_id']}"
    )


def build_shared_ground_truth_name(
    time_limit_days,
    n_subjects,
    total_internal_contacts,
    ground_truth_parameter_bundle,
):
    return (
        f"shared_ground_truth_t{time_limit_days}_s{n_subjects}_c{total_internal_contacts}"
        f"__{ground_truth_parameter_bundle['case_id']}"
    )


def compute_shared_ground_truth(
    repo_root,
    output_root,
    time_limit_days,
    n_subjects,
    total_internal_contacts,
    seed,
    time_step_hours=TIME_STEP_HOURS,
    dataset_source=DEFAULT_DATASET_SOURCE,
    ground_truth_parameter_bundle=None,
    include_moving_avg_metrics=DEFAULT_INCLUDE_MOVING_AVG_METRICS,
    save_plots=True,
    disable_tests=False,
):
    if ground_truth_parameter_bundle is None:
        raise ValueError("compute_shared_ground_truth requires a ground_truth_parameter_bundle.")

    shared_name = build_shared_ground_truth_name(
        time_limit_days,
        n_subjects,
        total_internal_contacts,
        ground_truth_parameter_bundle,
    )
    shared_root = ensure_dir(os.path.join(output_root, "_shared_ground_truth"))
    shared_dir = ensure_dir(os.path.join(shared_root, shared_name))
    fine_grained = time_step_hours < 1.0
    ground_truth_parameter_bundle_path = os.path.join(shared_dir, "ground_truth_parameter_bundle.json")
    write_json(ground_truth_parameter_bundle_path, ground_truth_parameter_bundle)

    dataset_path, dataset_payload = generate_dataset(
        repo_root,
        shared_dir,
        shared_name,
        time_limit_days,
        n_subjects,
        total_internal_contacts,
        seed,
        dataset_source=dataset_source,
        disable_tests=disable_tests,
    )
    convergence_result = run_dataset_simulations(
        dataset_path=dataset_path,
        run_until_convergence=True,
        iterations_cap=CONVERGENCE_ITERATIONS_CAP,
        convergence_threshold=CONVERGENCE_THRESHOLD,
        fine_grained=fine_grained,
        time_step_hours=time_step_hours,
        dataset_label=shared_name,
        seed=seed,
        prune_after_positive_test=False,
        export_observed_simulation=True,
        pruning_seed=seed,
        parameter_bundle=ground_truth_parameter_bundle,
        save_plots=save_plots,
    )

    summary = {
        "run_name": shared_name,
        "run_dir": shared_dir,
        "time_limit": time_limit_days,
        "n_subjects": n_subjects,
        "total_internal_contacts": total_internal_contacts,
        "seed": seed,
        "time_step_hours": time_step_hours,
        "dataset_source": dataset_source,
        "dataset_generation_method": dataset_generation_method_for_source(dataset_source),
        "moving_avg_metrics_enabled": include_moving_avg_metrics,
        "dataset_path": dataset_path,
        "dataset_events": len(dataset_payload["events"]),
        "tests_enabled": not disable_tests,
        "dataset_test_events": len([event for event in dataset_payload["events"] if event["type"] == "Test"]),
        "effective_dataset_path": convergence_result["effective_dataset_path"],
        "pruned_dataset_path": convergence_result["pruned_dataset_path"],
        "positive_test_pruning": convergence_result["positive_test_pruning"],
        "ground_truth_parameter_case_id": ground_truth_parameter_bundle["case_id"],
        "ground_truth_parameter_levels": ground_truth_parameter_bundle["levels"],
        "ground_truth_parameter_unit_measure": ground_truth_parameter_bundle["unit_measure"],
        "ground_truth_parameter_bundle_path": ground_truth_parameter_bundle_path,
        "status": "completed",
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "convergence": {
            "threshold": CONVERGENCE_THRESHOLD,
            "iterations": convergence_result["rep_done"],
            "reached": convergence_result["convergence_reached"],
            "scores": convergence_result["convergence_scores"],
            "ground_truth_path": convergence_result["averaged_results_path"],
            "observed_simulated_path": convergence_result["observed_simulated_path"],
            "effective_dataset_path": convergence_result["effective_dataset_path"],
            "pruned_dataset_path": convergence_result["pruned_dataset_path"],
            "granularity": convergence_result.get("granularity", time_step_hours),
            "time_step_hours": convergence_result.get("time_step_hours", time_step_hours),
            "actual_runtime_seconds": convergence_result.get("actual_runtime_seconds"),
            "suppressed_stdout_log_path": convergence_result.get("suppressed_stdout_log_path"),
            "parameterization": "fixed_mid_bundle",
            "parameter_case_id": ground_truth_parameter_bundle["case_id"],
            "parameter_levels": ground_truth_parameter_bundle["levels"],
            "parameter_bundle_path": ground_truth_parameter_bundle_path,
        },
    }
    write_json(os.path.join(shared_dir, "shared_ground_truth_summary.json"), summary)
    return summary


def prepare_run_inputs_from_shared_ground_truth(
    run_dir,
    run_name,
    shared_ground_truth,
    observed_test_ablation=OBSERVED_TEST_ABLATION_NONE,
):
    shared_dataset_path = shared_ground_truth["dataset_path"]
    local_dataset_path = os.path.join(run_dir, os.path.basename(shared_dataset_path))
    shutil.copy2(shared_dataset_path, local_dataset_path)

    shared_observed_simulated_path = shared_ground_truth["convergence"]["observed_simulated_path"]
    local_observed_simulated_path = None
    observed_test_ablation_stats = {
        "mode": observed_test_ablation,
        "tests_before": 0,
        "positive_tests_before": 0,
        "negative_tests_before": 0,
        "tests_after": 0,
        "positive_tests_after": 0,
        "negative_tests_after": 0,
        "tests_removed": 0,
    }
    if shared_observed_simulated_path:
        local_observed_simulated_path = os.path.join(
            run_dir,
            os.path.basename(shared_observed_simulated_path),
        )
        shutil.copy2(shared_observed_simulated_path, local_observed_simulated_path)
        observed_payload = read_json(local_observed_simulated_path)
        if observed_test_ablation != OBSERVED_TEST_ABLATION_NONE:
            observed_payload, observed_test_ablation_stats = apply_observed_test_ablation(
                observed_payload,
                observed_test_ablation,
            )
            write_json(local_observed_simulated_path, observed_payload)
        else:
            observed_test_ablation_stats["mode"] = OBSERVED_TEST_ABLATION_NONE
            observed_test_ablation_stats["tests_before"] = sum(
                1
                for event in observed_payload["events"]
                if event.get("type") == "Test"
            )
            observed_test_ablation_stats["positive_tests_before"] = sum(
                1
                for event in observed_payload["events"]
                if event.get("type") == "Test" and event.get("result") is True
            )
            observed_test_ablation_stats["negative_tests_before"] = sum(
                1
                for event in observed_payload["events"]
                if event.get("type") == "Test" and event.get("result") is False
            )
            observed_test_ablation_stats["tests_after"] = observed_test_ablation_stats["tests_before"]
            observed_test_ablation_stats["positive_tests_after"] = observed_test_ablation_stats["positive_tests_before"]
            observed_test_ablation_stats["negative_tests_after"] = observed_test_ablation_stats["negative_tests_before"]

    write_json(
        os.path.join(run_dir, "dataset_metadata.json"),
        {
            "run_name": run_name,
            "dataset_path": local_dataset_path,
            "dataset_source": shared_ground_truth.get("dataset_source", DEFAULT_DATASET_SOURCE),
            "dataset_generation_method": shared_ground_truth.get("dataset_generation_method"),
            "shared_dataset_path": shared_dataset_path,
            "shared_observed_simulated_path": shared_observed_simulated_path,
            "local_observed_simulated_path": local_observed_simulated_path,
            "n_subjects": shared_ground_truth["n_subjects"],
            "time_limit": shared_ground_truth["time_limit"],
            "n_contacts": shared_ground_truth["total_internal_contacts"],
            "seed": shared_ground_truth["seed"],
            "tests_enabled": shared_ground_truth.get("tests_enabled", True),
            "dataset_test_events": shared_ground_truth.get("dataset_test_events"),
            "observed_test_ablation": observed_test_ablation,
            "observed_test_ablation_stats": observed_test_ablation_stats,
        },
    )
    return {
        "dataset_path": local_dataset_path,
        "shared_dataset_path": shared_dataset_path,
        "observed_simulated_path": local_observed_simulated_path,
        "shared_observed_simulated_path": shared_observed_simulated_path,
        "observed_test_ablation": observed_test_ablation,
        "observed_test_ablation_stats": observed_test_ablation_stats,
    }


def run_single_pipeline(
    repo_root,
    output_root,
    time_limit_days,
    n_subjects,
    total_internal_contacts,
    seed,
    time_step_hours=TIME_STEP_HOURS,
    baseline_runtime_multiplier=BASELINE_RUNTIME_MULTIPLIER,
    parameter_bundle=None,
    ground_truth_parameter_bundle=None,
    shared_ground_truth=None,
    observed_test_ablation=OBSERVED_TEST_ABLATION_NONE,
    include_moving_avg_metrics=DEFAULT_INCLUDE_MOVING_AVG_METRICS,
    save_plots=True,
):
    if parameter_bundle is None:
        raise ValueError("run_single_pipeline requires a parameter_bundle.")
    if ground_truth_parameter_bundle is None:
        raise ValueError("run_single_pipeline requires a ground_truth_parameter_bundle.")
    if shared_ground_truth is None:
        raise ValueError("run_single_pipeline requires a shared_ground_truth.")

    run_name = build_run_name(
        time_limit_days,
        n_subjects,
        total_internal_contacts,
        parameter_bundle,
    )
    run_dir = ensure_dir(os.path.join(output_root, run_name))
    fine_grained = time_step_hours < 1.0
    parameter_bundle_path = os.path.join(run_dir, "parameter_bundle.json")
    write_json(parameter_bundle_path, parameter_bundle)
    ground_truth_parameter_bundle_path = shared_ground_truth["ground_truth_parameter_bundle_path"]
    run_inputs = prepare_run_inputs_from_shared_ground_truth(
        run_dir,
        run_name,
        shared_ground_truth,
        observed_test_ablation=observed_test_ablation,
    )
    dataset_path = run_inputs["dataset_path"]

    summary = {
        "run_name": run_name,
        "run_dir": run_dir,
        "time_limit": time_limit_days,
        "n_subjects": n_subjects,
        "total_internal_contacts": total_internal_contacts,
        "seed": shared_ground_truth["seed"],
        "baseline_seed": seed,
        "time_step_hours": time_step_hours,
        "baseline_runtime_multiplier": baseline_runtime_multiplier,
        "dataset_source": shared_ground_truth.get("dataset_source", DEFAULT_DATASET_SOURCE),
        "dataset_generation_method": shared_ground_truth.get("dataset_generation_method"),
        "moving_avg_metrics_enabled": include_moving_avg_metrics,
        "tests_enabled": shared_ground_truth.get("tests_enabled", True),
        "dataset_test_events": shared_ground_truth.get("dataset_test_events"),
        "observed_test_ablation": run_inputs["observed_test_ablation"],
        "observed_test_ablation_stats": run_inputs["observed_test_ablation_stats"],
        "shared_ground_truth_run_dir": shared_ground_truth["run_dir"],
        "parameter_case_id": parameter_bundle["case_id"],
        "parameter_levels": parameter_bundle["levels"],
        "parameter_unit_measure": parameter_bundle["unit_measure"],
        "parameter_bundle_path": parameter_bundle_path,
        "ground_truth_parameter_case_id": ground_truth_parameter_bundle["case_id"],
        "ground_truth_parameter_levels": ground_truth_parameter_bundle["levels"],
        "ground_truth_parameter_unit_measure": ground_truth_parameter_bundle["unit_measure"],
        "ground_truth_parameter_bundle_path": ground_truth_parameter_bundle_path,
        "dataset_path": dataset_path,
        "shared_dataset_path": run_inputs["shared_dataset_path"],
        "shared_observed_simulated_path": run_inputs["shared_observed_simulated_path"],
        "status": "running",
    }
    write_json(os.path.join(run_dir, "run_summary.json"), summary)
    summary["dataset_events"] = shared_ground_truth["dataset_events"]
    summary["effective_dataset_path"] = shared_ground_truth["effective_dataset_path"]
    summary["pruned_dataset_path"] = shared_ground_truth["pruned_dataset_path"]
    summary["positive_test_pruning"] = shared_ground_truth["positive_test_pruning"]
    convergence_result = copy.deepcopy(shared_ground_truth["convergence"])
    summary["convergence"] = convergence_result

    java_result = run_java_analysis(
        repo_root,
        run_dir,
        time_step_hours=time_step_hours,
        parameter_bundle_path=parameter_bundle_path,
        parameter_bundle=parameter_bundle,
    )
    analysis_path = find_generated_file(run_dir, "_tracks_it3.json")
    baseline_runtime_budget_seconds = max(
        baseline_runtime_multiplier * java_result["analysis_runtime_seconds"],
        0.0,
    )
    baseline_result = run_dataset_simulations(
        dataset_path=dataset_path,
        rep=BASELINE_ITERATIONS_CAP,
        run_until_convergence=False,
        fine_grained=fine_grained,
        time_step_hours=time_step_hours,
        dataset_label=run_name,
        seed=seed + 1,
        prune_after_positive_test=False,
        export_observed_simulation=False,
        pruning_seed=seed,
        max_runtime_seconds=baseline_runtime_budget_seconds,
        parameter_bundle=parameter_bundle,
        save_plots=False,
    )
    analysis_curve_plots = create_analysis_subject_curve_plots(
        run_dir=run_dir,
        run_name=run_name,
        analysis_path=analysis_path,
        ground_truth_path=convergence_result["ground_truth_path"],
        baseline_path=baseline_result["averaged_results_path"],
        granularity=convergence_result.get("granularity", 1.0),
        ground_truth_sample_size=convergence_result.get("iterations"),
        save_plots=save_plots,
    )
    summary["analysis"] = {
        "analysis_path": analysis_path,
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
        "parameterization": "swept_parameter_bundle",
        "parameter_case_id": parameter_bundle["case_id"],
        "parameter_levels": parameter_bundle["levels"],
        "parameter_bundle_path": parameter_bundle_path,
        "input_observed_simulated_path": run_inputs["observed_simulated_path"],
        "java_analysis_stdout_log_path": java_result["java_analysis_stdout_log_path"],
        "java_analysis_stderr_log_path": java_result["java_analysis_stderr_log_path"],
        "java_precompute_stdout_log_path": java_result["java_precompute_stdout_log_path"],
        "java_precompute_stderr_log_path": java_result["java_precompute_stderr_log_path"],
        "subject_curve_plots": analysis_curve_plots,
    }
    summary["baseline"] = {
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
        "parameterization": "swept_parameter_bundle",
        "parameter_case_id": parameter_bundle["case_id"],
        "parameter_levels": parameter_bundle["levels"],
        "parameter_bundle_path": parameter_bundle_path,
    }

    precision_dir = ensure_dir(os.path.join(run_dir, "precision_metrics"))
    prediction_metrics_path = os.path.join(precision_dir, "metrics_prediction.json")
    baseline_metrics_path = os.path.join(precision_dir, "metrics_baseline.json")
    metrics_stdout = io.StringIO()
    t0_metrics = time.time()
    with contextlib.redirect_stdout(metrics_stdout):
        process_and_save(
            analysis_path,
            convergence_result["ground_truth_path"],
            M=10,
            metrics_output=prediction_metrics_path,
            plots_dir=os.path.join(precision_dir, "numericalAnalysis", "plots"),
            save_plots=save_plots,
        )
        process_and_save(
            baseline_result["averaged_results_path"],
            convergence_result["ground_truth_path"],
            M=10,
            metrics_output=baseline_metrics_path,
            plots_dir=os.path.join(precision_dir, "simulatedBaseline", "plots"),
            save_plots=save_plots,
        )
    t1_metrics = time.time()
    metrics_runtime_seconds = t1_metrics - t0_metrics
    with open(os.path.join(precision_dir, "metrics_stdout.log"), "w", encoding="utf-8") as handle:
        handle.write(metrics_stdout.getvalue())

    comparison_summary = create_analysis_vs_simulation_plots(
        run_dir,
        run_name,
        convergence_result["ground_truth_path"],
        analysis_path,
        baseline_result["averaged_results_path"],
        include_moving_avg_metrics=include_moving_avg_metrics,
        save_plots=save_plots,
    )

    summary["precision_metrics"] = {
        "prediction_metrics_path": prediction_metrics_path,
        "baseline_metrics_path": baseline_metrics_path,
        "prediction_mean_brier": mean_metric_value(prediction_metrics_path, "Brier Score"),
        "prediction_mean_ece": mean_metric_value(prediction_metrics_path, "ECE"),
        "baseline_mean_brier": mean_metric_value(baseline_metrics_path, "Brier Score"),
        "baseline_mean_ece": mean_metric_value(baseline_metrics_path, "ECE"),
        "metrics_runtime_seconds": metrics_runtime_seconds,
    }
    summary["comparison_metrics"] = comparison_summary
    summary["moving_avg_metrics_enabled"] = include_moving_avg_metrics
    summary["status"] = "completed"
    summary["completed_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(os.path.join(run_dir, "run_summary.json"), summary)
    return summary


def write_aggregate_outputs(output_root, summaries):
    aggregate_json_path = os.path.join(output_root, "sweep_summary.json")
    write_json(aggregate_json_path, summaries)

    csv_rows = []
    for summary in summaries:
        parameter_levels = summary.get("parameter_levels", {})
        row = {
            "run_name": summary["run_name"],
            "status": summary["status"],
            "dataset_source": summary.get("dataset_source"),
            "dataset_generation_method": summary.get("dataset_generation_method"),
            "time_limit": summary["time_limit"],
            "n_subjects": summary["n_subjects"],
            "total_internal_contacts": summary["total_internal_contacts"],
            "time_step_hours": summary.get("time_step_hours"),
            "baseline_runtime_multiplier": summary.get("baseline_runtime_multiplier"),
            "moving_avg_metrics_enabled": summary.get("moving_avg_metrics_enabled"),
            "observed_test_ablation": summary.get("observed_test_ablation"),
            "observed_test_events_before": summary.get("observed_test_ablation_stats", {}).get("tests_before"),
            "observed_test_events_after": summary.get("observed_test_ablation_stats", {}).get("tests_after"),
            "observed_positive_tests_before": summary.get("observed_test_ablation_stats", {}).get("positive_tests_before"),
            "observed_positive_tests_after": summary.get("observed_test_ablation_stats", {}).get("positive_tests_after"),
            "parameter_case_id": summary.get("parameter_case_id"),
            "ground_truth_parameter_case_id": summary.get("ground_truth_parameter_case_id"),
            "baseline_seed": summary.get("baseline_seed"),
            "infectiousness_level": parameter_levels.get("infectiousness"),
            "healing_level": parameter_levels.get("healing"),
            "symptoms_level": parameter_levels.get("symptoms"),
            "isolating_level": parameter_levels.get("isolating"),
            "symptoms_onset_level": parameter_levels.get("symptomsOnset"),
            "notification_to_isolation_level": parameter_levels.get("notificationToIsolation"),
            "symptomatic_period_level": parameter_levels.get("symptomaticPeriod"),
            "dataset_events": summary.get("dataset_events"),
            "convergence_iterations": summary.get("convergence", {}).get("iterations"),
            "convergence_reached": summary.get("convergence", {}).get("reached"),
            "java_analysis_runtime_seconds": summary.get("analysis", {}).get("analysis_runtime_seconds"),
            "java_analysis_wall_runtime_seconds": summary.get("analysis", {}).get("analysis_wall_runtime_seconds"),
            "java_analysis_runtime_excludes_overhead": summary.get("analysis", {}).get("analysis_runtime_excludes_overhead"),
            "stpn_precomputation_runtime_seconds": summary.get("analysis", {}).get("stpn_precomputation_runtime_seconds"),
            "java_precompute_cache_hit": summary.get("analysis", {}).get("java_precompute_cache_hit"),
            "java_precompute_seed_run_dir": summary.get("analysis", {}).get("java_precompute_seed_run_dir"),
            "baseline_runtime_budget_seconds": summary.get("baseline", {}).get("runtime_budget_seconds"),
            "baseline_actual_runtime_seconds": summary.get("baseline", {}).get("actual_runtime_seconds"),
            "baseline_iterations": summary.get("baseline", {}).get("iterations"),
            "prediction_mean_brier": summary.get("precision_metrics", {}).get("prediction_mean_brier"),
            "baseline_mean_brier": summary.get("precision_metrics", {}).get("baseline_mean_brier"),
            "prediction_mean_ece": summary.get("precision_metrics", {}).get("prediction_mean_ece"),
            "baseline_mean_ece": summary.get("precision_metrics", {}).get("baseline_mean_ece"),
            "analysis_tau": summary.get("comparison_metrics", {}).get("analysis", {}).get("tau"),
            "simulation_tau": summary.get("comparison_metrics", {}).get("simulation", {}).get("tau"),
            "analysis_spearman": summary.get("comparison_metrics", {}).get("analysis", {}).get("spearman"),
            "simulation_spearman": summary.get("comparison_metrics", {}).get("simulation", {}).get("spearman"),
            "analysis_mrr": summary.get("comparison_metrics", {}).get("analysis", {}).get("mrr"),
            "simulation_mrr": summary.get("comparison_metrics", {}).get("simulation", {}).get("mrr"),
            "error": summary.get("error"),
        }
        csv_rows.append(row)

    csv_path = os.path.join(output_root, "sweep_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()) if csv_rows else ["run_name"])
        writer.writeheader()
        writer.writerows(csv_rows)


def select_summaries_for_reduced_plots(summaries):
    completed = [summary for summary in summaries if summary.get("status") == "completed"]
    if not completed:
        return [], []

    selected_by_run = {}

    def add_selected(summary, reason):
        run_name = summary["run_name"]
        info = selected_by_run.setdefault(
            run_name,
            {
                "summary": summary,
                "reasons": [],
                "analysis_spearman": finite_float(
                    summary.get("comparison_metrics", {}).get("analysis", {}).get("spearman")
                ),
                "analysis_tau": finite_float(
                    summary.get("comparison_metrics", {}).get("analysis", {}).get("tau")
                ),
            },
        )
        if reason not in info["reasons"]:
            info["reasons"].append(reason)

    def add_metric_buckets(metric_name, metric_getter):
        ranked = []
        for summary in completed:
            metric_value = metric_getter(summary)
            if metric_value is None:
                continue
            ranked.append((metric_value, summary))

        ranked.sort(key=lambda item: (item[0], item[1].get("run_name", "")))
        if not ranked:
            return

        bucket_size = min(REDUCED_PLOT_BUCKET_SIZE, len(ranked))
        median_bucket_size = min(REDUCED_PLOT_BUCKET_SIZE, len(ranked))
        median_center = len(ranked) // 2
        median_start = max(
            0,
            min(len(ranked) - median_bucket_size, median_center - median_bucket_size // 2),
        )

        for _, summary in ranked[:bucket_size]:
            add_selected(summary, f"worst_10_{metric_name}")
        for _, summary in ranked[-bucket_size:]:
            add_selected(summary, f"best_10_{metric_name}")
        for _, summary in ranked[median_start:median_start + median_bucket_size]:
            add_selected(summary, f"median_10_{metric_name}")

    add_metric_buckets(
        "spearman_analysis",
        lambda summary: finite_float(
            summary.get("comparison_metrics", {}).get("analysis", {}).get("spearman")
        ),
    )
    add_metric_buckets(
        "kendall_analysis",
        lambda summary: finite_float(
            summary.get("comparison_metrics", {}).get("analysis", {}).get("tau")
        ),
    )

    for summary in completed:
        if summary.get("parameter_case_id") == summary.get("ground_truth_parameter_case_id"):
            add_selected(summary, "mid_parameter_case")

    sorted_items = sorted(
        selected_by_run.values(),
        key=lambda item: (
            item["analysis_spearman"] if item["analysis_spearman"] is not None else float("inf"),
            item["analysis_tau"] if item["analysis_tau"] is not None else float("inf"),
            item["summary"]["run_name"],
        ),
    )
    selected_summaries = [item["summary"] for item in sorted_items]
    manifest_rows = [
        {
            "run_name": item["summary"]["run_name"],
            "run_dir": item["summary"]["run_dir"],
            "parameter_case_id": item["summary"].get("parameter_case_id"),
            "analysis_spearman": item["analysis_spearman"],
            "analysis_tau": item["analysis_tau"],
            "selection_reasons": "|".join(item["reasons"]),
        }
        for item in sorted_items
    ]
    return selected_summaries, manifest_rows


def write_reduced_plot_selection(output_root, manifest_rows):
    selection_json_path = os.path.join(output_root, "reduced_plot_selection.json")
    selection_csv_path = os.path.join(output_root, "reduced_plot_selection.csv")
    write_json(selection_json_path, manifest_rows)
    with open(selection_csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(manifest_rows[0].keys()) if manifest_rows else ["run_name"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    return {
        "json_path": selection_json_path,
        "csv_path": selection_csv_path,
    }


def regenerate_run_plots(summary):
    if summary.get("status") != "completed":
        return summary

    run_dir = summary["run_dir"]
    run_name = summary["run_name"]
    ground_truth_path = summary["convergence"]["ground_truth_path"]
    analysis_path = summary["analysis"]["analysis_path"]
    baseline_path = summary["baseline"]["baseline_path"]
    granularity = summary["convergence"].get("granularity", 1.0)
    include_moving_avg_metrics = summary.get("moving_avg_metrics_enabled", DEFAULT_INCLUDE_MOVING_AVG_METRICS)

    analysis_curve_plots = create_analysis_subject_curve_plots(
        run_dir=run_dir,
        run_name=run_name,
        analysis_path=analysis_path,
        ground_truth_path=ground_truth_path,
        baseline_path=baseline_path,
        granularity=granularity,
        ground_truth_sample_size=summary["convergence"].get("iterations"),
        save_plots=True,
    )
    summary["analysis"]["subject_curve_plots"] = analysis_curve_plots

    precision_dir = ensure_dir(os.path.join(run_dir, "precision_metrics"))
    prediction_metrics_path = summary["precision_metrics"]["prediction_metrics_path"]
    baseline_metrics_path = summary["precision_metrics"]["baseline_metrics_path"]
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

    summary["comparison_metrics"] = create_analysis_vs_simulation_plots(
        run_dir,
        run_name,
        ground_truth_path,
        analysis_path,
        baseline_path,
        include_moving_avg_metrics=include_moving_avg_metrics,
        save_plots=True,
    )
    write_json(os.path.join(run_dir, "run_summary.json"), summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run the full CHITA sweep pipeline.")
    parser.add_argument(
        "--output-root",
        default=os.path.join("sweeps", datetime.now().strftime("sweep_%Y%m%d_%H%M%S")),
        help="Directory where sweep outputs will be created.",
    )
    parser.add_argument("--seed-base", type=int, default=1000, help="Base seed used to derive per-run seeds.")
    parser.add_argument(
        "--time-step-hours",
        type=float,
        default=TIME_STEP_HOURS,
        help="Shared time step in hours for both Python simulation and Java analysis.",
    )
    parser.add_argument(
        "--baseline-runtime-multiplier",
        type=float,
        default=BASELINE_RUNTIME_MULTIPLIER,
        help=(
            "Multiplier applied to the measured Java analysis runtime to set the "
            "Python baseline simulation runtime budget. Use 1.0 to match the "
            "analysis runtime."
        ),
    )
    parser.add_argument(
        "--parameter-ods-path",
        default=DEFAULT_PARAMETER_ODS_PATH,
        help="ODS spreadsheet containing the lower/mid/upper transition parameters.",
    )
    parser.add_argument(
        "--parameter-case-mode",
        choices=["all-combinations", "aligned-scenarios"],
        default="all-combinations",
        help=(
            "How to build the parameter sweep: 'all-combinations' evaluates the full "
            "Cartesian product across parameter families, while 'aligned-scenarios' "
            "runs only the three lower/mid/upper bundles."
        ),
    )
    parser.add_argument(
        "--d2",
        action="store_true",
        help=(
            "Generate the raw source dataset with the same D2-style recipe used in "
            "run_n_simulations.py, then continue with the normal sweep pipeline."
        ),
    )
    parser.add_argument(
        "--disable-tests",
        action="store_true",
        help="Remove all raw Test events from the generated dataset before running the sweep.",
    )
    parser.add_argument(
        "--observed-test-ablation",
        choices=OBSERVED_TEST_ABLATION_CHOICES,
        default=OBSERVED_TEST_ABLATION_NONE,
        help=(
            "Optional filtering applied only to the shared observed_simulated.json copied "
            "into each run for Java analysis. The raw dataset and Python baseline inputs "
            "remain unchanged."
        ),
    )
    parser.add_argument(
        "--skip-plot-images",
        action="store_true",
        help="Skip PNG plot generation and keep only CSV/JSON outputs for a faster sweep.",
    )
    parser.add_argument(
        "--disable-moving-avg-metrics",
        action="store_true",
        help="Skip computing the moving-average summary metrics in the sweep outputs.",
    )
    parser.add_argument(
        "--reduce-plots",
        action="store_true",
        help=(
            "Defer plot generation and only render plots for the 10 worst cases, "
            "10 best cases, the mid-parameter case, and 10 cases around the median "
            "based on Java analysis Spearman and Kendall correlation (selected runs "
            "still generate the full per-run curve and comparison plots)."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help=(
            "Maximum number of parameter-case workers to run in parallel. "
            "Defaults to about half of the available CPU cores."
        ),
    )
    args = parser.parse_args()
    if args.time_step_hours <= 0:
        raise ValueError("--time-step-hours must be greater than 0.")
    if args.baseline_runtime_multiplier < 0:
        raise ValueError("--baseline-runtime-multiplier must be greater than or equal to 0.")

    repo_root = os.path.abspath(os.path.dirname(__file__))
    output_root = ensure_dir(os.path.abspath(args.output_root))
    save_plots_during_run = not args.skip_plot_images and not args.reduce_plots
    include_moving_avg_metrics = not args.disable_moving_avg_metrics
    dataset_source = DATASET_SOURCE_D2 if args.d2 else DATASET_SOURCE_GENERATED
    parameter_space = load_parameter_space_from_ods(os.path.abspath(args.parameter_ods_path))
    ground_truth_parameter_bundle = resolve_uniform_parameter_bundle(
        parameter_space,
        GROUND_TRUTH_PARAMETER_LEVEL,
    )
    parameter_cases = enumerate_parameter_cases(
        parameter_space,
        mode=args.parameter_case_mode,
    )
    parameter_manifest_paths = write_parameter_case_manifest(output_root, parameter_cases)
    write_dataset_generation_parameters_markdown(
        output_root=output_root,
        seed_base=args.seed_base,
        time_step_hours=args.time_step_hours,
        baseline_runtime_multiplier=args.baseline_runtime_multiplier,
        include_moving_avg_metrics=include_moving_avg_metrics,
        dataset_source=dataset_source,
        tests_enabled=not args.disable_tests,
        observed_test_ablation=args.observed_test_ablation,
        java_precompute_cache_root=os.path.abspath(os.path.join(repo_root, JAVA_PRECOMPUTE_CACHE_ROOT)),
        java_precompute_seed_sweep_root=os.path.abspath(os.path.join(repo_root, JAVA_PRECOMPUTE_SEED_SWEEP_ROOT)),
        parameter_space=parameter_space,
        ground_truth_parameter_bundle=ground_truth_parameter_bundle,
        parameter_case_mode=args.parameter_case_mode,
        parameter_case_count=len(parameter_cases),
        parameter_manifest_paths=parameter_manifest_paths,
        generate_plot_images=not args.skip_plot_images,
        reduce_plots=args.reduce_plots,
    )

    summaries = []
    dataset_combinations = list(itertools.product(
        TIME_LIMITS,
        N_SUBJECTS,
        TOTAL_INTERNAL_CONTACTS,
    ))
    parameter_case_count = len(parameter_cases)
    worker_count = resolve_worker_count(args.max_workers, parameter_case_count)
    total_progress_steps = len(dataset_combinations) * (parameter_case_count + 1)

    with tqdm(total=total_progress_steps, desc="Sweep progress", unit="step") as progress_bar:
        for dataset_index, (time_limit_days, n_subjects, total_internal_contacts) in enumerate(dataset_combinations):
            dataset_label = f"t{time_limit_days}/s{n_subjects}/c{total_internal_contacts}"
            shared_ground_truth_seed = args.seed_base + dataset_index
            progress_bar.set_postfix_str(f"{dataset_label} | shared ground truth")
            try:
                shared_ground_truth = compute_shared_ground_truth(
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
                    run_name = build_run_name(
                        time_limit_days,
                        n_subjects,
                        total_internal_contacts,
                        parameter_bundle,
                    )
                    run_dir = ensure_dir(os.path.join(output_root, run_name))
                    summary = {
                        "run_name": run_name,
                        "run_dir": run_dir,
                        "status": "failed",
                        "dataset_source": dataset_source,
                        "dataset_generation_method": dataset_generation_method_for_source(dataset_source),
                        "time_limit": time_limit_days,
                        "n_subjects": n_subjects,
                        "total_internal_contacts": total_internal_contacts,
                        "seed": shared_ground_truth_seed,
                        "baseline_seed": baseline_seed,
                        "time_step_hours": args.time_step_hours,
                        "baseline_runtime_multiplier": args.baseline_runtime_multiplier,
                        "moving_avg_metrics_enabled": include_moving_avg_metrics,
                        "observed_test_ablation": args.observed_test_ablation,
                        "parameter_case_id": parameter_bundle["case_id"],
                        "parameter_levels": parameter_bundle["levels"],
                        "parameter_unit_measure": parameter_bundle["unit_measure"],
                        "ground_truth_parameter_case_id": ground_truth_parameter_bundle["case_id"],
                        "ground_truth_parameter_levels": ground_truth_parameter_bundle["levels"],
                        "ground_truth_parameter_unit_measure": ground_truth_parameter_bundle["unit_measure"],
                        "error": f"Shared ground truth failed: {exc}",
                        "traceback": shared_traceback,
                    }
                    write_json(os.path.join(run_dir, "run_summary.json"), summary)
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
                try:
                    executor = ProcessPoolExecutor(max_workers=worker_count)
                except PermissionError:
                    # Some Windows environments disallow the pipe setup used by
                    # ProcessPoolExecutor. Fall back to threads so the sweep can continue.
                    executor = ThreadPoolExecutor(max_workers=worker_count)
                future_to_task = {
                    executor.submit(
                        run_single_pipeline,
                        repo_root=repo_root,
                        output_root=output_root,
                        time_limit_days=time_limit_days,
                        n_subjects=n_subjects,
                        total_internal_contacts=total_internal_contacts,
                        seed=task_spec["baseline_seed"],
                        time_step_hours=args.time_step_hours,
                        baseline_runtime_multiplier=args.baseline_runtime_multiplier,
                        observed_test_ablation=args.observed_test_ablation,
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
                    baseline_seed = task_spec["baseline_seed"]
                    progress_bar.set_postfix_str(
                        f"{dataset_label} | {completed_cases}/{parameter_case_count} cases done"
                    )
                    try:
                        if future is None:
                            summary = run_single_pipeline(
                                repo_root=repo_root,
                                output_root=output_root,
                                time_limit_days=time_limit_days,
                                n_subjects=n_subjects,
                                total_internal_contacts=total_internal_contacts,
                                seed=baseline_seed,
                                time_step_hours=args.time_step_hours,
                                baseline_runtime_multiplier=args.baseline_runtime_multiplier,
                                observed_test_ablation=args.observed_test_ablation,
                                include_moving_avg_metrics=include_moving_avg_metrics,
                                parameter_bundle=parameter_bundle,
                                ground_truth_parameter_bundle=ground_truth_parameter_bundle,
                                shared_ground_truth=shared_ground_truth,
                                save_plots=save_plots_during_run,
                            )
                        else:
                            summary = future.result()
                    except Exception as exc:
                        run_name = build_run_name(
                            time_limit_days,
                            n_subjects,
                            total_internal_contacts,
                            parameter_bundle,
                        )
                        run_dir = ensure_dir(os.path.join(output_root, run_name))
                        summary = {
                            "run_name": run_name,
                            "run_dir": run_dir,
                            "status": "failed",
                            "dataset_source": dataset_source,
                            "dataset_generation_method": dataset_generation_method_for_source(dataset_source),
                            "time_limit": time_limit_days,
                            "n_subjects": n_subjects,
                            "total_internal_contacts": total_internal_contacts,
                            "seed": shared_ground_truth_seed,
                            "baseline_seed": baseline_seed,
                            "time_step_hours": args.time_step_hours,
                            "baseline_runtime_multiplier": args.baseline_runtime_multiplier,
                            "moving_avg_metrics_enabled": include_moving_avg_metrics,
                            "observed_test_ablation": args.observed_test_ablation,
                            "shared_ground_truth_run_dir": shared_ground_truth["run_dir"],
                            "parameter_case_id": parameter_bundle["case_id"],
                            "parameter_levels": parameter_bundle["levels"],
                            "parameter_unit_measure": parameter_bundle["unit_measure"],
                            "ground_truth_parameter_case_id": ground_truth_parameter_bundle["case_id"],
                            "ground_truth_parameter_levels": ground_truth_parameter_bundle["levels"],
                            "ground_truth_parameter_unit_measure": ground_truth_parameter_bundle["unit_measure"],
                            "ground_truth_parameter_bundle_path": shared_ground_truth["ground_truth_parameter_bundle_path"],
                            "error": str(exc),
                            "traceback": traceback.format_exc() if future is None else "".join(
                                traceback.format_exception(type(exc), exc, exc.__traceback__)
                            ),
                        }
                        write_json(os.path.join(run_dir, "run_summary.json"), summary)
                    summaries.append(summary)
                    write_aggregate_outputs(output_root, summaries)
                    completed_cases += 1
                    progress_bar.set_postfix_str(
                        f"{dataset_label} | {completed_cases}/{parameter_case_count} cases done"
                    )
                    progress_bar.update(1)
            finally:
                if executor is not None:
                    executor.shutdown(wait=True, cancel_futures=False)

    if args.reduce_plots:
        selected_summaries, manifest_rows = select_summaries_for_reduced_plots(summaries)
        write_reduced_plot_selection(output_root, manifest_rows)
        if not args.skip_plot_images:
            for summary in selected_summaries:
                regenerate_run_plots(summary)
            write_aggregate_outputs(output_root, summaries)


if __name__ == "__main__":
    main()
