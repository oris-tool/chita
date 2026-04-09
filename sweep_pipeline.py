import argparse
import contextlib
import csv
import io
import itertools
import json
import math
import os
import re
import subprocess
import time
import traceback
import zipfile
from datetime import datetime
from xml.etree import ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np

import dataset_graph as dataset_generator
import metrics as ranking_metrics
from compute_precision_metrics import process_and_save
from run_n_simulations import compute_confidence_intervals, run_dataset_simulations


TIME_LIMITS = [84]
N_SUBJECTS = [8]
TOTAL_INTERNAL_CONTACTS = [84]
CONVERGENCE_THRESHOLD = 1e-5
CONVERGENCE_ITERATIONS_CAP = 100_000
BASELINE_ITERATIONS_CAP = 100_000
MAX_TOP_PRECISION = 7
TIME_STEP_HOURS = 1.0
DEFAULT_PARAMETER_ODS_PATH = os.path.join(os.path.expanduser("~"), "Downloads", "new-parameters.ods")

PARAMETER_FAMILY_ORDER = (
    "infectiousness",
    "healing",
    "symptoms",
    "isolating",
    "symptomsOnset",
)
PARAMETER_LEVEL_ORDER = ("lower", "mid", "upper")

FAMILY_ABBREVIATIONS = {
    "infectiousness": "inf",
    "healing": "heal",
    "symptoms": "sym",
    "isolating": "iso",
    "symptomsOnset": "onset",
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


def mean_metric_value(metrics_path, metric_name):
    metrics_payload = read_json(metrics_path)
    values = [subject_metrics[metric_name] for subject_metrics in metrics_payload.values()]
    return float(sum(values) / len(values)) if values else math.nan


def plot_overlay(series_a, series_b, title, ylabel, output_path, label_a="Analysis", label_b="Simulation"):
    plt.figure(figsize=(10, 6))
    plt.plot(series_a, marker="o", label=label_a)
    plt.plot(series_b, marker="x", label=label_b)
    plt.title(title)
    plt.xlabel("Timestep")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_series_csv(output_path, columns, rows):
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def sanitized_time_step_label(time_step_hours):
    return str(time_step_hours).replace(".", "p")


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
        transition = {"true": float(numeric_values[0])}
        if len(numeric_values) > 1:
            transition["false"] = float(numeric_values[1])
        parsed_levels[level] = transition

    missing_levels = [level for level in PARAMETER_LEVEL_ORDER if level not in parsed_levels]
    if missing_levels:
        raise ValueError(
            f"Missing symptomatic-probability rows: {', '.join(missing_levels)}"
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
    }
    return {
        "source_path": os.path.abspath(path),
        "source_format": "ods",
        "unit_measure": "hours",
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


def manifest_row_for_bundle(bundle):
    row = {
        "case_id": bundle["case_id"],
        "infectiousness_level": bundle["levels"]["infectiousness"],
        "healing_level": bundle["levels"]["healing"],
        "symptoms_level": bundle["levels"]["symptoms"],
        "isolating_level": bundle["levels"]["isolating"],
        "symptoms_onset_level": bundle["levels"]["symptomsOnset"],
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
    parameter_space,
    parameter_case_mode,
    parameter_case_count,
    parameter_manifest_paths,
):
    content = f"""# Dataset Generation Parameters

This file documents the parameters used by `sweep_pipeline.py` to create the datasets for this sweep.

## Sweep Configuration

- `TIME_LIMITS` (days): {TIME_LIMITS}
- `N_SUBJECTS`: {N_SUBJECTS}
- `TOTAL_INTERNAL_CONTACTS`: {TOTAL_INTERNAL_CONTACTS}
- `seed_base`: {seed_base}
- `TIME_STEP_HOURS`: {time_step_hours}
- `parameter_ods_path`: {parameter_space["source_path"]}
- `parameter_case_mode`: {parameter_case_mode}
- `parameter_case_count`: {parameter_case_count}
- `parameter_manifest_csv`: {parameter_manifest_paths["csv_path"]}
- `parameter_manifest_json`: {parameter_manifest_paths["json_path"]}
- Per-run seed rule: `seed = seed_base + combination_index`

## Parameter Cases

- The sweep loads the lower/mid/upper parameter values from the ODS spreadsheet.
- For the distribution sheets, the hour-based rows are used.
- This is the correct unit for the active Python simulation because event times and transition durations are tracked in hours, and the transition lambdas are consumed as per-hour rates.
- The `symptoms` sheet is unitless and is used as a probability.

## Per-Run Dataset Parameters

For each run, `generate_dataset()` calls `dataset_graph.simulate_external_introduction()` with:

- `n_nodes = n_subjects`
- `tmax_after_intro = time_limit_days * 24` hours
- `max_intro_time = min(48.0, time_limit_days * 24)` hours
- `total_internal_contacts = selected TOTAL_INTERNAL_CONTACTS value`
- `seed = derived per-run seed`

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
  - `randint(0, 2 * time_limit_days // 7)`
- Event times are stored as floating-point hours in the dataset JSON

## Dataset Event Construction Notes

- External events are generated before simulation output is converted into dataset events
- Internal transmission events come from the epidemic simulation and are supplemented with extra internal contacts when needed to reach the requested total
- Tests are added independently per subject using the sampler above
- The dataset-generation step itself does not depend on the lower/mid/upper transition bundle; those parameter cases are applied in the Python simulation runs that produce the ground-truth trajectories
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
    granularity=1.0,
):
    analysis = normalize_subject_series(read_json(analysis_path))
    ground_truth = normalize_subject_series(read_json(ground_truth_path))

    analysis_subjects = list(analysis.keys())
    ground_truth_subjects = list(ground_truth.keys())
    if analysis_subjects != ground_truth_subjects:
        raise ValueError("Analysis and ground-truth files do not contain the same subjects.")

    output_dir = ensure_dir(os.path.join(run_dir, "plots", "analysis_subject_curves"))
    dkw_result = compute_confidence_intervals(ground_truth)
    csv_rows = []
    generated_plots = []
    java_only_plots = []

    for subject_id in ground_truth_subjects:
        ground_truth_values = ground_truth[subject_id]
        analysis_values = analysis[subject_id]
        lower = dkw_result["band"][subject_id]["lower"]
        upper = dkw_result["band"][subject_id]["upper"]

        if not (
            len(ground_truth_values)
            == len(analysis_values)
            == len(lower)
            == len(upper)
        ):
            raise ValueError(f"Inconsistent trajectory length for subject {subject_id}.")

        time_axis = np.arange(len(ground_truth_values)) * granularity

        plt.figure(figsize=(10, 4.5))
        plt.plot(
            time_axis,
            ground_truth_values,
            linewidth=1.8,
            color="tab:blue",
            label="Ground truth mean",
        )
        plt.fill_between(
            time_axis,
            lower,
            upper,
            alpha=0.25,
            color="tab:blue",
            label="Ground truth DKW band",
        )
        plt.plot(
            time_axis,
            analysis_values,
            linewidth=1.8,
            color="tab:orange",
            label="Java analysis",
        )
        plt.ylim(0.0, 1.0)
        plt.xlabel("Time")
        plt.ylabel("Probability")
        plt.title(f"Subject {subject_id} Analysis Curve ({run_name})")
        plt.legend()
        plt.tight_layout()

        output_name = f"{run_name}_subject_{subject_id}_analysis_curve.png"
        output_path = os.path.join(output_dir, output_name)
        plt.savefig(output_path, dpi=150)
        plt.close()
        generated_plots.append(output_path)

        plt.figure(figsize=(10, 4.5))
        plt.plot(
            time_axis,
            analysis_values,
            linewidth=1.8,
            color="tab:orange",
            label="Java analysis",
        )
        plt.ylim(0.0, 1.0)
        plt.xlabel("Time")
        plt.ylabel("Probability")
        plt.title(f"Subject {subject_id} Java Analysis Curve ({run_name})")
        plt.legend()
        plt.tight_layout()

        java_only_name = f"{run_name}_subject_{subject_id}_java_curve.png"
        java_only_path = os.path.join(output_dir, java_only_name)
        plt.savefig(java_only_path, dpi=150)
        plt.close()
        java_only_plots.append(java_only_path)

        for time_index, (ground_truth_value, lower_value, upper_value, analysis_value) in enumerate(
            zip(ground_truth_values, lower, upper, analysis_values)
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
                ]
            )

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
        ],
        csv_rows,
    )

    return {
        "output_dir": output_dir,
        "csv_path": csv_path,
        "epsilon": dkw_result["epsilon"],
        "plot_paths": generated_plots,
        "java_curve_plot_paths": java_only_plots,
    }


def compute_candidate_summary(ground_truth, candidate):
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
        moving_precision = ranking_metrics.compute_top_n_precision_on_a_moving_window(
            ground_truth,
            candidate,
            top_n=top_k,
            window_size=1,
        )
        summary[f"top_{top_k}_precision_mean"] = float(np.mean(precision))
        summary[f"top_{top_k}_precision_moving_avg_mean"] = float(np.mean(moving_precision))
    return summary


def create_analysis_vs_simulation_plots(run_dir, run_name, ground_truth_path, analysis_path, baseline_path):
    ground_truth = read_json(ground_truth_path)
    analysis = read_json(analysis_path)
    baseline = read_json(baseline_path)

    output_dir = ensure_dir(os.path.join(run_dir, "plots", "analysis_vs_simulation"))
    window_size = 1

    tau_analysis, _ = ranking_metrics.compute_kendalls_tau_correlation_per_timestep(ground_truth, analysis)
    tau_baseline, _ = ranking_metrics.compute_kendalls_tau_correlation_per_timestep(ground_truth, baseline)
    plot_overlay(
        tau_analysis,
        tau_baseline,
        title=f"{run_name} Analysis vs Simulation",
        ylabel="Kendall's Tau",
        output_path=os.path.join(output_dir, "kendalls_tau_correlation.png"),
    )
    save_series_csv(
        os.path.join(output_dir, "kendall_correlation_data.csv"),
        ["timestep", "analysis_kendall", "simulation_kendall"],
        [[index, tau_analysis[index], tau_baseline[index]] for index in range(len(tau_analysis))],
    )

    tau_analysis_moving, _ = ranking_metrics.compute_kendalls_tau_correlation_moving_window(
        ground_truth, analysis, window_size=window_size
    )
    tau_baseline_moving, _ = ranking_metrics.compute_kendalls_tau_correlation_moving_window(
        ground_truth, baseline, window_size=window_size
    )
    plot_overlay(
        tau_analysis_moving,
        tau_baseline_moving,
        title=f"{run_name} Analysis vs Simulation (Moving Avg)",
        ylabel="Kendall's Tau",
        output_path=os.path.join(output_dir, "kendalls_tau_correlation_moving_avg.png"),
    )

    spearman_analysis, _ = ranking_metrics.compute_spearmans_correlation_per_timestep(ground_truth, analysis)
    spearman_baseline, _ = ranking_metrics.compute_spearmans_correlation_per_timestep(ground_truth, baseline)
    plot_overlay(
        spearman_analysis,
        spearman_baseline,
        title=f"{run_name} Analysis vs Simulation",
        ylabel="Spearman's Rho",
        output_path=os.path.join(output_dir, "spearmans_correlation.png"),
    )
    save_series_csv(
        os.path.join(output_dir, "spearman_correlation_data.csv"),
        ["timestep", "analysis_spearman", "simulation_spearman"],
        [[index, spearman_analysis[index], spearman_baseline[index]] for index in range(len(spearman_analysis))],
    )

    spearman_analysis_moving, _ = ranking_metrics.compute_spearman_correlation_moving_window(
        ground_truth, analysis, window_size=window_size
    )
    spearman_baseline_moving, _ = ranking_metrics.compute_spearman_correlation_moving_window(
        ground_truth, baseline, window_size=window_size
    )
    plot_overlay(
        spearman_analysis_moving,
        spearman_baseline_moving,
        title=f"{run_name} Analysis vs Simulation (Moving Avg)",
        ylabel="Spearman's Rho",
        output_path=os.path.join(output_dir, "spearmans_correlation_moving_avg.png"),
    )

    scalar_rows = []
    max_top_precision = min(MAX_TOP_PRECISION, len(ground_truth))
    for top_k in range(1, max_top_precision + 1):
        analysis_precision = ranking_metrics.compute_top_n_precision(ground_truth, analysis, top_k)
        baseline_precision = ranking_metrics.compute_top_n_precision(ground_truth, baseline, top_k)
        plot_overlay(
            analysis_precision,
            baseline_precision,
            title=f"{run_name} Analysis vs Simulation Top-{top_k} Precision",
            ylabel=f"Top-{top_k} Precision",
            output_path=os.path.join(output_dir, f"top_{top_k}_precision.png"),
        )
        save_series_csv(
            os.path.join(output_dir, f"top_{top_k}_precision_data.csv"),
            ["timestep", f"analysis_top_{top_k}_precision", f"simulation_top_{top_k}_precision"],
            [[index, analysis_precision[index], baseline_precision[index]] for index in range(len(analysis_precision))],
        )

        analysis_precision_moving = ranking_metrics.compute_top_n_precision_on_a_moving_window(
            ground_truth, analysis, top_n=top_k, window_size=window_size
        )
        baseline_precision_moving = ranking_metrics.compute_top_n_precision_on_a_moving_window(
            ground_truth, baseline, top_n=top_k, window_size=window_size
        )
        plot_overlay(
            analysis_precision_moving,
            baseline_precision_moving,
            title=f"{run_name} Analysis vs Simulation Top-{top_k} Precision (Moving Avg)",
            ylabel=f"Top-{top_k} Precision",
            output_path=os.path.join(output_dir, f"top_{top_k}_precision_moving_avg.png"),
        )
        save_series_csv(
            os.path.join(output_dir, f"top_{top_k}_precision_moving_avg_data.csv"),
            [
                "timestep",
                f"analysis_top_{top_k}_precision_moving_avg",
                f"simulation_top_{top_k}_precision_moving_avg",
            ],
            [
                [index, analysis_precision_moving[index], baseline_precision_moving[index]]
                for index in range(len(analysis_precision_moving))
            ],
        )

        scalar_rows.append(
            {
                "comparison": f"top_{top_k}",
                "analysis_mean": float(np.mean(analysis_precision)),
                "simulation_mean": float(np.mean(baseline_precision)),
                "analysis_moving_avg_mean": float(np.mean(analysis_precision_moving)),
                "simulation_moving_avg_mean": float(np.mean(baseline_precision_moving)),
            }
        )

    analysis_summary = compute_candidate_summary(ground_truth, analysis)
    baseline_summary = compute_candidate_summary(ground_truth, baseline)
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


def run_java_analysis(repo_root, run_dir, time_step_hours=TIME_STEP_HOURS):
    class_root = os.path.join(repo_root, "out", "production", "chita-main-test")
    if not os.path.exists(os.path.join(class_root, "com", "chita", "analysis", "STPNAnalysis.class")):
        raise FileNotFoundError("Compiled STPNAnalysis.class not found under out/production/chita-main-test.")

    java_executable = resolve_java_executable()
    gson_jar = resolve_optional_jar(
        [
            os.path.join(repo_root, "lib__", "gson.jar"),
            os.path.join(repo_root, "lib__", "gson-2.13.1.jar"),
            os.path.join(repo_root, "lib__", "gson-2.11.0.jar"),
            os.path.join(os.path.expanduser("~"), ".m2", "repository", "com", "google", "code", "gson", "gson", "2.13.1", "gson-2.13.1.jar"),
            os.path.join(os.path.expanduser("~"), ".m2", "repository", "com", "google", "code", "gson", "gson", "2.11.0", "gson-2.11.0.jar"),
            os.path.join(os.path.expanduser("~"), ".gradle", "caches", "modules-2", "files-2.1", "com.google.code.gson", "gson", "2.10.1", "b3add478d4382b78ea20b1671390a858002feb6c", "gson-2.10.1.jar"),
        ],
        "gson",
    )
    classpath = os.pathsep.join(
        [
            class_root,
            os.path.join(repo_root, "lib__", "*"),
            gson_jar,
        ]
    )
    stpn_solution_filename = f"stpn_solution_ts{sanitized_time_step_label(time_step_hours)}.csv"
    command = [
        java_executable,
        "-cp",
        classpath,
        "com.chita.analysis.STPNAnalysis",
        "--time-step",
        str(time_step_hours),
        "--stpn-solution-path",
        stpn_solution_filename,
    ]
    java_analysis_stdout_log_path = os.path.join(run_dir, "java_analysis_stdout.log")
    java_analysis_stderr_log_path = os.path.join(run_dir, "java_analysis_stderr.log")
    java_precompute_stdout_log_path = os.path.join(run_dir, "java_precompute_stdout.log")
    java_precompute_stderr_log_path = os.path.join(run_dir, "java_precompute_stderr.log")

    def execute_java_command(extra_args=None, stdout_log_path=None, stderr_log_path=None):
        started_at = time.perf_counter()
        result = subprocess.run(
            command + ([] if extra_args is None else extra_args),
            cwd=run_dir,
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

    stpn_solution_path = os.path.join(run_dir, stpn_solution_filename)
    precomputation_runtime_seconds = 0.0
    precomputation_performed = False
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
        "stpn_solution_path": stpn_solution_path,
        "time_step_hours": time_step_hours,
        "analysis_runtime_seconds": analysis_runtime_seconds,
        "analysis_wall_runtime_seconds": analysis_wall_runtime_seconds,
        "analysis_runtime_excludes_overhead": analysis_core_runtime_seconds is not None,
        "stpn_precomputation_runtime_seconds": precomputation_runtime_seconds,
        "stpn_precomputation_performed": precomputation_performed,
        "java_analysis_stdout_log_path": java_analysis_stdout_log_path,
        "java_analysis_stderr_log_path": java_analysis_stderr_log_path,
        "java_precompute_stdout_log_path": java_precompute_stdout_log_path if precomputation_performed else None,
        "java_precompute_stderr_log_path": java_precompute_stderr_log_path if precomputation_performed else None,
    }


def generate_dataset(run_dir, run_name, time_limit_days, n_subjects, total_internal_contacts, seed):
    result = dataset_generator.simulate_external_introduction(
        n_nodes=n_subjects,
        tmax_after_intro=float(time_limit_days * 24),
        max_intro_time=min(48.0, float(time_limit_days * 24)),
        total_internal_contacts=total_internal_contacts,
        seed=seed,
    )
    dataset_path = os.path.join(
        run_dir,
        f"dataset_s{n_subjects}_t{time_limit_days}_c{total_internal_contacts}.json",
    )
    payload = dataset_generator.save_dataset_event_sequence(result, dataset_path)
    write_json(
        os.path.join(run_dir, "dataset_metadata.json"),
        {
            "run_name": run_name,
            "dataset_path": dataset_path,
            "n_subjects": payload["n_subjects"],
            "time_limit": payload["time_limit"],
            "n_contacts": payload["n_contacts"],
            "seed": seed,
        },
    )
    return dataset_path, payload


def build_run_name(time_limit_days, n_subjects, total_internal_contacts, parameter_bundle):
    return (
        f"run_t{time_limit_days}_s{n_subjects}_c{total_internal_contacts}"
        f"__{parameter_bundle['case_id']}"
    )


def run_single_pipeline(
    repo_root,
    output_root,
    time_limit_days,
    n_subjects,
    total_internal_contacts,
    seed,
    time_step_hours=TIME_STEP_HOURS,
    parameter_bundle=None,
):
    if parameter_bundle is None:
        raise ValueError("run_single_pipeline requires a parameter_bundle.")

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

    summary = {
        "run_name": run_name,
        "run_dir": run_dir,
        "time_limit": time_limit_days,
        "n_subjects": n_subjects,
        "total_internal_contacts": total_internal_contacts,
        "seed": seed,
        "time_step_hours": time_step_hours,
        "parameter_case_id": parameter_bundle["case_id"],
        "parameter_levels": parameter_bundle["levels"],
        "parameter_unit_measure": parameter_bundle["unit_measure"],
        "parameter_bundle_path": parameter_bundle_path,
        "status": "running",
    }
    write_json(os.path.join(run_dir, "run_summary.json"), summary)

    dataset_path, dataset_payload = generate_dataset(
        run_dir,
        run_name,
        time_limit_days,
        n_subjects,
        total_internal_contacts,
        seed,
    )
    summary["dataset_path"] = dataset_path
    summary["dataset_events"] = len(dataset_payload["events"])

    convergence_result = run_dataset_simulations(
        dataset_path=dataset_path,
        run_until_convergence=True,
        iterations_cap=CONVERGENCE_ITERATIONS_CAP,
        convergence_threshold=CONVERGENCE_THRESHOLD,
        fine_grained=fine_grained,
        time_step_hours=time_step_hours,
        dataset_label=run_name,
        seed=seed,
        prune_after_positive_test=False,
        export_observed_simulation=True,
        pruning_seed=seed,
        parameter_bundle=parameter_bundle,
    )
    summary["effective_dataset_path"] = convergence_result["effective_dataset_path"]
    summary["pruned_dataset_path"] = convergence_result["pruned_dataset_path"]
    summary["positive_test_pruning"] = convergence_result["positive_test_pruning"]
    summary["convergence"] = {
        "threshold": CONVERGENCE_THRESHOLD,
        "iterations": convergence_result["rep_done"],
        "reached": convergence_result["convergence_reached"],
        "scores": convergence_result["convergence_scores"],
        "ground_truth_path": convergence_result["averaged_results_path"],
        "observed_simulated_path": convergence_result["observed_simulated_path"],
        "effective_dataset_path": convergence_result["effective_dataset_path"],
        "pruned_dataset_path": convergence_result["pruned_dataset_path"],
        "time_step_hours": convergence_result.get("time_step_hours", time_step_hours),
        "actual_runtime_seconds": convergence_result.get("actual_runtime_seconds"),
        "suppressed_stdout_log_path": convergence_result.get("suppressed_stdout_log_path"),
    }

    java_result = run_java_analysis(repo_root, run_dir, time_step_hours=time_step_hours)
    analysis_path = find_generated_file(run_dir, "_tracks_it3.json")
    analysis_curve_plots = create_analysis_subject_curve_plots(
        run_dir=run_dir,
        run_name=run_name,
        analysis_path=analysis_path,
        ground_truth_path=convergence_result["averaged_results_path"],
        granularity=convergence_result.get("granularity", 1.0),
    )
    summary["analysis"] = {
        "analysis_path": analysis_path,
        "stpn_solution_path": java_result["stpn_solution_path"],
        "analysis_runtime_seconds": java_result["analysis_runtime_seconds"],
        "analysis_wall_runtime_seconds": java_result["analysis_wall_runtime_seconds"],
        "analysis_runtime_excludes_overhead": java_result["analysis_runtime_excludes_overhead"],
        "stpn_precomputation_runtime_seconds": java_result["stpn_precomputation_runtime_seconds"],
        "stpn_precomputation_performed": java_result["stpn_precomputation_performed"],
        "parameterization": "fixed_java_defaults",
        "java_analysis_stdout_log_path": java_result["java_analysis_stdout_log_path"],
        "java_analysis_stderr_log_path": java_result["java_analysis_stderr_log_path"],
        "java_precompute_stdout_log_path": java_result["java_precompute_stdout_log_path"],
        "java_precompute_stderr_log_path": java_result["java_precompute_stderr_log_path"],
        "subject_curve_plots": analysis_curve_plots,
    }

    baseline_runtime_budget_seconds = max(2.0 * java_result["analysis_runtime_seconds"], 0.0)
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
    )
    summary["baseline"] = {
        "iterations": baseline_result["rep_done"],
        "baseline_path": baseline_result["averaged_results_path"],
        "dkw_csv_path": baseline_result["dkw_csv_path"],
        "effective_dataset_path": baseline_result["effective_dataset_path"],
        "pruned_dataset_path": baseline_result["pruned_dataset_path"],
        "time_step_hours": baseline_result.get("time_step_hours", time_step_hours),
        "runtime_budget_seconds": baseline_runtime_budget_seconds,
        "actual_runtime_seconds": baseline_result.get("actual_runtime_seconds"),
        "suppressed_stdout_log_path": baseline_result.get("suppressed_stdout_log_path"),
    }

    precision_dir = ensure_dir(os.path.join(run_dir, "precision_metrics"))
    prediction_metrics_path = os.path.join(precision_dir, "metrics_prediction.json")
    baseline_metrics_path = os.path.join(precision_dir, "metrics_baseline.json")
    metrics_stdout = io.StringIO()
    with contextlib.redirect_stdout(metrics_stdout):
        process_and_save(
            analysis_path,
            convergence_result["averaged_results_path"],
            M=10,
            metrics_output=prediction_metrics_path,
            plots_dir=os.path.join(precision_dir, "numericalAnalysis", "plots"),
        )
        process_and_save(
            baseline_result["averaged_results_path"],
            convergence_result["averaged_results_path"],
            M=10,
            metrics_output=baseline_metrics_path,
            plots_dir=os.path.join(precision_dir, "simulatedBaseline", "plots"),
        )
    with open(os.path.join(precision_dir, "metrics_stdout.log"), "w", encoding="utf-8") as handle:
        handle.write(metrics_stdout.getvalue())

    comparison_summary = create_analysis_vs_simulation_plots(
        run_dir,
        run_name,
        convergence_result["averaged_results_path"],
        analysis_path,
        baseline_result["averaged_results_path"],
    )

    summary["precision_metrics"] = {
        "prediction_metrics_path": prediction_metrics_path,
        "baseline_metrics_path": baseline_metrics_path,
        "prediction_mean_brier": mean_metric_value(prediction_metrics_path, "Brier Score"),
        "prediction_mean_ece": mean_metric_value(prediction_metrics_path, "ECE"),
        "baseline_mean_brier": mean_metric_value(baseline_metrics_path, "Brier Score"),
        "baseline_mean_ece": mean_metric_value(baseline_metrics_path, "ECE"),
    }
    summary["comparison_metrics"] = comparison_summary
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
            "time_limit": summary["time_limit"],
            "n_subjects": summary["n_subjects"],
            "total_internal_contacts": summary["total_internal_contacts"],
            "time_step_hours": summary.get("time_step_hours"),
            "parameter_case_id": summary.get("parameter_case_id"),
            "infectiousness_level": parameter_levels.get("infectiousness"),
            "healing_level": parameter_levels.get("healing"),
            "symptoms_level": parameter_levels.get("symptoms"),
            "isolating_level": parameter_levels.get("isolating"),
            "symptoms_onset_level": parameter_levels.get("symptomsOnset"),
            "dataset_events": summary.get("dataset_events"),
            "convergence_iterations": summary.get("convergence", {}).get("iterations"),
            "convergence_reached": summary.get("convergence", {}).get("reached"),
            "java_analysis_runtime_seconds": summary.get("analysis", {}).get("analysis_runtime_seconds"),
            "java_analysis_wall_runtime_seconds": summary.get("analysis", {}).get("analysis_wall_runtime_seconds"),
            "java_analysis_runtime_excludes_overhead": summary.get("analysis", {}).get("analysis_runtime_excludes_overhead"),
            "stpn_precomputation_runtime_seconds": summary.get("analysis", {}).get("stpn_precomputation_runtime_seconds"),
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
    args = parser.parse_args()
    if args.time_step_hours <= 0:
        raise ValueError("--time-step-hours must be greater than 0.")

    repo_root = os.path.abspath(os.path.dirname(__file__))
    output_root = ensure_dir(os.path.abspath(args.output_root))
    parameter_space = load_parameter_space_from_ods(os.path.abspath(args.parameter_ods_path))
    parameter_cases = enumerate_parameter_cases(
        parameter_space,
        mode=args.parameter_case_mode,
    )
    parameter_manifest_paths = write_parameter_case_manifest(output_root, parameter_cases)
    write_dataset_generation_parameters_markdown(
        output_root=output_root,
        seed_base=args.seed_base,
        time_step_hours=args.time_step_hours,
        parameter_space=parameter_space,
        parameter_case_mode=args.parameter_case_mode,
        parameter_case_count=len(parameter_cases),
        parameter_manifest_paths=parameter_manifest_paths,
    )

    summaries = []
    combinations = itertools.product(
        TIME_LIMITS,
        N_SUBJECTS,
        TOTAL_INTERNAL_CONTACTS,
        parameter_cases,
    )
    for index, (time_limit_days, n_subjects, total_internal_contacts, parameter_bundle) in enumerate(combinations):
        seed = args.seed_base + index
        try:
            summary = run_single_pipeline(
                repo_root=repo_root,
                output_root=output_root,
                time_limit_days=time_limit_days,
                n_subjects=n_subjects,
                total_internal_contacts=total_internal_contacts,
                seed=seed,
                time_step_hours=args.time_step_hours,
                parameter_bundle=parameter_bundle,
            )
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
                "time_limit": time_limit_days,
                "n_subjects": n_subjects,
                "total_internal_contacts": total_internal_contacts,
                "seed": seed,
                "time_step_hours": args.time_step_hours,
                "parameter_case_id": parameter_bundle["case_id"],
                "parameter_levels": parameter_bundle["levels"],
                "parameter_unit_measure": parameter_bundle["unit_measure"],
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            write_json(os.path.join(run_dir, "run_summary.json"), summary)
        summaries.append(summary)
        write_aggregate_outputs(output_root, summaries)


if __name__ == "__main__":
    main()
