# This program is part of the ORIS Tool.
# Copyright (C) 2011-2025 The ORIS Authors.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

import contextlib
import csv
import dataset
import simulation
from tqdm import tqdm
import json
import os
import argparse
import copy
import time
from compute_precision_metrics import calculate_brier_score
import numpy as np
import matplotlib.pyplot as plt


def get_suppressed_stdout_log_path(dataset_dir, dataset_stem, run_until_convergence, max_runtime_seconds):
    if run_until_convergence:
        suffix = "convergence"
    elif max_runtime_seconds is not None:
        suffix = "runtime_limited"
    else:
        suffix = "fixed_reps"
    return os.path.join(dataset_dir, f"{os.path.basename(dataset_stem)}_{suffix}_stdout.log")


def check_convergence(current_state, last_state, threshold=1e-4):
    p_subjects = sorted(list(current_state.keys()), key=int)
    g_subjects = sorted(list(last_state.keys()), key=int)
    if p_subjects != g_subjects:
        error_msg = "Error: The states do not contain the same subjects.\n"        
        raise ValueError(error_msg)
    scores = []
    converged = True
    for subject_id in g_subjects:
        current = current_state[subject_id]
        last = last_state[subject_id]
        brier_score = calculate_brier_score(current, last)
        scores.append(brier_score)
        if brier_score > threshold:
            converged = False
    return {"converged" : converged, "scores" : scores}

def compute_confidence_intervals(current_state, confidence=0.95):
    if not current_state:
        raise ValueError("Error: current_state cannot be empty.")

    subjects = sorted(list(current_state.keys()), key=int)
    horizon = len(current_state[subjects[0]])
    if horizon == 0:
        raise ValueError("Error: subject trajectories cannot be empty.")

    for subject_id in subjects:
        if len(current_state[subject_id]) != horizon:
            raise ValueError("Error: all subjects must have trajectories with the same length.")

    alpha = 1 - confidence
    if alpha <= 0 or alpha >= 1:
        raise ValueError("Error: confidence must be between 0 and 1.")

    # Dvoretzky-Kiefer-Wolfowitz epsilon for empirical CDF bands.
    epsilon = np.sqrt(-np.log(alpha / 2) / (2 * horizon))
    dkw_band = {
        subject_id: {
            "lower": [max(0.0, value - epsilon) for value in current_state[subject_id]],
            "upper": [min(1.0, value + epsilon) for value in current_state[subject_id]],
        }
        for subject_id in subjects
    }

    return {"epsilon": float(epsilon), "band": dkw_band}


def plot_dkw_bands(
    current_state,
    dkw_result,
    output_dir,
    granularity=1.0,
    dataset_label="",
    save_plots=True,
):
    if not current_state:
        raise ValueError("Error: current_state cannot be empty.")
    if "band" not in dkw_result:
        raise ValueError("Error: dkw_result must contain a 'band' entry.")
    if not save_plots:
        return

    os.makedirs(output_dir, exist_ok=True)
    subjects = sorted(list(current_state.keys()), key=int)
    horizon = len(current_state[subjects[0]])
    time_axis = np.arange(horizon) * granularity
    band = dkw_result["band"]

    for subject_id in subjects:
        if subject_id not in band:
            raise ValueError(f"Error: missing DKW band for subject {subject_id}.")
        lower = band[subject_id]["lower"]
        upper = band[subject_id]["upper"]
        mean_values = current_state[subject_id]
        if not (len(lower) == len(upper) == len(mean_values)):
            raise ValueError(f"Error: inconsistent DKW band length for subject {subject_id}.")

        plt.figure(figsize=(10, 4.5))
        plt.plot(time_axis, mean_values, label="Mean probability", linewidth=1.8, color="tab:blue")
        plt.fill_between(time_axis, lower, upper, alpha=0.25, color="tab:blue", label="DKW band")
        plt.ylim(0.0, 1.0)
        plt.xlabel("Time")
        plt.ylabel("Probability")
        title = f"Subject {subject_id} DKW Confidence Band"
        if dataset_label:
            title += f" ({dataset_label})"
        plt.title(title)
        plt.legend()
        plt.tight_layout()

        output_name = f"subject_{subject_id}_dkw_band.png"
        if dataset_label:
            output_name = f"{dataset_label}_subject_{subject_id}_dkw_band.png"
        plt.savefig(os.path.join(output_dir, output_name), dpi=150)
        plt.close()


def export_dkw_bands_to_csv(current_state, dkw_result, output_csv_path, granularity=1.0):
    if not current_state:
        raise ValueError("Error: current_state cannot be empty.")
    if "band" not in dkw_result:
        raise ValueError("Error: dkw_result must contain a 'band' entry.")

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    subjects = sorted(list(current_state.keys()), key=int)
    horizon = len(current_state[subjects[0]])
    band = dkw_result["band"]
    epsilon = dkw_result.get("epsilon", "")

    with open(output_csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "subject_id",
            "time_index",
            "time",
            "mean_probability",
            "lower_band",
            "upper_band",
            "epsilon",
        ])

        for subject_id in subjects:
            if subject_id not in band:
                raise ValueError(f"Error: missing DKW band for subject {subject_id}.")
            lower = band[subject_id]["lower"]
            upper = band[subject_id]["upper"]
            mean_values = current_state[subject_id]
            if not (len(lower) == len(upper) == len(mean_values) == horizon):
                raise ValueError(f"Error: inconsistent DKW band length for subject {subject_id}.")

            for time_index, mean_value in enumerate(mean_values):
                writer.writerow([
                    subject_id,
                    time_index,
                    time_index * granularity,
                    mean_value,
                    lower[time_index],
                    upper[time_index],
                    epsilon,
                ])


def get_infectious_simulated_data(numerical_results, avg_state):
    bin_data = {}
    for subject, results in numerical_results.items():
        bin_data[subject] = [1 if value == 1 else 0 for value in results]
    return bin_data


def get_granularity(fine_grained=False, time_step_hours=None):
    if time_step_hours is not None:
        if time_step_hours <= 0:
            raise ValueError("Error: time_step_hours must be greater than 0.")
        return float(time_step_hours)
    return 0.1 if fine_grained else 1.0


def run_dataset_simulations(
    dataset_path,
    rep=10_000,
    run_until_convergence=False,
    iterations_cap=100_000,
    convergence_threshold=1e-4,
    convergence_check_every=1000,
    fine_grained=False,
    dataset_label=None,
    seed=None,
    distortion=1.0,
    progress_bar=None,
    prune_after_positive_test=False,
    export_observed_simulation=True,
    pruning_seed=None,
    time_step_hours=None,
    max_runtime_seconds=None,
    parameter_bundle=None,
    save_plots=True,
):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    effective_fine_grained = fine_grained or (
        time_step_hours is not None and float(time_step_hours) < 1.0
    )

    raw_dataset_path = dataset_path
    effective_dataset_path = dataset_path
    pruned_dataset_path = None
    positive_test_pruning = None

    if prune_after_positive_test:
        raise NotImplementedError(
            "prune_after_positive_test is not implemented in this pipeline."
        )

    source_data = simulation.read_data(effective_dataset_path)
    n_subjects = source_data["n_subjects"]
    timelimit = source_data["time_limit"]
    granularity = get_granularity(
        fine_grained=effective_fine_grained,
        time_step_hours=time_step_hours,
    )
    horizon = int(timelimit * 24 / granularity)
    dataset_stem = os.path.splitext(effective_dataset_path)[0]
    dataset_dir = os.path.dirname(effective_dataset_path) or "."
    suppressed_stdout_log_path = get_suppressed_stdout_log_path(
        dataset_dir=dataset_dir,
        dataset_stem=dataset_stem,
        run_until_convergence=run_until_convergence,
        max_runtime_seconds=max_runtime_seconds,
    )

    avg_state = {subject_id: [0] * horizon for subject_id in range(1, n_subjects + 1)}
    if run_until_convergence:
        max_iterations = iterations_cap
    elif rep is None:
        max_iterations = iterations_cap
    else:
        max_iterations = rep
    rep_done = 0
    save_first_iteration = export_observed_simulation
    convergence_reached = False
    conv_check = None
    last_state = None
    started_at = time.perf_counter()
    with open(suppressed_stdout_log_path, "w", encoding="utf-8", buffering=1) as suppressed_stdout_handle:
        suppressed_stdout_handle.write(
            f"# Suppressed stdout log for {os.path.basename(dataset_stem)}\n"
            f"# run_until_convergence={run_until_convergence}\n"
            f"# max_runtime_seconds={max_runtime_seconds}\n"
        )

        for iteration_index in range(max_iterations):
            if (
                max_runtime_seconds is not None
                and rep_done > 0
                and (time.perf_counter() - started_at) >= max_runtime_seconds
            ):
                break
            # Per-iteration log markers are disabled to reduce I/O overhead.
            # suppressed_stdout_handle.write(f"\n## iteration={iteration_index}\n")
            # suppressed_stdout_handle.flush()
            with contextlib.redirect_stdout(suppressed_stdout_handle):
                data = copy.deepcopy(source_data)
                data["events"].sort(key=lambda x: x["time"])
                simulated_data = simulation.simulate_one_iteration(
                    data,
                    n_subjects,
                    export_data=save_first_iteration,
                    exported_data_filename=f"{dataset_stem}_simulated",
                    seed=seed if iteration_index == 0 else None,
                    fine_grained=effective_fine_grained,
                    distortion=distortion if iteration_index == 0 else 1.0,
                    parameter_bundle=parameter_bundle,
                )
                save_first_iteration = False
                numerical_results = simulation.get_numerical_results(simulated_data, granularity)
                infectious_results = get_infectious_simulated_data(numerical_results, avg_state)

            for subject, infectious_track in infectious_results.items():
                subject_state = avg_state[subject]
                for time_index, value in enumerate(infectious_track):
                    subject_state[time_index] += value

            rep_done += 1

            if run_until_convergence:
                current_state = {
                    subject_id: [avg_state[subject_id][time_index] / rep_done for time_index in range(horizon)]
                    for subject_id in range(1, n_subjects + 1)
                }
                if iteration_index == 0:
                    last_state = current_state
                elif iteration_index % convergence_check_every == 0:
                    conv_check = check_convergence(
                        current_state=current_state,
                        last_state=last_state,
                        threshold=convergence_threshold,
                    )
                    convergence_reached = conv_check["converged"]
                    if convergence_reached:
                        break
                    last_state = current_state

            if progress_bar is not None:
                progress_bar.update(1)

    actual_runtime_seconds = time.perf_counter() - started_at
    if rep_done == 0:
        raise RuntimeError("No simulation repetitions were completed.")

    avg_results = {
        subject_id: [avg_state[subject_id][time_index] / rep_done for time_index in range(horizon)]
        for subject_id in range(1, n_subjects + 1)
    }

    confidence_intervals = compute_confidence_intervals(avg_results)
    effective_label = dataset_label or os.path.basename(dataset_dir)
    plots_dir = os.path.join(dataset_dir, "plots")
    plot_dkw_bands(
        current_state=avg_results,
        dkw_result=confidence_intervals,
        output_dir=plots_dir,
        granularity=granularity,
        dataset_label=f"{effective_label}_t{timelimit}_{rep_done}reps",
        save_plots=save_plots,
    )
    dkw_csv_path = os.path.join(
        plots_dir,
        f"{effective_label}_t{timelimit}_{rep_done}reps_dkw_band.csv",
    )
    export_dkw_bands_to_csv(
        current_state=avg_results,
        dkw_result=confidence_intervals,
        output_csv_path=dkw_csv_path,
        granularity=granularity,
    )

    averaged_results_path = f"{dataset_stem}_simulated_{rep_done}_reps.json"
    with open(averaged_results_path, "w", encoding="utf-8") as handle:
        json.dump(avg_results, handle, indent=4)

    return {
        "dataset_path": raw_dataset_path,
        "effective_dataset_path": effective_dataset_path,
        "pruned_dataset_path": pruned_dataset_path,
        "dataset_dir": dataset_dir,
        "dataset_stem": dataset_stem,
        "n_subjects": n_subjects,
        "time_limit": timelimit,
        "rep_done": rep_done,
        "actual_runtime_seconds": actual_runtime_seconds,
        "max_runtime_seconds": max_runtime_seconds,
        "run_until_convergence": run_until_convergence,
        "convergence_reached": convergence_reached,
        "convergence_scores": [] if conv_check is None else conv_check["scores"],
        "convergence_threshold": convergence_threshold,
        "observed_simulated_path": f"{dataset_stem}_simulated.json" if export_observed_simulation else None,
        "averaged_results_path": averaged_results_path,
        "plots_dir": plots_dir,
        "dkw_csv_path": dkw_csv_path,
        "granularity": granularity,
        "time_step_hours": granularity,
        "positive_test_pruning": positive_test_pruning,
        "suppressed_stdout_log_path": suppressed_stdout_log_path,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulations")
    parser.add_argument("--rep", type=int, default=10_000, help="Number of repetitions per configuration")
    parser.add_argument("--run_until_convergence", action="store_true", help="Run simulations until convergence", default=False)
    parser.add_argument("--iterations_cap", type=int, default=100_000, help="Maximum number of iterations when running until convergence")
    args = parser.parse_args()

    FINE_GRAINED = True
    GRANULARITY = get_granularity(fine_grained=FINE_GRAINED)

    rep = args.rep
    timelimits = {84}
    subjects = {15}
    if args.run_until_convergence:
        rep = args.iterations_cap
    total_iterations = len(timelimits) * len(subjects) * 3 * rep
    datasets = ["D0", "D1", "D2", "D2+15", "D2-15", "D2+25", "D2-25", "D3"]
    # datasets = ["D0"]
    total_iterations = len(datasets) * rep
    seeds = [i for i in range(2, len(datasets) + 2)]
    seed_index = 0
    with tqdm(total = total_iterations, desc="Progress") as pbar:
        for timelimit in sorted(timelimits):
            for n_subjects in sorted(subjects):
                for dataset_path in datasets:
                    if dataset_path == "":
                        seed_index += 1
                        continue
                    if dataset_path == "D0":
                        n_subjects = 8
                        internal_contacts = {84}
                    elif dataset_path == "D1":
                        n_subjects = 4
                        internal_contacts = {84}
                    elif dataset_path == "D2" or dataset_path == "D2+15" or dataset_path == "D2-15" or dataset_path == "D2+25" or dataset_path == "D2-25":
                        n_subjects = 8
                        internal_contacts = {84}
                    elif dataset_path == "D3":
                        n_subjects = 16
                        internal_contacts = {84}
                    os.makedirs(f"{dataset_path}", exist_ok=True)
                    filenames = dataset.create_datasets(
                        f"{dataset_path}/dataset_s{n_subjects}_t{timelimit}",
                        [n_subjects],
                        [timelimit],
                        seeds[seed_index],
                        FINE_GRAINED,
                        internal_contacts=internal_contacts,
                        max_contacts=6,
                    )
                    for filename in filenames:
                        print("\033[91m" + filename + "\033[0m")
                        distortion = 1.0
                        if dataset_path == "D2+15":
                            distortion = 1.15
                        elif dataset_path == "D2-15":
                            distortion = 0.85
                        elif dataset_path == "D2+25":
                            distortion = 1.25
                        elif dataset_path == "D2-25":
                            distortion = 0.75
                        result = run_dataset_simulations(
                            dataset_path=filename,
                            rep=rep,
                            run_until_convergence=args.run_until_convergence,
                            iterations_cap=args.iterations_cap,
                            convergence_threshold=1e-4,
                            convergence_check_every=1000,
                            fine_grained=FINE_GRAINED,
                            dataset_label=dataset_path,
                            seed=seeds[seed_index],
                            distortion=distortion,
                            progress_bar=pbar,
                        )
                        if args.run_until_convergence and result["convergence_scores"]:
                            print(f"Convergence scores: {result['convergence_scores']}")
                    seed_index += 1
