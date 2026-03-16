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
from compute_precision_metrics import calculate_brier_score
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def check_convergence(current_state, last_state, threshold=1e-6):
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
    print(f"Convergence check: Brier scores for all subjects: {scores}")
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


def plot_dkw_bands(current_state, dkw_result, output_dir, granularity=1.0, dataset_label=""):
    if not current_state:
        raise ValueError("Error: current_state cannot be empty.")
    if "band" not in dkw_result:
        raise ValueError("Error: dkw_result must contain a 'band' entry.")

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulations")
    parser.add_argument("--rep", type=int, default=10_000, help="Number of repetitions per configuration")
    parser.add_argument("--run_until_convergence", action="store_true", help="Run simulations until convergence", default=False)
    parser.add_argument("--iterations_cap", type=int, default=100_000, help="Maximum number of iterations when running until convergence")
    args = parser.parse_args()

    FINE_GRAINED = True
    if FINE_GRAINED:
        GRANULARITY = 0.1
    else:
        GRANULARITY = 1.0

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
                    filenames = dataset.create_datasets(f"{dataset_path}/dataset_s{n_subjects}_t{timelimit}", [n_subjects], [timelimit], seeds[seed_index], FINE_GRAINED, internal_contacts = internal_contacts, max_contacts = 6)
                    for filename in filenames:
                        print("\033[91m" + filename + "\033[0m")
                        avg_state ={i : [0] * int(timelimit * 24 / GRANULARITY) for i in range(1, n_subjects + 1)}
                        save = True
                        convergence_reached = False
                        rep_done = 0
                        for i in range(rep):
                            with contextlib.redirect_stdout(None):
                                data = simulation.read_data(f"{filename}")
                                data["events"].sort(key=lambda x: x["time"])
                                if i == 0:
                                    distortion = 1.0
                                    if dataset_path == "D2+15":
                                        distortion = 1.15
                                    elif dataset_path == "D2-15":
                                        distortion = 0.85
                                    elif dataset_path == "D2+25":
                                        distortion = 1.25
                                    elif dataset_path == "D2-25":
                                        distortion = 0.75
                                    simulated_data = simulation.simulate_one_iteration(data, n_subjects, save, f"{filename.split('.')[0]}_simulated", seeds[seed_index], distortion=distortion)
                                else:
                                    simulated_data  = simulation.simulate_one_iteration(data, n_subjects, save, f"{filename.split('.')[0]}_simulated", distortion=1.0)
                                save = False
                                numerical_results = simulation.get_numerical_results(simulated_data, GRANULARITY)
                                infectious_results = get_infectious_simulated_data(numerical_results, avg_state)
                                for subject in infectious_results:
                                    for j in range(int(timelimit * 24 / GRANULARITY)):
                                        avg_state[subject][j] += infectious_results[subject][j]
                                rep_done += 1
                                if args.run_until_convergence:
                                    if i == 0:
                                        last_state = {i : [avg_state[i][j] / rep_done for j in range(int(timelimit * 24 / GRANULARITY))] for i in range(1, n_subjects + 1)}
                                    if i > 0 and i % 1000 == 0:
                                        # Check for convergence every 1000 iterations
                                        current_state = {i : [avg_state[i][j] / rep_done for j in range(int(timelimit * 24 / GRANULARITY))] for i in range(1, n_subjects + 1)}
                                        conv_check = check_convergence(current_state=current_state, last_state=last_state)
                                        convergence_reached = conv_check["converged"]
                                        if convergence_reached:
                                            print(f"Convergence reached at iteration {i} for dataset {dataset_path} with timelimit {timelimit} and n_subjects {n_subjects}.")
                                            break
                                        else:
                                            last_state = current_state
                                pbar.update(1)
                        if args.run_until_convergence:
                            print(f"Convergence scores: {conv_check['scores']}")     
                        avg_results = {i : [avg_state[i][j] / rep_done for j in range(int(timelimit * 24 / GRANULARITY))] for i in range(1, n_subjects + 1)}
                        confidence_intervals = compute_confidence_intervals(avg_results)
                        plot_dkw_bands(
                            current_state=avg_results,
                            dkw_result=confidence_intervals,
                            output_dir=os.path.join(os.path.dirname(filename), "plots"),
                            granularity=GRANULARITY,
                            dataset_label=f"{dataset_path}_t{timelimit}_{rep_done}reps",
                        )
                        export_dkw_bands_to_csv(
                            current_state=avg_results,
                            dkw_result=confidence_intervals,
                            output_csv_path=os.path.join(
                                os.path.dirname(filename),
                                "plots",
                                f"{dataset_path}_t{timelimit}_{rep_done}reps_dkw_band.csv",
                            ),
                            granularity=GRANULARITY,
                        )
                        filename_surgery = filename.split('_t')[0] + f"_t{timelimit}_{internal_contacts.pop()}_simulated_{rep_done}_reps.json" 
                        with open(filename_surgery, "w") as f:
                            json.dump(avg_results, f, indent=4)
                    seed_index += 1