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

import json
import os
import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt

import scipy.stats as stats

import metrics as cm

import csv
import pandas as pd

def plot_results(results, title="", xlabel="Timestep", ylabel="Correlation", output_path="./unknown.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(results, marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()  


if __name__ == "__main__":
    MAX_TOP_PRECISION = 7
    columns = ["filename", "N", "time limit", "internal contacts", "window (hours)", "tau", "p_value_tau", "spearman", "p_value_sp"]
    for i in range(1, MAX_TOP_PRECISION + 1):
        columns.append(f"top_{i}_precision")
    for i in range(1, MAX_TOP_PRECISION + 1):
        columns.append(f"top_{i}_precision_moving_avg")
    columns.append("mrr")
    df = pd.DataFrame(columns=columns)

    PRIORS_SUFFIXES = ["0.25,0.25,0.75", "0.5,0.5,0.5", "0.75,0.75,0.25"]

    granularity = 0.1 # 6 minutes

    WINDOWS = [1] # HOURS
    WINDOWS = [int(w / granularity) for w in WINDOWS] # convert to 6-minutes


    datasets = ["D0", "D1", "D2", "D2+15", "D2-15", "D2+25", "D2-25", "D3"]
    for dataset in datasets:
        if not os.path.exists(f"{dataset}"):
            continue
        print(f"Processing dataset {dataset}")
        OUTPUT_PLOTS_DIR = os.path.join(dataset, f"plots")
        os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)
        gt = os.path.join(f"{dataset}", [f for f in os.listdir(f"{dataset}") if f.endswith("10000_reps.json")][0])
        print(f"Using ground truth file: {gt}")
        with open(gt, 'r') as f:
            data = json.load(f)
            gt_data = data
        ana_files = [f for f in os.listdir(f"{dataset}") if "tracks" in f and f.endswith(".json")]
        for ana_file in ana_files:
            file_path = os.path.join(f"{dataset}", f"{ana_file}")
            with open(file_path, 'r') as f:
                ana_data = json.load(f)
            current_output_plot_dir = os.path.join(OUTPUT_PLOTS_DIR, f"{dataset}_{ana_file.replace('.json','')}")
            os.makedirs(current_output_plot_dir, exist_ok=True)
            taus, _ = cm.compute_kendalls_tau_correlation_per_timestep(gt_data, ana_data)
            # Plot the results
            plot_results(taus, title="Kendall's Tau Correlation", xlabel="Timestep", ylabel="Correlation", output_path=os.path.join(current_output_plot_dir, "kendalls_tau_correlation.png"))
            taus_moving_avg, _ = cm.compute_kendalls_tau_correlation_moving_window(gt_data, ana_data, window_size=WINDOWS[0])
            # Plot the results
            plot_results(taus_moving_avg, title="Kendall's Tau Correlation Moving Average", xlabel="Timestep", ylabel="Correlation", output_path=os.path.join(current_output_plot_dir, "kendalls_tau_correlation_moving_avg.png"))
            sp_corrs, _ = cm.compute_spearmans_correlation_per_timestep(gt_data, ana_data)
            # Plot the results
            plot_results(sp_corrs, title="Spearman's Correlation", xlabel="Timestep", ylabel="Correlation", output_path=os.path.join(current_output_plot_dir, "spearmans_correlation.png"))
            sp_corrs_moving_avg, _ = cm.compute_spearman_correlation_moving_window(gt_data, ana_data, window_size=WINDOWS[0])
            # Plot the results
            plot_results(sp_corrs_moving_avg, title="Spearman's Correlation Moving Average", xlabel="Timestep", ylabel="Correlation", output_path=os.path.join(current_output_plot_dir, "spearmans_correlation_moving_avg.png"))
            top_n_avg_precisions = {}
            for i in range(1, MAX_TOP_PRECISION + 1):
                if i > len(gt_data):
                    break
                precisions = cm.compute_top_n_precision(gt_data, ana_data, i)
                top_n_avg_precisions[i] = precisions
                # Plot the results
                plot_results(precisions, title=f"Top {i} Precision", xlabel="Timestep", ylabel="Precision", output_path=os.path.join(current_output_plot_dir, f"top_{i}_precision.png"))
            for i in range(1, MAX_TOP_PRECISION + 1):
                if i > len(gt_data):
                    break
                precisions = cm.compute_top_n_precision_on_a_moving_window(gt_data, ana_data, i, window_size=WINDOWS[0])
                # Plot the results
                plot_results(precisions, title=f"Top {i} Precision Moving Average", xlabel="Timestep", ylabel="Precision", output_path=os.path.join(current_output_plot_dir, f"top_{i}_precision_moving_avg.png"))
            
            (tau, p_value_tau) = cm.compute_kendalls_tau_correlation(gt_data, ana_data)
            (sp_corr, p_value_sp) = cm.compute_spearmans_correlation(gt_data, ana_data)
            top_n_avg_precisions = []
            for i in range(1, MAX_TOP_PRECISION + 1):
                if i > len(gt_data):
                    top_n_avg_precisions.append(pd.NA)
                    continue
                precisions = cm.compute_top_n_precision(gt_data, ana_data, i)
                top_n_avg_precisions.append(np.mean(precisions))
            top_n_avg_precisions_moving_avg = []
            for i in range(1, MAX_TOP_PRECISION + 1):
                if i > len(gt_data):
                    top_n_avg_precisions_moving_avg.append(pd.NA)
                    continue
                precisions = cm.compute_top_n_precision_on_a_moving_window(gt_data, ana_data, i, window_size=WINDOWS[0])
                top_n_avg_precisions_moving_avg.append(np.mean(precisions))
            mrr = cm.compute_mrr(gt_data, ana_data)
            # Append the results to the DataFrame
            df = pd.concat([df, pd.DataFrame([[ana_file, len(gt_data), 84, pd.NA, 1, tau, p_value_tau, sp_corr, p_value_sp] + top_n_avg_precisions + top_n_avg_precisions_moving_avg + [mrr]], columns=columns)], ignore_index=True)
        sim_files = [f for f in os.listdir(f"{dataset}") if "simulated" in f and f.endswith(".json") and "tracks" not in f and "10000_reps" not in f and "reps" in f]
        for sim_file in sim_files:
            file_path = os.path.join(f"{dataset}", f"{sim_file}")
            with open(file_path, 'r') as f:
                ana_data = json.load(f)
            current_output_plot_dir = os.path.join(OUTPUT_PLOTS_DIR, f"{dataset}_{sim_file.replace('.json','')}")
            os.makedirs(current_output_plot_dir, exist_ok=True)
            taus, _ = cm.compute_kendalls_tau_correlation_per_timestep(gt_data, ana_data)
            # Plot the results
            plot_results(taus, title="Kendall's Tau Correlation", xlabel="Timestep", ylabel="Correlation", output_path=os.path.join(current_output_plot_dir, "kendalls_tau_correlation.png"))
            taus_moving_avg, _ = cm.compute_kendalls_tau_correlation_moving_window(gt_data, ana_data, window_size=WINDOWS[0])
            # Plot the results
            plot_results(taus_moving_avg, title="Kendall's Tau Correlation Moving Average", xlabel="Timestep", ylabel="Correlation", output_path=os.path.join(current_output_plot_dir, "kendalls_tau_correlation_moving_avg.png"))
            sp_corrs, _ = cm.compute_spearmans_correlation_per_timestep(gt_data, ana_data)
            # Plot the results
            plot_results(sp_corrs, title="Spearman's Correlation", xlabel="Timestep", ylabel="Correlation", output_path=os.path.join(current_output_plot_dir, "spearmans_correlation.png"))
            sp_corrs_moving_avg, _ = cm.compute_spearman_correlation_moving_window(gt_data, ana_data, window_size=WINDOWS[0])
            # Plot the results
            plot_results(sp_corrs_moving_avg, title="Spearman's Correlation Moving Average", xlabel="Timestep", ylabel="Correlation", output_path=os.path.join(current_output_plot_dir, "spearmans_correlation_moving_avg.png"))
            top_n_avg_precisions = {} # {1: [], 2: [], 3
            for i in range(1, MAX_TOP_PRECISION + 1):
                if i > len(gt_data):
                    break
                precisions = cm.compute_top_n_precision(gt_data, ana_data, i)
                top_n_avg_precisions[i] = precisions
                # Plot the results
                plot_results(precisions, title=f"Top {i} Precision", xlabel="Timestep", ylabel="Precision", output_path=os.path.join(current_output_plot_dir, f"top_{i}_precision.png"))
            for i in range(1, MAX_TOP_PRECISION + 1):
                if i > len(gt_data):
                    break
                precisions = cm.compute_top_n_precision_on_a_moving_window(gt_data, ana_data, i, window_size=WINDOWS[0])
                # Plot the results
                plot_results(precisions, title=f"Top {i} Precision Moving Average", xlabel="Timestep", ylabel="Precision", output_path=os.path.join(current_output_plot_dir, f"top_{i}_precision_moving_avg.png"))
            (tau, p_value_tau) = cm.compute_kendalls_tau_correlation(gt_data, ana_data)
            (sp_corr, p_value_sp) = cm.compute_spearmans_correlation(gt_data, ana_data)
            top_n_avg_precisions = []
            for i in range(1, MAX_TOP_PRECISION + 1):
                if i > len(gt_data):
                    top_n_avg_precisions.append(pd.NA)
                    continue
                precisions = cm.compute_top_n_precision(gt_data, ana_data, i)
                top_n_avg_precisions.append(np.mean(precisions))
            top_n_avg_precisions_moving_avg = []
            for i in range(1, MAX_TOP_PRECISION + 1):
                if i > len(gt_data):
                    top_n_avg_precisions_moving_avg.append(pd.NA)
                    continue
                precisions = cm.compute_top_n_precision_on_a_moving_window(gt_data, ana_data, i, window_size=WINDOWS[0])
                top_n_avg_precisions_moving_avg.append(np.mean(precisions))
            mrr = cm.compute_mrr(gt_data, ana_data)
            # Append the results to the DataFrame
            df = pd.concat([df, pd.DataFrame([[sim_file, len(gt_data), 84, pd.NA, 1, tau, p_value_tau, sp_corr, p_value_sp] + top_n_avg_precisions + top_n_avg_precisions_moving_avg + [mrr]], columns=columns)], ignore_index=True)
            
    # Save the DataFrame to a CSV file
    output_file = f"{OUTPUT_PLOTS_DIR}/metrics_results.csv"
    df.to_csv(output_file, index=False)
    print(f"Metrics saved to {output_file}")
