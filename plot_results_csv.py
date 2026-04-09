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

import pandas as pd
import os
import metrics as cm
import json
import matplotlib.pyplot as plt


def plot_more_results(result_ana, result_sim, title="", xlabel="Timestep", ylabel="Correlation", output_path="./unknown.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(result_ana, marker='o', label='Analysis')
    plt.plot(result_sim, marker='x', label='Simulation')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    MAX_TOP_PRECISION = 7
    columns = ["filename",
               "N",
               "time limit",
               "internal contacts",
               "analysis steps",
               "window (hours)",
               "tau",
               "spearman"]
    for i in range(1, MAX_TOP_PRECISION + 1):
        columns.append(f"top_{i}_precision")
    for i in range(1, MAX_TOP_PRECISION + 1):
        columns.append(f"top_{i}_precision_moving_avg")
    df = pd.DataFrame(columns=columns)
    granularity = 0.1 # 6 minutes
    WINDOW = int(1 / granularity)
    datasets = ["D2", "D2+15", "D2-15", "D2+25", "D2-25", "D3"]
    for dataset in datasets:
        print("Processing dataset:", dataset)
        ana_files = [f for f in sorted(os.listdir(dataset)) if f.endswith(".json") and "tracks" in f and "excluded" not in f]
        sim_files = [f for f in sorted(os.listdir(dataset)) if f.endswith(".json") and  "reps" in f and "10000" not in f and "excluded" not in f]
        ground_truth_files = [f for f in sorted(os.listdir(dataset)) if f.endswith(".json") and "10000" in f][0]
        print(ground_truth_files)
        print(f"Found {len(ana_files)} analysis files, {len(sim_files)} simulation files, and {len(ground_truth_files)} ground truth files.")
        with open(os.path.join(dataset, ground_truth_files), 'r') as f:
            gt_data = json.load(f)
        sim_data = {}
        for sim_file in sim_files:
            with open(os.path.join(dataset, sim_file), 'r') as f:
                sim_data[sim_file] = json.load(f)
        ana_data = {}
        for ana_file in ana_files:
            with open(os.path.join(dataset, ana_file), 'r') as f:
                ana_data[ana_file] = json.load(f)

        if dataset == "D0":
            dataset = "D0_" + ana_file.split("_")[-1].replace(".json", "")
            dataset_folder = "D0"
            os.makedirs(f"{dataset_folder}/D0_{ana_file.split('_')[-1].replace('.json', '')}", exist_ok=True)
        else:
            dataset_folder = dataset
    
        # Compute kendall for ana_files against simulation_files
        taus_analysis = [cm.compute_kendalls_tau_correlation_per_timestep(gt_data, ana_file)[0] for ana_file in ana_data.values()]
        # Compute kendall for sim_files against ground_truth_files
        taus_simulation = [cm.compute_kendalls_tau_correlation_per_timestep(gt_data, sim_file)[0] for sim_file in sim_data.values()]
        for a, s in zip(taus_analysis, taus_simulation):
            # Save the correlation data to CSV
            correlation_data = pd.DataFrame({
                'timestep': range(len(a)),
                'analysis_kendall': a,
                'simulation_kendall': s
            })
            correlation_data.to_csv(f"{dataset_folder}/{dataset}_kendall_correlation_data.csv", index=False)
            plot_more_results(a, s, title=f"{dataset} Analysis vs Simulation", xlabel="Timestep (6-min intervals)", ylabel="Kendall's Tau", output_path=f"{dataset_folder}/{dataset}_analysis_vs_simulation_kendall.png")
        # Compute kendalls for ana_files against sim_files with moving average
        taus_analysis_moving_avg = [cm.compute_kendalls_tau_correlation_moving_window(gt_data, ana_file, window_size=WINDOW)[0] for ana_file in ana_data.values()]
        # Compute kendalls for sim_files against ground_truth_files with moving average
        taus_simulation_moving_avg = [cm.compute_kendalls_tau_correlation_moving_window(gt_data, sim_file, window_size=WINDOW)[0] for sim_file in sim_data.values()]
        for a, s in zip(taus_analysis_moving_avg, taus_simulation_moving_avg):
            # Save the moving average correlation data to CSV
            correlation_data_moving_avg = pd.DataFrame({
                'timestep': range(len(a)),
                'analysis_kendall_moving_avg': a,
                'simulation_kendall_moving_avg': s
            })
            correlation_data_moving_avg.to_csv(f"{dataset_folder}/{dataset}_kendall_correlation_moving_avg_data.csv", index=False)
            plot_more_results(a, s, title=f"{dataset} Analysis vs Simulation (Moving Avg)", xlabel="Timestep (6-min intervals)", ylabel="Kendall's Tau (Moving Avg)", output_path=f"{dataset_folder}/{dataset}_analysis_vs_simulation_kendall_moving_avg.png")
        # Compute spearman for ana_files against simulation_files
        spearmans_analysis = [cm.compute_spearmans_correlation_per_timestep(gt_data, ana_file)[0] for ana_file in ana_data.values()]
        # Compute spearman for sim_files against ground_truth_files
        spearmans_simulation = [cm.compute_spearmans_correlation_per_timestep(gt_data, sim_file)[0] for sim_file in sim_data.values()]
        for a, s in zip(spearmans_analysis, spearmans_simulation):
            # Save the correlation data to CSV
            correlation_data = pd.DataFrame({
                'timestep': range(len(a)),
                'analysis_spearman': a,
                'simulation_spearman': s
            })
            correlation_data.to_csv(f"{dataset_folder}/{dataset}_spearman_correlation_data.csv", index=False)
            plot_more_results(a, s, title=f"{dataset} Analysis vs Simulation", xlabel="Timestep (6-min intervals)", ylabel="Spearman's Rho", output_path=f"{dataset_folder}/{dataset}_analysis_vs_simulation_spearman.png")
        # Compute spearman for ana_files against sim_files with moving average
        spearmans_analysis_moving_avg = [cm.compute_spearman_correlation_moving_window(gt_data, ana_file, window_size=WINDOW)[0] for ana_file in ana_data.values()]
        # Compute spearman for sim_files against ground_truth_files with moving average
        spearmans_simulation_moving_avg = [cm.compute_spearman_correlation_moving_window(gt_data, sim_file, window_size=WINDOW)[0] for sim_file in sim_data.values()]
        for a, s in zip(spearmans_analysis_moving_avg, spearmans_simulation_moving_avg):
            # Save the moving average correlation data to CSV
            correlation_data_moving_avg = pd.DataFrame({
                'timestep': range(len(a)),
                'analysis_spearman_moving_avg': a,
                'simulation_spearman_moving_avg': s
            })
            correlation_data_moving_avg.to_csv(f"{dataset_folder}/{dataset}_spearman_correlation_moving_avg_data.csv", index=False)
            plot_more_results(a, s, title=f"{dataset} Analysis vs Simulation (Moving Avg)", xlabel="Timestep (6-min intervals)", ylabel="Spearman's Rho (Moving Avg)", output_path=f"{dataset_folder}/{dataset}_analysis_vs_simulation_spearman_moving_avg.png")
        # Compute top-k precision for ana_files against simulation_files
        max_k = min(MAX_TOP_PRECISION, min([len(ana_file) for ana_file in ana_data.values()]), min([len(sim_file) for sim_file in sim_data.values()]))
        topk_precisions_analysis_all = []
        topk_precisions_simulation_all = []
        topk_precisions_analysis_moving_avg_all = []
        topk_precisions_simulation_moving_avg_all = []
        for k in range(1, max_k + 1):
            topk_precisions_analysis = [cm.compute_top_n_precision(gt_data, ana_file, top_n=k) for ana_file in ana_data.values()]
            topk_precisions_simulation = [cm.compute_top_n_precision(gt_data, sim_file, top_n=k) for sim_file in sim_data.values()]
            for a, s in zip(topk_precisions_analysis, topk_precisions_simulation):
                # Save the top-k precision data to CSV
                topk_precision_data = pd.DataFrame({
                    'timestep': range(len(a)),
                    f'analysis_top_{k}_precision': a,
                    f'simulation_top_{k}_precision': s
                })
                topk_precision_data.to_csv(f"{dataset_folder}/{dataset}_top_{k}_precision_data.csv", index=False)
                plot_more_results(a, s, title=f"{dataset} Analysis vs Simulation Top-{k} Precision", xlabel="Timestep (6-min intervals)", ylabel=f"Top-{k} Precision", output_path=f"{dataset_folder}/{dataset}_analysis_vs_simulation_top_{k}_precision.png")
            # Compute top-k precision for ana_files against sim_files with moving average
            topk_precisions_analysis_moving_avg = [cm.compute_top_n_precision_on_a_moving_window(gt_data, ana_file, top_n=k, window_size=WINDOW) for ana_file in ana_data.values()]
            topk_precisions_simulation_moving_avg = [cm.compute_top_n_precision_on_a_moving_window(gt_data, sim_file, top_n=k, window_size=WINDOW) for sim_file in sim_data.values()]
            for a, s in zip(topk_precisions_analysis_moving_avg, topk_precisions_simulation_moving_avg):
                # Save the moving average top-k precision data to CSV
                topk_precision_moving_avg_data = pd.DataFrame({
                    'timestep': range(len(a)),
                    f'analysis_top_{k}_precision_moving_avg': a,
                    f'simulation_top_{k}_precision_moving_avg': s
                })
                topk_precision_moving_avg_data.to_csv(f"{dataset_folder}/{dataset}_top_{k}_precision_moving_avg_data.csv", index=False)
                plot_more_results(a, s, title=f"{dataset} Analysis vs Simulation Top-{k} Precision (Moving Avg)", xlabel="Timestep (6-min intervals)", ylabel=f"Top-{k} Precision (Moving Avg)", output_path=f"{dataset_folder}/{dataset}_analysis_vs_simulation_top_{k}_precision_moving_avg.png")
            topk_precisions_analysis_all.append(topk_precisions_analysis)
            topk_precisions_simulation_all.append(topk_precisions_simulation)
            topk_precisions_analysis_moving_avg_all.append(topk_precisions_analysis_moving_avg)
            topk_precisions_simulation_moving_avg_all.append(topk_precisions_simulation_moving_avg)
