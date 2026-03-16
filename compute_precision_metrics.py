import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_brier_score(p, g):
    """
    Calculate the Brier Score for a single subject.
    """
    p = np.array(p)
    g = np.array(g)
    brier_score = np.mean((p - g) ** 2)
    return float(brier_score)

def calculate_ece(p, g, M=10):
    """
    Calculate the Expected Calibration Error (ECE) for a single subject.
    """
    p = np.array(p)
    g = np.array(g)
    T = len(p)
    
    bins = np.linspace(0, 1, M + 1)
    ece = 0.0
    
    for m in range(M):
        if m == 0:
            in_bin = (p >= bins[m]) & (p <= bins[m+1])
        else:
            in_bin = (p > bins[m]) & (p <= bins[m+1])
            
        B_m = np.sum(in_bin)
        
        if B_m > 0:
            conf = np.mean(p[in_bin])
            acc = np.mean(g[in_bin])
            ece += (B_m / T) * np.abs(acc - conf)
    
    return float(ece)

def calculate_metrics(p, g, M=10):
    brier_score = calculate_brier_score(p, g)
    ece = calculate_ece(p, g, M)
    return float(brier_score), float(ece)

def process_and_save(json_path_p, json_path_g, M=10, metrics_output='metrics.json', plots_dir='plots'):
    """
    Reads the two JSON files, calculates metrics, saves individual plots per subject,
    and saves all metrics (including scatter coordinates) to a single JSON file.
    """
    # Create directory for saving plots if it doesn't exist
    os.makedirs(plots_dir, exist_ok=True)
    # Ensure the parent directory for metrics_output exists too
    os.makedirs(os.path.dirname(metrics_output), exist_ok=True)

    # Load JSON files
    with open(json_path_p, 'r') as fp:
        dict_p = json.load(fp)
        
    with open(json_path_g, 'r') as fg:
        dict_g = json.load(fg)
        
    results = []
    metrics_dict = {}
    
    # Identify present subjects and sort them numerically
    p_subjects = sorted(list(dict_p.keys()), key=int)
    g_subjects = sorted(list(dict_g.keys()), key=int)

    if p_subjects != g_subjects:
        error_msg = "Error: The JSON files do not contain the same subjects.\n"        
        raise ValueError(error_msg)

    # Calculate metrics and plot for each subject
    for subject_id in g_subjects:
        p = dict_p[subject_id]
        g = dict_g[subject_id]
        
        # Verify that p and g have the same number of time steps T
        if len(p) != len(g):
            error_msg = f"Error: Subject {subject_id} has a different T (p:{len(p)}, g:{len(g)}).\n"        
            raise ValueError(error_msg)
            
        bs, ece = calculate_metrics(p, g, M=M)
        
        # Add to results list for the DataFrame printout
        results.append({
            'Subject': subject_id,
            'Brier Score': bs,
            'ECE': ece
        })
        
        # Extract coordinates for the JSON: a list of [x, y] format points
        coordinates = [[float(p_val), float(g_val)] for p_val, g_val in zip(p, g)]
        
        # Add to the dictionary for JSON export
        metrics_dict[subject_id] = {
            'Brier Score': bs,
            'ECE': ece,
            'scatter_coordinates': coordinates
        }
        
        # -----------------------------------------------------
        # Save individual plot for the subject
        # -----------------------------------------------------
        plt.figure(figsize=(6, 6))
        plt.scatter(p, g, color='blue', alpha=0.5, edgecolors='none', label='<p(t), g(t)>')
        
        # 45-degree line
        plt.plot([0, 1], [0, 1], 'r--', label='45° Line (p=g)')
        
        plt.title(f"Subject {subject_id} - Reliability diagram")
        plt.xlabel('Predicted Probability p(t)')
        plt.ylabel('Ground Truth g(t)')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save plot and close figure to free memory
        plot_path = os.path.join(plots_dir, f"subject_{subject_id}_plot.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=150)
        plt.close()

    # Create DataFrame and keep subjects in numerical ascending order for plotting/printing.
    results_df = pd.DataFrame(results)
    results_df['Subject_num'] = pd.to_numeric(results_df['Subject'])
    results_df = results_df.sort_values(by='Subject_num').reset_index(drop=True)

    # Save one combined Brier/ECE plot for all subjects.
    metrics_table_df = results_df[['Subject', 'Brier Score', 'ECE']].copy()
    metrics_table_df['Subject'] = metrics_table_df['Subject'].astype(str)
    metrics_table_df = metrics_table_df.set_index('Subject')

    plt.figure(figsize=(6, max(3, 0.45 * len(metrics_table_df) + 1.5)))
    ax = sns.heatmap(
        metrics_table_df,
        annot=True,
        fmt='.6f',
        cbar=False,
        linewidths=0.8,
        linecolor='white',
        cmap='Blues',
    )
    ax.set_title('Brier Score and ECE by Subject')
    ax.set_xlabel('Metric')
    ax.set_ylabel('Subject')
    combined_metrics_plot_path = os.path.join(plots_dir, 'all_subjects_metrics_table.png')
    plt.tight_layout()
    plt.savefig(combined_metrics_plot_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print("Coefficients Table per Subject:")
    print("-" * 45)
    print(results_df[['Subject', 'Brier Score', 'ECE']].to_string(index=False))
    print("-" * 45)
    
    # Save metrics to the output JSON file
    with open(metrics_output, 'w') as fm:
        json.dump(metrics_dict, fm, indent=4)
        
    print(f"\n✅ All metrics and coordinates saved successfully to: {metrics_output}")
    print(f"✅ All subject plots saved successfully in the directory: ./{plots_dir}/")

# ==========================================
# Execution
# ==========================================
if __name__ == "__main__":
    # gather all data
    for experiment_folder in [f for f in os.listdir(".") if os.path.isdir(f) and f.startswith("D")]:
        print(f"Processing experiment folder: {experiment_folder}")

        content = os.listdir(experiment_folder)
        baseline_file_path = os.path.basename([f for f in content if f.startswith("dataset") and f.endswith("_10_reps.json")][0])
        baseline_file_path = os.path.join(experiment_folder, baseline_file_path)

        prediction_file_path = os.path.basename([f for f in content if f.startswith("dataset") and "_0.5,0.5,0.5_tracks_" in f and f.endswith(".json")][0])
        prediction_file_path = os.path.join(experiment_folder, prediction_file_path)
        
        ground_truth_file_path = os.path.basename([f for f in content if f.startswith("dataset") and f.endswith("_reps.json") and "_10_" not in f][0])
        ground_truth_file_path = os.path.join(experiment_folder, ground_truth_file_path)

        print(f"  Baseline file: {baseline_file_path}")
        print(f"  Prediction file: {prediction_file_path}")
        print(f"  Ground Truth file: {ground_truth_file_path}")

        results_dir = f"{experiment_folder}/precision_metrics"
        # numerical analysis
        process_and_save(
            prediction_file_path, 
            ground_truth_file_path, 
            M=10, 
            metrics_output=os.path.join(results_dir, 'metrics_prediction.json'), 
            plots_dir=os.path.join(results_dir, "numericalAnalysis", 'plots')
        )
        # simulated (baseline)
        process_and_save(
            baseline_file_path, 
            ground_truth_file_path, 
            M=10, 
            metrics_output=os.path.join(results_dir, 'metrics_baseline.json'), 
            plots_dir=os.path.join(results_dir, "simulatedBaseline", 'plots')
        )

    # #results_dir = f"{root_dir}/precision_metrics/s8_t84_84_simulated"
    # results_dir = f"{root_dir}/precision_metrics/s8_t84_84_numericalAnalysys"
    # process_and_save(
    #     prediction_file_path, 
    #     ground_truth_file_path, 
    #     M=10, 
    #     metrics_output=os.path.join(results_dir, 'metrics.json'), 
    #     plots_dir=os.path.join(results_dir, 'plots')
    # )