import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
        
    # Create DataFrame and sort by Brier Score
    results_df = pd.DataFrame(results).sort_values(by='Brier Score').reset_index(drop=True)
    
    print("Coefficients Table per Subject:")
    print("-" * 45)
    print(results_df.to_string(index=False))
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
    root_dir = "jsons"
    #prediction_file_path = f"{root_dir}/dataset_s8_t84_84_simulated_10_reps.json"
    prediction_file_path = f"{root_dir}/dataset_s8_t84_84_simulated_0.5,0.5,0.5_tracks_it3.json"
    ground_truth_file_path = f"{root_dir}/dataset_s8_t84_84_simulated_10000_reps.json"

    #results_dir = "precision_metrics/s8_t84_84_simulated"
    results_dir = "precision_metrics/s8_t84_84_numericalAnalysys"
    process_and_save(
        prediction_file_path, 
        ground_truth_file_path, 
        M=10, 
        metrics_output=os.path.join(results_dir, 'metrics.json'), 
        plots_dir=os.path.join(results_dir, 'plots')
    )