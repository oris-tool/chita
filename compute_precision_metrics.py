import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COMPACT_PLOT_DPI = 80
GRID_ROWS = 4
GRID_COLS = 2


def _safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except (OSError, ValueError):
        pass


def calculate_brier_score(p, g):
    """
    Calculate the Brier Score for a single subject.
    """
    p = np.asarray(p)
    g = np.asarray(g)
    brier_score = np.mean((p - g) ** 2)
    return float(brier_score)


def calculate_ece(p, g, M=10):
    """
    Calculate the Expected Calibration Error (ECE) for a single subject.
    """
    p = np.asarray(p)
    g = np.asarray(g)
    T = len(p)

    bins = np.linspace(0, 1, M + 1)
    ece = 0.0

    for m in range(M):
        if m == 0:
            in_bin = (p >= bins[m]) & (p <= bins[m + 1])
        else:
            in_bin = (p > bins[m]) & (p <= bins[m + 1])

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


def chunked(sequence, chunk_size):
    for start in range(0, len(sequence), chunk_size):
        yield sequence[start:start + chunk_size]


def hide_unused_axes(axes, used_count):
    for axis in axes[used_count:]:
        axis.set_visible(False)


def save_metrics_table(results_df, output_path):
    import seaborn as sns

    metrics_table_df = results_df[["Subject", "Brier Score", "ECE"]].copy()
    metrics_table_df["Subject"] = metrics_table_df["Subject"].astype(str)
    metrics_table_df = metrics_table_df.set_index("Subject")

    fig, ax = plt.subplots(figsize=(6, max(3, 0.45 * len(metrics_table_df) + 1.5)))
    sns.heatmap(
        metrics_table_df,
        annot=True,
        fmt=".6f",
        cbar=False,
        linewidths=0.8,
        linecolor="white",
        cmap="Blues",
        ax=ax,
    )
    ax.set_title("Brier Score and ECE by Subject")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Subject")
    fig.tight_layout()
    fig.savefig(output_path, dpi=COMPACT_PLOT_DPI)
    plt.close(fig)


def save_reliability_grids(dict_p, dict_g, subject_ids, plots_dir):
    subjects_per_page = GRID_ROWS * GRID_COLS
    multiple_pages = len(subject_ids) > subjects_per_page

    for page_index, subject_page in enumerate(chunked(subject_ids, subjects_per_page), start=1):
        fig, axes = plt.subplots(
            GRID_ROWS,
            GRID_COLS,
            figsize=(12, 16),
            sharex=True,
            sharey=True,
        )
        flat_axes = list(axes.flat)

        for axis, subject_id in zip(flat_axes, subject_page):
            p = dict_p[subject_id]
            g = dict_g[subject_id]
            axis.scatter(
                p,
                g,
                color="tab:blue",
                alpha=0.45,
                s=12,
                edgecolors="none",
            )
            axis.plot([0, 1], [0, 1], "r--", linewidth=1.0)
            axis.set_title(f"Subject {subject_id}")
            axis.set_xlim([-0.05, 1.05])
            axis.set_ylim([-0.05, 1.05])
            axis.grid(True, alpha=0.3)

        hide_unused_axes(flat_axes, len(subject_page))
        fig.suptitle("Reliability Diagrams", fontsize=14)
        fig.supxlabel("Predicted Probability p(t)")
        fig.supylabel("Ground Truth g(t)")
        fig.tight_layout(rect=(0.03, 0.03, 1.0, 0.96))

        output_name = "all_subjects_reliability_diagrams.png"
        if multiple_pages:
            output_name = f"all_subjects_reliability_diagrams_page_{page_index}.png"
        fig.savefig(os.path.join(plots_dir, output_name), dpi=COMPACT_PLOT_DPI)
        plt.close(fig)


def process_and_save(
    json_path_p,
    json_path_g,
    M=10,
    metrics_output="metrics.json",
    plots_dir="plots",
    save_plots=True,
    verbose=True,
    include_scatter_coordinates=True,
):
    """
    Reads the two JSON files, calculates metrics, optionally saves compact plots,
    and saves all metrics (including scatter coordinates) to a single JSON file.
    """
    os.makedirs(os.path.dirname(metrics_output), exist_ok=True)
    if save_plots:
        os.makedirs(plots_dir, exist_ok=True)

    with open(json_path_p, "r", encoding="utf-8") as fp:
        dict_p = json.load(fp)

    with open(json_path_g, "r", encoding="utf-8") as fg:
        dict_g = json.load(fg)

    results = []
    metrics_dict = {}
    total_brier_score = 0.0
    total_ece = 0.0

    p_subjects = sorted(list(dict_p.keys()), key=int)
    g_subjects = sorted(list(dict_g.keys()), key=int)

    if p_subjects != g_subjects:
        raise ValueError("Error: The JSON files do not contain the same subjects.\n")

    for subject_id in g_subjects:
        p = dict_p[subject_id]
        g = dict_g[subject_id]

        if len(p) != len(g):
            raise ValueError(
                f"Error: Subject {subject_id} has a different T (p:{len(p)}, g:{len(g)}).\n"
            )

        p_array = np.asarray(p)
        g_array = np.asarray(g)
        bs, ece = calculate_metrics(p_array, g_array, M=M)
        total_brier_score += bs
        total_ece += ece
        results.append(
            {
                "Subject": subject_id,
                "Brier Score": bs,
                "ECE": ece,
            }
        )

        subject_metrics = {
            "Brier Score": bs,
            "ECE": ece,
        }
        if include_scatter_coordinates:
            subject_metrics["scatter_coordinates"] = [
                [float(p_val), float(g_val)]
                for p_val, g_val in zip(p, g)
            ]
        metrics_dict[subject_id] = subject_metrics

    subject_count = len(results)
    mean_brier_score = float(total_brier_score / subject_count) if subject_count else float("nan")
    mean_ece = float(total_ece / subject_count) if subject_count else float("nan")

    results_df = None
    if save_plots or verbose:
        import pandas as pd

        results_df = pd.DataFrame(results)
        results_df["Subject_num"] = pd.to_numeric(results_df["Subject"])
        results_df = results_df.sort_values(by="Subject_num").reset_index(drop=True)

    if save_plots:
        save_metrics_table(
            results_df,
            os.path.join(plots_dir, "all_subjects_metrics_table.png"),
        )
        save_reliability_grids(dict_p, dict_g, g_subjects, plots_dir)

    if verbose:
        _safe_print("Coefficients Table per Subject:")
        _safe_print("-" * 45)
        _safe_print(results_df[["Subject", "Brier Score", "ECE"]].to_string(index=False))
        _safe_print("-" * 45)

    with open(metrics_output, "w", encoding="utf-8") as fm:
        json.dump(metrics_dict, fm, indent=4)

    if verbose:
        _safe_print(f"\nAll metrics and coordinates saved successfully to: {metrics_output}")
        if save_plots:
            _safe_print(f"Compact plot set saved successfully in the directory: ./{plots_dir}/")
        else:
            _safe_print("Plot image generation was skipped.")

    return {
        "metrics_output": metrics_output,
        "plots_dir": plots_dir,
        "plots_generated": bool(save_plots),
        "subject_count": subject_count,
        "mean_brier_score": mean_brier_score,
        "mean_ece": mean_ece,
    }


if __name__ == "__main__":
    for experiment_folder in [f for f in os.listdir(".") if os.path.isdir(f) and f.startswith("D")]:
        _safe_print(f"Processing experiment folder: {experiment_folder}")

        content = os.listdir(experiment_folder)
        baseline_file_path = os.path.basename(
            [f for f in content if f.startswith("dataset") and f.endswith("_10_reps.json")][0]
        )
        baseline_file_path = os.path.join(experiment_folder, baseline_file_path)

        prediction_file_path = os.path.basename(
            [f for f in content if f.startswith("dataset") and "_0.5,0.5,0.5_tracks_" in f and f.endswith(".json")][0]
        )
        prediction_file_path = os.path.join(experiment_folder, prediction_file_path)

        ground_truth_file_path = os.path.basename(
            [f for f in content if f.startswith("dataset") and f.endswith("_reps.json") and "_10_" not in f][0]
        )
        ground_truth_file_path = os.path.join(experiment_folder, ground_truth_file_path)

        _safe_print(f"  Baseline file: {baseline_file_path}")
        _safe_print(f"  Prediction file: {prediction_file_path}")
        _safe_print(f"  Ground Truth file: {ground_truth_file_path}")

        results_dir = f"{experiment_folder}/precision_metrics"
        process_and_save(
            prediction_file_path,
            ground_truth_file_path,
            M=10,
            metrics_output=os.path.join(results_dir, "metrics_prediction.json"),
            plots_dir=os.path.join(results_dir, "numericalAnalysis", "plots"),
        )
        process_and_save(
            baseline_file_path,
            ground_truth_file_path,
            M=10,
            metrics_output=os.path.join(results_dir, "metrics_baseline.json"),
            plots_dir=os.path.join(results_dir, "simulatedBaseline", "plots"),
        )
