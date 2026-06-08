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

import scipy.stats
import numpy as np

def compute_kendalls_tau_correlation(simulated_tracks, numerical_tracks):
    """
    Compute the overall correlation between simulated_tracks and numerical_tracks using Kendall's tau.
    Compares all probability values directly.

    Returns:
    - (tau, p value)
    """
    # Flatten all values for direct comparison
    simulated_values = [prob for probs in simulated_tracks.values() for prob in probs]
    numerical_values = [prob for probs in numerical_tracks.values() for prob in probs]
    
    kendall_tau = scipy.stats.kendalltau(simulated_values, numerical_values)
    return kendall_tau.correlation, kendall_tau.pvalue

def compute_kendalls_tau_correlation_per_timestep(simulated_tracks, numerical_tracks):
    """
    Compute the Kendall's tau correlation between simulated_tracks and numerical_tracks
    for each timestep separately (comparing subject rankings at each timestep).

    Returns:
    - (list of tau values, list of p values)
    """
    simulated_subjects = list(simulated_tracks.keys())
    numerical_subjects = list(numerical_tracks.keys())
    timesteps = len(next(iter(simulated_tracks.values())))
    
    taus = []
    p_values = []
    
    for t in range(timesteps):
        simulated_at_t = [simulated_tracks[subj][t] for subj in simulated_subjects]
        numerical_at_t = [numerical_tracks[subj][t] for subj in numerical_subjects]
        
        kendall_tau = scipy.stats.kendalltau(simulated_at_t, numerical_at_t)
        taus.append(kendall_tau.correlation)
        p_values.append(kendall_tau.pvalue)
    
    return taus, p_values

def compute_kendalls_tau_correlation_moving_window(simulated_tracks, numerical_tracks, window_size):
    """
    Compute the Kendall's tau correlation between simulated_tracks and numerical_tracks 
    using a moving window approach. For each window, compute average values and then correlate.
    
    Args:
        simulated_tracks: dict with subjects as keys, lists of probabilities as values
        numerical_tracks: dict with subjects as keys, lists of probabilities as values
        window_size: int, size of the moving window
    
    Returns:
        tuple: (list of tau values, list of p-values) for each window position
    """
    # Verify inputs
    assert len(simulated_tracks) == len(numerical_tracks), "Dictionaries must have same length"
    
    simulated_subjects = list(simulated_tracks.keys())
    numerical_subjects = list(numerical_tracks.keys())
    n_timesteps = len(next(iter(simulated_tracks.values())))
    
    # Verify window size is valid
    assert window_size > 0, "Window size must be positive"
    assert window_size <= n_timesteps, "Window size cannot be larger than number of timesteps"
    
    taus = []
    p_values = []
    
    # Slide the window across timesteps
    for t in range(window_size - 1, n_timesteps):
        # For each subject, compute the average value over the window
        sim_window_avg = {subj: np.mean([simulated_tracks[subj][i] for i in range(t-window_size+1, t+1)])
                         for subj in simulated_subjects}
        num_window_avg = {subj: np.mean([numerical_tracks[subj][i] for i in range(t-window_size+1, t+1)])
                         for subj in numerical_subjects}
        
        # Get averaged values for correlation
        sim_values = list(sim_window_avg.values())
        num_values = list(num_window_avg.values())
        
        # Compute Kendall's tau correlation
        tau, p_value = scipy.stats.kendalltau(sim_values, num_values)
        
        taus.append(tau)
        p_values.append(p_value)
    
    return taus, p_values

def compute_spearmans_correlation(simulated_tracks, numerical_tracks):
    """
    Compute the overall correlation between simulated_tracks and numerical_tracks using Spearman's correlation.
    Compares all probability values directly.

    Returns:
    - (correlation, p value)
    """
    # Flatten all values for direct comparison
    simulated_values = [prob for probs in simulated_tracks.values() for prob in probs]
    numerical_values = [prob for probs in numerical_tracks.values() for prob in probs]
    
    spearman = scipy.stats.spearmanr(simulated_values, numerical_values)
    return spearman.correlation, spearman.pvalue

def compute_spearmans_correlation_per_timestep(simulated_tracks, numerical_tracks):
    """
    Compute the Spearman correlation between simulated_tracks and numerical_tracks
    for each timestep separately (comparing subject rankings at each timestep).

    Returns:
    - (list of correlation values, list of p values)
    """
    simulated_subjects = list(simulated_tracks.keys())
    numerical_subjects = list(numerical_tracks.keys())
    timesteps = len(next(iter(simulated_tracks.values())))
    
    correlations = []
    p_values = []
    
    for t in range(timesteps):
        simulated_at_t = [simulated_tracks[subj][t] for subj in simulated_subjects]
        numerical_at_t = [numerical_tracks[subj][t] for subj in numerical_subjects]
        
        spearman = scipy.stats.spearmanr(simulated_at_t, numerical_at_t)
        correlations.append(spearman.correlation)
        p_values.append(spearman.pvalue)
    
    return correlations, p_values

def compute_spearman_correlation_moving_window(simulated_tracks, numerical_tracks, window_size):
    """
    Compute the Spearman's rank correlation between simulated_tracks and numerical_tracks 
    using a moving window approach. For each window, compute average values and then correlate.
    
    Args:
        simulated_tracks: dict with subjects as keys, lists of probabilities as values
        numerical_tracks: dict with subjects as keys, lists of probabilities as values
        window_size: int, size of the moving window
    
    Returns:
        tuple: (list of correlation values, list of p-values) for each window position
    """
    # Verify inputs
    assert len(simulated_tracks) == len(numerical_tracks), "Dictionaries must have same length"
    
    simulated_subjects = list(simulated_tracks.keys())
    numerical_subjects = list(numerical_tracks.keys())
    n_timesteps = len(next(iter(simulated_tracks.values())))
    
    # Verify window size is valid
    assert window_size > 0, "Window size must be positive"
    assert window_size <= n_timesteps, "Window size cannot be larger than number of timesteps"
    
    correlations = []
    p_values = []
    
    # Slide the window across timesteps
    for t in range(window_size - 1, n_timesteps):
        # For each subject, compute the average value over the window
        sim_window_avg = {subj: np.mean([simulated_tracks[subj][i] for i in range(t-window_size+1, t+1)])
                         for subj in simulated_subjects}
        num_window_avg = {subj: np.mean([numerical_tracks[subj][i] for i in range(t-window_size+1, t+1)])
                         for subj in numerical_subjects}
        
        # Get averaged values for correlation
        sim_values = list(sim_window_avg.values())
        num_values = list(num_window_avg.values())
        
        # Compute Spearman's rank correlation
        correlation, p_value = scipy.stats.spearmanr(sim_values, num_values)
        
        correlations.append(correlation)
        p_values.append(p_value)
    
    return correlations, p_values

def top_n_precision(rank1, rank2, n):
    """
    Compute top-n precision between two rankings.

    *Given the predicted ranking, how many subjects in the top-n are relevant in the ground truth?*
    
    Parameters:
        rank1 (list): The first ranking (e.g., simulated ranking).
        rank2 (list): The second ranking (e.g., analyzed ranking).
        n (int): The top-n cutoff.
        
    Returns:
        precision (float): The fraction of subjects in the top-n of rank1 that also appear in the top-n of rank2.
                        This is computed as the size of the intersection divided by n.
    """
    # Handle edge case where n exceeds available subjects
    n = min(n, len(rank1), len(rank2))
    
    if n == 0:
        return 0.0
    
    # Get top n elements from each ranking
    top_n_rank1 = set(rank1[:n])
    top_n_rank2 = set(rank2[:n])
    
    # Count common elements
    common = top_n_rank1.intersection(top_n_rank2)
    precision = len(common) / n
    return precision

def compute_top_n_precision(simulated_tracks, numerical_tracks, top_n=1):
    """
    Computes the top-n precision between the rankings derived from the simulated and analyzed data
    for each timestep.

    Parameters:
        simulated_tracks (dict): Dictionary with subjects as keys and lists of probabilities over timesteps as values.
        numerical_tracks (dict): Dictionary with subjects as keys and lists of probabilities over timesteps as values.
        top_n (int): The top-n cutoff.

    Returns:
        A list of top-n precision values for each timestep.
    """
    simulated_subjects = list(simulated_tracks.keys())
    numerical_subjects = list(numerical_tracks.keys())
    example_key = next(iter(simulated_tracks))
    timesteps = len(simulated_tracks[example_key])
    precisions = []

    for t in range(timesteps):
        # Build the ranking for simulated data at timestep t (higher probability first).
        rank_sim = sorted(simulated_subjects, key=lambda subj: simulated_tracks[subj][t], reverse=True)
        # Build the ranking for analyzed data at timestep t.
        rank_ana = sorted(numerical_subjects, key=lambda subj: numerical_tracks[subj][t], reverse=True)
        
        precision = top_n_precision(rank_sim, rank_ana, top_n)
        precisions.append(precision)

    return precisions

def compute_mrr(simulated_tracks, numerical_tracks, top_n=None):
    """
    Compute the Mean Reciprocal Rank (MRR) between two rankings.
    For each timestep, finds where the top simulated subject ranks in the numerical ranking.

    Parameters:
        simulated_tracks (dict): Dictionary with subjects as keys and lists of probabilities over timesteps as values.
        numerical_tracks (dict): Dictionary with subjects as keys and lists of probabilities over timesteps as values.
        top_n (int, optional): If specified, only consider reciprocal rank if within top_n, otherwise 0.

    Returns:
        mrr (float): The mean reciprocal rank.
    """
    simulated_subjects = list(simulated_tracks.keys())
    numerical_subjects = list(numerical_tracks.keys())
    example_key = next(iter(simulated_tracks))
    timesteps = len(simulated_tracks[example_key])
    rrs = []
    
    for t in range(timesteps):
        # Build the ranking for simulated data at timestep t (higher probability first).
        rank_sim = sorted(simulated_subjects, key=lambda subj: simulated_tracks[subj][t], reverse=True)
        # Build the ranking for analyzed data at timestep t.
        rank_ana = sorted(numerical_subjects, key=lambda subj: numerical_tracks[subj][t], reverse=True)
        
        # Find the rank of the top simulated subject in the numerical rankings
        query = rank_sim[0]  # Top subject from simulated
        try:
            rank = rank_ana.index(query) + 1  # 1-indexed rank
            if top_n is None or rank <= top_n:
                rrs.append(1 / rank)
            else:
                rrs.append(0)
        except ValueError:
            # Subject not found in numerical ranking
            rrs.append(0)
    
    return np.mean(rrs) if rrs else 0.0
    
def compute_top_n_precision_on_a_moving_window(simulated_tracks, numerical_tracks, top_n=1, window_size=1):
    """
    Computes the top-n precision between the rankings derived from the simulated and analyzed data
    for each timestep, using a mobile window approach.

    Parameters:
        simulated_tracks (dict): Dictionary with subjects as keys and lists of probabilities over timesteps as values.
        numerical_tracks (dict): Dictionary with subjects as keys and lists of probabilities over timesteps as values.
        top_n (int): The top-n cutoff.
        window_size (int): Size of the sliding window to compute average rankings.

    Returns:
        A list of top-n precision values for each timestep after the initial window.
    """
    simulated_subjects = list(simulated_tracks.keys())
    numerical_subjects = list(numerical_tracks.keys())
    example_key = next(iter(simulated_tracks))
    timesteps = len(simulated_tracks[example_key])
    precisions = []

    # Start from window_size-1 to have enough previous data points for the window
    for t in range(window_size-1, timesteps):
        # For each subject, compute the average value over the window
        sim_window_avg = {subj: np.mean([simulated_tracks[subj][i] for i in range(t-window_size+1, t+1)])
                         for subj in simulated_subjects}
        num_window_avg = {subj: np.mean([numerical_tracks[subj][i] for i in range(t-window_size+1, t+1)])
                         for subj in numerical_subjects}
        
        # Build the ranking based on window averages (higher probability first)
        rank_sim = sorted(sim_window_avg.keys(), key=lambda subj: sim_window_avg[subj], reverse=True)
        rank_ana = sorted(num_window_avg.keys(), key=lambda subj: num_window_avg[subj], reverse=True)
        
        precision = top_n_precision(rank_sim, rank_ana, top_n)
        precisions.append(precision)

    return precisions
