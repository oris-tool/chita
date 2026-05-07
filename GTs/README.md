# GTs folder — structure and file descriptions

This document describes the folder and file layout found under the `GTs` directory used by the project.

- dataset_noise/
  - dataset_400_contact_noise_summary.json
    - Contact-noise summary for `dataset_400` (aggregated noise statistics for contact/noisy-contact experiments).
  - dataset_400_observation_noise_summary.json
    - Observation-noise summary for `dataset_400` (observation/noise statistics for that dataset).
  - dataset_scale_free_2500_contact_noise_summary.json
  - dataset_scale_free_2500_observation_noise_summary.json
  - dataset_small_world_3600_contact_noise_summary.json
  - dataset_small_world_3600_observation_noise_summary.json

- ground_truth/
  - dataset_400/
    - Directory containing ground-truth files for `dataset_400` (e.g. adjacency, node/edge metadata, or original graph sources).
  - dataset_scale_free_2500/
    - Ground-truth files for the scale-free dataset with 2500 nodes.
  - dataset_small_world_3600/
    - Ground-truth files for the small-world dataset with 3600 nodes.

- observed_one_run/
  - dataset_400_contact_noisy_one_run_summary.json
    - Summary of a single observed noisy run for contact data on `dataset_400`.
  - dataset_scale_free_2500_contact_noisy_one_run_summary.json
  - dataset_small_world_3600_contact_noisy_one_run_summary.json

Notes:
- JSON summary files under `dataset_noise` and `observed_one_run` contain precomputed metrics and noise statistics used by analysis and plotting scripts.
- The `ground_truth` subdirectories typically contain the raw or canonical dataset representations used as references when computing precision/recall or other metrics.
- If you add new datasets, follow the existing naming conventions (e.g. `dataset_<name>_contact_noise_summary.json`).

If you want, I can also list the exact contents of each ground-truth subdirectory and the JSON schemas for the summary files.
