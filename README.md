# CHITA Experiments

This repository contains the CHITA Java analysis code and Python experiment
pipeline used to evaluate infection-risk predictions on synthetic contact
datasets.

CHITA builds a Stochastic Time Petri Net (STPN) model of disease progression and
combines it with observations of contacts, symptoms, and diagnostic tests. The
Python pipeline generates datasets, runs ground-truth simulations, runs the Java
analysis, runs the simulation baseline, and writes comparison metrics and plots.

## Repository Layout

- `src/main/java/com/chita/analysis/`: Java STPN analysis.
- `sweep_pipeline_final.py`: supported end-to-end experiment pipeline.
- `dataset_graph.py`, `scale_free_dataset_graph.py`, `small_world_dataset_graph.py`: dataset generators.
- `run_n_simulations.py`: Python simulation baseline and ground-truth runner used by the sweep.
- `compute_precision_metrics.py`: Brier score, ECE, and reliability outputs.

## Setup

Use Python 3.8+ and JDK 11+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The Java classes are compiled manually because this repository does not include a
Maven or Gradle build file:

```bash
mkdir -p out/production/chita-main-test
javac -cp "lib/*" -d out/production/chita-main-test src/main/java/com/chita/analysis/*.java
```

If `javac` is missing, install a JDK rather than a JRE.

## Running Experiments

The supported reproducibility entrypoint is `sweep_pipeline_final.py`. It writes
outputs under `results/sweep_*`.

These commands are long-running:

```bash
MPLCONFIGDIR=.cache/matplotlib python sweep_pipeline_final.py --dataset bubble
MPLCONFIGDIR=.cache/matplotlib python sweep_pipeline_final.py --dataset scale_free
MPLCONFIGDIR=.cache/matplotlib python sweep_pipeline_final.py --dataset small_world
```

Published dataset targets are:

- `bubble`: 8 subjects, complete graph, `200`, `400`, and `800` internal contacts.
- `scale_free`: 100 subjects, Barabasi-Albert graph, `1250`, `2500`, and `5000` internal contacts.
- `small_world`: 100 subjects, Watts-Strogatz graph, `1800`, `3600`, and `7200` internal contacts.

To reuse an existing run directory:

```bash
MPLCONFIGDIR=.cache/matplotlib python sweep_pipeline_final.py --reuse-run results/sweep_YYYYMMDD-HHMM --dataset scale_free
```

To regenerate only the selected plots from an existing run:

```bash
MPLCONFIGDIR=.cache/matplotlib python sweep_pipeline_final.py --reuse-run results/sweep_YYYYMMDD-HHMM --only-selected-plots --quartile-label q4
```

## License

CHITA is released under the GNU Affero General Public License v3.0. See
`LICENSE.txt`.
