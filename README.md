# CHITA Library

This repository provides CHITA, a Java library implementing a quantitative approach to predict the spread of infectious diseases within a cluster. CHITA is presented in a paper titled "An observation-based quantitative approach to predict the spread of infectious diseases within a cluster", authored by Laura Carnevali, Silvia Dani, Niccolò Niccoli, Benedetta Picano, and Enrico Vicario, currently submitted for a journal publication.

The most distinctive features of CHITA are:
- definition of a custom-made extensible metamodel of an infection chain of a disease;
- automated translation of a metamodel instance into a Stochastic Time Petri Net (STPN) characterizing the disease evolution from contact to infectiousness;
- implementation of an efficient quantitative approach to predict the spread of infectious diseases within a cluster, exploiting not only the STPN model of disease evolution in an individual but also observations of contacts, symptoms, and results of diagnostic tests;
- randomly generated data sets of observations of contacts, symptoms, and results of diagnostic tests.

This repository contains the CHITA Java analysis code and Python experiment pipeline used to evaluate infection-risk predictions on synthetic contact datasets. The Python pipeline generates datasets, runs ground-truth simulations, runs the Java analysis, runs the simulation baseline, and writes comparison metrics and plots.

The main components of this repository are:
- `src/main/java/com/chita/analysis/`: Java STPN analysis;
- `sweep_pipeline_final.py`: supported end-to-end experiment pipeline;
- `dataset_graph.py`, `scale_free_dataset_graph.py`, `small_world_dataset_graph.py`: dataset generators;
- `run_n_simulations.py`: Python simulation baseline and ground-truth runner used by the sweep;
- `compute_precision_metrics.py`: Brier score, ECE, and reliability outputs.

## Experimental reproducibility

To support reproducibility of the experimental results, use `sweep_pipeline_final.py`. It writes outputs under `results/sweep_*`.

1. Run the supported reproducibility pipeline for each dataset.

   These commands are long-running:

   ```bash
   MPLCONFIGDIR=.cache/matplotlib python sweep_pipeline_final.py --dataset bubble
   MPLCONFIGDIR=.cache/matplotlib python sweep_pipeline_final.py --dataset scale_free
   MPLCONFIGDIR=.cache/matplotlib python sweep_pipeline_final.py --dataset small_world
   ```

   - `bubble`: 8 subjects, complete graph, `200`, `400`, and `800` internal contacts;
   - `scale_free`: 100 subjects, Barabasi-Albert graph, `1250`, `2500`, and `5000` internal contacts;
   - `small_world`: 100 subjects, Watts-Strogatz graph, `1800`, `3600`, and `7200` internal contacts.


2. To generate again only the selected plots from an existing run: the best 10 runs, the worst 10 runs, and the 10 runs closest to the median for each correlation metric.

   ```bash
   MPLCONFIGDIR=.cache/matplotlib python sweep_pipeline_final.py --reuse-run results/sweep_YYYYMMDD-HHMM --only-selected-plots --quartile-label q4
   ```

## Installation

Use Python 3.8+ and JDK 11+.

1. **Install Python dependencies.**

   macOS / Linux:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

   Windows PowerShell:

   ```powershell
   python -m venv .venv
   . .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **Compile the Java analysis classes.**

   The Java classes are compiled manually because this repository does not include a Maven or Gradle build file.

   ```bash
   mkdir -p out/production/chita-main-test
   javac -cp "lib/*" -d out/production/chita-main-test src/main/java/com/chita/analysis/*.java
   ```

   If `javac` is missing, install a JDK rather than a JRE.

## Licence

CHITA is released under the [GNU Affero General Public License v3.0](https://choosealicense.com/licenses/agpl-3.0).
