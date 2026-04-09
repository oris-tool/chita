# Using `dataset_graph.py`

`dataset_graph.py` generates a synthetic CHITA dataset by simulating an outbreak on a contact graph and converting the simulated history into JSON events.

The generated JSON has the same top-level structure used by the rest of the project:

```json
{
  "events": [],
  "n_subjects": 8,
  "time_limit": 84,
  "n_contacts": 400
}
```

Event times are stored in hours. The top-level `time_limit` is stored in days.

## Requirements

Install the Python dependencies first:

```bash
pip install -r requirements.txt
```

The relevant packages for this file are:

- `networkx`, used to build the contact graph
- `EoN`, used to run the SIR epidemic simulation
- `matplotlib`, used only by the optional plotting function and the direct script demo

If `EoN` is missing, `simulate_external_introduction(...)` raises an import error with an installation hint.

## Quick Run

Run the file directly:

```bash
python dataset_graph.py
```

This uses the demo block at the bottom of the file. It calls:

```python
simulate_external_introduction(seed=7, total_internal_contacts=400)
```

Then it writes:

```text
dataset_sample_output.json
```

It also opens a plot with the contact network and the S/I/R epidemic curves. If you only want a dataset file and do not want a plot window, use the import-based workflow below.

## Recommended Usage From Python

Create a dataset from another Python script or notebook:

```python
from dataset_graph import simulate_external_introduction, save_dataset_event_sequence

result = simulate_external_introduction(
    n_nodes=8,
    transmission_rate=0.6,
    recovery_rate=0.2,
    max_intro_time=48.0,
    tmax_after_intro=84 * 24,
    max_external_contacts_per_node=8,
    total_internal_contacts=400,
    seed=7,
)

payload = save_dataset_event_sequence(result, "dataset_graph_output.json")
print(payload["n_subjects"], payload["time_limit"], payload["n_contacts"])
```

This creates a dataset with:

- `n_nodes` subjects
- an 84-day horizon if `tmax_after_intro=84 * 24`
- at least `400` internal contacts if `total_internal_contacts=400`
- deterministic output for the same `seed`

## Main Parameters

`simulate_external_introduction(...)` controls the simulation and dataset source:

- `n_nodes`: number of subjects in the contact graph.
- `transmission_rate`: EoN SIR transmission rate, passed as `tau`.
- `recovery_rate`: EoN SIR recovery rate, passed as `gamma`.
- `max_intro_time`: latest possible time, in hours, for the external introduction of the first infected subject.
- `tmax_after_intro`: simulation horizon in hours.
- `max_external_contacts_per_node`: maximum number of random external contacts sampled for each subject.
- `total_internal_contacts`: optional target for the number of internal contacts in the final dataset.
- `seed`: random seed for reproducible generation.

There is also an `edge_probability` parameter, but it is not active in the current code. The current graph is a complete graph:

```python
graph = nx.complete_graph(n_nodes)
```

The older Erdos-Renyi graph line is present but commented out.

## What EoN Does

EoN is used to simulate the hidden epidemic:

```python
simulation = EoN.fast_SIR(
    graph,
    tau=transmission_rate,
    gamma=recovery_rate,
    initial_infecteds=[infected_node],
    tmax=tmax_after_intro,
    return_full_data=True,
)
```

EoN receives the graph, transmission rate, recovery rate, one initially infected node, and a time horizon. It returns a simulation object.

The code uses that object in two ways:

- `simulation.t()`, `simulation.S()`, `simulation.I()`, and `simulation.R()` provide the S/I/R curves.
- `simulation.transmissions()` provides the person-to-person transmissions that are converted into `Internal` events.

EoN uses zero-based graph node IDs. The dataset uses one-based subject IDs, so the code converts graph node `0` into subject `1`, node `1` into subject `2`, and so on.

## Internal Contacts

The real EoN transmissions are converted into `Internal` events. Each event gets:

- `type`: `"Internal"`
- `involved_subjects`: the source and target subjects, with possible extra group members
- `time`: `introduction_time + EoN_transmission_time`
- `risk_factor`: a random value between `0.0` and `0.99`
- `result`: `None`

The helper `sample_internal_group(...)` always includes the EoN source and target. With probability `0.35`, it may add neighboring subjects, up to a maximum group size of `4`.

If `total_internal_contacts` is provided and EoN produces fewer transmissions than requested, the code adds extra synthetic internal contacts sampled from graph edges. If EoN produces more transmissions than requested, the code keeps them all. This means `total_internal_contacts` is a lower bound, not an exact cap.

## External Contacts

Before the EoN result is converted, the code creates random `External` events for every subject. It also adds one external event for the subject selected as the initially infected node.

External events have:

- `type`: `"External"`
- `involved_subjects`: one subject
- `time`: sampled in hours
- `risk_factor`: a random value between `0.0` and `0.99`
- `result`: `None`

## Test Events

After internal and external events are generated, the code adds `Test` events for each subject.

Test events have:

- `type`: `"Test"`
- `involved_subjects`: one subject
- `time`: sampled in hours
- `risk_factor`: `None`
- `result`: `"to be defined"`

The test result is not determined in `dataset_graph.py`. It is left for later pipeline stages.

## Output Helpers

Use `build_dataset_event_sequence(result)` when you want the dataset payload as a Python dictionary:

```python
from dataset_graph import simulate_external_introduction, build_dataset_event_sequence

result = simulate_external_introduction(seed=7)
payload = build_dataset_event_sequence(result)
```

Use `save_dataset_event_sequence(result, output_path)` when you want to write the JSON file:

```python
from dataset_graph import simulate_external_introduction, save_dataset_event_sequence

result = simulate_external_introduction(seed=7)
save_dataset_event_sequence(result, "my_dataset.json")
```

Use `plot_network_and_epidemic(result)` only when you want a diagnostic plot:

```python
import matplotlib.pyplot as plt
from dataset_graph import simulate_external_introduction, plot_network_and_epidemic

result = simulate_external_introduction(seed=7)
plot_network_and_epidemic(result)
plt.show()
```

## Notes

- The current graph is complete, so every subject is connected to every other subject.
- The direct script run writes `dataset_sample_output.json`; rename the output path in the bottom demo block if you want a different default.
- For reproducible output, pass a fixed `seed`.
- The generated `Internal` contact count can exceed `total_internal_contacts` if EoN creates more transmissions than the requested target.
