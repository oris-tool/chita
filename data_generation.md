# Data Generation

This document explains how the dataset generation code works now, from an empty run directory to the final raw dataset JSON that the sweep consumes.

There are two dataset families:

- `bubble`: the original complete-graph generator in [`dataset_graph.py`](dataset_graph.py)
- `scale_free`: the new Barabasi-Albert generator in [`scale_free_dataset_graph.py`](scale_free_dataset_graph.py)

The sweep entrypoint is [`sweep_pipeline_final.py`](sweep_pipeline_final.py), which now accepts:

```bash
python sweep_pipeline_final.py --dataset bubble
python sweep_pipeline_final.py --dataset scale_free
```

## Shared JSON schema

Both generators write the same raw dataset schema:

```json
{
    "events": [...],
    "n_subjects": 8,
    "time_limit": 84,
    "n_contacts": 400
}
```

Each event has the same shape:

```json
{
    "type": "Internal",
    "involved_subjects": [1, 4],
    "time": 123.5,
    "risk_factor": 0.42,
    "result": null
}
```

The important event families are:

- `External`: a raw outside exposure for one subject
- `Internal`: a contact inside the network
- `Symptoms`: a symptom observation for one subject
- `Test`: a test observation for one subject

The common final payload step is:

- `dataset_graph.finalize_dataset_payload()`
- `scale_free_dataset_graph.finalize_dataset_payload()`

Reference code:

```python
return {
    "events": events,
    "n_subjects": graph.number_of_nodes(),
    "time_limit": int(round(time_limit_hours / 24.0)),
    "n_contacts": len([event for event in events if event["type"] == "Internal"]),
}
```

## Step 1: choose the dataset family

The sweep resolves the requested dataset profile in [`sweep_pipeline_final.py`](sweep_pipeline_final.py):

```python
args = parse_args(argv)
dataset_profile = resolve_dataset_profile(args.dataset)
```

That profile holds the core generation settings:

- `bubble`
  - `n_subjects = 8`
  - `internal_contacts = (32, 200, 400, 800)`
  - `effective_external_contacts = 3`
- `scale_free`
  - `n_subjects = 100`
  - `internal_contacts = (32, 2500, 5000, 10000)`
  - `effective_external_contacts = 15`
  - `total_external_contacts = 1000`
  - `total_symptom_observations = 1000`
  - `total_test_observations = 1000`
  - `barabasi_m = 3`

Reference code:

```python
DATASET_PROFILES = {
    "bubble": DatasetProfile(...),
    "scale_free": DatasetProfile(...),
}
```

## Step 2: build the output filename

Each raw dataset gets a family-specific filename:

```python
def dataset_filename(dataset_profile, internal_contacts):
    return f"dataset_{dataset_profile.family}_{internal_contacts}.json"
```

Examples:

- `dataset_bubble_32.json`
- `dataset_bubble_400.json`
- `dataset_scale_free_2500.json`

This matters later because selection, caching, and summaries stay separated by `dataset_stem`.

## Step 3: generate the raw dataset

The raw dataset is created by `generate_dataset_for_profile()` in [`sweep_pipeline_final.py`](sweep_pipeline_final.py).

Reference code:

```python
if dataset_profile.family == DATASET_FAMILY_BUBBLE:
    dataset = dg.simulate_external_introduction(...)
    payload = dg.save_dataset_event_sequence(dataset, dataset_path)
elif dataset_profile.family == DATASET_FAMILY_SCALE_FREE:
    dataset = sfdg.simulate_scale_free_introduction(...)
    payload = sfdg.save_dataset_event_sequence(dataset, dataset_path)
```

What changes between families is the generator module. What stays the same is the output schema.

## Bubble generator flow

The bubble generator lives in [`dataset_graph.py`](dataset_graph.py). It was refactored into small steps so the full path is readable.

### 3.1 Build the contact graph

The bubble network is a complete graph:

```python
graph = build_contact_graph(n_nodes)
```

Reference:

- `dataset_graph.build_contact_graph()`

### 3.2 Sample raw external contacts

For each node, the generator samples how many outside contacts it has and when they happen:

```python
external_contacts = generate_external_contacts(
    graph=graph,
    total_time_limit=total_time_limit,
    max_external_contacts_per_node=max_external_contacts_per_node,
)
external_contacts = ensure_external_contacts(...)
```

Reference:

- `dataset_graph.generate_external_contacts()`
- `dataset_graph.ensure_external_contacts()`

### 3.3 Choose the effective external introductions

The generator does not treat every raw external contact as a real importation. It chooses a small number of effective introductions:

```python
introduced_nodes, effective_contacts = select_effective_external_contacts(
    external_contacts=external_contacts,
    effective_external_contacts=effective_external_contacts,
)
introduction_time = align_effective_introduction_times(effective_contacts)
```

Reference:

- `dataset_graph.select_effective_external_contacts()`
- `dataset_graph.align_effective_introduction_times()`

### 3.4 Run the hidden epidemic

The epidemic itself is simulated over the graph with `EoN.fast_SIR(...)`:

```python
simulation = run_hidden_outbreak(
    graph=graph,
    transmission_rate=transmission_rate,
    recovery_rate=recovery_rate,
    introduced_nodes=introduced_nodes,
    tmax_after_intro=tmax_after_intro,
)
```

Reference:

- `dataset_graph.run_hidden_outbreak()`
- `dataset_graph.shift_epidemic_curves()`

### 3.5 Convert transmissions into raw internal contacts

Every transmission inside the hidden epidemic becomes a raw `Internal` event:

```python
internal_events = generate_transmission_internal_events(result)
```

Reference:

- `dataset_graph.generate_transmission_internal_events()`

### 3.6 Top up internal contacts to the requested total

If the hidden epidemic creates fewer internal contacts than requested, the generator samples extra graph-based internal contacts:

```python
internal_events = top_up_internal_events(
    graph=graph,
    internal_events=internal_events,
    requested_internal_contacts=requested_internal_contacts,
    time_limit_hours=result["time_limit"],
)
```

Reference:

- `dataset_graph.top_up_internal_events()`
- `dataset_graph.sample_internal_contact_from_graph()`
- `dataset_graph.sample_internal_group()`

### 3.7 Add raw test events

The bubble raw dataset directly includes `Test` events:

```python
events.extend(
    generate_test_events(
        graph=graph,
        time_limit_hours=result["time_limit"],
        fine_grained_rng=fine_grained_rng,
    )
)
```

Reference:

- `dataset_graph.generate_test_events()`
- `dataset_graph.sample_tests()`
- `dataset_graph.sample_timestamps()`

### 3.8 Finalize and save

The final bubble payload is assembled here:

```python
return finalize_dataset_payload(
    graph=graph,
    events=events,
    time_limit_hours=result["time_limit"],
)
```

Reference:

- `dataset_graph.build_dataset_event_sequence()`
- `dataset_graph.save_dataset_event_sequence()`

## Scale-free generator flow

The scale-free generator lives in [`scale_free_dataset_graph.py`](scale_free_dataset_graph.py). It is fully separate from the bubble generator.

### 4.1 Build the scale-free graph

The graph is a Barabasi-Albert network:

```python
graph = build_scale_free_graph(
    n_nodes=n_nodes,
    barabasi_m=barabasi_m,
    graph_seed=seed,
)
```

Reference:

- `scale_free_dataset_graph.build_scale_free_graph()`

`m` in `networkx.barabasi_albert_graph(n, m, seed)` means:

- every new node attaches to `m` existing nodes
- larger `m` gives a denser graph
- here we use `m = 3`

### 4.2 Generate exact raw external contacts

The scale-free generator creates exactly `1000` raw external contacts:

```python
external_contacts = generate_exact_external_contacts(
    n_subjects=n_nodes,
    total_time_limit=tmax_after_intro,
    total_external_contacts=total_external_contacts,
    effective_external_contacts=effective_external_contacts,
)
```

Reference:

- `scale_free_dataset_graph.generate_exact_external_contacts()`
- `scale_free_dataset_graph.sample_uniform_subject_assignments()`
- `scale_free_dataset_graph.sample_continuous_timestamps()`

### 4.3 Choose the effective external introductions

Exactly `15` effective external introductions are selected from the raw external events:

```python
introduced_nodes, effective_contacts = select_effective_external_contacts(...)
introduction_time = align_effective_introduction_times(effective_contacts)
```

Reference:

- `scale_free_dataset_graph.select_effective_external_contacts()`
- `scale_free_dataset_graph.align_effective_introduction_times()`

### 4.4 Run the hidden epidemic

The hidden outbreak is simulated over the scale-free graph:

```python
simulation = run_hidden_outbreak(
    graph=graph,
    transmission_rate=transmission_rate,
    recovery_rate=recovery_rate,
    introduced_nodes=introduced_nodes,
    tmax_after_intro=tmax_after_intro,
)
```

Reference:

- `scale_free_dataset_graph.run_hidden_outbreak()`
- `scale_free_dataset_graph.shift_epidemic_curves()`

### 4.5 Convert transmissions to internal events and force the exact requested count

The scale-free generator always produces the exact requested internal-contact count:

```python
internal_events = generate_transmission_internal_events(result)
internal_events = adjust_internal_events_to_exact_count(
    graph=graph,
    internal_events=internal_events,
    requested_internal_contacts=result.get("total_internal_contacts"),
    time_limit_hours=result["time_limit"],
)
```

Reference:

- `scale_free_dataset_graph.generate_transmission_internal_events()`
- `scale_free_dataset_graph.adjust_internal_events_to_exact_count()`

For the small warm-up case (`32`), the generator trims the internal-event list if the hidden epidemic produced more than `32`. For the larger cases, it usually tops up with extra graph-based contacts.

### 4.6 Add exact raw symptoms and tests

Unlike the bubble generator, the scale-free raw dataset directly contains both `Symptoms` and `Test` observations in exact counts:

```python
events.extend(generate_exact_observation_events("Symptoms", ...))
events.extend(generate_exact_observation_events("Test", ...))
```

Reference:

- `scale_free_dataset_graph.generate_exact_observation_events()`

The exact counts are:

- `1000 External`
- `1000 Symptoms`
- `1000 Test`
- `32 / 2500 / 5000 / 10000 Internal`

### 4.7 Finalize and save

The final scale-free payload is assembled here:

```python
return finalize_dataset_payload(
    graph=graph,
    events=events,
    time_limit_hours=result["time_limit"],
)
```

Reference:

- `scale_free_dataset_graph.build_dataset_event_sequence()`
- `scale_free_dataset_graph.save_dataset_event_sequence()`

## Step 5: verify the refactored bubble generator

The bubble generator was refactored for readability, so there is a regression test that checks it against controlled reference fixtures:

- [`tests/test_bubble_generation.py`](tests/test_bubble_generation.py)
- [`tests/fixtures/bubble_generation/`](tests/fixtures/bubble_generation)

The test seeds both Python `random` and NumPy before generation, because `EoN.fast_SIR(...)` uses hidden random sources outside the module-level `rng`.

Reference code:

```python
random.seed(case["seed"])
np.random.seed(case["seed"])
result = dg.simulate_external_introduction(...)
payload = dg.build_dataset_event_sequence(result)
self.assertEqual(payload, expected)
```

## Step 6: what the sweep does after raw dataset generation

Once the raw dataset exists, [`sweep_pipeline_final.py`](sweep_pipeline_final.py) turns it into the actual experimental inputs.

### 6.1 Clean ground-truth run

The sweep first runs the clean ground-truth simulation on the clean raw dataset:

```python
gt_results = run_dataset_simulations(
    dataset_path=ground_truth_dataset_path,
    run_until_convergence=True,
    export_observed_simulation=True,
    parameter_bundle=ground_truth_case["parameter_bundle"],
)
```

### 6.2 Contact noise on the raw dataset

Then it creates the baseline raw dataset by corrupting only `Internal` and `External` events:

```python
contact_noise_summary = apply_contact_noise_to_dataset(
    source_dataset_path=dataset_path,
    noisy_dataset_path=contact_noisy_dataset_path,
    seed_label=f"{dataset_stem}:contact",
)
```

### 6.3 One observed simulation on the noisy raw dataset

Next it runs exactly one simulation with the ground-truth parameters:

```python
one_run_result = run_dataset_simulations(
    dataset_path=contact_noisy_dataset_path,
    rep=1,
    export_observed_simulation=True,
    parameter_bundle=ground_truth_case["parameter_bundle"],
)
```

### 6.4 Observation noise on the one-run observed file

Then it corrupts only `Test` and `Symptoms` events:

```python
observation_noise_summary = apply_observation_noise_to_dataset(
    source_dataset_path=clean_observed_one_run_path,
    noisy_dataset_path=final_observed_simulated_path,
    seed_label=f"{dataset_stem}:observation",
)
```

### 6.5 Final routing

The sweep keeps two different inputs:

- Python baseline uses the contact-noisy raw dataset
- Java analysis uses the observation-noisy `*_simulated.json`

Reference code:

```python
dataset_runs.append(
    {
        "dataset_path": contact_noisy_dataset_paths_by_stem[dataset_stem],
        "observed_simulated_path": final_observed_simulated_path,
        "ground_truth_path": gt_results["averaged_results_path"],
    }
)
```

## Minimal examples

Generate and run the bubble sweep:

```bash
python sweep_pipeline_final.py --dataset bubble
```

Generate and run the scale-free sweep:

```bash
python sweep_pipeline_final.py --dataset scale_free
```

Force a new run instead of resuming:

```bash
CHITA_FORCE_NEW_RUN=1 python sweep_pipeline_final.py --dataset scale_free
```
