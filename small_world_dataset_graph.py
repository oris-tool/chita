import random

import networkx as nx

import scale_free_dataset_graph as sfdg


def build_small_world_graph(n_nodes, watts_k, rewire_probability, graph_seed):
    if watts_k <= 0:
        raise ValueError("watts_k must be greater than 0.")
    if watts_k >= n_nodes:
        raise ValueError("watts_k must be smaller than n_nodes.")
    if not 0.0 <= rewire_probability <= 1.0:
        raise ValueError("rewire_probability must be between 0 and 1.")
    return nx.watts_strogatz_graph(
        n=n_nodes,
        k=watts_k,
        p=rewire_probability,
        seed=graph_seed,
    )


def simulate_small_world_introduction(
    n_nodes=100,
    transmission_rate=0.6,
    recovery_rate=0.2,
    tmax_after_intro=2016.0,
    total_external_contacts=1000,
    effective_external_contacts=15,
    total_internal_contacts=1800,
    total_symptom_observations=1000,
    total_test_observations=1000,
    watts_k=5,
    rewire_probability=0.1,
    seed=None,
):
    if sfdg.EoN is None:
        raise ImportError(
            "EoN is required for this sample. Install it with `pip install EoN`."
        ) from sfdg._EON_IMPORT_ERROR

    if seed is not None:
        sfdg.rng = random.Random(seed)

    graph = build_small_world_graph(
        n_nodes=n_nodes,
        watts_k=watts_k,
        rewire_probability=rewire_probability,
        graph_seed=seed,
    )

    if int(effective_external_contacts) <= 0:
        raise ValueError("effective_external_contacts must be greater than 0.")
    if total_external_contacts < effective_external_contacts:
        raise ValueError("total_external_contacts must be >= effective_external_contacts.")

    external_contacts = sfdg.generate_exact_external_contacts(
        n_subjects=n_nodes,
        total_time_limit=tmax_after_intro,
        total_external_contacts=total_external_contacts,
        effective_external_contacts=effective_external_contacts,
    )
    introduced_nodes, effective_contacts = sfdg.select_effective_external_contacts(
        external_contacts=external_contacts,
        effective_external_contacts=effective_external_contacts,
    )
    introduction_time = sfdg.align_effective_introduction_times(effective_contacts)

    simulation = sfdg.run_hidden_outbreak(
        graph=graph,
        transmission_rate=transmission_rate,
        recovery_rate=recovery_rate,
        introduced_nodes=introduced_nodes,
        tmax_after_intro=tmax_after_intro,
    )
    shifted_times, susceptible, infected, recovered = sfdg.shift_epidemic_curves(
        simulation=simulation,
        introduction_time=introduction_time,
        n_nodes=n_nodes,
        n_effective_contacts=len(effective_contacts),
    )

    return {
        "graph": graph,
        "simulation": simulation,
        "introduction_time": introduction_time,
        "introduced_node": introduced_nodes[0],
        "introduced_nodes": introduced_nodes,
        "effective_external_contacts": len(effective_contacts),
        "external_contacts": sorted(external_contacts, key=lambda event: event["time"]),
        "total_internal_contacts": total_internal_contacts,
        "total_symptom_observations": total_symptom_observations,
        "total_test_observations": total_test_observations,
        "watts_k": watts_k,
        "rewire_probability": rewire_probability,
        "time_limit": tmax_after_intro,
        "t": shifted_times,
        "S": susceptible,
        "I": infected,
        "R": recovered,
    }


def build_dataset_event_sequence(result):
    return sfdg.build_dataset_event_sequence(result)


def save_dataset_event_sequence(result, output_path):
    return sfdg.save_dataset_event_sequence(result, output_path)


if __name__ == "__main__":
    result = simulate_small_world_introduction(seed=30)
    dataset_payload = save_dataset_event_sequence(result, "dataset_small_world_sample_output.json")
    print(
        f"Saved {len(dataset_payload['events'])} events "
        f"({dataset_payload['n_contacts']} internal contacts) to dataset_small_world_sample_output.json"
    )
