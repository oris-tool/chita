import random
import json

import matplotlib.pyplot as plt
import networkx as nx

try:
    import EoN
except ImportError as exc:
    EoN = None
    _EON_IMPORT_ERROR = exc
else:
    _EON_IMPORT_ERROR = None

seed = 777
rng = random.Random(seed)


def create_event(event_type, involved_subjects, timestamp, risk_factor=None, result=None):
    return {
        "type": event_type,
        "involved_subjects": involved_subjects,
        "time": float(timestamp),
        "risk_factor": risk_factor,
        "result": result,
    }


def sample_tests(time_limit_hours):
    time_limit_days = max(1, int(time_limit_hours // 24))
    return rng.randint(0, 2 * time_limit_days // 7)


def sample_timestamps(n_events, time_limit_hours, fine_grained_rng=None):
    if n_events <= 0:
        return []

    upper_bound = max(1, int(time_limit_hours))
    sampled = rng.sample(range(0, upper_bound + 1), min(n_events, upper_bound + 1))
    if fine_grained_rng is not None:
        sampled = [timestamp + fine_grained_rng.random() for timestamp in sampled]
    return sorted(sampled)


def sample_continuous_timestamps(n_events, time_limit_hours):
    if n_events <= 0:
        return []
    return sorted(rng.uniform(0.0, time_limit_hours) for _ in range(n_events))


def sample_internal_group(graph, source, target, max_group_size=4, group_event_probability=0.35):
    involved_subjects = {source + 1, target + 1}
    if rng.random() >= group_event_probability:
        return sorted(involved_subjects)

    candidate_nodes = set(graph.neighbors(source)) | set(graph.neighbors(target))
    candidate_nodes.discard(source)
    candidate_nodes.discard(target)
    candidate_subjects = [node + 1 for node in candidate_nodes]
    if not candidate_subjects:
        return sorted(involved_subjects)

    max_extra_nodes = min(len(candidate_subjects), max(0, max_group_size - len(involved_subjects)))
    if max_extra_nodes == 0:
        return sorted(involved_subjects)

    n_extra_nodes = rng.randint(1, max_extra_nodes)
    involved_subjects.update(rng.sample(candidate_subjects, n_extra_nodes))
    return sorted(involved_subjects)


def sample_internal_contact_from_graph(
    graph,
    max_group_size=4,
    group_event_probability=0.35,
    edges=None,
):
    if edges is None:
        edges = list(graph.edges())
    source, target = rng.choice(edges)
    return sample_internal_group(
        graph,
        source,
        target,
        max_group_size=max_group_size,
        group_event_probability=group_event_probability,
    )


def simulate_external_introduction(
    n_nodes=8,
    edge_probability=0.8,
    transmission_rate=0.6,
    recovery_rate=0.2,
    max_intro_time=48.0,
    tmax_after_intro=2016.0,
    max_external_contacts_per_node=8,
    total_internal_contacts=None,
    seed=None,
):
    """
    Build a random contact network and simulate SIR diffusion with EoN.

    The disease is introduced from outside the network by infecting one random
    node at a random time. Since ``EoN.fast_SIR`` starts at time 0, the sample
    runs the epidemic from the imported case and then shifts the output times
    so that the introduction happens at the sampled external-arrival time.

    Returns:
        dict: Graph, sampled introduction info, and S/I/R time series.
    """
    if EoN is None:
        raise ImportError(
            "EoN is required for this sample. Install it with `pip install EoN`."
        ) from _EON_IMPORT_ERROR

    if seed is not None:
        global rng
        rng = random.Random(seed)

    graph = nx.complete_graph(n_nodes)
    if graph.number_of_edges() == 0 and n_nodes > 1:
        for node in range(n_nodes - 1):
            graph.add_edge(node, node + 1)

    infected_node = rng.choice(list(graph.nodes))
    introduction_time = rng.uniform(0.0, max_intro_time)
    total_time_limit = tmax_after_intro

    external_contacts = []
    for node in graph.nodes:
        n_contacts = rng.randint(0, max_external_contacts_per_node)
        sampled_times = sorted(
            rng.uniform(0.0, total_time_limit) for _ in range(n_contacts)
        )
        for contact_time in sampled_times:
            external_contacts.append(
                {
                    "type": "External",
                    "involved_subjects": [node + 1],
                    "time": float(contact_time),
                    "risk_factor": rng.uniform(0.0, 0.99),
                    "result": None,
                }
            )

    external_contacts.append(
        {
            "type": "External",
            "involved_subjects": [infected_node + 1],
            "time": float(introduction_time),
            "risk_factor": rng.uniform(0.0, 0.99),
            "result": None,
        }
    )

    simulation = EoN.fast_SIR(
        graph,
        tau=transmission_rate,
        gamma=recovery_rate,
        initial_infecteds=[infected_node],
        tmax=tmax_after_intro,
        return_full_data=True,
    )
    times = simulation.t()
    susceptible = simulation.S()
    infected = simulation.I()
    recovered = simulation.R()

    shifted_times = [introduction_time + t for t in times]

    if introduction_time > 0.0:
        shifted_times = [0.0, introduction_time] + shifted_times
        susceptible = [n_nodes, n_nodes - 1] + list(susceptible)
        infected = [0, 1] + list(infected)
        recovered = [0, 0] + list(recovered)
    else:
        susceptible = list(susceptible)
        infected = list(infected)
        recovered = list(recovered)

    return {
        "graph": graph,
        "simulation": simulation,
        "introduction_time": introduction_time,
        "introduced_node": infected_node,
        "external_contacts": sorted(external_contacts, key=lambda event: event["time"]),
        "total_internal_contacts": total_internal_contacts,
        "time_limit": total_time_limit,
        "t": shifted_times,
        "S": susceptible,
        "I": infected,
        "R": recovered,
    }


def plot_network_and_epidemic(result):
    """
    Plot the contact network and the SIR curves for one simulated outbreak.
    """
    graph = result["graph"]
    introduced_node = result["introduced_node"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    layout = nx.spring_layout(graph, seed=42)
    node_colors = [
        "crimson" if node == introduced_node else "lightsteelblue"
        for node in graph.nodes
    ]
    nx.draw_networkx(
        graph,
        pos=layout,
        ax=axes[0],
        with_labels=True,
        node_color=node_colors,
        edge_color="silver",
        node_size=500,
        font_size=8,
    )
    axes[0].set_title(
        "Contact network\n"
        f"external introduction on node {introduced_node} at t={result['introduction_time']:.2f}"
    )
    axes[0].axis("off")

    axes[1].plot(result["t"], result["S"], label="Susceptible", color="royalblue", linewidth=2)
    axes[1].plot(result["t"], result["I"], label="Infected", color="crimson", linewidth=2)
    axes[1].plot(result["t"], result["R"], label="Recovered", color="seagreen", linewidth=2)
    axes[1].axvline(
        result["introduction_time"],
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="External introduction",
    )
    axes[1].set_title("Contagion evolution")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Number of nodes")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    return fig, axes


def build_dataset_event_sequence(result):
    """
    Convert the simulated outbreak to the JSON event format used by dataset.py.
    """
    graph = result["graph"]
    simulation = result["simulation"]
    introduction_time = result["introduction_time"]
    fine_grained_rng = random.Random(seed)
    requested_internal_contacts = result.get("total_internal_contacts")

    events = list(result["external_contacts"])
    internal_events = []

    for time, source, target in simulation.transmissions():
        if source is None:
            continue
        internal_events.append(
            create_event(
                "Internal",
                sample_internal_group(graph, source, target),
                introduction_time + time,
                rng.uniform(0.0, 0.99),
            )
        )

    if requested_internal_contacts is None:
        events.extend(internal_events)
    else:
        target_internal_contacts = max(int(requested_internal_contacts), len(internal_events))
        additional_contacts_needed = target_internal_contacts - len(internal_events)
        additional_timestamps = sample_continuous_timestamps(
            additional_contacts_needed,
            result["time_limit"],
        )
        graph_edges = list(graph.edges())
        for timestamp in additional_timestamps:
            internal_events.append(
                create_event(
                    "Internal",
                    sample_internal_contact_from_graph(graph, edges=graph_edges),
                    timestamp,
                    rng.uniform(0.0, 0.99),
                )
            )
        events.extend(internal_events)

    for subject in range(1, graph.number_of_nodes() + 1):
        n_tests = sample_tests(result["time_limit"])
        test_timestamps = sample_timestamps(n_tests, result["time_limit"], fine_grained_rng)
        for timestamp in test_timestamps:
            events.append(
                create_event("Test", [subject], timestamp, None, "to be defined")
            )

    events.sort(key=lambda event: event["time"])

    return {
        "events": events,
        "n_subjects": graph.number_of_nodes(),
        "time_limit": int(round(result["time_limit"] / 24.0)),
        "n_contacts": len([event for event in events if event["type"] == "Internal"]),
    }


def save_dataset_event_sequence(result, output_path):
    """
    Save the simulated outbreak in the same JSON schema as the existing datasets.
    """
    dataset_payload = build_dataset_event_sequence(result)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(dataset_payload, handle, indent=4)
    return dataset_payload


if __name__ == "__main__":
    result = simulate_external_introduction(seed=7, total_internal_contacts=400)
    print(
        f"External introduction at t={result['introduction_time']:.2f} "
        f"on node {result['introduced_node']}"
    )
    print(f"Network: {result['graph'].number_of_nodes()} nodes, {result['graph'].number_of_edges()} edges")
    print(f"External contacts saved: {len(result['external_contacts'])}")
    print("First 5 timeline points:")
    for t, s, i, r in zip(result["t"][:5], result["S"][:5], result["I"][:5], result["R"][:5]):
        print(f"t={t:.2f}, S={s}, I={i}, R={r}")
    dataset_payload = save_dataset_event_sequence(result, "dataset_sample_output.json")
    print(f"Saved {len(dataset_payload['events'])} events to dataset_sample_output.json")
    plot_network_and_epidemic(result)
    plt.show()
