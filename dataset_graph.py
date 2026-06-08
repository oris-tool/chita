import json
import random

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


def build_contact_graph(n_nodes):
    graph = nx.complete_graph(n_nodes)
    if graph.number_of_edges() == 0 and n_nodes > 1:
        for node in range(n_nodes - 1):
            graph.add_edge(node, node + 1)
    return graph


def generate_external_contacts(graph, total_time_limit, max_external_contacts_per_node):
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
                    "risk_factor": rng.uniform(0.0, 1.0),
                    "result": None,
                }
            )
    return external_contacts


def ensure_external_contacts(graph, external_contacts, total_time_limit):
    if external_contacts:
        return external_contacts

    fallback_node = rng.choice(list(graph.nodes))
    external_contacts.append(
        {
            "type": "External",
            "involved_subjects": [fallback_node + 1],
            "time": float(rng.uniform(0.0, total_time_limit)),
            "risk_factor": rng.uniform(0.0, 1.0),
            "result": None,
        }
    )
    return external_contacts


def index_external_contacts_by_node(external_contacts):
    contact_indices_by_node = {}
    for index, event in enumerate(external_contacts):
        subject_id = int(event["involved_subjects"][0])
        node = subject_id - 1
        contact_indices_by_node.setdefault(node, []).append(index)
    return contact_indices_by_node


def select_effective_external_contacts(external_contacts, effective_external_contacts):
    contact_indices_by_node = index_external_contacts_by_node(external_contacts)
    candidate_nodes = sorted(contact_indices_by_node.keys())
    n_effective_contacts = min(int(effective_external_contacts), len(candidate_nodes))
    introduced_nodes = rng.sample(candidate_nodes, n_effective_contacts)
    effective_contact_indices = [
        rng.choice(contact_indices_by_node[node])
        for node in introduced_nodes
    ]
    effective_contacts = [external_contacts[index] for index in effective_contact_indices]
    return introduced_nodes, effective_contacts


def align_effective_introduction_times(effective_contacts):
    introduction_time = min(event["time"] for event in effective_contacts)
    for effective_event in effective_contacts:
        effective_event["time"] = float(introduction_time)
        effective_event["result"] = True
    return introduction_time


def run_hidden_outbreak(graph, transmission_rate, recovery_rate, introduced_nodes, tmax_after_intro):
    return EoN.fast_SIR(
        graph,
        tau=transmission_rate,
        gamma=recovery_rate,
        initial_infecteds=introduced_nodes,
        tmax=tmax_after_intro,
        return_full_data=True,
    )


def shift_epidemic_curves(simulation, introduction_time, n_nodes, n_effective_contacts):
    times = simulation.t()
    susceptible = simulation.S()
    infected = simulation.I()
    recovered = simulation.R()

    shifted_times = [introduction_time + t for t in times]

    if introduction_time > 0.0:
        shifted_times = [0.0, introduction_time] + shifted_times
        susceptible = [n_nodes, n_nodes - n_effective_contacts] + list(susceptible)
        infected = [0, n_effective_contacts] + list(infected)
        recovered = [0, 0] + list(recovered)
    else:
        susceptible = list(susceptible)
        infected = list(infected)
        recovered = list(recovered)

    return shifted_times, susceptible, infected, recovered


def simulate_external_introduction(
    n_nodes=8,
    edge_probability=0.8,
    transmission_rate=0.6,
    recovery_rate=0.2,
    max_intro_time=48.0,
    tmax_after_intro=2016.0,
    max_external_contacts_per_node=8,
    effective_external_contacts=1,
    total_internal_contacts=None,
    seed=None,
):
    """
    Build a random contact network and simulate SIR diffusion with EoN.

    The disease is introduced from outside the network by selecting one or more
    already-generated external contacts as effective introductions. Since
    ``EoN.fast_SIR`` starts at time 0, the sample runs the epidemic from the
    imported case(s) and then shifts the output times so that the introduction
    happens at the selected external-arrival time.

    Returns:
        dict: Graph, sampled introduction info, and S/I/R time series.
    """
    del edge_probability
    del max_intro_time

    if EoN is None:
        raise ImportError(
            "EoN is required for this sample. Install it with `pip install EoN`."
        ) from _EON_IMPORT_ERROR

    if seed is not None:
        global rng
        rng = random.Random(seed)

    graph = build_contact_graph(n_nodes)

    if int(effective_external_contacts) <= 0:
        raise ValueError("effective_external_contacts must be greater than 0.")

    total_time_limit = tmax_after_intro
    external_contacts = generate_external_contacts(
        graph=graph,
        total_time_limit=total_time_limit,
        max_external_contacts_per_node=max_external_contacts_per_node,
    )
    external_contacts = ensure_external_contacts(
        graph=graph,
        external_contacts=external_contacts,
        total_time_limit=total_time_limit,
    )

    introduced_nodes, effective_contacts = select_effective_external_contacts(
        external_contacts=external_contacts,
        effective_external_contacts=effective_external_contacts,
    )
    introduction_time = align_effective_introduction_times(effective_contacts)

    simulation = run_hidden_outbreak(
        graph=graph,
        transmission_rate=transmission_rate,
        recovery_rate=recovery_rate,
        introduced_nodes=introduced_nodes,
        tmax_after_intro=tmax_after_intro,
    )
    shifted_times, susceptible, infected, recovered = shift_epidemic_curves(
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
    introduced_nodes = result.get("introduced_nodes", [result["introduced_node"]])
    introduced_nodes_set = set(int(node) for node in introduced_nodes)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    layout = nx.spring_layout(graph, seed=42)
    node_colors = [
        "crimson" if node in introduced_nodes_set else "lightsteelblue"
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
        f"external introductions on nodes {sorted(introduced_nodes_set)} at t={result['introduction_time']:.2f}"
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


def generate_transmission_internal_events(result):
    graph = result["graph"]
    simulation = result["simulation"]
    introduction_time = result["introduction_time"]
    internal_events = []

    for time, source, target in simulation.transmissions():
        if source is None:
            continue
        internal_events.append(
            create_event(
                "Internal",
                sample_internal_group(graph, source, target),
                introduction_time + time,
                rng.uniform(0.0, 1.0),
            )
        )

    return internal_events


def top_up_internal_events(graph, internal_events, requested_internal_contacts, time_limit_hours):
    target_internal_contacts = max(int(requested_internal_contacts), len(internal_events))
    additional_contacts_needed = target_internal_contacts - len(internal_events)
    additional_timestamps = sample_continuous_timestamps(
        additional_contacts_needed,
        time_limit_hours,
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
    return internal_events


def generate_test_events(graph, time_limit_hours, fine_grained_rng):
    test_events = []
    for subject in range(1, graph.number_of_nodes() + 1):
        n_tests = sample_tests(time_limit_hours)
        test_timestamps = sample_timestamps(n_tests, time_limit_hours, fine_grained_rng)
        for timestamp in test_timestamps:
            test_events.append(
                create_event("Test", [subject], timestamp, None, "to be defined")
            )
    return test_events


def finalize_dataset_payload(graph, events, time_limit_hours):
    events.sort(key=lambda event: event["time"])
    return {
        "events": events,
        "n_subjects": graph.number_of_nodes(),
        "time_limit": int(round(time_limit_hours / 24.0)),
        "n_contacts": len([event for event in events if event["type"] == "Internal"]),
    }


def build_dataset_event_sequence(result):
    """
    Convert the simulated outbreak to the JSON event format used by dataset.py.
    """
    graph = result["graph"]
    fine_grained_rng = random.Random(seed)
    requested_internal_contacts = result.get("total_internal_contacts")

    events = list(result["external_contacts"])
    internal_events = generate_transmission_internal_events(result)

    if requested_internal_contacts is None:
        events.extend(internal_events)
    else:
        internal_events = top_up_internal_events(
            graph=graph,
            internal_events=internal_events,
            requested_internal_contacts=requested_internal_contacts,
            time_limit_hours=result["time_limit"],
        )
        events.extend(internal_events)

    events.extend(
        generate_test_events(
            graph=graph,
            time_limit_hours=result["time_limit"],
            fine_grained_rng=fine_grained_rng,
        )
    )
    return finalize_dataset_payload(
        graph=graph,
        events=events,
        time_limit_hours=result["time_limit"],
    )


def save_dataset_event_sequence(result, output_path):
    """
    Save the simulated outbreak in the same JSON schema as the existing datasets.
    """
    dataset_payload = build_dataset_event_sequence(result)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(dataset_payload, handle, indent=4)
    return dataset_payload


if __name__ == "__main__":
    result = simulate_external_introduction(
        seed=7,
        total_internal_contacts=400,
        effective_external_contacts=3,
    )
    print(
        f"External introduction at t={result['introduction_time']:.2f} "
        f"on nodes {result.get('introduced_nodes', [result['introduced_node']])}"
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
