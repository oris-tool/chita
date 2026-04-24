import json
import random

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


def sample_continuous_timestamps(n_events, time_limit_hours):
    if n_events <= 0:
        return []
    return sorted(rng.uniform(0.0, time_limit_hours) for _ in range(n_events))


def sample_uniform_subject_assignments(n_events, n_subjects, minimum_unique_subjects=0):
    if n_events <= 0:
        return []

    if minimum_unique_subjects > n_subjects:
        raise ValueError("minimum_unique_subjects cannot exceed n_subjects.")

    for _ in range(1000):
        assignments = [rng.randint(1, n_subjects) for _ in range(n_events)]
        if len(set(assignments)) >= minimum_unique_subjects:
            return assignments

    raise RuntimeError(
        "Could not sample enough unique subject assignments for the requested event count."
    )


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


def build_scale_free_graph(n_nodes, barabasi_m, graph_seed):
    if barabasi_m <= 0:
        raise ValueError("barabasi_m must be greater than 0.")
    if barabasi_m >= n_nodes:
        raise ValueError("barabasi_m must be smaller than n_nodes.")
    return nx.barabasi_albert_graph(n=n_nodes, m=barabasi_m, seed=graph_seed)


def generate_exact_external_contacts(
    n_subjects,
    total_time_limit,
    total_external_contacts,
    effective_external_contacts,
):
    subject_assignments = sample_uniform_subject_assignments(
        total_external_contacts,
        n_subjects,
        minimum_unique_subjects=effective_external_contacts,
    )
    timestamps = sample_continuous_timestamps(total_external_contacts, total_time_limit)
    return [
        create_event(
            "External",
            [subject_id],
            timestamp,
            rng.uniform(0.0, 1.0),
        )
        for subject_id, timestamp in zip(subject_assignments, timestamps)
    ]


def index_external_contacts_by_subject(external_contacts):
    contact_indices_by_subject = {}
    for index, event in enumerate(external_contacts):
        subject_id = int(event["involved_subjects"][0])
        contact_indices_by_subject.setdefault(subject_id, []).append(index)
    return contact_indices_by_subject


def select_effective_external_contacts(external_contacts, effective_external_contacts):
    contact_indices_by_subject = index_external_contacts_by_subject(external_contacts)
    candidate_subjects = sorted(contact_indices_by_subject.keys())
    if len(candidate_subjects) < effective_external_contacts:
        raise ValueError("Not enough unique external-contact subjects to select effective introductions.")

    introduced_subjects = rng.sample(candidate_subjects, int(effective_external_contacts))
    effective_contact_indices = [
        rng.choice(contact_indices_by_subject[subject_id])
        for subject_id in introduced_subjects
    ]
    effective_contacts = [external_contacts[index] for index in effective_contact_indices]
    introduced_nodes = [subject_id - 1 for subject_id in introduced_subjects]
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


def simulate_scale_free_introduction(
    n_nodes=100,
    transmission_rate=0.6,
    recovery_rate=0.2,
    tmax_after_intro=2016.0,
    total_external_contacts=1000,
    effective_external_contacts=15,
    total_internal_contacts=2500,
    total_symptom_observations=1000,
    total_test_observations=1000,
    barabasi_m=3,
    seed=None,
):
    if EoN is None:
        raise ImportError(
            "EoN is required for this sample. Install it with `pip install EoN`."
        ) from _EON_IMPORT_ERROR

    if seed is not None:
        global rng
        rng = random.Random(seed)

    graph = build_scale_free_graph(
        n_nodes=n_nodes,
        barabasi_m=barabasi_m,
        graph_seed=seed,
    )

    if int(effective_external_contacts) <= 0:
        raise ValueError("effective_external_contacts must be greater than 0.")
    if total_external_contacts < effective_external_contacts:
        raise ValueError("total_external_contacts must be >= effective_external_contacts.")

    external_contacts = generate_exact_external_contacts(
        n_subjects=n_nodes,
        total_time_limit=tmax_after_intro,
        total_external_contacts=total_external_contacts,
        effective_external_contacts=effective_external_contacts,
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
        "total_symptom_observations": total_symptom_observations,
        "total_test_observations": total_test_observations,
        "barabasi_m": barabasi_m,
        "time_limit": tmax_after_intro,
        "t": shifted_times,
        "S": susceptible,
        "I": infected,
        "R": recovered,
    }


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


def adjust_internal_events_to_exact_count(graph, internal_events, requested_internal_contacts, time_limit_hours):
    if requested_internal_contacts is None:
        return internal_events

    target_internal_contacts = int(requested_internal_contacts)
    if len(internal_events) >= target_internal_contacts:
        internal_events.sort(key=lambda event: event["time"])
        return internal_events[:target_internal_contacts]

    additional_contacts_needed = target_internal_contacts - len(internal_events)
    additional_timestamps = sample_continuous_timestamps(additional_contacts_needed, time_limit_hours)
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


def generate_exact_observation_events(event_type, total_events, n_subjects, time_limit_hours):
    default_result = "to be defined" if event_type == "Test" else None
    subject_assignments = sample_uniform_subject_assignments(total_events, n_subjects)
    timestamps = sample_continuous_timestamps(total_events, time_limit_hours)
    return [
        create_event(event_type, [subject_id], timestamp, None, default_result)
        for subject_id, timestamp in zip(subject_assignments, timestamps)
    ]


def finalize_dataset_payload(graph, events, time_limit_hours):
    events.sort(key=lambda event: event["time"])
    return {
        "events": events,
        "n_subjects": graph.number_of_nodes(),
        "time_limit": int(round(time_limit_hours / 24.0)),
        "n_contacts": len([event for event in events if event["type"] == "Internal"]),
    }


def build_dataset_event_sequence(result):
    graph = result["graph"]
    events = list(result["external_contacts"])

    internal_events = generate_transmission_internal_events(result)
    internal_events = adjust_internal_events_to_exact_count(
        graph=graph,
        internal_events=internal_events,
        requested_internal_contacts=result.get("total_internal_contacts"),
        time_limit_hours=result["time_limit"],
    )
    events.extend(internal_events)

    events.extend(
        generate_exact_observation_events(
            "Symptoms",
            result.get("total_symptom_observations", 0),
            graph.number_of_nodes(),
            result["time_limit"],
        )
    )
    events.extend(
        generate_exact_observation_events(
            "Test",
            result.get("total_test_observations", 0),
            graph.number_of_nodes(),
            result["time_limit"],
        )
    )

    return finalize_dataset_payload(
        graph=graph,
        events=events,
        time_limit_hours=result["time_limit"],
    )


def save_dataset_event_sequence(result, output_path):
    dataset_payload = build_dataset_event_sequence(result)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(dataset_payload, handle, indent=4)
    return dataset_payload


if __name__ == "__main__":
    result = simulate_scale_free_introduction(
        seed=30,
        total_internal_contacts=2500,
    )
    dataset_payload = save_dataset_event_sequence(result, "dataset_scale_free_sample_output.json")
    print(
        f"Saved {len(dataset_payload['events'])} events "
        f"({dataset_payload['n_contacts']} internal contacts) to dataset_scale_free_sample_output.json"
    )
