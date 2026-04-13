import argparse
import copy
import csv
import json
import os

import simulation
import sweep_pipeline as sweep_utils


DEFAULT_INPUT_DATASET_PATH = os.path.join("D2", "dataset_s8_t84_84.json")


def load_parameter_bundle(args):
    if args.parameter_bundle_path:
        bundle_path = os.path.abspath(args.parameter_bundle_path)
        return sweep_utils.read_json(bundle_path), bundle_path

    parameter_space = sweep_utils.load_parameter_space_from_ods(
        os.path.abspath(args.parameter_ods_path)
    )
    bundle = sweep_utils.resolve_uniform_parameter_bundle(
        parameter_space,
        args.parameter_level,
    )
    return bundle, None


def simulate_with_positive_test_audit(dataset_path, parameter_bundle, seed):
    raw_data = simulation.read_data(dataset_path)
    audit_rows = []
    original_handler = simulation.schedule_isolation_after_positive_test

    def audited_handler(
        data,
        subjects,
        subject_id,
        current_time,
        fine_grained_rng=None,
        distortion=1.0,
        parameter_bundle=None,
    ):
        if subjects[subject_id - 1].state == simulation.ISOLATED:
            audit_rows.append(
                {
                    "subject_id": subject_id,
                    "test_time_hours": current_time,
                    "state_before": "ISOLATED",
                    "sampled_delay_hours": None,
                    "scheduled_enter_time_hours": None,
                    "effective_delay_hours": None,
                    "reason": "already_isolated",
                    "pending_enter_before_hours": None,
                }
            )
            return subjects

        transition_parameters = simulation._transition_parameters(parameter_bundle)
        sampled_delay = simulation._sample_positive_test_isolation_delay(
            fine_grained_rng,
            transition_parameters,
        )
        target_enter_time = current_time + sampled_delay
        pending_enter, pending_exit = simulation._find_pending_isolation_events(
            data,
            subject_id,
            current_time,
        )
        pending_enter_before = None if pending_enter is None else pending_enter["time"]

        if pending_enter is None:
            enter_event = simulation.dataset.create_event(
                "Enter_Isolation",
                [subject_id],
                target_enter_time,
                risk_factor=None,
                result=None,
            )
            data["events"].append(enter_event)
            exit_event = simulation.dataset.create_event(
                "Exit_Isolation",
                [subject_id],
                target_enter_time + simulation._sample_isolation_duration(
                    fine_grained_rng,
                    distortion,
                    transition_parameters,
                    subject_symptoms=subjects[subject_id - 1].symptoms,
                ),
                risk_factor=None,
                result=None,
            )
            data["events"].append(exit_event)
            scheduled_enter_time = target_enter_time
            reason = "new_isolation_scheduled"
        elif pending_enter["time"] > target_enter_time:
            delay_delta = pending_enter["time"] - target_enter_time
            pending_enter["time"] = target_enter_time
            if pending_exit is not None:
                pending_exit["time"] = max(target_enter_time, pending_exit["time"] - delay_delta)
            else:
                exit_event = simulation.dataset.create_event(
                    "Exit_Isolation",
                    [subject_id],
                    target_enter_time + simulation._sample_isolation_duration(
                        fine_grained_rng,
                        distortion,
                        transition_parameters,
                        subject_symptoms=subjects[subject_id - 1].symptoms,
                    ),
                    risk_factor=None,
                    result=None,
                )
                data["events"].append(exit_event)
            scheduled_enter_time = target_enter_time
            reason = "pending_isolation_brought_forward"
        else:
            scheduled_enter_time = pending_enter["time"]
            reason = "pending_isolation_already_earlier"

        audit_rows.append(
            {
                "subject_id": subject_id,
                "test_time_hours": current_time,
                "state_before": "INFECTIOUS",
                "sampled_delay_hours": sampled_delay,
                "scheduled_enter_time_hours": scheduled_enter_time,
                "effective_delay_hours": scheduled_enter_time - current_time,
                "reason": reason,
                "pending_enter_before_hours": pending_enter_before,
            }
        )
        return subjects

    simulation.schedule_isolation_after_positive_test = audited_handler
    try:
        simulated_payload = simulation.simulate_one_iteration(
            copy.deepcopy(raw_data),
            raw_data["n_subjects"],
            seed=seed,
            parameter_bundle=parameter_bundle,
        )
    finally:
        simulation.schedule_isolation_after_positive_test = original_handler

    return raw_data, simulated_payload, audit_rows


def write_audit_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "subject_id",
                "test_time_hours",
                "state_before",
                "sampled_delay_hours",
                "scheduled_enter_time_hours",
                "effective_delay_hours",
                "reason",
                "pending_enter_before_hours",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def build_audit_summary(dataset_path, simulated_payload, parameter_bundle, seed, audit_rows):
    positive_tests = [
        event
        for event in simulated_payload["events"]
        if event["type"] == "Test" and bool(event["result"])
    ]
    non_isolated_rows = [row for row in audit_rows if row["state_before"] == "INFECTIOUS"]
    effective_delays = [
        row["effective_delay_hours"]
        for row in non_isolated_rows
        if row["effective_delay_hours"] is not None
    ]

    reason_counts = {}
    for row in audit_rows:
        reason_counts[row["reason"]] = reason_counts.get(row["reason"], 0) + 1

    return {
        "input_dataset_path": dataset_path,
        "parameter_case_id": parameter_bundle.get("case_id"),
        "parameter_levels": parameter_bundle.get("levels"),
        "seed": seed,
        "positive_test_count": len(positive_tests),
        "audit_row_count": len(audit_rows),
        "non_isolated_positive_test_count": len(non_isolated_rows),
        "all_non_isolated_positive_tests_have_isolation_schedule": all(
            row["scheduled_enter_time_hours"] is not None
            and row["scheduled_enter_time_hours"] >= row["test_time_hours"]
            for row in non_isolated_rows
        ),
        "reason_counts": reason_counts,
        "effective_delay_hours_min": min(effective_delays) if effective_delays else None,
        "effective_delay_hours_max": max(effective_delays) if effective_delays else None,
        "effective_delay_hours_mean": (
            sum(effective_delays) / len(effective_delays) if effective_delays else None
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Adapt the legacy D2 dataset to the current simulator behavior by "
            "materializing one derived simulated dataset and auditing the "
            "positive-test-to-isolation scheduling."
        )
    )
    parser.add_argument(
        "--input-dataset",
        default=DEFAULT_INPUT_DATASET_PATH,
        help="Raw dataset JSON to adapt. Defaults to D2/dataset_s8_t84_84.json.",
    )
    parser.add_argument(
        "--output-root",
        default=os.path.join("D2", "adapted_positive_test_isolation"),
        help="Directory where the adapted files will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Seed used for the adaptation simulation.",
    )
    parser.add_argument(
        "--parameter-ods-path",
        default=sweep_utils.DEFAULT_PARAMETER_ODS_PATH,
        help="ODS spreadsheet used to resolve the default parameter bundle.",
    )
    parser.add_argument(
        "--parameter-level",
        choices=["lower", "mid", "upper"],
        default=sweep_utils.GROUND_TRUTH_PARAMETER_LEVEL,
        help="Uniform bundle level to use when --parameter-bundle-path is not provided.",
    )
    parser.add_argument(
        "--parameter-bundle-path",
        default=None,
        help="Optional explicit parameter bundle JSON. Overrides --parameter-level.",
    )
    args = parser.parse_args()

    input_dataset_path = os.path.abspath(args.input_dataset)
    if not os.path.exists(input_dataset_path):
        raise FileNotFoundError(f"Input dataset not found: {input_dataset_path}")

    parameter_bundle, parameter_bundle_path = load_parameter_bundle(args)
    output_root = sweep_utils.ensure_dir(os.path.abspath(args.output_root))
    dataset_stem = os.path.splitext(os.path.basename(input_dataset_path))[0]
    bundle_case_id = parameter_bundle.get("case_id", "parameter_bundle")
    output_prefix = f"{dataset_stem}__{bundle_case_id}"

    raw_payload, simulated_payload, audit_rows = simulate_with_positive_test_audit(
        dataset_path=input_dataset_path,
        parameter_bundle=parameter_bundle,
        seed=args.seed,
    )

    adapted_dataset_path = os.path.join(output_root, f"{output_prefix}_simulated.json")
    audit_json_path = os.path.join(output_root, f"{output_prefix}_positive_test_isolation_audit.json")
    audit_csv_path = os.path.join(output_root, f"{output_prefix}_positive_test_isolation_audit.csv")
    summary_path = os.path.join(output_root, f"{output_prefix}_adaptation_summary.json")
    bundle_output_path = os.path.join(output_root, f"{output_prefix}_parameter_bundle.json")

    sweep_utils.write_json(adapted_dataset_path, simulated_payload)
    sweep_utils.write_json(audit_json_path, audit_rows)
    write_audit_csv(audit_csv_path, audit_rows)
    sweep_utils.write_json(bundle_output_path, parameter_bundle)

    summary = build_audit_summary(
        dataset_path=input_dataset_path,
        simulated_payload=simulated_payload,
        parameter_bundle=parameter_bundle,
        seed=args.seed,
        audit_rows=audit_rows,
    )
    summary.update(
        {
            "raw_dataset_path": input_dataset_path,
            "raw_dataset_events": len(raw_payload["events"]),
            "adapted_dataset_path": adapted_dataset_path,
            "audit_json_path": audit_json_path,
            "audit_csv_path": audit_csv_path,
            "parameter_bundle_path": bundle_output_path,
            "parameter_bundle_source_path": parameter_bundle_path,
        }
    )
    sweep_utils.write_json(summary_path, summary)

    print(f"Adapted dataset written to: {adapted_dataset_path}")
    print(f"Positive-test isolation audit written to: {audit_csv_path}")
    print(
        "Non-isolated positive tests: "
        f"{summary['non_isolated_positive_test_count']} | "
        f"all scheduled: {summary['all_non_isolated_positive_tests_have_isolation_schedule']}"
    )


if __name__ == "__main__":
    main()
