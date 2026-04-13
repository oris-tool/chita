# This program is part of the ORIS Tool.
# Copyright (C) 2011-2025 The ORIS Authors.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

import json
import random
import dataset
import time
import numpy as np

from inversion_method import sample_from_hyper_exp, sample_generalized_erlang


HEALTHY = 0
INFECTED = 1
INFECTIOUS = 2
HEALED = 3
ISOLATED = 4

ASYMPTOMATIC = 0
DEVELOPING_SYMPTOMS = 1
SYMPTOMATIC = 2

TRACK_HEALTHY = 0
TRACK_INFECTIOUS = 1
TRACK_HEALED = 3
TRACK_ISOLATED = 4

HOURS_PER_DAY = 24.0
POSITIVE_TEST_ISOLATION_PSI = {
    "p1": 0.88188,
    "p2": 0.11812,
    "lambda1_per_day": 4.64146,
    "lambda2_per_day": 0.62170,
}


def _transition_parameters(parameter_bundle):
    if parameter_bundle is None:
        return None
    if "transitions" in parameter_bundle:
        return parameter_bundle["transitions"]
    return parameter_bundle


def _transition_value(transition_spec, key):
    if key in transition_spec:
        return transition_spec[key]
    legacy_key = key.replace("_", " ")
    if legacy_key in transition_spec:
        return transition_spec[legacy_key]
    raise KeyError(f"Missing transition key '{key}' in specification: {transition_spec}")


def _sample_bundle_transition_duration(transition_spec):
    erlang_stages = int(_transition_value(transition_spec, "erlang_stages"))
    erlang_lambda = float(_transition_value(transition_spec, "erlang_lambda"))
    exponential_lambda = float(_transition_value(transition_spec, "exponential_lambda"))
    lambdas = []
    if erlang_stages > 0:
        lambdas.extend([erlang_lambda] * erlang_stages)
    lambdas.append(exponential_lambda)
    sampled_duration = float(sample_generalized_erlang(lambdas))
    unit_measure = str(transition_spec.get("unit_measure", "hours")).strip().lower()
    # if unit_measure in {"day", "days"}:
    #     return sampled_duration * HOURS_PER_DAY
    return sampled_duration


def _sample_bundle_hyper_exp_duration(transition_spec):
    sampled_duration = float(
        sample_from_hyper_exp(
            float(_transition_value(transition_spec, "p1")),
            float(_transition_value(transition_spec, "p2")),
            float(_transition_value(transition_spec, "lambda1")),
            float(_transition_value(transition_spec, "lambda2")),
        )
    )
    unit_measure = str(transition_spec.get("unit_measure", "hours")).strip().lower()
    # if unit_measure in {"day", "days"}:
    #     return sampled_duration * HOURS_PER_DAY
    return sampled_duration

class Subject:
    def __init__(self, id, state, symptoms = 0):
        """
        state: HEALTHY, INFECTED, INFECTIOUS, HEALED, or ISOLATED.
        symptoms: ASYMPTOMATIC, DEVELOPING_SYMPTOMS, or SYMPTOMATIC.
        """
        self.id = id
        self.state = state
        self.symptoms = symptoms

def read_data(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def print_event(event):
    if event["type"] == "External":
        print("\033[33m{}\033[00m".format(event))
    elif event["type"] == "Symptoms":
        print("\033[93m{}\033[00m".format(event))
    elif event["type"] == "Test":
        print("\033[90m{}\033[00m".format(event))
    elif event["type"] == "Infectious":
        print("\033[91m{}\033[00m".format(event))
    elif event["type"] == "Internal":
        print("\033[35m{}\033[00m".format(event))
    elif event["type"] == "Enter_Isolation":
        print("\033[94m{}\033[00m".format(event))
    elif event["type"] == "Heal" or event["type"] == "Exit_Isolation":
        print("\033[92m{}\033[00m".format(event))
    else:
        print(event)


def print_subject_snapshot(subjects, involved_subjects, prefix):
    for subject in involved_subjects:
        subject_state = subjects[subject - 1].state
        subject_symptoms = subjects[subject - 1].symptoms
        print(
            f"{prefix} subject={subject} state={subject_state} symptoms={subject_symptoms}"
        )


def infect_subject(data, subjects, subject_id, current_time, N, fine_grained_rng=None, distortion=1.0, parameter_bundle=None):
    transition_parameters = _transition_parameters(parameter_bundle)
    if transition_parameters is not None:
        symptomatic_probability = float(_transition_value(transition_parameters["symptoms"], "true"))
        symptoms_onset_spec = transition_parameters["symptomsOnset"]
        infectiousness_spec = transition_parameters["infectiousness"]
    elif abs(distortion - 1.0) < 1e-6:
        erl_k_dev_symptoms = 2
        lambda_erl_dev_symptoms = 0.022553849
        lambda_exp_dev_symptoms = 0.013418615

        erl_k_infectious = 2
        lambda_erl_infectious = 0.0608792530475255
        lambda_exp_infectious = 0.0205136260095492
    elif abs(distortion - 1.15) < 1e-6:
        p1_develop_symptoms = 0.76
        lambda1_develop_symptoms = 0.56385 / 24.0
        p2_develop_symptoms = 0.24
        lambda2_develop_symptoms = 0.182557 / 24.0
        p1_infectious = 0.85662
        p2_infectious = 0.14338
        lambda1_infectious = 1.137232/24.0
        lambda2_infectious = 0.190348/24.0
    elif abs(distortion - 0.85) < 1e-6:
        p1_develop_symptoms = 0.86
        lambda1_develop_symptoms = 0.867248 / 24.0
        p2_develop_symptoms = 0.14
        lambda2_develop_symptoms = 0.142598 / 24.0
        p1_infectious = 0.9177
        p2_infectious = 0.082286
        lambda1_infectious = 1.648341/24.0
        lambda2_infectious = 0.147798/24.0
    elif abs(distortion - 1.25) < 1e-6:
        p1_develop_symptoms = 0.713
        lambda1_develop_symptoms = 0.4897 / 24.0
        p2_develop_symptoms = 0.287
        lambda2_develop_symptoms = 0.1973 / 24.0
        p1_infectious = 0.833
        p2_infectious = 0.167
        lambda1_infectious = 1.0177/24.0
        lambda2_infectious = 0.2036/24.0
    elif abs(distortion - 0.75) < 1e-6:
        p1_develop_symptoms = 0.888
        lambda1_develop_symptoms = 1.0158 / 24.0
        p2_develop_symptoms = 0.112
        lambda2_develop_symptoms = 0.1286 / 24.0
        p1_infectious = 0.935
        p2_infectious = 0.065
        lambda1_infectious = 1.903/24.0
        lambda2_infectious = 0.132/24.0
        
    symptomatic_threshold = 0.351
    generate_symptoms_threshold = 0.75


    subjects[subject_id-1].state = INFECTED
    if transition_parameters is not None:
        is_asymptomatic = random.random() >= symptomatic_probability
    else:
        is_asymptomatic = random.random() < symptomatic_threshold
    if is_asymptomatic:
        subjects[subject_id-1].symptoms = ASYMPTOMATIC
    else:
        subjects[subject_id-1].symptoms = DEVELOPING_SYMPTOMS
        if transition_parameters is not None:
            develop_symptoms_in = _sample_bundle_transition_duration(symptoms_onset_spec)
        elif abs(distortion - 1.0) < 1e-6:
            develop_symptoms_in = sample_generalized_erlang(
                [lambda_erl_dev_symptoms] * erl_k_dev_symptoms + [lambda_exp_dev_symptoms]
            )
        else:
            develop_symptoms_in = sample_from_hyper_exp(p1_develop_symptoms, p2_develop_symptoms, lambda1_develop_symptoms, lambda2_develop_symptoms)
        event = dataset.create_event("Develop_Symptoms", [subject_id], current_time + develop_symptoms_in, risk_factor=None, result=None)
        data["events"].append(event)
        n_symptoms = random.randint(1, N)
        for i in range(n_symptoms):
            if random.random() < generate_symptoms_threshold:
                # Generate symptoms
                event_offset = random.randint(1, 14 * 24)
                if fine_grained_rng is not None:
                    minute = fine_grained_rng.random()
                    event_offset += minute
                event = dataset.create_event("Symptoms", [subject_id], current_time + event_offset, risk_factor=None, result=None) #XXX Creazione dei sintomi
                data["events"].append(event)

    if transition_parameters is not None:
        infectious_in = _sample_bundle_transition_duration(infectiousness_spec)
    elif abs(distortion - 1.0) < 1e-6:
        infectious_in = sample_generalized_erlang(
            [lambda_erl_infectious] * erl_k_infectious + [lambda_exp_infectious]
        )
    else:
        infectious_in = sample_from_hyper_exp(p1_develop_symptoms, p2_develop_symptoms, lambda1_develop_symptoms, lambda2_develop_symptoms) + sample_from_hyper_exp(p1_infectious, p2_infectious, lambda1_infectious, lambda2_infectious)
    event = dataset.create_event("Infectious", [subject_id], current_time + infectious_in, risk_factor=None, result=None)
    data["events"].append(event)
    return subjects


def set_healing_time(data, subjects, subject_id, current_time, distortion=1.0, parameter_bundle=None):
    subjects[subject_id-1].state = INFECTIOUS
    transition_parameters = _transition_parameters(parameter_bundle)
    if transition_parameters is not None:
        heal_in = _sample_bundle_transition_duration(transition_parameters["healing"])
    elif abs(distortion - 1.0) < 1e-6:
        ## Generalized Erlang
        lambda1 = 0.011156175
        lambda2 = 0.005581493
        heal_in = sample_generalized_erlang([lambda1, lambda2])
    elif abs(distortion - 1.15) < 1e-6:
        lambda1 = 0.0993 / 24.0
        lambda2 = 0.272390 / 24.0
        heal_in = sample_generalized_erlang([lambda1, lambda2])
    elif abs(distortion - 0.85) < 1e-6:
        p1 = 0.12217
        p2 = 0.07468
        lambda1 = 0.62063 / 24.0
        lambda2 = 0.37937 / 24.0
        heal_in = sample_from_hyper_exp(p1, p2, lambda1, lambda2)
    elif abs(distortion - 1.25) < 1e-6:
        lambda1 = 0.1123/24.0
        lambda2 = 0.1656/24.0
        heal_in = sample_generalized_erlang([lambda1, lambda2])
    elif abs(distortion - 0.75) < 1e-6:
        p1 = 0.1590
        p2 = 0.0642
        lambda1 = 0.7123 / 24.0
        lambda2 = 0.2877 / 24.0
        heal_in = sample_from_hyper_exp(p1, p2, lambda1, lambda2)


    event = dataset.create_event("Heal", [subject_id], current_time + heal_in, risk_factor=None, result=None)
    data["events"].append(event)
    return subjects


def _sample_isolation_delay(fine_grained_rng=None, distortion=1.0, transition_parameters=None):
    if transition_parameters is not None:
        isolate_in = _sample_bundle_transition_duration(transition_parameters["isolating"])
    elif abs(distortion - 1.0) < 1e-6:
        erl_k = 3
        lambda_erl = 0.033632585
        lambda_exp = 0.016447155
        isolate_in = sample_generalized_erlang([lambda_erl] * erl_k + [lambda_exp])
    else:
        min_isolate_in_hours = 0
        max_isolate_in_hours = int(24 * distortion)
        isolate_in = random.randint(min_isolate_in_hours, max_isolate_in_hours)
    if fine_grained_rng is not None:
        isolate_in += fine_grained_rng.random()
    return isolate_in


def _sample_positive_test_isolation_delay(fine_grained_rng=None, transition_parameters=None):
    if transition_parameters is not None and "notificationToIsolation" in transition_parameters:
        return _sample_bundle_hyper_exp_duration(transition_parameters["notificationToIsolation"])

    psi_parameters = POSITIVE_TEST_ISOLATION_PSI
    delay_days = sample_from_hyper_exp(
        psi_parameters["p1"],
        psi_parameters["p2"],
        psi_parameters["lambda1_per_day"] / HOURS_PER_DAY,
        psi_parameters["lambda2_per_day"] / HOURS_PER_DAY,
    )
    return delay_days


def _sample_isolation_duration(
    fine_grained_rng=None,
    distortion=1.0,
    transition_parameters=None,
    subject_symptoms=None,
):
    if transition_parameters is not None:
        if (
            subject_symptoms in (DEVELOPING_SYMPTOMS, SYMPTOMATIC)
            and "symptomaticPeriod" in transition_parameters
        ):
            isolation_duration = _sample_bundle_transition_duration(
                transition_parameters["symptomaticPeriod"]
            )
        else:
            isolation_duration = _sample_bundle_transition_duration(
                transition_parameters["healing"]
            )
    else:
        max_isolate_in_hours = int(24 * distortion)
        isolation_duration = random.randint(
            int(max_isolate_in_hours * 7),
            int(max_isolate_in_hours * 28),
        )
    if fine_grained_rng is not None:
        isolation_duration += fine_grained_rng.random()
    return isolation_duration


def _find_pending_isolation_events(data, subject_id, current_time):
    pending_enter = None
    for event in data["events"]:
        if (
            event["type"] == "Enter_Isolation"
            and subject_id in event["involved_subjects"]
            and event["time"] >= current_time
        ):
            if pending_enter is None or event["time"] < pending_enter["time"]:
                pending_enter = event

    pending_exit = None
    if pending_enter is None:
        return None, None

    for event in data["events"]:
        if (
            event["type"] == "Exit_Isolation"
            and subject_id in event["involved_subjects"]
            and event["time"] >= pending_enter["time"]
        ):
            if pending_exit is None or event["time"] < pending_exit["time"]:
                pending_exit = event

    return pending_enter, pending_exit


def set_isolation_time(data, subjects, subject_id, current_time, fine_grained_rng=None, distortion=1.0, parameter_bundle=None):
    transition_parameters = _transition_parameters(parameter_bundle)
    isolate_in = _sample_isolation_delay(fine_grained_rng, distortion, transition_parameters)
    isolation_duration = _sample_isolation_duration(
        fine_grained_rng,
        distortion,
        transition_parameters,
        subject_symptoms=subjects[subject_id - 1].symptoms,
    )
    subjects[subject_id-1].state = INFECTIOUS
    enter_time = current_time + isolate_in
    event = dataset.create_event("Enter_Isolation", [subject_id], enter_time, risk_factor=None, result=None)
    data["events"].append(event)
    event = dataset.create_event(
        "Exit_Isolation",
        [subject_id],
        enter_time + isolation_duration,
        risk_factor=None,
        result=None,
    )
    data["events"].append(event)
    return subjects


def schedule_isolation_after_positive_test(
    data,
    subjects,
    subject_id,
    current_time,
    fine_grained_rng=None,
    distortion=1.0,
    parameter_bundle=None,
):
    if subjects[subject_id - 1].state == ISOLATED:
        return subjects

    transition_parameters = _transition_parameters(parameter_bundle)
    target_enter_time = current_time + _sample_positive_test_isolation_delay(
        fine_grained_rng,
        transition_parameters,
    )
    pending_enter, pending_exit = _find_pending_isolation_events(data, subject_id, current_time)

    if pending_enter is None:
        enter_event = dataset.create_event(
            "Enter_Isolation",
            [subject_id],
            target_enter_time,
            risk_factor=None,
            result=None,
        )
        data["events"].append(enter_event)
        exit_event = dataset.create_event(
            "Exit_Isolation",
            [subject_id],
            target_enter_time + _sample_isolation_duration(
                fine_grained_rng,
                distortion,
                transition_parameters,
                subject_symptoms=subjects[subject_id - 1].symptoms,
            ),
            risk_factor=None,
            result=None,
        )
        data["events"].append(exit_event)
    elif pending_enter["time"] > target_enter_time:
        delay_delta = pending_enter["time"] - target_enter_time
        pending_enter["time"] = target_enter_time
        if pending_exit is not None:
            pending_exit["time"] = max(target_enter_time, pending_exit["time"] - delay_delta)
        else:
            exit_event = dataset.create_event(
                "Exit_Isolation",
                [subject_id],
                target_enter_time + _sample_isolation_duration(
                    fine_grained_rng,
                    distortion,
                    transition_parameters,
                    subject_symptoms=subjects[subject_id - 1].symptoms,
                ),
                risk_factor=None,
                result=None,
            )
            data["events"].append(exit_event)

    return subjects

def _export_data(data, filename):
    data["events"].sort(key=lambda x: x["time"])
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)



    

def simulate_one_iteration(
    data,
    n_subjects,
    export_data=False,
    exported_data_filename=None,
    seed=None,
    fine_grained=False,
    distortion=1.0,
    parameter_bundle=None,
):
    # Create a list of subjects
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    if fine_grained:
        if seed is not None:
            fine_grained_rng = random.Random(seed)
        else:
            fine_grained_rng = random.Random()
    else:
        fine_grained_rng = None

    subjects = []
    for i in range(n_subjects):
        subjects.append(Subject(i, HEALTHY))
    past_events = []
    while data["events"]:
        data["events"].sort(key=lambda x: x["time"])
        event = data["events"].pop(0)
        involved_subjects = event["involved_subjects"]
        event_type = event["type"]
        current_time = event["time"]
        if event_type == "External":
            if event["risk_factor"] > random.random():
                for subject in involved_subjects:
                    if subjects[subject - 1].state == HEALTHY:
                        subjects = infect_subject(
                            data,
                            subjects,
                            subject,
                            current_time,
                            n_subjects,
                            fine_grained_rng,
                            distortion,
                            parameter_bundle=parameter_bundle,
                        )
        elif event_type == "Infectious":
            for subject in involved_subjects:
                if subjects[subject - 1].state == INFECTED:
                    if subjects[subject - 1].symptoms == ASYMPTOMATIC:
                        subjects = set_healing_time(
                            data,
                            subjects,
                            subject,
                            current_time,
                            distortion,
                            parameter_bundle=parameter_bundle,
                        )
                    elif subjects[subject - 1].symptoms in (DEVELOPING_SYMPTOMS, SYMPTOMATIC):
                        subjects = set_isolation_time(
                            data,
                            subjects,
                            subject,
                            current_time,
                            fine_grained_rng,
                            distortion,
                            parameter_bundle=parameter_bundle,
                        )
        elif event_type == "Heal":
            for subject in involved_subjects:
                if subjects[subject - 1].state == INFECTIOUS:
                    subjects[subject - 1].state = HEALED
                    subjects[subject - 1].symptoms = ASYMPTOMATIC
        elif event_type == "Develop_Symptoms":
            for subject in involved_subjects:
                if subjects[subject - 1].symptoms == DEVELOPING_SYMPTOMS:
                    subjects[subject - 1].symptoms = SYMPTOMATIC
        elif event_type == "Enter_Isolation":
            for subject in involved_subjects:
                if subjects[subject - 1].state == INFECTIOUS:
                    subjects[subject - 1].state = ISOLATED
        elif event_type == "Exit_Isolation":
            for subject in involved_subjects:
                if subjects[subject - 1].state == ISOLATED:
                    subjects[subject - 1].state = HEALED
                    subjects[subject - 1].symptoms = ASYMPTOMATIC
        elif event_type == "Internal":
            infectious = any(subjects[subject - 1].state == INFECTIOUS for subject in involved_subjects)
            if infectious:
                for subject in involved_subjects:
                    if event["risk_factor"] > random.random() and subjects[subject - 1].state == HEALTHY:
                        subjects = infect_subject(
                            data,
                            subjects,
                            subject,
                            current_time,
                            n_subjects,
                            fine_grained_rng,
                            distortion,
                            parameter_bundle=parameter_bundle,
                        )
        elif event_type == "Symptoms": # XXX Sintomi
            is_symptomatic = False
            for subject in involved_subjects:
                if subjects[subject - 1].symptoms == SYMPTOMATIC:
                    is_symptomatic = True
                elif subjects[subject - 1].symptoms == DEVELOPING_SYMPTOMS:
                    is_symptomatic = random.random() < 0.5
            event["result"] = is_symptomatic
        elif event_type == "Test": # XXX Test
            is_positive = False
            for subject in involved_subjects:
                if subjects[subject - 1].state == INFECTIOUS or subjects[subject - 1].state == ISOLATED:
                    is_positive = True
                    subjects = schedule_isolation_after_positive_test(
                        data,
                        subjects,
                        subject,
                        current_time,
                        fine_grained_rng=fine_grained_rng,
                        distortion=distortion,
                        parameter_bundle=parameter_bundle,
                    )
            event["result"] = is_positive
        past_events.append(event)

    summary = {"events" : past_events, "n_subjects" : n_subjects, "time_limit" : data["time_limit"], "n_contacts" : data["n_contacts"]}
    if export_data:
        _export_data(summary, f"{exported_data_filename}.json")   

    return summary
        

def get_numerical_results(simulated_data, granularity=1.0):
    n_subjects = simulated_data["n_subjects"]
    time_limit = simulated_data["time_limit"]
    state = {i : [TRACK_HEALTHY] * int(time_limit * 24 / granularity) for i in range(1, n_subjects + 1)}

    for event in sorted(simulated_data["events"], key=lambda x: x["time"]):
        involved_subjects = event["involved_subjects"]
        current_time = int(np.round(event["time"] / granularity))
        if event["type"] == "Infectious":
            for subject in involved_subjects:
                state[subject][current_time:] = [TRACK_INFECTIOUS] * len(state[subject][current_time:])
        elif event["type"] == "Heal":
            for subject in involved_subjects:
                state[subject][current_time:] = [TRACK_HEALED] * len(state[subject][current_time:])
        elif event["type"] == "Enter_Isolation":
            for subject in involved_subjects:
                state[subject][current_time:] = [TRACK_ISOLATED] * len(state[subject][current_time:])
    return state
        



if __name__=="__main__":
    print("Running simulation")

    # Read dataset
    data_filename = "dataset_s5_t14_20.json"
    data = read_data(data_filename)
    n_subjects = data["n_subjects"]
    timelimit = data["time_limit"]

    # Sort dataset by timestamp
    data["events"].sort(key=lambda x: x["time"])

    # Print dataset
    for event in data["events"]:
        print(event)

    # run 1 iteration of the simulation
    seed = 27
    tic = time.time()
    simulate_one_iteration(data, n_subjects, False, data_filename.split(".")[0] + "_simulated", seed=seed)
    toc = time.time()
    print(f"Simulation took {toc - tic} seconds")

