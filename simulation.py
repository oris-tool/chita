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

class Subject:
    def __init__(self, id, state, symptoms = 0):
        """
        id: int
        state: int (0 = healthy, 1 = infected, 2 = infectious, 3 = healed, 4 = isolated)
        symptoms: int (0 = no symptoms, 1 = developing symptoms, 2 = symptomatic)
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


def infect_subject(data, subjects, subject_id, current_time, N, fine_grained_rng=None, distortion=1.0):
    if abs(distortion - 1.0) < 1e-6: 
        # PARAMETERS
        ## Develop Symptoms
        p1_develop_symptoms = 0.81
        lambda1_develop_symptoms = 0.6958 / 24.0
        p2_develop_symptoms = 0.19
        lambda2_develop_symptoms = 0.1626 / 24.0
        ## Infectious
        ### First part is like Develop Symptoms
        ### Second part:
        p1_infectious = 0.89
        p2_infectious = 0.11
        lambda1_infectious = 1.357/24.0
        lambda2_infectious = 0.170/24.0
    elif abs(distortion - 1.15) < 1e-6:
        p1_develop_symptoms = 0.76
        lambda1_develop_symptoms = 0.56385 / 24.0
        p2_develop_symptoms = 0.24
        lambda2_develop_symptoms = 0.182557 / 24.0
        ## Infectious
        p1_infectious = 0.85662
        p2_infectious = 0.14338
        lambda1_infectious = 1.137232/24.0
        lambda2_infectious = 0.190348/24.0
    elif abs(distortion - 0.85) < 1e-6:
        p1_develop_symptoms = 0.86
        lambda1_develop_symptoms = 0.867248 / 24.0
        p2_develop_symptoms = 0.14
        lambda2_develop_symptoms = 0.142598 / 24.0
        ## Infectious
        p1_infectious = 0.9177
        p2_infectious = 0.082286
        lambda1_infectious = 1.648341/24.0
        lambda2_infectious = 0.147798/24.0
    elif abs(distortion - 1.25) < 1e-6:
        p1_develop_symptoms = 0.713
        lambda1_develop_symptoms = 0.4897 / 24.0
        p2_develop_symptoms = 0.287
        lambda2_develop_symptoms = 0.1973 / 24.0
        ## Infectious
        p1_infectious = 0.833
        p2_infectious = 0.167
        lambda1_infectious = 1.0177/24.0
        lambda2_infectious = 0.2036/24.0
    elif abs(distortion - 0.75) < 1e-6:
        p1_develop_symptoms = 0.888
        lambda1_develop_symptoms = 1.0158 / 24.0
        p2_develop_symptoms = 0.112
        lambda2_develop_symptoms = 0.1286 / 24.0
        ## Infectious
        p1_infectious = 0.935
        p2_infectious = 0.065
        lambda1_infectious = 1.903/24.0
        lambda2_infectious = 0.132/24.0
        
    symptomatic_threshold = 0.35
    generate_symptoms_threshold = 0.75


    subjects[subject_id-1].state = 1
    if random.random() < symptomatic_threshold:
        subjects[subject_id-1].symptoms = 0 # asymptomatic
    else:
        subjects[subject_id-1].symptoms = 1
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
                event = dataset.create_event("Symptoms", [subject_id], current_time + event_offset, risk_factor=None, result=None)
                data["events"].append(event)


    infectious_in = sample_from_hyper_exp(p1_develop_symptoms, p2_develop_symptoms, lambda1_develop_symptoms, lambda2_develop_symptoms) + sample_from_hyper_exp(p1_infectious, p2_infectious, lambda1_infectious, lambda2_infectious)
    event = dataset.create_event("Infectious", [subject_id], current_time + infectious_in, risk_factor=None, result=None)
    data["events"].append(event)
    data["events"].sort(key=lambda x: x["time"])
    print(f"\033[1mSubject {subject_id} is now infected\033[0m")
    return subjects


def set_healing_time(data, subjects, subject_id, current_time, distortion=1.0):
    subjects[subject_id-1].state = 2
    # PARAMETERS
    if abs(distortion - 1.0) < 1e-6:
        ## Generalized Erlang
        lambda1 = 1 / (10.68 * 24.0) # 0.093633 / 24.0
        lambda2 = 1 / (1.27 * 24.0) # 0.787227 / 24.0
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
    data["events"].sort(key=lambda x: x["time"])
    print(f"\033[1mSubject {subject_id} is now infectious\033[0m")
    return subjects

def set_isolation_time(data, subjects, subject_id, current_time, fine_grained_rng=None, distortion=1.0):
    min_isolate_in_hours = 0
    max_isolate_in_hours = int(24 * distortion)
    # max_isolate_in_hours = int(24)
    isolate_in = random.randint(min_isolate_in_hours, max_isolate_in_hours)
    isolate_for = random.randint(isolate_in, max_isolate_in_hours * 21)
    if fine_grained_rng is not None:
        isolate_in += fine_grained_rng.random()
        isolate_for += fine_grained_rng.random()
    subjects[subject_id-1].state = 2 # so they get isolated
    event = dataset.create_event("Enter_Isolation", [subject_id], current_time + isolate_in, risk_factor=None, result=None)
    data["events"].append(event)
    event = dataset.create_event("Exit_Isolation", [subject_id], current_time + isolate_for, risk_factor=None, result=None)
    data["events"].append(event)
    data["events"].sort(key=lambda x: x["time"])
    return subjects

def _export_data(data, filename):
    data["events"].sort(key=lambda x: x["time"])
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"Data exported to {filename}")



    

def simulate_one_iteration(data, n_subjects, export_data=False, exported_data_filename=None, seed = None, fine_grained = False, distortion = 1.0):
    # Create a list of subjects
    if seed is not None:
        random.seed(seed)
    if fine_grained:
        if seed is not None:
            fine_grained_rng = random.Random(seed)
        else:
            fine_grained_rng = random.Random()
    else:
        fine_grained_rng = None

    subjects = []
    for i in range(n_subjects):
        subjects.append(Subject(i, 0))
    past_events = []
    while data["events"]:
        event = data["events"].pop(0)
        print_event(event)
        # Get involved subjects
        involved_subjects = event["involved_subjects"]
        type = event["type"]
        current_time = event["time"]
        # Infection through external event
        if event["type"] == "External":
            if event["risk_factor"] > random.random(): # XXX
                for subject in involved_subjects:
                    if subjects[subject - 1].state == 0:
                        subjects = infect_subject(data, subjects, subject, current_time, n_subjects, fine_grained_rng, distortion) # subjects[subject - 1].state : 0 -> 1
        elif event["type"] == "Infectious":
            for subject in involved_subjects:
                print(subject)
                print("state", subjects[subject-1].state)
                if subjects[subject - 1].state == 1:
                    if subjects[subject - 1].symptoms == 0:
                        subjects = set_healing_time(data, subjects, subject, current_time, distortion) # subjects[subject - 1].state : 1 -> 3. Subject is asymptomatic
                    elif subjects[subject - 1].symptoms == 1 or subjects[subject - 1].symptoms == 2:
                        subjects = set_isolation_time(data, subjects, subject, current_time, fine_grained_rng, distortion) # subjects[subject - 1].state : 1 -> 2 -> 4. Subject is symptomatic
        elif event["type"] == "Heal":
            for subject in involved_subjects:
                if subjects[subject - 1].state == 2:
                    subjects[subject - 1].state = 3
                    print(f"\033[1mSubject {subject} is now healed\033[0m")
                    # Reset symptoms when subject is healed
                    subjects[subject - 1].symptoms = 0
        elif event["type"] == "Develop_Symptoms":
            for subject in involved_subjects:
                if subjects[subject - 1].symptoms == 1:
                    subjects[subject - 1].symptoms = 2
                    print(f"\033[1mSubject {subject} is now symptomatic\033[0m")
        elif event["type"] == "Enter_Isolation":
            for subject in involved_subjects:
                if subjects[subject - 1].state == 2:
                    subjects[subject - 1].state = 4
                    print(f"\033[1mSubject {subject} is now isolated\033[0m")
        elif event["type"] == "Exit_Isolation":
            for subject in involved_subjects:
                if subjects[subject - 1].state == 4:
                    subjects[subject - 1].state = 3
                    print(f"\033[1mSubject {subject} is now healthy\033[0m")
                    # Reset symptoms when subject is healed
                    subjects[subject - 1].symptoms = 0
        elif event["type"] == "Internal":
            # Remove isolated and immune subjects from the event
            # involved_subjects = [subject for subject in involved_subjects if subjects[subject - 1].state == 0 or subjects[subject - 1].state == 2] # only healthy and infectious subjects
            # Check if at least one subject is infectious
            infectious = False
            for subject in involved_subjects:
                if subjects[subject - 1].state == 2:
                    infectious = True
                    break
            
            # If at least one subject is infectious, infect the others
            if infectious:
                for subject in involved_subjects:
                    if event["risk_factor"] > random.random() and subjects[subject - 1].state == 0: # XXX
                        subjects = infect_subject(data, subjects, subject, current_time, n_subjects, fine_grained_rng, distortion) # subject[subject - 1].state : 0 -> 1
                        # input("--->")
        elif event["type"] == "Symptoms":
            # Check if subject is symptomatic. if asymptomatic (0) -> symptoms = False, if they are developing symptoms (1) -> symptoms = random, if they is symptomatic (2) -> symptoms = True
            is_symptomatic = False
            for subject in involved_subjects:
                if subjects[subject - 1].symptoms == 2:
                    is_symptomatic = True
                    print(f"\033[1mSubject {subject} is symptomatic\033[0m")
                elif subjects[subject - 1].symptoms == 1:
                    is_symptomatic = random.random() < 0.5
                    print(f"\033[1mSubject {subject} is not symptomatic\033[0m")
                else:
                    print(f"\033[1mSubject {subject} is not symptomatic\033[0m")
            event["result"] = is_symptomatic
        elif event["type"] == "Test":
            is_positive = False
            for subject in involved_subjects:
                if subjects[subject - 1].state == 2 or subjects[subject - 1].state == 4:
                    is_positive = True
                    print(f"\033[38;5;208m\033[3mSubject {subject} is positive\033[0m")
                else:
                    print(f"\033[38;5;30m\033[3mSubject {subject} is negative\033[0m")
            event["result"] = is_positive
        past_events.append(event)

    summary = {"events" : past_events, "n_subjects" : n_subjects, "time_limit" : data["time_limit"], "n_contacts" : data["n_contacts"]}
    if export_data:
        _export_data(summary, f"{exported_data_filename}.json")   

    return summary
        

def get_numerical_results(simulated_data, granularity=1.0):
    # Get the number of the subjects and the time limit
    n_subjects = simulated_data["n_subjects"]
    time_limit = simulated_data["time_limit"]
    # Create n_subjects lists of 0s, each list is long time_limit * 24
    state = {i : [0] * int(time_limit * 24 / granularity) for i in range(1, n_subjects + 1)}

    decimals = 0
    if granularity < 1.0:
        decimals = int(-np.log10(granularity))
    
    # Iterate over the events of the simulated data
    for event in sorted(simulated_data["events"], key=lambda x: x["time"]):
        involved_subjects = event["involved_subjects"]
        current_time = int(np.round(event["time"], decimals) / granularity) # round to the nearest granularity
        # If the event is Infectious, set the state of the involeved subjects to 1 for the current time and all the next hours
        if event["type"] == "Infectious":
            for subject in involved_subjects:
                state[subject][current_time:] = [1] * len(state[subject][current_time:])
        # If the event is Heal, set the state of the involeved subjects to 3 for the current time and all the next hours
        elif event["type"] == "Heal":
            for subject in involved_subjects:
                state[subject][current_time:] = [3] * len(state[subject][current_time:])
        # If the event is Enter_Isolation, set the state of the involeved subjects to 4 for the current time and all the next hours
        elif event["type"] == "Enter_Isolation":
            for subject in involved_subjects:
                state[subject][current_time:] = [4] * len(state[subject][current_time:])
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

