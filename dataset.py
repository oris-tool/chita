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

import os
import json
import random

# Hyper-parameters
file = "path/filename"
n_subjects = {5, 10, 15}
time_limit = {14, 21, 28}

# Create a JSON dataset file.
def create_empty_json_file(filename):
    # Check if the file already exists
    if os.path.exists(filename):
        print(f"File '{filename}' already exists.")
        user_input = input("Do you want to delete the existing file and create a new one? (Y/N): ").strip().lower()
        if user_input == 'y':
            os.remove(filename)
            print(f"File '{filename}' deleted.")
            # Create an empty JSON file
            with open(filename, 'w') as file:
                json.dump({"events" : []}, file)  # Dump an empty JSON object
            print(f"File '{filename}' created successfully.")
        else:
            print("Exiting without creating a new file.")
            exit(1)
    else:
        # Create an empty JSON file
        with open(filename, 'w') as file:
            json.dump({"events" : []}, file)  # Dump an empty JSON object
        print(f"File '{filename}' created successfully.")

# Sample involved subjects for a single event.
def sample_involved_subjects(N):
    lower_bound = 2
    upper_bound = (N // 2) - 1 
    if upper_bound < lower_bound:
        upper_bound = lower_bound + 1
    n_involved_subjects = random.randint(lower_bound, upper_bound)
    return random.sample(range(1, N + 1), n_involved_subjects)

# Sample event timestamps.
def sample_timestamps(M, T, fine_grained_rng = None):
    T_hours = T * 24
    sampled = random.sample(range(0, T_hours + 1), M)
    if fine_grained_rng is not None:
        minutes = [fine_grained_rng.random() for _ in range(M)]
        sampled = [sampled[i] + minutes[i] for i in range(M)]
    return sorted(sampled)

# Sample the risk factor of a single event.
def sample_risk_factor():
    return random.uniform(0, 1)

# Sample the number of external contacts of a subject.
def sample_external_contacts(max_contacts = 4):
    return random.randint(0, max_contacts)

# Sample the number of symptoms of a subject.
def sample_symptoms(T):
    return random.randint(0, 2*T//7)

# Sample symptoms of a subject.
def get_sampled_symptoms(n_symptoms):
    symptoms = get_symptoms()  # Already in event format.
    if symptoms == None:
        return None
    sampled_symptoms = random.sample(symptoms, n_symptoms)
    return sampled_symptoms

# Run the simulation to get symptoms.
def get_symptoms():
    pass

# Sample the number of tests of a subject.
def sample_tests(T):
    return random.randint(0, 2*T//7)

# Sample tests of a subject.
def get_sampled_tests(n_tests):
    tests = get_tests()  # Already in event format.
    if tests == None:
        return None
    sampled_tests = random.sample(tests, n_tests)
    return sampled_tests

# Run the simulation to get tests.
def get_tests():
    pass

# Create the JSON event representation.
def create_event(event_type, involved_subjects, t, risk_factor = None, result = None, duration_hours = None):
    event = {
        "type": event_type,
        "involved_subjects": involved_subjects,
        "time": t,
        "risk_factor": risk_factor,
        "result": result
    }
    if duration_hours is not None:
        event["duration_hours"] = duration_hours
    return event

def create_datasets(file, n_subjects, time_limit, seed = None, fine_grained = False, internal_contacts = None, max_contacts = 4):
    """
    Create datasets with the given parameters.
    
    Parameters:
    file (str): The name of the file to create. Without extension.

    Returns:
    filenames (list): A list of the filenames created.
    """
    if seed is not None:
        random.seed(seed)
    if fine_grained:
        if seed is None:
            fine_grained_rng = random.Random()
        else:
            fine_grained_rng = random.Random(seed)
    else:
        fine_grained_rng = None
    created_files = []
    for T in time_limit:
        for N in n_subjects:
            n_contacts = internal_contacts
            if n_contacts is None:
                n_contacts = {N}
            for M in n_contacts:
                filename = f"{file}_{M}.json"
                created_files.append(filename)
                dataset_placeholder = {"events" : [], "n_subjects" : N, "time_limit" : T, "n_contacts" : M}
                timestamps = sample_timestamps(M, T, fine_grained_rng)
                for t in timestamps:
                    involved_subjects = sample_involved_subjects(N)
                    risk_factor = sample_risk_factor()
                    internal_contact = create_event("Internal", involved_subjects, t, risk_factor)
                    dataset_placeholder["events"].append(internal_contact)
                for subject in range (1, N + 1):
                    n_external_contacts = sample_external_contacts(max_contacts = max_contacts)
                    timestamps = sample_timestamps(n_external_contacts, T, fine_grained_rng)
                    for t in timestamps:
                        risk_factor = sample_risk_factor()
                        external_contact = create_event("External", [subject], t, risk_factor) 
                        dataset_placeholder["events"].append(external_contact)

                for subject in range(1, N + 1):
                    n_tests = sample_tests(T)
                    test_timestamps = sample_timestamps(n_tests, T, fine_grained_rng)
                    for t in test_timestamps:
                        test = create_event("Test", [subject], t, None, "to be defined")
                        dataset_placeholder["events"].append(test)
                 
                with open(filename, 'w') as f:
                    json.dump(dataset_placeholder, f, indent=4)
                    print(f"Dataset '{filename}' created successfully.")
    return created_files
