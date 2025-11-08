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

import contextlib
import dataset
import simulation
from tqdm import tqdm
import json
import os
import argparse

def get_infectious_simulated_data(numerical_results, avg_state):
    bin_data = {}
    for subject, results in numerical_results.items():
        bin_data[subject] = [1 if value == 1 else 0 for value in results]
    return bin_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulations")
    parser.add_argument("--rep", type=int, default=10_000, help="Number of repetitions per configuration")
    args = parser.parse_args()

    FINE_GRAINED = True
    if FINE_GRAINED:
        GRANULARITY = 0.1
    else:
        GRANULARITY = 1.0

    rep = args.rep
    timelimits = {84}
    subjects = {15}
    total_iterations = len(timelimits) * len(subjects) * 3 * rep
    datasets = ["D0", "D1", "D2", "D2+15", "D2-15", "D2+25", "D2-25", "D3"]
    total_iterations = len(datasets) * rep
    seeds = [i for i in range(2, len(datasets) + 2)]
    seed_index = 0
    with tqdm(total = total_iterations, desc="Progress") as pbar:
        for timelimit in sorted(timelimits):
            for n_subjects in sorted(subjects):
                for dataset_path in datasets:
                    if dataset_path == "":
                        seed_index += 1
                        continue
                    if dataset_path == "D0":
                        n_subjects = 8
                        internal_contacts = {84}
                    elif dataset_path == "D1":
                        n_subjects = 4
                        internal_contacts = {84}
                    elif dataset_path == "D2" or dataset_path == "D2+15" or dataset_path == "D2-15" or dataset_path == "D2+25" or dataset_path == "D2-25":
                        n_subjects = 8
                        internal_contacts = {84}
                    elif dataset_path == "D3":
                        n_subjects = 16
                        internal_contacts = {84}
                    os.makedirs(f"{dataset_path}", exist_ok=True)
                    filenames = dataset.create_datasets(f"{dataset_path}/dataset_s{n_subjects}_t{timelimit}", [n_subjects], [timelimit], seeds[seed_index], FINE_GRAINED, internal_contacts = internal_contacts, max_contacts = 6)
                    for filename in filenames:
                        print("\033[91m" + filename + "\033[0m")
                        avg_state ={i : [0] * int(timelimit * 24 / GRANULARITY) for i in range(1, n_subjects + 1)}
                        save = True
                        for i in range(rep):
                            with contextlib.redirect_stdout(None):
                                data = simulation.read_data(f"{filename}")
                                data["events"].sort(key=lambda x: x["time"])
                                if i == 0:
                                    distortion = 1.0
                                    if dataset_path == "D2+15":
                                        distortion = 1.15
                                    elif dataset_path == "D2-15":
                                        distortion = 0.85
                                    elif dataset_path == "D2+25":
                                        distortion = 1.25
                                    elif dataset_path == "D2-25":
                                        distortion = 0.75
                                    simulated_data = simulation.simulate_one_iteration(data, n_subjects, save, f"{filename.split('.')[0]}_simulated", seeds[seed_index], distortion=distortion)
                                else:
                                    simulated_data  = simulation.simulate_one_iteration(data, n_subjects, save, f"{filename.split('.')[0]}_simulated", distortion=1.0)
                                save = False
                                numerical_results = simulation.get_numerical_results(simulated_data, GRANULARITY)
                                infectious_results = get_infectious_simulated_data(numerical_results, avg_state)
                                for subject in infectious_results:
                                    for j in range(int(timelimit * 24 / GRANULARITY)):
                                        avg_state[subject][j] += infectious_results[subject][j]
                                pbar.update(1)
                        avg_results = {i : [avg_state[i][j] / rep for j in range(int(timelimit * 24 / GRANULARITY))] for i in range(1, n_subjects + 1)}
                        filename_surgery = filename.split('_t')[0] + f"_t{timelimit}_{internal_contacts.pop()}_simulated_{rep}_reps.json" 
                        with open(filename_surgery, "w") as f:
                            json.dump(avg_results, f, indent=4)
                    seed_index += 1