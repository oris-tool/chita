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
import math
import random
import hashlib
import dataset
import time
import numpy as np
from pathlib import Path

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

THETA_CACHE_VERSION = 1
THETA_TIME_LIMIT_DAYS = 60.0
THETA_BASE_DT_DAYS = 0.05
THETA_TEST_GAMMA_SHAPE = 7.85
THETA_TEST_GAMMA_SCALE = 2.14
THETA_TEST_GAMMA_SCALING = 0.94
DEFAULT_THETA_TIME_STEP_HOURS = 1.0
THETA_CACHE_DIR = Path(".cache")
THETA_MEMORY_CACHE = {}


def _theta_default_symptoms_onset_spec():
    # Matches Java default analysis parameters for symptomsOnset.
    return {
        "unit_measure": "hours",
        "erlang_stages": 2,
        "erlang_lambda": 0.02255375,
        "exponential_lambda": 0.01341875,
    }


def _normalize_unit_measure(unit_measure):
    normalized = str(unit_measure or "").strip().lower()
    if normalized in {"hour", "hours"}:
        return "hours"
    if normalized in {"day", "days"}:
        return "days"
    raise ValueError(f"Unsupported unit measure: {unit_measure}")


def _convert_rate(rate, unit_measure, target_unit_measure):
    source = _normalize_unit_measure(unit_measure)
    target = _normalize_unit_measure(target_unit_measure)
    if source == target:
        return float(rate)
    if source == "hours" and target == "days":
        return float(rate) * HOURS_PER_DAY
    if source == "days" and target == "hours":
        return float(rate) / HOURS_PER_DAY
    raise ValueError(f"Cannot convert rate from {unit_measure} to {target_unit_measure}")


def _clamp01(value):
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return max(0.0, min(1.0, value))


def _log_factorial(n):
    result = 0.0
    for i in range(2, n + 1):
        result += math.log(i)
    return result


def _erlang_pdf(t_days, stages, rate):
    if t_days < 0.0 or stages <= 0 or rate <= 0.0:
        return 0.0
    if t_days == 0.0:
        return rate if stages == 1 else 0.0
    log_pdf = (
        stages * math.log(rate)
        + (stages - 1) * math.log(t_days)
        - rate * t_days
        - _log_factorial(stages - 1)
    )
    return math.exp(log_pdf)


def _exponential_pdf(t_days, rate):
    if t_days < 0.0 or rate <= 0.0:
        return 0.0
    return rate * math.exp(-rate * t_days)


def _convolve(a, b, length, dt_days):
    result = np.zeros(length, dtype=float)
    for i in range(length):
        summed = 0.0
        for j in range(i + 1):
            summed += a[j] * b[i - j]
        result[i] = summed * dt_days
    return result


def _sample_base_curve(curve, t_days):
    if t_days < 0.0:
        return 0.0
    position = t_days / THETA_BASE_DT_DAYS
    lower_index = int(math.floor(position))
    if lower_index < 0:
        return 0.0
    if lower_index >= len(curve) - 1:
        return float(curve[lower_index]) if lower_index == len(curve) - 1 else 0.0
    fraction = position - lower_index
    return float(curve[lower_index] * (1.0 - fraction) + curve[lower_index + 1] * fraction)


def _regularized_gamma_p_series(a, x):
    max_iterations = 10000
    epsilon = 1e-14
    summed = 1.0 / a
    term = summed

    for n in range(1, max_iterations + 1):
        term *= x / (a + n)
        summed += term
        if abs(term) < abs(summed) * epsilon:
            log_term = -x + a * math.log(x) - math.lgamma(a)
            return _clamp01(summed * math.exp(log_term))

    log_term = -x + a * math.log(x) - math.lgamma(a)
    return _clamp01(summed * math.exp(log_term))


def _regularized_gamma_q_continued_fraction(a, x):
    max_iterations = 10000
    epsilon = 1e-14
    fp_min = 1e-300
    b = x + 1.0 - a
    c = 1.0 / fp_min
    d = 1.0 / max(b, fp_min)
    h = d

    for i in range(1, max_iterations + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < fp_min:
            d = fp_min
        c = b + an / c
        if abs(c) < fp_min:
            c = fp_min
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < epsilon:
            break

    log_term = -x + a * math.log(x) - math.lgamma(a)
    return _clamp01(math.exp(log_term) * h)


def _regularized_gamma_q(a, x):
    if a <= 0.0 or x < 0.0:
        return float("nan")
    if x == 0.0:
        return 1.0
    if x < a + 1.0:
        return 1.0 - _regularized_gamma_p_series(a, x)
    return _regularized_gamma_q_continued_fraction(a, x)


def _gamma_survival(shape, scale, t_days):
    if t_days <= 0.0:
        return 1.0
    return _regularized_gamma_q(shape, t_days / scale)


def _generalized_erlang_pdf(erlang_stages, erlang_rate_per_day, exp_rate_per_day, samples):
    erlang_pdf = np.zeros(samples, dtype=float)
    exp_pdf = np.zeros(samples, dtype=float)
    for i in range(samples):
        t_days = i * THETA_BASE_DT_DAYS
        erlang_pdf[i] = _erlang_pdf(t_days, erlang_stages, erlang_rate_per_day)
        exp_pdf[i] = _exponential_pdf(t_days, exp_rate_per_day)
    return _convolve(erlang_pdf, exp_pdf, samples, THETA_BASE_DT_DAYS)


def _base_curve_samples():
    return int(math.ceil(THETA_TIME_LIMIT_DAYS / THETA_BASE_DT_DAYS))


def _theta_horizon_steps(time_step_hours):
    return int(math.ceil(THETA_TIME_LIMIT_DAYS * HOURS_PER_DAY / float(time_step_hours)))


def _symptoms_onset_spec_from_parameter_bundle(parameter_bundle):
    transition_parameters = _transition_parameters(parameter_bundle)
    if transition_parameters is None:
        return _theta_default_symptoms_onset_spec()

    symptoms_onset_spec = (
        transition_parameters.get("symptomsOnset")
        or transition_parameters.get("symptoms_onset")
    )
    if symptoms_onset_spec is None:
        return _theta_default_symptoms_onset_spec()
    return symptoms_onset_spec


def _theta_signature(symptoms_onset_spec, time_step_hours):
    unit_measure = _normalize_unit_measure(symptoms_onset_spec.get("unit_measure", "hours"))
    payload = {
        "version": THETA_CACHE_VERSION,
        "time_limit_days": THETA_TIME_LIMIT_DAYS,
        "base_dt_days": THETA_BASE_DT_DAYS,
        "shape": THETA_TEST_GAMMA_SHAPE,
        "scale": THETA_TEST_GAMMA_SCALE,
        "scaling": THETA_TEST_GAMMA_SCALING,
        "time_step_hours": float(time_step_hours),
        "onset": {
            "unit_measure": unit_measure,
            "erlang_stages": int(_transition_value(symptoms_onset_spec, "erlang_stages")),
            "erlang_lambda": float(_transition_value(symptoms_onset_spec, "erlang_lambda")),
            "exponential_lambda": float(_transition_value(symptoms_onset_spec, "exponential_lambda")),
        },
    }
    signature_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    signature_hash = hashlib.sha1(signature_json.encode("utf-8")).hexdigest()
    return signature_hash, payload


def _theta_cache_paths(signature_hash):
    base_path = THETA_CACHE_DIR / f"theta_curve_{signature_hash}"
    return base_path.with_suffix(".json"), base_path.with_suffix(".npy")


def _load_cached_theta_curve(signature_hash, expected_payload, expected_horizon_steps):
    metadata_path, values_path = _theta_cache_paths(signature_hash)
    if not metadata_path.exists() or not values_path.exists():
        return None

    try:
        with open(metadata_path, "r", encoding="utf-8") as metadata_file:
            cached_payload = json.load(metadata_file)
        if cached_payload != expected_payload:
            return None
        curve = np.load(values_path)
        if curve.ndim != 1 or len(curve) != expected_horizon_steps:
            return None
        return curve.astype(float)
    except (OSError, ValueError, TypeError):
        return None


def _write_cached_theta_curve(signature_hash, payload, theta_curve):
    THETA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path, values_path = _theta_cache_paths(signature_hash)
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(payload, metadata_file, indent=2, sort_keys=True)
    np.save(values_path, np.asarray(theta_curve, dtype=float))


def _compute_theta_curve(symptoms_onset_spec, time_step_hours, horizon_steps):
    erlang_stages = int(_transition_value(symptoms_onset_spec, "erlang_stages"))
    unit_measure = symptoms_onset_spec.get("unit_measure", "hours")
    erlang_rate_per_day = _convert_rate(
        float(_transition_value(symptoms_onset_spec, "erlang_lambda")),
        unit_measure,
        "days",
    )
    exp_rate_per_day = _convert_rate(
        float(_transition_value(symptoms_onset_spec, "exponential_lambda")),
        unit_measure,
        "days",
    )

    base_samples = _base_curve_samples()
    onset_pdf = _generalized_erlang_pdf(erlang_stages, erlang_rate_per_day, exp_rate_per_day, base_samples)
    positive_since_symptom_onset = np.zeros(base_samples, dtype=float)
    for i in range(base_samples):
        t_days = i * THETA_BASE_DT_DAYS
        positive_since_symptom_onset[i] = _clamp01(
            THETA_TEST_GAMMA_SCALING
            * _gamma_survival(THETA_TEST_GAMMA_SHAPE, THETA_TEST_GAMMA_SCALE, t_days)
        )

    theta_base = _convolve(onset_pdf, positive_since_symptom_onset, base_samples, THETA_BASE_DT_DAYS)
    theta_curve = np.zeros(horizon_steps, dtype=float)
    for i in range(horizon_steps):
        t_days = i * float(time_step_hours) / HOURS_PER_DAY
        theta_curve[i] = _clamp01(_sample_base_curve(theta_base, t_days))

    return theta_curve


def _load_or_create_theta_curve(parameter_bundle, time_step_hours=DEFAULT_THETA_TIME_STEP_HOURS):
    symptoms_onset_spec = _symptoms_onset_spec_from_parameter_bundle(parameter_bundle)
    signature_hash, payload = _theta_signature(symptoms_onset_spec, time_step_hours)
    horizon_steps = _theta_horizon_steps(time_step_hours)
    memory_key = (signature_hash, horizon_steps)

    if memory_key in THETA_MEMORY_CACHE:
        return THETA_MEMORY_CACHE[memory_key]

    cached_curve = _load_cached_theta_curve(signature_hash, payload, horizon_steps)
    if cached_curve is not None:
        THETA_MEMORY_CACHE[memory_key] = cached_curve
        return cached_curve

    theta_curve = _compute_theta_curve(symptoms_onset_spec, time_step_hours, horizon_steps)
    _write_cached_theta_curve(signature_hash, payload, theta_curve)
    THETA_MEMORY_CACHE[memory_key] = theta_curve
    return theta_curve


def _theta_value_for_elapsed_time(theta_curve, elapsed_hours, time_step_hours):
    if elapsed_hours is None or elapsed_hours < 0.0:
        return 0.0
    sample_index = int(round(float(elapsed_hours) / float(time_step_hours)))
    if sample_index < 0 or sample_index >= len(theta_curve):
        return 0.0
    return _clamp01(float(theta_curve[sample_index]))


def _transition_parameters(parameter_bundle):
    if parameter_bundle is None:
        return None
    if "transitions" in parameter_bundle:
        return parameter_bundle["transitions"]
    return parameter_bundle


def _transition_value(transition_spec, key):
    if key in transition_spec:
        return transition_spec[key]

    # Backward-compatible aliases for raw parameter bundles used by sweep scripts.
    alias_map = {
        "erlang_stages": ["n"],
        "erlang_lambda": ["lambdaErl", "lambda_erl"],
        "exponential_lambda": ["lambdaExp", "lambda_exp"],
        "p1": ["p"],
        "p2": [],
        "true": ["p"],
        "false": [],
    }
    for alias in alias_map.get(key, []):
        if alias in transition_spec:
            if key == "p1" and alias == "p":
                return float(transition_spec[alias])
            return transition_spec[alias]

    # Raw hyperexponential specs often provide only p; derive p2 as 1-p.
    if key == "p2" and "p" in transition_spec:
        return 1.0 - float(transition_spec["p"])

    # Raw switch specs often provide only p; derive false as 1-p.
    if key == "false" and "p" in transition_spec:
        return 1.0 - float(transition_spec["p"])

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
        self.infected_time = None

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


def safe_print_event(event):
    pass
    # try:
    #     print_event(event)
    # except (ValueError, OSError, BrokenPipeError):
    #     # Some sweep paths suppress/redirect stdout and may close it early.
    #     # Event printing is best-effort and should never fail the simulation.
    #     pass


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
    subjects[subject_id-1].infected_time = current_time
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
    data["events"].sort(key=lambda x: x["time"])
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
    data["events"].sort(key=lambda x: x["time"])
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
    data["events"].sort(key=lambda x: x["time"])
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

            data["events"].sort(key=lambda x: x["time"])
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
    theta_time_step_hours=DEFAULT_THETA_TIME_STEP_HOURS,
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

    theta_curve = _load_or_create_theta_curve(
        parameter_bundle=parameter_bundle,
        time_step_hours=theta_time_step_hours,
    )

    subjects = []
    for i in range(n_subjects):
        subjects.append(Subject(i, HEALTHY))
    past_events = []
    while data["events"]:
        event = data["events"].pop(0)
        safe_print_event(event)
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
                        pass

        elif event_type == "Heal":
            for subject in involved_subjects:
                if subjects[subject - 1].state == INFECTIOUS:
                    subjects[subject - 1].state = HEALED
                    subjects[subject - 1].symptoms = ASYMPTOMATIC
        elif event_type == "Develop_Symptoms":
            for subject in involved_subjects:
                if subjects[subject - 1].symptoms == DEVELOPING_SYMPTOMS:
                    subjects[subject - 1].symptoms = SYMPTOMATIC     
                    subjects = set_isolation_time(
                            data,
                            subjects,
                            subject,
                            current_time,
                            fine_grained_rng,
                            distortion,
                            parameter_bundle=parameter_bundle,
                        )
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
        elif event_type == "Symptoms":
            is_symptomatic = False
            for subject in involved_subjects:
                if subjects[subject - 1].symptoms == SYMPTOMATIC:
                    is_symptomatic = True
                elif subjects[subject - 1].symptoms == DEVELOPING_SYMPTOMS:
                    is_symptomatic = random.random() < 0.5
            event["result"] = is_symptomatic
        elif event_type == "Test":
            is_positive = False
            for subject in involved_subjects:
                subject_state = subjects[subject - 1].state
                subject_positive = False

                if subject_state in (INFECTIOUS, ISOLATED):
                    infected_time = subjects[subject - 1].infected_time
                    elapsed_hours = None if infected_time is None else (current_time - infected_time)
                    theta_value = _theta_value_for_elapsed_time(
                        theta_curve,
                        elapsed_hours,
                        theta_time_step_hours,
                    )
                    subject_positive = random.random() < theta_value
                elif subject_state == HEALED:
                    subject_positive = random.random() < 0.025
                else:
                    subject_positive = random.random() < 0.05

                if subject_positive:
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
