import os
import shutil
import subprocess
from pathlib import Path

from sweep_pipeline import (
    file_sha256,
    java_cache_label,
    java_time_step_label,
    prepare_java_precompute_cache,
    resolve_java_executable,
    resolve_optional_jar,
    sanitized_time_step_label,
    write_json,
)


_RAW_TO_JAVA_TRANSITION_KEYS = {
    "infectiousness": "infectiousness",
    "healing": "healing",
    "symptoms": "symptoms",
    "isolating": "isolating",
    "symptomsOnset": "symptomsOnset",
    "notification_to_isolation": "notificationToIsolation",
    "symptomatic_period": "symptomaticPeriod",
}

_CASE_ID_PARTS = (
    ("infectiousness", "inf"),
    ("healing", "heal"),
    ("symptoms", "sym"),
    ("isolating", "iso"),
    ("symptomsOnset", "onset"),
    ("notification_to_isolation", "notif"),
    ("symptomatic_period", "symdur"),
)


def _repo_root(repo_root=None):
    return os.path.abspath("." if repo_root is None else repo_root)


def _resolve_java_artifacts(repo_root):
    class_root = os.path.join(repo_root, "out", "production", "chita-main-test")
    stpn_analysis_class_path = os.path.join(
        class_root,
        "com",
        "chita",
        "analysis",
        "STPNAnalysis.class",
    )
    if not os.path.exists(stpn_analysis_class_path):
        raise FileNotFoundError(
            "Compiled STPNAnalysis.class not found under out/production/chita-main-test."
        )

    gson_jar = resolve_optional_jar(
        [
            os.path.join(repo_root, "lib", "gson.jar"),
            os.path.join(repo_root, "lib", "gson-2.13.1.jar"),
            os.path.join(repo_root, "lib", "gson-2.11.0.jar"),
            os.path.join(
                os.path.expanduser("~"),
                ".m2",
                "repository",
                "com",
                "google",
                "code",
                "gson",
                "gson",
                "2.13.1",
                "gson-2.13.1.jar",
            ),
            os.path.join(
                os.path.expanduser("~"),
                ".m2",
                "repository",
                "com",
                "google",
                "code",
                "gson",
                "gson",
                "2.11.0",
                "gson-2.11.0.jar",
            ),
            os.path.join(
                os.path.expanduser("~"),
                ".gradle",
                "caches",
                "modules-2",
                "files-2.1",
                "com.google.code.gson",
                "gson",
                "2.10.1",
                "b3add478d4382b78ea20b1671390a858002feb6c",
                "gson-2.10.1.jar",
            ),
        ],
        "gson",
    )
    classpath = os.pathsep.join(
        [
            class_root,
            os.path.join(repo_root, "lib", "*"),
            gson_jar,
        ]
    )
    return {
        "class_root": class_root,
        "classpath": classpath,
        "stpn_analysis_class_path": stpn_analysis_class_path,
    }


def _build_case_id(raw_parameters):
    labels = []
    for raw_key, prefix in _CASE_ID_PARTS:
        spec = raw_parameters.get(raw_key, {})
        labels.append(f"{prefix}_{spec.get('level', 'custom')}")
    return "__".join(labels)


def _convert_raw_parameter_bundle(raw_parameters, case_id=None):
    levels = {}
    transitions = {}

    for raw_key, java_key in _RAW_TO_JAVA_TRANSITION_KEYS.items():
        if raw_key not in raw_parameters:
            raise KeyError(f"Missing parameter transition '{raw_key}'.")
        spec = raw_parameters[raw_key]
        level = spec.get("level", "custom")
        levels[java_key] = level

        if raw_key == "symptoms":
            probability = float(spec["p"])
            transitions[java_key] = {
                "unit_measure": "probability",
                "true": probability,
                "false": 1.0 - probability,
            }
        elif raw_key == "notification_to_isolation":
            transitions[java_key] = {
                "unit_measure": "hours",
                "distribution": "hyperexponential",
                "p1": float(spec["p"]),
                "p2": 1.0 - float(spec["p"]),
                "lambda1": float(spec["lambda1"]),
                "lambda2": float(spec["lambda2"]),
            }
        elif raw_key == "symptomatic_period":
            transitions[java_key] = {
                "unit_measure": "hours",
                "erlang_stages": int(spec["n"]),
                "erlang_lambda": float(spec["lambdaErl"]),
                "exponential_lambda": float(spec["lambdaExp"]),
            }
        else:
            transitions[java_key] = {
                "unit_measure": "hours",
                "erlang_stages": int(spec["n"]),
                "erlang_lambda": float(spec["lambdaErl"]),
                "exponential_lambda": float(spec["lambdaExp"]),
            }

    return {
        "unit_measure": "mixed",
        "case_id": case_id or _build_case_id(raw_parameters),
        "levels": levels,
        "transitions": transitions,
    }


def normalize_stpn_parameter_bundle(parameter_bundle, case_id=None):
    if "transitions" in parameter_bundle:
        normalized = dict(parameter_bundle)
        if case_id is not None:
            normalized["case_id"] = case_id
        normalized.setdefault("case_id", "custom_parameter_bundle")
        normalized.setdefault("unit_measure", "mixed")
        return normalized
    return _convert_raw_parameter_bundle(parameter_bundle, case_id=case_id)


def _run_java_stpn_command(
    repo_root,
    command_args,
    working_dir,
):
    java_executable = resolve_java_executable()
    java_artifacts = _resolve_java_artifacts(repo_root)
    command = [
        java_executable,
        "-cp",
        java_artifacts["classpath"],
        "com.chita.analysis.STPNAnalysis",
        *command_args,
    ]
    result = subprocess.run(
        command,
        cwd=working_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Java STPNAnalysis failed.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    return result


def precompute_stpn_solution(
    parameter_bundle,
    cache_dir=None,
    repo_root=None,
    time_step_hours=1,
    case_id=None,
):
    repo_root = _repo_root(repo_root)
    normalized_bundle = normalize_stpn_parameter_bundle(parameter_bundle, case_id=case_id)
    java_artifacts = _resolve_java_artifacts(repo_root)
    cache_entry = prepare_java_precompute_cache(
        repo_root=repo_root,
        parameter_bundle=normalized_bundle,
        time_step_hours=time_step_hours,
        java_class_fingerprint=file_sha256(java_artifacts["stpn_analysis_class_path"]),
    )

    if cache_dir is not None:
        cache_dir = os.path.abspath(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        cache_entry["cache_dir"] = cache_dir
        cache_entry["parameter_bundle_path"] = os.path.join(cache_dir, "parameter_bundle.json")
        cache_entry["stpn_solution_path"] = os.path.join(
            cache_dir,
            cache_entry["stpn_solution_filename"],
        )
        cache_entry["observation_curve_path"] = os.path.join(
            cache_dir,
            cache_entry["observation_curve_filename"],
        )
        cache_entry["manifest_path"] = os.path.join(cache_dir, "cache_manifest.json")

    write_json(cache_entry["parameter_bundle_path"], normalized_bundle)
    write_json(
        cache_entry["manifest_path"],
        {
            "case_id": normalized_bundle["case_id"],
            "time_step_hours": time_step_hours,
            "parameter_bundle_path": cache_entry["parameter_bundle_path"],
            "stpn_solution_path": cache_entry["stpn_solution_path"],
            "observation_curve_path": cache_entry["observation_curve_path"],
        },
    )

    cache_hit = (
        os.path.exists(cache_entry["stpn_solution_path"])
        and os.path.exists(cache_entry["observation_curve_path"])
    )
    if not cache_hit:
        result = _run_java_stpn_command(
            repo_root=repo_root,
            command_args=[
                "--time-step",
                str(time_step_hours),
                "--stpn-solution-path",
                os.path.abspath(cache_entry["stpn_solution_path"]),
                "--parameter-bundle",
                os.path.abspath(cache_entry["parameter_bundle_path"]),
                "--precompute-only",
            ],
            working_dir=cache_entry["cache_dir"],
        )
    else:
        result = None

    return {
        "parameter_bundle": normalized_bundle,
        "parameter_bundle_path": cache_entry["parameter_bundle_path"],
        "cache_dir": cache_entry["cache_dir"],
        "stpn_solution_path": cache_entry["stpn_solution_path"],
        "observation_curve_path": cache_entry["observation_curve_path"],
        "cache_hit": cache_hit,
        "stdout": None if result is None else result.stdout,
        "stderr": None if result is None else result.stderr,
    }


def run_stpn_analysis(
    parameter_bundle,
    analysis_dir,
    iterations,
    cache_dir=None,
    repo_root=None,
    time_step_hours=1,
    case_id=None,
    require_simulated_json=True,
):
    repo_root = _repo_root(repo_root)
    analysis_dir = os.path.abspath(analysis_dir)
    os.makedirs(analysis_dir, exist_ok=True)

    precomputed = precompute_stpn_solution(
        parameter_bundle=parameter_bundle,
        cache_dir=cache_dir,
        repo_root=repo_root,
        time_step_hours=time_step_hours,
        case_id=case_id,
    )

    if require_simulated_json:
        simulated_files = list(Path(analysis_dir).rglob("*simulated.json"))
        if not simulated_files:
            raise FileNotFoundError(
                f"No '*simulated.json' files found under {analysis_dir}. "
                "STPNAnalysis scans its working directory recursively for those files."
            )

    local_observation_curve_path = os.path.join(
        analysis_dir,
        os.path.basename(precomputed["observation_curve_path"]),
    )
    if os.path.abspath(local_observation_curve_path) != os.path.abspath(
        precomputed["observation_curve_path"]
    ):
        shutil.copy2(precomputed["observation_curve_path"], local_observation_curve_path)

    result = _run_java_stpn_command(
        repo_root=repo_root,
        command_args=[
            "--time-step",
            str(time_step_hours),
            "--iterations",
            str(iterations),
            "--stpn-solution-path",
            os.path.abspath(precomputed["stpn_solution_path"]),
            "--parameter-bundle",
            os.path.abspath(precomputed["parameter_bundle_path"]),
        ],
        working_dir=analysis_dir,
    )

    return {
        "parameter_bundle": precomputed["parameter_bundle"],
        "parameter_bundle_path": precomputed["parameter_bundle_path"],
        "analysis_dir": analysis_dir,
        "iterations": iterations,
        "time_step_hours": time_step_hours,
        "stpn_solution_path": precomputed["stpn_solution_path"],
        "observation_curve_path": precomputed["observation_curve_path"],
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
