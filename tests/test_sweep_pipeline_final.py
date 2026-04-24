import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sweep_pipeline_final import (
    DATASET_PROFILES,
    build_notes_map_and_selection_manifest,
    dataset_filename,
    refresh_selected_run_outputs,
    resolve_quartile_label_for_existing_run,
)


class SweepPipelineFinalSelectionTest(unittest.TestCase):
    def test_dataset_profiles_keep_warmup_case_for_both_families(self):
        self.assertIn(32, DATASET_PROFILES["bubble"].internal_contacts)
        self.assertIn(32, DATASET_PROFILES["scale_free"].internal_contacts)
        self.assertEqual(dataset_filename(DATASET_PROFILES["bubble"], 200), "dataset_bubble_200.json")
        self.assertEqual(dataset_filename(DATASET_PROFILES["scale_free"], 2500), "dataset_scale_free_2500.json")

    def test_selection_manifest_stays_grouped_by_dataset_stem(self):
        comparison_summaries = [
            {
                "run_id": "dataset_bubble_32__0__q4",
                "dataset_stem": "dataset_bubble_32",
                "parameter_case_id": "case_a",
                "parameter_case_code": "0",
                "comparison_metrics": {"analysis": {"tau": 0.95, "spearman": 0.80}},
            },
            {
                "run_id": "dataset_bubble_32__1__q4",
                "dataset_stem": "dataset_bubble_32",
                "parameter_case_id": "case_b",
                "parameter_case_code": "1",
                "comparison_metrics": {"analysis": {"tau": 0.15, "spearman": 0.20}},
            },
            {
                "run_id": "dataset_bubble_200__0__q4",
                "dataset_stem": "dataset_bubble_200",
                "parameter_case_id": "case_c",
                "parameter_case_code": "0",
                "comparison_metrics": {"analysis": {"tau": 0.40, "spearman": 0.90}},
            },
            {
                "run_id": "dataset_bubble_200__1__q4",
                "dataset_stem": "dataset_bubble_200",
                "parameter_case_id": "case_d",
                "parameter_case_code": "1",
                "comparison_metrics": {"analysis": {"tau": 0.35, "spearman": 0.10}},
            },
            {
                "run_id": "dataset_scale_free_32__0__q4",
                "dataset_stem": "dataset_scale_free_32",
                "parameter_case_id": "case_e",
                "parameter_case_code": "0",
                "comparison_metrics": {"analysis": {"tau": 0.60, "spearman": 0.70}},
            },
            {
                "run_id": "dataset_scale_free_32__1__q4",
                "dataset_stem": "dataset_scale_free_32",
                "parameter_case_id": "case_f",
                "parameter_case_code": "1",
                "comparison_metrics": {"analysis": {"tau": 0.05, "spearman": 0.30}},
            },
        ]

        notes_by_run, manifest = build_notes_map_and_selection_manifest(comparison_summaries)

        self.assertCountEqual(
            manifest["datasets"].keys(),
            ["dataset_bubble_32", "dataset_bubble_200", "dataset_scale_free_32"],
        )
        self.assertEqual(
            manifest["datasets"]["dataset_bubble_32"]["kendall"]["top"][0]["run_id"],
            "dataset_bubble_32__0__q4",
        )
        self.assertEqual(
            manifest["datasets"]["dataset_bubble_200"]["spearman"]["top"][0]["run_id"],
            "dataset_bubble_200__0__q4",
        )
        self.assertEqual(
            manifest["datasets"]["dataset_scale_free_32"]["kendall"]["top"][0]["run_id"],
            "dataset_scale_free_32__0__q4",
        )
        self.assertIn("Top 1 Kendall", notes_by_run["dataset_bubble_32__0__q4"])
        self.assertIn("Top 1 Spearman", notes_by_run["dataset_bubble_200__0__q4"])

    def test_resolve_quartile_label_from_completion_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            sentinel_path = Path(temp_dir) / "_run_completed.json"
            sentinel_path.write_text(json.dumps({"quartile_label": "q4"}), encoding="utf-8")
            self.assertEqual(resolve_quartile_label_for_existing_run(temp_dir), "q4")

    def test_refresh_selected_run_outputs_rewrites_existing_run_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            comparison_dir_a = run_dir / "comparison" / "q4" / "dataset_bubble_32" / "0"
            comparison_dir_b = run_dir / "comparison" / "q4" / "dataset_bubble_32" / "1"
            comparison_dir_a.mkdir(parents=True)
            comparison_dir_b.mkdir(parents=True)

            comparison_summaries = [
                {
                    "run_id": "dataset_bubble_32__0__q4",
                    "dataset_stem": "dataset_bubble_32",
                    "parameter_case_id": "case_a",
                    "parameter_case_code": "0",
                    "parameter_levels": {},
                    "precision_metrics": {},
                    "java_analysis": {},
                    "python_analysis": {},
                    "comparison_metrics": {
                        "analysis": {"tau": 0.9, "spearman": 0.8},
                        "simulation": {},
                    },
                    "comparison_dir": str(comparison_dir_a),
                    "ground_truth_path": "gt_a.json",
                    "note": "",
                },
                {
                    "run_id": "dataset_bubble_32__1__q4",
                    "dataset_stem": "dataset_bubble_32",
                    "parameter_case_id": "case_b",
                    "parameter_case_code": "1",
                    "parameter_levels": {},
                    "precision_metrics": {},
                    "java_analysis": {},
                    "python_analysis": {},
                    "comparison_metrics": {
                        "analysis": {"tau": 0.1, "spearman": 0.2},
                        "simulation": {},
                    },
                    "comparison_dir": str(comparison_dir_b),
                    "ground_truth_path": "gt_b.json",
                    "note": "",
                },
            ]
            (run_dir / "comparison_summary_q4.json").write_text(
                json.dumps(comparison_summaries, indent=4),
                encoding="utf-8",
            )
            (run_dir / "_run_completed.json").write_text(
                json.dumps({"quartile_label": "q4"}, indent=4),
                encoding="utf-8",
            )

            with mock.patch(
                "sweep_pipeline_final.regenerate_selected_run_plots",
                side_effect=lambda summary, *_args, **_kwargs: summary,
            ):
                refresh_summary = refresh_selected_run_outputs(
                    save_path=str(run_dir),
                    quartile_label="q4",
                    time_step_hours=1,
                    iterations=2,
                )

            self.assertEqual(refresh_summary["selected_runs_for_plots"], 2)
            manifest = json.loads((run_dir / "selected_run_manifest.json").read_text(encoding="utf-8"))
            self.assertIn("dataset_bubble_32", manifest["datasets"])

            updated_summary = json.loads((run_dir / "comparison_summary_q4.json").read_text(encoding="utf-8"))
            self.assertTrue(all(item["note"] for item in updated_summary))
            self.assertTrue((run_dir / "sweep_summary.csv").exists())
            self.assertTrue((run_dir / "comparison_summary_q4.csv").exists())
            completion_payload = json.loads((run_dir / "_run_completed.json").read_text(encoding="utf-8"))
            self.assertEqual(completion_payload["quartile_label"], "q4")
            self.assertEqual(completion_payload["selected_runs_for_plots"], 2)


if __name__ == "__main__":
    unittest.main()
