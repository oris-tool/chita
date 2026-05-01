import os
import tempfile
import unittest

from export_roi_from_comparison_summary import (
    build_copy_record,
    extract_selection_notes,
    parse_run_id,
    resolve_source_dir,
)


class ExportRoiFromComparisonSummaryTest(unittest.TestCase):
    def test_parse_run_id_extracts_dataset_and_run_numbers(self):
        parsed = parse_run_id("dataset_400__153__q4")
        self.assertEqual(parsed["dataset_stem"], "dataset_400")
        self.assertEqual(parsed["dataset_number"], "400")
        self.assertEqual(parsed["run_number"], "153")
        self.assertEqual(parsed["suffix"], "q4")

    def test_parse_run_id_supports_dataset_stem_with_embedded_label(self):
        parsed = parse_run_id("dataset_scale_free_2500__88__q4")
        self.assertEqual(parsed["dataset_stem"], "dataset_scale_free_2500")
        self.assertEqual(parsed["dataset_number"], "2500")
        self.assertEqual(parsed["run_number"], "88")
        self.assertEqual(parsed["suffix"], "q4")

    def test_extract_selection_notes_ignores_non_roi_labels(self):
        notes = extract_selection_notes(
            "Worst 1 Kendall - Top 2 Spearman - Median 3 Kendall - something else"
        )
        self.assertEqual(notes, ["Top 2 Spearman", "Median 3 Kendall"])

    def test_build_copy_record_uses_dataset_run_and_note_label(self):
        copy_record = build_copy_record(
            {
                "row": {"run_id": "dataset_scale_free_800__22__q4"},
                "dataset_stem": "dataset_scale_free_800",
                "dataset_number": "800",
                "run_number": "22",
                "suffix": "q4",
                "matched_notes": ["Top 1 Kendall", "Median 4 Spearman"],
            },
            comparison_root="/tmp/comparison",
            output_root="/tmp/roi",
        )
        self.assertEqual(copy_record["source_dir"], "/tmp/comparison/dataset_scale_free_800/22")
        self.assertEqual(
            copy_record["destination_dir"],
            "/tmp/roi/dataset_scale_free_800/22_top_1_kendall__median_4_spearman",
        )

    def test_resolve_source_dir_supports_quartile_layout(self):
        selection_row = {
            "dataset_stem": "dataset_scale_free_800",
            "dataset_number": "800",
            "run_number": "22",
            "suffix": "q4",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            comparison_root = os.path.join(temp_dir, "comparison")
            expected_source_dir = os.path.join(comparison_root, "q4", "dataset_scale_free_800", "22")
            os.makedirs(expected_source_dir)
            self.assertEqual(resolve_source_dir(selection_row, comparison_root), expected_source_dir)


if __name__ == "__main__":
    unittest.main()
