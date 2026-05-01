import argparse
import csv
import json
import os
import re
import shutil


DEFAULT_SUMMARY_CSV = "/mnt/nniccoli/chita_results/sweep_20260423-0202/comparison_summary_q4.csv"
DEFAULT_COMPARISON_ROOT = "/mnt/nniccoli/chita_results/sweep_20260423-0202/comparison/q4"
DEFAULT_OUTPUT_ROOT = "/mnt/nniccoli/chita_results/roi"

SELECTION_NOTE_PATTERN = re.compile(r"\b(?:Top|Median)\s+\d+\s+(?:Kendall|Spearman)\b")
RUN_ID_PATTERN = re.compile(r"^(?P<dataset_stem>dataset_.+?)__(?P<run_number>\d+)__(?P<suffix>.+)$")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def sanitize_path_fragment(value):
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_")
    if not sanitized:
        return "selection"
    return sanitized.lower()


def parse_run_id(run_id):
    match = RUN_ID_PATTERN.match(run_id or "")
    if not match:
        raise ValueError(
            f"Unsupported run_id format: {run_id!r}. Expected dataset_<stem>__<run_number>__<suffix>."
        )
    dataset_stem = match.group("dataset_stem")
    dataset_number_match = re.search(r"(\d+)$", dataset_stem)
    return {
        "dataset_stem": dataset_stem,
        "dataset_number": dataset_number_match.group(1) if dataset_number_match else None,
        "run_number": match.group("run_number"),
        "suffix": match.group("suffix"),
    }


def extract_selection_notes(note_text):
    if not note_text:
        return []
    return SELECTION_NOTE_PATTERN.findall(note_text)


def iter_selected_rows(summary_csv_path):
    with open(summary_csv_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            matched_notes = extract_selection_notes(row.get("note", ""))
            if not matched_notes:
                continue
            run_info = parse_run_id(row.get("run_id", ""))
            yield {
                "row": row,
                "matched_notes": matched_notes,
                "dataset_stem": run_info["dataset_stem"],
                "dataset_number": run_info["dataset_number"],
                "run_number": run_info["run_number"],
                "suffix": run_info["suffix"],
            }


def resolve_source_dir(selection_row, comparison_root):
    dataset_stem = selection_row["dataset_stem"]
    run_number = selection_row["run_number"]
    suffix = selection_row["suffix"]
    candidate_dirs = [
        os.path.join(comparison_root, dataset_stem, run_number),
        os.path.join(comparison_root, suffix, dataset_stem, run_number),
    ]

    dataset_number = selection_row.get("dataset_number")
    if dataset_number:
        legacy_dataset_dir = f"dataset_{dataset_number}"
        candidate_dirs.extend(
            [
                os.path.join(comparison_root, legacy_dataset_dir, run_number),
                os.path.join(comparison_root, suffix, legacy_dataset_dir, run_number),
            ]
        )

    for candidate_dir in candidate_dirs:
        if os.path.isdir(candidate_dir):
            return candidate_dir

    return candidate_dirs[0]


def build_copy_record(selection_row, comparison_root, output_root):
    dataset_stem = selection_row["dataset_stem"]
    note_label = "__".join(sanitize_path_fragment(note) for note in selection_row["matched_notes"])
    source_dir = resolve_source_dir(selection_row, comparison_root)
    destination_dir = os.path.join(
        output_root,
        dataset_stem,
        f"{selection_row['run_number']}_{note_label}",
    )
    return {
        "run_id": selection_row["row"]["run_id"],
        "dataset_stem": dataset_stem,
        "dataset_number": selection_row["dataset_number"],
        "run_number": selection_row["run_number"],
        "suffix": selection_row["suffix"],
        "matched_notes": selection_row["matched_notes"],
        "source_dir": source_dir,
        "destination_dir": destination_dir,
    }


def export_selected_runs(summary_csv_path, comparison_root, output_root, overwrite=False, dry_run=False):
    copied_rows = []
    for selection_row in iter_selected_rows(summary_csv_path):
        copy_record = build_copy_record(selection_row, comparison_root, output_root)
        source_dir = copy_record["source_dir"]
        destination_dir = copy_record["destination_dir"]

        if not os.path.isdir(source_dir):
            raise FileNotFoundError(f"Source comparison directory not found: {source_dir}")

        if os.path.exists(destination_dir):
            if not overwrite:
                raise FileExistsError(
                    f"Destination already exists: {destination_dir}. Use --overwrite to replace it."
                )
            if not dry_run:
                shutil.rmtree(destination_dir)

        if not dry_run:
            ensure_dir(os.path.dirname(destination_dir))
            shutil.copytree(source_dir, destination_dir)

        copied_rows.append(copy_record)

    return copied_rows


def write_manifest(output_root, copied_rows):
    manifest_path = os.path.join(output_root, "roi_export_manifest.json")
    ensure_dir(output_root)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(copied_rows, handle, indent=4)
    return manifest_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Copy selected comparison run folders into an ROI directory based on "
            "Top/Median Kendall/Spearman markers in comparison_summary_q4.csv."
        )
    )
    parser.add_argument(
        "--summary-csv",
        default=DEFAULT_SUMMARY_CSV,
        help="Path to the comparison summary CSV.",
    )
    parser.add_argument(
        "--comparison-root",
        default=DEFAULT_COMPARISON_ROOT,
        help=(
            "Root directory containing comparison folders in either "
            "<dataset_stem>/<run_number> or <suffix>/<dataset_stem>/<run_number> layout."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory where ROI folders will be created.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace destination folders if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be copied without writing anything.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    copied_rows = export_selected_runs(
        summary_csv_path=os.path.abspath(args.summary_csv),
        comparison_root=os.path.abspath(args.comparison_root),
        output_root=os.path.abspath(args.output_root),
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    for copy_record in copied_rows:
        print(
            f"{copy_record['run_id']} -> {copy_record['destination_dir']} "
            f"[{', '.join(copy_record['matched_notes'])}]"
        )

    if args.dry_run:
        print(f"Dry run complete. {len(copied_rows)} folders would be copied.")
        return

    manifest_path = write_manifest(os.path.abspath(args.output_root), copied_rows)
    print(f"Copied {len(copied_rows)} folders.")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
