"""Aggregate paired WESAD backbone and quality-adapter LOSO metrics."""

import argparse
import csv
import json
import statistics
from pathlib import Path


MODELS = ("cocoa", "semipfl", "stress_calibration_2026", "fg_dualnet", "husformer")
METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "kappa")


def describe(values):
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True, help="suite output directory")
    parser.add_argument("--data", required=True, help="Husformer-input cache")
    parser.add_argument("--output", help="defaults to RUNS/quality_summary")
    args = parser.parse_args()
    root = Path(args.runs)
    output = Path(args.output) if args.output else root / "quality_summary"
    manifest = json.loads((Path(args.data) / "manifest.json").read_text(encoding="utf-8"))
    subjects = sorted(int(value) for value in manifest["subjects"])
    summaries, rows = {}, []
    for model in MODELS:
        paired = []
        for subject in subjects:
            base_path = root / model / f"S{subject}" / "metrics.json"
            quality_path = root / f"{model}_quality" / f"S{subject}" / "metrics.json"
            if not base_path.exists() or not quality_path.exists():
                print(f"MISSING model={model} test_subject=S{subject}", flush=True)
                continue
            baseline = json.loads(base_path.read_text(encoding="utf-8"))
            quality = json.loads(quality_path.read_text(encoding="utf-8"))
            checksums = {
                baseline["canonical_checksum"], quality["canonical_checksum"],
                manifest["checksum"],
            }
            if len(checksums) != 1:
                raise RuntimeError(f"checksum mismatch for {model} S{subject}")
            row = {"model": model, "test_subject": subject}
            for metric in METRICS:
                before = float(baseline["test"][metric])
                after = float(quality["test"][metric])
                row[f"baseline_{metric}"] = before
                row[f"quality_{metric}"] = after
                row[f"delta_{metric}"] = after - before
            paired.append(row)
            rows.append(row)
        if len(paired) != len(subjects):
            continue
        summaries[model] = {
            metric: {
                "baseline": describe([row[f"baseline_{metric}"] for row in paired]),
                "quality": describe([row[f"quality_{metric}"] for row in paired]),
                "paired_delta": describe([row[f"delta_{metric}"] for row in paired]),
            }
            for metric in METRICS
        }
    result = {
        "dataset": "WESAD",
        "canonical_checksum": manifest["checksum"],
        "subjects": subjects,
        "models": summaries,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "paired_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if rows:
        with (output / "paired_folds.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(
        f"completed_models={len(summaries)}/{len(MODELS)} "
        f"saved={output / 'paired_summary.json'}", flush=True,
    )


if __name__ == "__main__":
    main()
