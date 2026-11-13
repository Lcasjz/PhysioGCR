"""Aggregate strict-LOSO WESAD metrics produced by ``scripts/run_suite.py``."""

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
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True, help="suite output directory")
    parser.add_argument("--data", required=True, help="Husformer-input cache")
    parser.add_argument("--output", help="defaults to RUNS/summary")
    args = parser.parse_args()
    root = Path(args.runs)
    output = Path(args.output) if args.output else root / "summary"
    manifest = json.loads((Path(args.data) / "manifest.json").read_text(encoding="utf-8"))
    subjects = sorted(int(value) for value in manifest["subjects"])
    rows, model_results = [], {}
    for model in MODELS:
        folds = {}
        for subject in subjects:
            path = root / model / f"S{subject}" / "metrics.json"
            if not path.exists():
                print(f"MISSING model={model} test_subject=S{subject}", flush=True)
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            if data["canonical_checksum"] != manifest["checksum"]:
                raise RuntimeError(f"checksum mismatch: {path}")
            folds[str(subject)] = data
        if len(folds) != len(subjects):
            continue
        aggregate = {
            metric: describe([folds[str(subject)]["test"][metric] for subject in subjects])
            for metric in METRICS
        }
        model_results[model] = {"aggregate": aggregate, "folds": folds}
        row = {"model": model}
        for metric in METRICS:
            row[f"test_{metric}_mean"] = aggregate[metric]["mean"]
            row[f"test_{metric}_std"] = aggregate[metric]["std"]
        rows.append(row)
    result = {
        "dataset": "WESAD",
        "claim": "common-input architecture adaptations under strict LOSO",
        "canonical_checksum": manifest["checksum"],
        "subjects": subjects,
        "models": model_results,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if rows:
        with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"completed_models={len(rows)}/{len(MODELS)} saved={output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
