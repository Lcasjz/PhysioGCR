"""Run a resumable dataset benchmark suite from the repository root."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


PAMAP2_MODELS = (
    "deepconvlstm",
    "deepsense",
    "attend_discriminate",
    "tinyhar",
    "mcnn",
    "sahar",
)
WESAD_MODELS = (
    "cocoa",
    "semipfl",
    "stress_calibration_2026",
    "fg_dualnet",
    "husformer",
)


def command(module, *arguments):
    return [sys.executable, "-m", module, *map(str, arguments)]


def build_jobs(args):
    output = Path(args.output)
    epoch_args = [] if args.epochs is None else ["--epochs", args.epochs]
    jobs = []
    if args.dataset == "sleep_edf":
        jobs.extend((
            (
                "router",
                command(
                    "experiments.train", "--dataset", "sleep", "--variant",
                    "full_no_calibration", "--data", args.data, "--output",
                    output / "router", *epoch_args,
                ),
                output / "router" / "validation_metrics.json",
            ),
            (
                "tokenizer_full",
                command(
                    "benchmarks.sleep_edf.train_tokenizer_baseline", "--variant", "full",
                    "--data", args.data, "--output", output / "tokenizer_full", *epoch_args,
                ),
                output / "tokenizer_full" / "metrics.json",
            ),
        ))
    elif args.dataset == "pamap2":
        for model in PAMAP2_MODELS:
            model_output = output / model
            jobs.append((
                model,
                command(
                    "benchmarks.pamap2.train_baselines", "--model", model,
                    "--data", args.data, "--output", model_output, *epoch_args,
                ),
                model_output / "metrics.json",
            ))
        router_output = output / "router"
        jobs.append((
            "router",
            command(
                "experiments.train", "--dataset", "pamap2", "--variant",
                "full_no_calibration", "--data", args.data, "--output", router_output,
                *epoch_args,
            ),
            router_output / "validation_metrics.json",
        ))
        if args.loso:
            for subject in range(101, 109):
                fold_output = output / "router_loso" / f"S{subject}"
                jobs.append((
                    f"router_loso_S{subject}",
                    command(
                        "benchmarks.pamap2.train_loso", "--test_subject", subject,
                        "--data", args.data, "--output", fold_output, *epoch_args,
                    ),
                    fold_output / "metrics.json",
                ))
    else:
        manifest = json.loads((Path(args.data) / "manifest.json").read_text(encoding="utf-8"))
        subjects = args.subjects or sorted(int(value) for value in manifest["subjects"])
        for model in WESAD_MODELS:
            for subject in subjects:
                base_output = output / model / f"S{subject}"
                jobs.append((
                    f"{model}_S{subject}",
                    command(
                        "benchmarks.wesad.train_loso", "--model", model, "--data", args.data,
                        "--loso_test_subject", subject, "--output", base_output, *epoch_args,
                    ),
                    base_output / "metrics.json",
                ))
                if args.quality:
                    quality_output = output / f"{model}_quality" / f"S{subject}"
                    jobs.append((
                        f"{model}_quality_S{subject}",
                        command(
                            "benchmarks.wesad.train_loso", "--model", model, "--data", args.data,
                            "--loso_test_subject", subject, "--output", quality_output,
                            "--quality_module", "--initial_checkpoint", base_output / "best.pt",
                            "--adaptation_mode", "parameter_efficient", *epoch_args,
                        ),
                        quality_output / "metrics.json",
                    ))
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("sleep_edf", "pamap2", "wesad"), required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--loso", action="store_true", help="also run PAMAP2 router LOSO")
    parser.add_argument("--quality", action="store_true", help="also run WESAD quality adapters")
    parser.add_argument("--subjects", type=int, nargs="+", help="optional WESAD LOSO subset")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--keep_going", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "suite_status.json"
    status = {}
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
    for name, invocation, completion_file in build_jobs(args):
        if completion_file.exists():
            print(f"SKIP {name} complete={completion_file}", flush=True)
            status[name] = {"status": "complete", "artifact": str(completion_file)}
            continue
        print("RUN", name, subprocess.list2cmdline(invocation), flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(invocation, check=False)
        status[name] = {"status": "complete" if result.returncode == 0 else "failed",
                        "returncode": result.returncode, "artifact": str(completion_file)}
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        if result.returncode and not args.keep_going:
            raise SystemExit(result.returncode)
    if not args.dry_run:
        print(f"SUITE_FINISHED status={status_path}", flush=True)


if __name__ == "__main__":
    main()
