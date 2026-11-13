import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from heterophysiorouter import QualityAwareRouter
from experiments.train import evaluate, load_pamap2, load_sleep, missing_conditions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("pamap2", "sleep"), required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_protocol = checkpoint["protocol"]
    loader_function = load_pamap2 if args.dataset == "pamap2" else load_sleep
    loaders, checksum, counts, config = loader_function(
        args.data, args.batch_size, args.workers, splits=("train", "test"),
    )
    if source_protocol["dataset"] != args.dataset:
        raise RuntimeError("checkpoint dataset does not match requested dataset")
    if source_protocol["canonical_checksum"] != checksum:
        raise RuntimeError("checkpoint and test cache checksums do not match")
    for key in ("channels", "samples", "patch", "classes"):
        if int(source_protocol[key]) != int(config[key]):
            raise RuntimeError(f"checkpoint/test protocol mismatch for {key}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QualityAwareRouter(
        config["channels"], config["samples"], config["patch"], config["classes"],
        config["groups"], source_protocol["variant"],
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    weights = np.minimum(counts.sum() / (config["classes"] * np.maximum(counts, 1)), 5.0)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device),
        label_smoothing=0.05,
    )

    conditions = {}
    for name, missing in missing_conditions(
        config["groups"], config["group_names"], include_pairs=args.dataset == "sleep",
    ):
        conditions[name] = evaluate(
            model, loaders["test"], criterion, device, config["classes"],
            config["groups"], missing=missing,
        )
        print(f"LOCKED_TEST dataset={args.dataset} condition={name} metrics={conditions[name]}", flush=True)

    result = {
        "dataset": args.dataset,
        "protocol": "single locked-test access after validation-only configuration freeze",
        "canonical_checksum": checksum,
        "source_checkpoint": str(checkpoint_path.resolve()),
        "source_protocol": source_protocol,
        "test_conditions": conditions,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"SAVED_LOCKED_TEST={output}", flush=True)


if __name__ == "__main__":
    main()
