import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from heterophysiorouter import QualityAwareRouter, corrupt_groups
from experiments.train import PAMAP2_GROUPS, evaluate, pamap2_patch_length


SUBJECTS = tuple(range(101, 109))
GROUP_NAMES = ("heart_rate", "hand", "chest", "ankle")
VARIANTS = ("full_no_calibration", "no_degradation")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_for(test_subject):
    index = SUBJECTS.index(test_subject)
    validation_subject = SUBJECTS[(index + 1) % len(SUBJECTS)]
    return {
        "train": [subject for subject in SUBJECTS if subject not in (test_subject, validation_subject)],
        "val": [validation_subject],
        "test": [test_subject],
    }


def load_loso(root, split, batch_size, workers):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["checksum"] != "2d6f32edff37755e4623dd56fff5fbb00f4fe47499a76575a603ffdc80a3149d":
        raise RuntimeError("unexpected PAMAP2 canonical checksum")
    arrays = {}
    for name, subjects in split.items():
        parts = [np.load(root / f"subject{subject}.npz") for subject in subjects]
        arrays[name] = (
            np.concatenate([part["x"] for part in parts]),
            np.concatenate([part["y"] for part in parts]),
        )
    mean = arrays["train"][0].mean((0, 1), keepdims=True)
    std = arrays["train"][0].std((0, 1), keepdims=True) + 1e-6
    loaders = {}
    train_counts = None
    for name, (signal, labels) in arrays.items():
        raw_signal = signal.astype(np.float32)
        normalized = ((signal - mean) / std).astype(np.float32)
        counts = np.bincount(labels, minlength=12)
        if name == "train":
            train_counts = counts
        dataset = TensorDataset(
            torch.from_numpy(normalized),
            torch.from_numpy(raw_signal),
            torch.from_numpy(labels.astype(np.int64)),
        )
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=name == "train",
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0,
        )
        print(f"samples.{name}={len(labels)} subjects={split[name]} labels={counts.tolist()}", flush=True)
    split_text = json.dumps(split, sort_keys=True, separators=(",", ":"))
    split_checksum = hashlib.sha256((manifest["checksum"] + split_text).encode("ascii")).hexdigest()
    return loaders, manifest["checksum"], split_checksum, train_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_subject", type=int, choices=SUBJECTS, required=True)
    parser.add_argument("--variant", choices=VARIANTS, default="full_no_calibration")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask_probability", type=float, default=.25)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_eval_batches", type=int, default=0)
    args = parser.parse_args()

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = split_for(args.test_subject)
    loaders, checksum, split_checksum, counts = load_loso(
        args.data, split, args.batch_size, args.workers
    )
    manifest = json.loads((Path(args.data) / "manifest.json").read_text(encoding="utf-8"))
    window = int(manifest["window"])
    config = {
        "channels": 52,
        "samples": window,
        "patch": pamap2_patch_length(window),
        "classes": 12,
        "groups": PAMAP2_GROUPS,
        "group_names": GROUP_NAMES,
    }
    model = QualityAwareRouter(
        config["channels"], config["samples"], config["patch"], config["classes"],
        config["groups"], args.variant,
    ).to(device)
    weights = np.minimum(counts.sum() / (config["classes"] * np.maximum(counts, 1)), 5.0)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device),
        label_smoothing=.05,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "dataset": "pamap2",
        "variant": args.variant,
        "canonical_checksum": checksum,
        "split_checksum": split_checksum,
        "split": split,
        "protocol": "LOSO with next subject cyclically fixed as validation",
        "evaluation_only": True,
        "selection_forbidden": True,
        "selection_metric": "validation complete macro_f1",
        "test_subject_used_for_early_stopping": False,
        "training_normalization": "fold training subjects only",
        "training_corruption": {
            "enabled": args.variant == "full_no_calibration",
            "mode": "mixed",
            "mask_probability": args.mask_probability,
            "rng_isolated_from_model_dropout": True,
        },
        **config,
    }
    protocol["groups"] = [list(group) for group in protocol["groups"]]
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(
        f"PAMAP2 LOSO EVALUATION_ONLY test={args.test_subject} val={split['val'][0]} "
        f"split_checksum={split_checksum} trainable={sum(p.numel() for p in model.parameters()):,}",
        flush=True,
    )

    best = -1.0
    stale = 0
    state = None
    best_validation = None
    best_epoch = 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for step, (signal, quality_signal, target) in enumerate(loaders["train"], 1):
            if args.max_train_batches and step > args.max_train_batches:
                break
            signal = signal.to(device, non_blocking=True)
            quality_signal = quality_signal.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if args.variant == "full_no_calibration":
                cpu_rng = torch.random.get_rng_state()
                cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                corrupt_signal, corrupt_quality, _ = corrupt_groups(
                    signal, quality_signal, args.mask_probability, PAMAP2_GROUPS, mode="mixed"
                )
                torch.random.set_rng_state(cpu_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state_all(cuda_rng)
                logits, _ = model(corrupt_signal, corrupt_quality, return_aux=True)
                model(signal, quality_signal, return_aux=True)
            else:
                logits, _ = model(signal, quality_signal, return_aux=True)
            loss = criterion(logits, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(signal)
            count += len(signal)
            if step == 1 or step % 100 == 0:
                print(f"epoch {epoch}/{args.epochs} step {step}/{len(loaders['train'])} loss={loss.item():.5f}", flush=True)

        validation = evaluate(
            model, loaders["val"], criterion, device, config["classes"], config["groups"],
            max_batches=args.max_eval_batches,
        )
        print(f"epoch {epoch} train_loss={total / count:.6f} val_complete={validation}", flush=True)
        if validation["macro_f1"] > best:
            best = validation["macro_f1"]
            stale = 0
            best_validation = validation
            best_epoch = epoch
            state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            torch.save({"model": state, "protocol": protocol, "args": vars(args)}, output / "best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stopping at epoch {epoch}", flush=True)
                break

    model.load_state_dict(state)
    test_conditions = {
        "complete": evaluate(
            model, loaders["test"], criterion, device, 12, PAMAP2_GROUPS,
            max_batches=args.max_eval_batches,
        )
    }
    for index, name in enumerate(GROUP_NAMES):
        test_conditions[f"missing_{name}"] = evaluate(
            model, loaders["test"], criterion, device, 12, PAMAP2_GROUPS,
            missing=(index,), max_batches=args.max_eval_batches,
        )
    result = {
        "protocol": protocol,
        "best_epoch": best_epoch,
        "best_validation_complete": best_validation,
        "test_conditions": test_conditions,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": time.time() - start,
        "args": vars(args),
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"FINAL_LOSO test_subject={args.test_subject} test={test_conditions}", flush=True)


if __name__ == "__main__":
    main()
