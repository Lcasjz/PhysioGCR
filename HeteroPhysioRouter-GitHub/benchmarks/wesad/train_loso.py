"""Strict LOSO benchmark on the shared public Husformer input geometry."""

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from benchmarks.wesad.metrics import (
    HusformerFocalLoss, calibration_metrics, classification_metrics, fit_temperature,
)
from benchmarks.wesad.models import HUSFORMER_INPUT_MODELS, MODALITY_NAMES
from benchmarks.wesad.quality_module import corrupt_modalities


MODEL_PROTOCOLS = {
    "cocoa": "four modality-specific encoders plus COCOA contrastive pretraining",
    "semipfl": "four-modal projector plus personalized hypernetwork autoencoder",
    "stress_calibration_2026": "four-modal masked reconstruction and validation-only temperature calibration",
    "fg_dualnet": "per-modality statistical features and feature-gated classifier",
    "husformer": "public four-stream geometry and cross-modal/self-attention fusion",
}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_husformer_input(root, test_subject):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    subjects = sorted(int(subject) for subject in manifest["subjects"])
    if test_subject not in subjects:
        raise ValueError(f"unknown test subject S{test_subject}; available={subjects}")
    index = subjects.index(test_subject)
    val_subject = subjects[(index + 1) % len(subjects)]
    split = {
        "train": [subject for subject in subjects if subject not in (test_subject, val_subject)],
        "val": [val_subject],
        "test": [test_subject],
    }
    arrays = {}
    for split_name, split_subjects in split.items():
        values = {name: [] for name in MODALITY_NAMES}
        labels, subject_ids = [], []
        for subject in split_subjects:
            with np.load(root / f"subject{subject}.npz") as part:
                for name in MODALITY_NAMES:
                    values[name].append(part[name].astype(np.float32))
                labels.append(part["y"].astype(np.int64))
                subject_ids.append(np.full(len(part["y"]), subject, dtype=np.int64))
        arrays[split_name] = {
            **{name: np.concatenate(values[name]) for name in MODALITY_NAMES},
            "y": np.concatenate(labels),
            "subjects": np.concatenate(subject_ids),
        }
    normalization = {}
    for name in MODALITY_NAMES:
        mean = arrays["train"][name].mean((0, 1), keepdims=True)
        std = arrays["train"][name].std((0, 1), keepdims=True) + 1e-6
        normalization[name] = {
            "mean": mean.reshape(-1).tolist(), "std": std.reshape(-1).tolist(),
        }
        for split_name in arrays:
            arrays[split_name][name] = ((arrays[split_name][name] - mean) / std).astype(np.float32)
    manifest = copy.deepcopy(manifest)
    manifest["split"] = split
    return manifest, arrays, normalization


class FourModalDataset(Dataset):
    def __init__(self, values, client_map):
        self.modalities = [torch.from_numpy(values[name]) for name in MODALITY_NAMES]
        self.labels = torch.from_numpy(values["y"])
        self.clients = torch.as_tensor(
            [client_map.get(int(subject), -1) for subject in values["subjects"]], dtype=torch.long,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return tuple(value[index] for value in self.modalities) + (self.labels[index], self.clients[index])


def make_loaders(arrays, train_subjects, batch_size, workers, seed, balanced):
    client_map = {subject: index for index, subject in enumerate(train_subjects)}
    loaders = {}
    for split_name, values in arrays.items():
        dataset = FourModalDataset(values, client_map)
        sampler, shuffle = None, split_name == "train"
        if split_name == "train" and balanced:
            counts = np.bincount(values["y"], minlength=3)
            weights = 1.0 / np.maximum(counts, 1)
            sampler = WeightedRandomSampler(
                torch.as_tensor(weights[values["y"]], dtype=torch.double),
                len(values["y"]), replacement=True,
                generator=torch.Generator().manual_seed(seed),
            )
            shuffle = False
        loaders[split_name] = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
            num_workers=workers, pin_memory=torch.cuda.is_available(),
            persistent_workers=workers > 0,
        )
    return loaders


def unpack(batch, device):
    modalities = tuple(value.to(device, non_blocking=True) for value in batch[:4])
    labels = batch[4].to(device, non_blocking=True)
    clients = batch[5].to(device, non_blocking=True)
    return modalities, labels, clients


def evaluate(model, loader, device, temperature=1.0, return_logits=False):
    model.eval()
    labels, predictions, logits_all, total_loss = [], [], [], 0.0
    quality_totals = {}
    with torch.no_grad():
        for batch in loader:
            modalities, target, clients = unpack(batch, device)
            output = model(
                modalities, clients,
                return_quality_aux=model.quality_router is not None,
            )
            if model.quality_router is not None:
                logits, auxiliary = output
                for key, value in auxiliary.items():
                    quality_totals[key] = quality_totals.get(key, 0.0) + value.detach().sum(0).cpu().numpy()
            else:
                logits = output
            logits = logits / temperature
            total_loss += F.cross_entropy(logits, target, reduction="sum").item()
            labels.extend(target.cpu().tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
            logits_all.append(logits.cpu())
    result = classification_metrics(labels, predictions, total_loss / max(len(labels), 1))
    if quality_totals:
        result["quality_diagnostics_mean"] = {
            key: (value / len(labels)).tolist() for key, value in quality_totals.items()
        }
    if return_logits:
        return result, torch.cat(logits_all).numpy(), np.asarray(labels)
    return result


def pretrain(model_name, model, loader, device, epochs, lr):
    if epochs <= 0 or model_name not in ("cocoa", "semipfl", "stress_calibration_2026"):
        return []
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for step, batch in enumerate(loader, 1):
            modalities, _, clients = unpack(batch, device)
            if model_name == "cocoa":
                loss = model.ssl_loss(modalities)
            elif model_name == "semipfl":
                loss = model.reconstruction_loss(modalities, clients)
            else:
                loss = model.masked_reconstruction_loss(modalities)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * len(clients)
            seen += len(clients)
            if step == 1 or step % 100 == 0:
                print(f"PRETRAIN epoch={epoch}/{epochs} step={step}/{len(loader)} loss={loss.item():.5f}", flush=True)
        average = total / max(seen, 1)
        history.append(average)
        print(f"PRETRAIN_EPOCH {epoch:02d}/{epochs} loss={average:.5f}", flush=True)
    return history


def train_epoch(model_name, model, loader, device, optimizer, criterion,
                corruption_generator=None, mask_probability=0.25,
                training_corruption=True):
    model.train()
    if getattr(model, "parameter_efficient_adaptation", False):
        for name, module in model.named_children():
            if name not in ("quality_router", "quality_film"):
                module.eval()
        model.quality_router.train()
        model.quality_film.train()
    labels, predictions, total_loss, seen = [], [], 0.0, 0
    for step, batch in enumerate(loader, 1):
        modalities, target, clients = unpack(batch, device)
        if model.quality_router is not None and training_corruption:
            modalities, _ = corrupt_modalities(
                modalities, mask_probability, generator=corruption_generator,
            )
        # Quality proxies must observe the same degraded view used by the router.
        quality_modalities = modalities
        logits = model(modalities, clients, quality_modalities=quality_modalities)
        loss = criterion(logits, target)
        if model_name == "semipfl":
            loss = loss + 0.1 * model.reconstruction_loss(
                modalities, clients, quality_modalities=quality_modalities,
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.8 if model_name == "husformer" else 5.0)
        optimizer.step()
        total_loss += loss.item() * len(target)
        seen += len(target)
        labels.extend(target.detach().cpu().tolist())
        predictions.extend(logits.argmax(1).detach().cpu().tolist())
        if step == 1 or step % 100 == 0:
            print(f"TRAIN_STEP step={step}/{len(loader)} loss={loss.item():.5f}", flush=True)
    return classification_metrics(labels, predictions, total_loss / max(seen, 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(HUSFORMER_INPUT_MODELS), required=True)
    parser.add_argument("--data", default="data/husformer_input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--loso_test_subject", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--pretrain_epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--pretrain_lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balanced_sampling", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--quality_module", action="store_true")
    parser.add_argument("--quality_hidden", type=int, default=128)
    parser.add_argument("--mask_probability", type=float, default=0.25)
    parser.add_argument("--training_corruption", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--initial_checkpoint")
    parser.add_argument(
        "--adaptation_mode", choices=("full", "parameter_efficient"), default="full",
    )
    args = parser.parse_args()
    seed_all(args.seed)

    manifest, arrays, normalization = load_husformer_input(args.data, args.loso_test_subject)
    train_subjects = manifest["split"]["train"]
    loaders = make_loaders(
        arrays, train_subjects, args.batch_size, args.workers, args.seed, args.balanced_sampling,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == "semipfl":
        model = HUSFORMER_INPUT_MODELS[args.model](
            clients=len(train_subjects), quality_module=args.quality_module,
            quality_hidden=args.quality_hidden,
        ).to(device)
    else:
        model = HUSFORMER_INPUT_MODELS[args.model](
            quality_module=args.quality_module, quality_hidden=args.quality_hidden,
        ).to(device)
    initialization = None
    if args.initial_checkpoint:
        initial = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=False)
        if initial.get("canonical_checksum") != manifest["checksum"]:
            raise RuntimeError("initial checkpoint and current WESAD cache checksums differ")
        missing, unexpected = model.load_state_dict(initial["model"], strict=False)
        allowed_missing = tuple(
            name for name in missing
            if name.startswith("quality_router.") or name.startswith("quality_film.")
        )
        if len(allowed_missing) != len(missing) or unexpected:
            raise RuntimeError(
                f"incompatible initial checkpoint missing={missing} unexpected={unexpected}"
            )
        initialization = {
            "checkpoint": str(Path(args.initial_checkpoint).resolve()),
            "missing_new_adapter_parameters": list(missing),
        }
    if args.adaptation_mode == "parameter_efficient":
        if not args.quality_module or not args.initial_checkpoint:
            parser.error(
                "parameter_efficient mode requires --quality_module and --initial_checkpoint"
            )
        for parameter in model.parameters():
            parameter.requires_grad = False
        trainable_prefixes = ("quality_router.", "quality_film.")
        for name, parameter in model.named_parameters():
            if name.startswith(trainable_prefixes):
                parameter.requires_grad = True
        model.parameter_efficient_adaptation = True
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    counts = {name: np.bincount(values["y"], minlength=3).tolist() for name, values in arrays.items()}
    protocol = {
        "dataset": "WESAD",
        "track": "husformer_public_input_quality_router_strict_loso" if args.quality_module else "husformer_public_input_strict_loso",
        "claim": "shared Husformer four-modality input comparison; not original-paper score reproduction",
        "model": args.model,
        "canonical_checksum": manifest["checksum"],
        "sample_identity_checksum": manifest["sample_identity_checksum"],
        "split": manifest["split"],
        "input": manifest["modalities"],
        "classes": manifest["class_names"],
        "normalization": normalization,
        "selection_metric": "validation macro_f1",
        "test_locked": True,
        "balanced_sampling": args.balanced_sampling,
        "preserved_model_mechanism": MODEL_PROTOCOLS[args.model],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "adaptation_mode": args.adaptation_mode,
        "initialization": initialization,
        "quality_module": None if not args.quality_module else {
            "variant": "full_no_calibration" if args.training_corruption else "no_degradation",
            "implementation_version": "v2_quality_estimator_observes_training_corruption",
            "hidden": args.quality_hidden,
            "components": ["modality_quality", "temporal_quality", "cross_granular_attention", "dynamic_three_branch_routing", "modality_group_corruption"],
            "training_corruption": {
                "enabled": args.training_corruption,
                "mode": "mixed_dropout_noise_flatline",
                "probability": args.mask_probability,
            },
            "integration": "zero-initialized residual FiLM at each backbone's native pre-classifier representation",
            "source_design": "heterophysiorouter.model.QualityAwareRouter",
        },
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print("=== WESAD HUSFORMER-INPUT STRICT LOSO BENCHMARK ===", flush=True)
    print("input=Resp(50x14) BVP(16x4) ECG(50x14) EDA(1x4), one label-pure second", flush=True)
    print(f"model={args.model} device={device} parameters={protocol['parameters']:,}", flush=True)
    print(
        f"adaptation_mode={args.adaptation_mode} "
        f"trainable_parameters={protocol['trainable_parameters']:,} "
        f"trainable_fraction={protocol['trainable_parameters'] / protocol['parameters']:.6f}",
        flush=True,
    )
    if args.quality_module:
        print(f"quality_module=full_no_calibration training_corruption={args.training_corruption} probability={args.mask_probability}", flush=True)
    print(f"split=train:{train_subjects} val:{manifest['split']['val']} test:{manifest['split']['test']}", flush=True)
    print(f"checksum={manifest['checksum']}", flush=True)
    for split_name in ("train", "val", "test"):
        print(f"samples.{split_name}={len(arrays[split_name]['y'])} labels={counts[split_name]}", flush=True)

    start = time.time()
    history = [] if args.adaptation_mode == "parameter_efficient" else pretrain(
        args.model, model, loaders["train"], device, args.pretrain_epochs, args.pretrain_lr,
    )
    train_counts = np.bincount(arrays["train"]["y"], minlength=3)
    class_weights = (
        np.ones(3, dtype=np.float32) if args.balanced_sampling else
        np.minimum(train_counts.sum() / (3 * np.maximum(train_counts, 1)), 5.0).astype(np.float32)
    )
    if args.model == "husformer":
        criterion = HusformerFocalLoss().to(device)
        optimizer = torch.optim.Adam(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=args.lr,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.1)
        protocol["supervised_loss"] = "focal alpha=[0.4,0.3,0.3], gamma=2"
        protocol["optimizer"] = "Adam with validation-loss plateau decay (public-code default)"
    else:
        criterion = nn.CrossEntropyLoss(
            weight=torch.as_tensor(class_weights, dtype=torch.float32, device=device),
            label_smoothing=args.label_smoothing,
        )
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=args.lr, weight_decay=args.weight_decay,
        )
        scheduler = None
        protocol["supervised_loss"] = "class-weighted cross entropy"
        protocol["optimizer"] = "AdamW"
    protocol["class_weights"] = None if args.model == "husformer" else class_weights.tolist()
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    corruption_generator = None
    if args.quality_module:
        corruption_generator = torch.Generator(device=device).manual_seed(args.seed + 20261031)
    best, best_state, best_val, best_epoch, stale = -1.0, None, None, 0, 0
    for epoch in range(1, args.epochs + 1):
        train = train_epoch(
            args.model, model, loaders["train"], device, optimizer, criterion,
            corruption_generator=corruption_generator,
            mask_probability=args.mask_probability,
            training_corruption=args.training_corruption,
        )
        val = evaluate(model, loaders["val"], device)
        if scheduler is not None:
            scheduler.step(val["loss"])
        print(
            f"EPOCH {epoch:02d}/{args.epochs} | "
            f"TRAIN loss={train['loss']:.5f} acc={train['accuracy']:.5f} bacc={train['balanced_accuracy']:.5f} f1={train['macro_f1']:.5f} kappa={train['kappa']:.5f} | "
            f"VAL loss={val['loss']:.5f} acc={val['accuracy']:.5f} bacc={val['balanced_accuracy']:.5f} f1={val['macro_f1']:.5f} kappa={val['kappa']:.5f}",
            flush=True,
        )
        if val["macro_f1"] > best:
            best, best_val, best_epoch, stale = val["macro_f1"], val, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= args.patience:
            print(f"EARLY_STOPPING epoch={epoch} patience={args.patience}", flush=True)
            break
    if best_state is None:
        raise RuntimeError("training produced no validation checkpoint")
    model.load_state_dict(best_state)
    temperature, calibration = 1.0, None
    if args.model == "stress_calibration_2026":
        _, val_logits, val_labels = evaluate(model, loaders["val"], device, return_logits=True)
        temperature = fit_temperature(val_logits, val_labels)
        print(f"CALIBRATION temperature={temperature:.6f} source=validation_only", flush=True)
    test, test_logits, test_labels = evaluate(
        model, loaders["test"], device, temperature=temperature, return_logits=True,
    )
    if args.model == "stress_calibration_2026":
        probabilities = torch.softmax(torch.from_numpy(test_logits), dim=1).numpy()
        calibration = calibration_metrics(probabilities, test_labels)
        test["calibration"] = calibration
    torch.save(
        {"model": model.state_dict(), "model_name": args.model,
         "canonical_checksum": manifest["checksum"], "temperature": temperature},
        output / "best.pt",
    )
    result = {
        "model": args.model, "canonical_checksum": manifest["checksum"],
        "split": manifest["split"], "protocol": protocol,
        "best_val_macro_f1": best, "best_epoch": best_epoch, "best_val": best_val,
        "test": test, "temperature": temperature, "pretrain_history": history,
        "args": vars(args), "elapsed_seconds": time.time() - start,
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("=== FINAL TEST ===", flush=True)
    print(f"model={args.model} best_val_macro_f1={best:.5f} best_epoch={best_epoch}", flush=True)
    print(f"TEST loss={test['loss']:.5f} acc={test['accuracy']:.5f} bacc={test['balanced_accuracy']:.5f} f1={test['macro_f1']:.5f} kappa={test['kappa']:.5f}", flush=True)
    if calibration is not None:
        print(f"CALIBRATION nll={calibration['nll']:.5f} brier={calibration['brier']:.5f} ece15={calibration['ece_15']:.5f}", flush=True)
    print(f"saved_metrics={output / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
