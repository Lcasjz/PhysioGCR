import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

from heterophysiorouter import QualityAwareRouter
from experiments.train_evidence import mask_groups, per_sample_task_loss
from experiments.train import basic_metrics, load_pamap2, load_sleep


UNSEEN_CORRUPTIONS = ("gain", "drift", "spike", "segment_loss")


def rankdata(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def correlation(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if len(first) < 2 or first.std() < 1e-12 or second.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def spearman(first, second):
    return correlation(rankdata(first), rankdata(second))


def calibration_summary(score, target, bins=10):
    score = np.asarray(score, dtype=np.float64).clip(0, 1)
    target = np.asarray(target, dtype=np.float64).clip(0, 1)
    edges = np.linspace(0, 1, bins + 1)
    records, weighted_error = [], 0.0
    for index in range(bins):
        if index == bins - 1:
            selected = (score >= edges[index]) & (score <= edges[index + 1])
        else:
            selected = (score >= edges[index]) & (score < edges[index + 1])
        count = int(selected.sum())
        if not count:
            continue
        predicted = float(score[selected].mean())
        observed = float(target[selected].mean())
        weighted_error += count * abs(predicted - observed)
        records.append({
            "lower": float(edges[index]), "upper": float(edges[index + 1]),
            "count": count, "predicted": predicted, "observed": observed,
        })
    return {
        "mae": float(np.abs(score - target).mean()),
        "brier": float(np.square(score - target).mean()),
        "ece_10": float(weighted_error / max(len(score), 1)),
        "bins": records,
    }


def assign_group(value, rows, indices, replacement):
    subset = value[rows].clone()
    subset[:, :, list(indices)] = replacement
    value[rows] = subset


def corrupt_one_group(x, quality_x, group, kind, seed):
    x = x.clone()
    quality_x = quality_x.clone()
    indices = list(group)
    generator = torch.Generator(device=x.device).manual_seed(seed)
    for value in (x, quality_x):
        segment = value[:, :, indices]
        scale = segment.std(1, keepdim=True, unbiased=False).clamp_min(1e-4)
        if kind == "gain":
            value[:, :, indices] = segment * 3.0
        elif kind == "drift":
            ramp = torch.linspace(-1, 1, value.shape[1], device=x.device, dtype=x.dtype)
            ramp = ramp.view(1, -1, 1)
            value[:, :, indices] = segment + 2.0 * scale * ramp
        elif kind == "spike":
            replacement = segment.clone()
            count = max(1, value.shape[1] // 20)
            positions = torch.randint(
                value.shape[1], (len(value), count), device=x.device, generator=generator,
            )
            signs = torch.where(
                torch.rand((len(value), count), device=x.device, generator=generator) > 0.5,
                1.0, -1.0,
            )
            for row in range(len(value)):
                replacement[row, positions[row]] += signs[row].view(-1, 1) * 6.0 * scale[row, 0]
            value[:, :, indices] = replacement
        elif kind == "segment_loss":
            start = value.shape[1] // 3
            end = start + max(1, value.shape[1] // 4)
            replacement = segment.clone()
            replacement[:, start:end] = 0
            value[:, :, indices] = replacement
        else:
            raise ValueError(kind)
    return x, quality_x


def build_model(checkpoint, device):
    protocol = checkpoint["protocol"]
    input_config = protocol["input"]
    representation_groups = tuple(
        tuple(group) for group in protocol["representation_groups"]
    )
    range_groups = tuple((min(group), max(group) + 1) for group in representation_groups)
    model = QualityAwareRouter(
        input_config["channels"], input_config["samples"], input_config["patch"],
        input_config["classes"], range_groups, "full_no_calibration",
        channel_group_indices=representation_groups,
        sensor_fusion=protocol.get("sensor_fusion", "weighted"),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model


@torch.no_grad()
def diagnose(model, loader, device, criterion, classes, groups, seed):
    clean_labels, clean_predictions, clean_loss_total = [], [], 0.0
    reliability_values, damage_values, target_values = [], [], []
    selection = {
        name: {"labels": [], "predictions": [], "loss": 0.0}
        for name in ("predicted", "oracle", "random", "anti_oracle")
    }
    corruption_records = {
        kind: {"labels": [], "predictions": [], "loss": 0.0,
               "score_drop": [], "task_harm": [], "detector_labels": [],
               "detector_scores": []}
        for kind in UNSEEN_CORRUPTIONS
    }
    seen = 0
    random_generator = np.random.default_rng(seed)
    for batch_index, (x, quality_x, target) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        quality_x = quality_x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        clean_logits, clean_aux = model(x, quality_x, return_aux=True)
        clean_losses = per_sample_task_loss(clean_logits, target, criterion)
        clean_loss_total += clean_losses.sum().item()
        clean_labels.extend(target.cpu().tolist())
        clean_predictions.extend(clean_logits.argmax(1).cpu().tolist())
        reliability = clean_aux["sensor_reliability"]

        group_logits, group_losses = [], []
        for group_index in range(len(groups)):
            selected = torch.zeros(
                len(x), len(groups), dtype=torch.bool, device=device,
            )
            selected[:, group_index] = True
            masked_x, masked_quality = mask_groups(x, quality_x, selected, groups)
            logits = model(masked_x, masked_quality)
            group_logits.append(logits)
            group_losses.append(per_sample_task_loss(logits, target, criterion))
        stacked_logits = torch.stack(group_logits, dim=1)
        stacked_losses = torch.stack(group_losses, dim=1)
        damage = stacked_losses - clean_losses.unsqueeze(1)
        utility = torch.softmax(damage / 0.5, dim=1)
        utility = utility / utility.amax(1, keepdim=True).clamp_min(1e-6)
        reliability_values.extend(reliability.flatten().cpu().tolist())
        damage_values.extend(damage.flatten().cpu().tolist())
        target_values.extend(utility.flatten().cpu().tolist())

        selectors = {
            "predicted": reliability.argmin(1),
            "oracle": damage.argmin(1),
            "anti_oracle": damage.argmax(1),
            "random": torch.as_tensor(
                random_generator.integers(0, len(groups), size=len(x)), device=device,
            ),
        }
        rows = torch.arange(len(x), device=device)
        for name, chosen in selectors.items():
            chosen_logits = stacked_logits[rows, chosen]
            chosen_loss = per_sample_task_loss(
                chosen_logits, target, criterion,
            ).sum().item()
            selection[name]["loss"] += chosen_loss
            selection[name]["labels"].extend(target.cpu().tolist())
            selection[name]["predictions"].extend(chosen_logits.argmax(1).cpu().tolist())

        for kind_index, kind in enumerate(UNSEEN_CORRUPTIONS):
            corrupted_logits, corrupted_losses, corrupted_scores = [], [], []
            for group_index, group in enumerate(groups):
                corrupted_x, corrupted_quality = corrupt_one_group(
                    x, quality_x, group, kind,
                    seed + batch_index * 1009 + kind_index * 101 + group_index,
                )
                logits, auxiliary = model(corrupted_x, corrupted_quality, return_aux=True)
                corrupted_logits.append(logits)
                corrupted_losses.append(per_sample_task_loss(logits, target, criterion))
                corrupted_scores.append(auxiliary["sensor_reliability"])
            all_logits = torch.stack(corrupted_logits, dim=1)
            all_losses = torch.stack(corrupted_losses, dim=1)
            all_scores = torch.stack(corrupted_scores, dim=1)
            chosen = torch.as_tensor(
                np.arange(seen, seen + len(x)) % len(groups), device=device,
            )
            rows = torch.arange(len(x), device=device)
            chosen_logits = all_logits[rows, chosen]
            chosen_losses = all_losses[rows, chosen]
            chosen_scores = all_scores[rows, chosen]
            chosen_group_score = chosen_scores[rows, chosen]
            original_group_score = reliability[rows, chosen]
            record = corruption_records[kind]
            record["loss"] += chosen_losses.sum().item()
            record["labels"].extend(target.cpu().tolist())
            record["predictions"].extend(chosen_logits.argmax(1).cpu().tolist())
            record["score_drop"].extend(
                (original_group_score - chosen_group_score).cpu().tolist()
            )
            record["task_harm"].extend(
                (chosen_losses - clean_losses).cpu().tolist()
            )
            detector_label = torch.zeros_like(chosen_scores)
            detector_label[rows, chosen] = 1
            record["detector_labels"].extend(detector_label.flatten().cpu().tolist())
            record["detector_scores"].extend((-chosen_scores).flatten().cpu().tolist())
        seen += len(x)

    clean = basic_metrics(
        clean_labels, clean_predictions, clean_loss_total / max(seen, 1), classes,
    )
    selection_metrics = {
        name: basic_metrics(
            record["labels"], record["predictions"], record["loss"] / max(seen, 1), classes,
        )
        for name, record in selection.items()
    }
    corruption_metrics = {}
    for kind, record in corruption_records.items():
        metrics = basic_metrics(
            record["labels"], record["predictions"], record["loss"] / max(seen, 1), classes,
        )
        metrics["score_drop_vs_task_harm_spearman"] = spearman(
            record["score_drop"], record["task_harm"],
        )
        try:
            metrics["corrupted_group_detection_auroc"] = float(roc_auc_score(
                record["detector_labels"], record["detector_scores"],
            ))
        except ValueError:
            metrics["corrupted_group_detection_auroc"] = None
        corruption_metrics[kind] = metrics
    return {
        "complete": clean,
        "task_relevance": {
            "reliability_vs_counterfactual_damage_pearson": correlation(
                reliability_values, damage_values,
            ),
            "reliability_vs_counterfactual_damage_spearman": spearman(
                reliability_values, damage_values,
            ),
            "calibration_to_normalized_task_utility": calibration_summary(
                reliability_values, target_values,
            ),
        },
        "group_removal_controls": selection_metrics,
        "unseen_corruptions": corruption_metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("pamap2", "sleep"), required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diagnostic_samples", type=int, default=8192)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    protocol = checkpoint["protocol"]
    if protocol["dataset"] != args.dataset:
        raise RuntimeError("checkpoint dataset mismatch")
    loader_function = load_pamap2 if args.dataset == "pamap2" else load_sleep
    loaders, checksum, counts, config = loader_function(
        args.data, args.batch_size, args.workers, splits=("train", "test"),
    )
    if checksum != protocol["canonical_checksum"]:
        raise RuntimeError("canonical checksum mismatch")
    groups = tuple(
        tuple(group) for group in protocol["physical_evaluation_groups"]
    )
    test_loader = loaders["test"]
    if args.diagnostic_samples and len(test_loader.dataset) > args.diagnostic_samples:
        generator = np.random.default_rng(args.seed + 20261107)
        indices = np.sort(generator.choice(
            len(test_loader.dataset), size=args.diagnostic_samples, replace=False,
        ))
        test_loader = DataLoader(
            Subset(test_loader.dataset, indices.tolist()),
            batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
            pin_memory=torch.cuda.is_available(), persistent_workers=args.workers > 0,
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(checkpoint, device)
    weights = np.minimum(
        counts.sum() / (config["classes"] * np.maximum(counts, 1)), 5.0,
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.as_tensor(weights, dtype=torch.float32, device=device),
        label_smoothing=0.05,
    )
    result = {
        "dataset": args.dataset,
        "source_checkpoint": str(Path(args.checkpoint).resolve()),
        "canonical_checksum": checksum,
        "diagnostic_policy": (
            "predeclared deterministic test subset; oracle uses labels only as an upper-bound control"
        ),
        "diagnostic_samples": len(test_loader.dataset),
        "results": diagnose(
            model, test_loader, device, criterion, config["classes"], groups,
            args.seed,
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"TASK_RELIABILITY_COMPLETE dataset={args.dataset} "
        f"checkpoint={args.checkpoint} output={output}", flush=True,
    )
    print(json.dumps(result["results"]["task_relevance"], indent=2), flush=True)


if __name__ == "__main__":
    main()
