import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from heterophysiorouter.local_corruption import corrupt_group_time
from heterophysiorouter.local_correction import (
    GATE_RULES,
    PILOT_MODES,
    TaskRelevanceLocalAdapter,
    balanced_reliability_loss,
    clean_identity_loss,
    combine_joint_gate,
    sampled_relevance_loss,
)
from experiments.train import apply_missing, basic_metrics, load_pamap2

SEEN_CORRUPTIONS = ("dropout", "noise", "flatline")
UNSEEN_CORRUPTIONS = ("drift", "spike", "gain")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def per_sample_loss(logits, target, criterion):
    return nn.functional.cross_entropy(
        logits, target, weight=criterion.weight,
        label_smoothing=criterion.label_smoothing, reduction="none",
    )


def mask_selected_regions(x, quality_x, severity, groups, patch):
    masked_x = x.clone()
    masked_quality = quality_x.clone()
    selected = severity > 0
    for group, (start, end) in enumerate(groups):
        for patch_index in range(severity.size(2)):
            rows = selected[:, group, patch_index]
            if rows.any():
                left = patch_index * patch
                right = left + patch
                masked_x[rows, left:right, start:end] = 0
                masked_quality[rows, left:right, start:end] = 0
    return masked_x, masked_quality


@torch.no_grad()
def task_relevance_target(model, clean_x, clean_quality, target, severity,
                          groups, patch, criterion, temperature):
    selected = severity > 0
    target_grid = torch.zeros_like(severity)
    if not selected.any():
        return target_grid, target_grid, selected
    base_logits = model.base(clean_x, clean_quality)
    masked_x, masked_quality = mask_selected_regions(
        clean_x, clean_quality, severity, groups, patch,
    )
    masked_logits = model.base(masked_x, masked_quality)
    damage = (
        per_sample_loss(masked_logits, target, criterion)
        - per_sample_loss(base_logits, target, criterion)
    )
    relevance = 1.0 - torch.exp(-torch.relu(damage) / max(temperature, 1e-6))
    target_grid[selected] = relevance.unsqueeze(1).unsqueeze(2).expand_as(severity)[selected]
    damage_grid = torch.zeros_like(severity)
    damage_grid[selected] = damage.unsqueeze(1).unsqueeze(2).expand_as(severity)[selected]
    return target_grid, damage_grid, selected


def load_base_checkpoint(model, path, checksum):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["protocol"]["canonical_checksum"] != checksum:
        raise RuntimeError("base checkpoint canonical checksum mismatch")
    model.base.load_state_dict(checkpoint["model"], strict=True)
    model.freeze_base()
    return checkpoint["protocol"]


def oracle_kwargs(mode, severity, relevance):
    if mode != "oracle":
        return {}
    return {
        "severity_override": severity,
        "relevance_override": relevance,
        "force_gate": True,
    }


@torch.no_grad()
def evaluate_condition(model, loader, criterion, device, config,
                       condition="complete", missing_group=None, seed=0,
                       relevance_temperature=0.25, max_batches=0):
    model.eval()
    labels, predictions = [], []
    total_loss, count = 0.0, 0
    gate_values, correction_values = [], []
    detection_targets, detection_scores = [], []
    relevance_targets, relevance_scores, relevance_damages = [], [], []
    generator = torch.Generator(device=device).manual_seed(seed)
    for step, (clean_x, clean_quality, target) in enumerate(loader, 1):
        if max_batches and step > max_batches:
            break
        clean_x = clean_x.to(device, non_blocking=True)
        clean_quality = clean_quality.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        x, quality_x = clean_x, clean_quality
        severity = torch.zeros(
            len(x), len(config["groups"]), config["samples"] // config["patch"],
            device=device, dtype=x.dtype,
        )
        if missing_group is not None:
            x, quality_x = apply_missing(
                x, quality_x, (missing_group,), config["groups"],
            )
            severity[:, missing_group] = 1.0
        elif condition != "complete":
            x, quality_x, severity = corrupt_group_time(
                x, quality_x, config["groups"], config["patch"],
                probability=1.0, full_span_probability=0.0,
                fixed_kind=None if condition == "mixed" else condition,
                fixed_span_fraction=1.0 / 3.0, generator=generator,
            )
        relevance, damage, supervised = task_relevance_target(
            model, clean_x, clean_quality, target, severity,
            config["groups"], config["patch"], criterion,
            relevance_temperature,
        )
        logits, auxiliary = model(
            x, quality_x, return_aux=True,
            **oracle_kwargs(model.mode, severity, relevance),
        )
        total_loss += criterion(logits, target).item() * len(target)
        count += len(target)
        labels.extend(target.cpu().tolist())
        predictions.extend(logits.argmax(1).cpu().tolist())
        gate_values.append(auxiliary["adapter_gate"].cpu())
        correction_values.append(auxiliary["applied_correction_norm"].cpu())
        if supervised.any():
            detection_targets.append((severity > 0).flatten().cpu())
            detection_scores.append(auxiliary["local_severity"].flatten().cpu())
            relevance_targets.append(relevance[supervised].cpu())
            relevance_scores.append(auxiliary["local_relevance"][supervised].cpu())
            relevance_damages.append(damage[supervised].cpu())
    result = basic_metrics(
        labels, predictions, total_loss / max(count, 1), config["classes"],
    )
    gates = torch.cat(gate_values)
    corrections = torch.cat(correction_values)
    result.update({
        "adapter_gate_mean": gates.mean().item(),
        "adapter_gate_active_fraction": (gates > 0.10).float().mean().item(),
        "applied_correction_norm": corrections.mean().item(),
    })
    if detection_targets:
        truth = torch.cat(detection_targets).numpy().astype(np.int64)
        score = torch.cat(detection_scores).numpy()
        result["localization_auroc"] = float(roc_auc_score(truth, score))
        result["localization_auprc"] = float(average_precision_score(truth, score))
        expected = torch.cat(relevance_targets).numpy()
        predicted = torch.cat(relevance_scores).numpy()
        damage_values = torch.cat(relevance_damages).numpy()
        correlation = spearmanr(expected, predicted).statistic
        damage_correlation = spearmanr(damage_values, predicted).statistic
        result["relevance_spearman"] = (
            None if not np.isfinite(correlation) else float(correlation)
        )
        result["task_damage_spearman"] = (
            None if not np.isfinite(damage_correlation) else float(damage_correlation)
        )
        result["relevance_target_mean"] = float(expected.mean())
        result["relevance_prediction_mean"] = float(predicted.mean())
    return result


def mean_macro_f1(results):
    return float(np.mean([value["macro_f1"] for value in results.values()]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=PILOT_MODES, required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--base_checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--gate_rule", choices=GATE_RULES, default="product")
    parser.add_argument("--gate_scale", type=float, default=1.0)
    parser.add_argument("--gate_temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_probability", type=float, default=0.25)
    parser.add_argument("--reliability_weight", type=float, default=0.2)
    parser.add_argument("--relevance_weight", type=float, default=0.2)
    parser.add_argument("--identity_weight", type=float, default=1.0)
    parser.add_argument("--sparsity_weight", type=float, default=0.1)
    parser.add_argument("--relevance_temperature", type=float, default=0.25)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_eval_batches", type=int, default=0)
    args = parser.parse_args()

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, checksum, counts, config = load_pamap2(
        args.data, args.batch_size, args.workers, splits=("train", "val"),
    )
    model = TaskRelevanceLocalAdapter(
        config["channels"], config["samples"], config["patch"],
        config["classes"], config["groups"], args.mode,
        hidden=args.hidden, rank=args.rank, gate_rule=args.gate_rule,
        gate_scale=args.gate_scale, gate_temperature=args.gate_temperature,
    ).to(device)
    base_protocol = load_base_checkpoint(model, args.base_checkpoint, checksum)
    trainable = model.trainable_parameters()
    trainable_count = sum(parameter.numel() for parameter in trainable)
    total_count = sum(parameter.numel() for parameter in model.parameters())
    class_weights = np.minimum(
        counts.sum() / (config["classes"] * np.maximum(counts, 1)), 5.0,
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.as_tensor(class_weights, dtype=torch.float32, device=device),
        label_smoothing=0.05,
    )
    optimizer = None if args.mode == "frozen" else torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=0.01,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "version": "task_relevance_local_adapter_pilot_v1",
        "dataset": "pamap2",
        "split": "train_validation_only",
        "test_accessed": False,
        "mode": args.mode,
        "seed": args.seed,
        "canonical_checksum": checksum,
        "base_checkpoint": str(Path(args.base_checkpoint).resolve()),
        "base_protocol": base_protocol,
        "base_frozen": True,
        "classifier_frozen": True,
        "classification_gradient_to_reliability": False,
        "classification_gradient_to_relevance": False,
        "pre_pooling_local_correction": True,
        "gate_rule": args.gate_rule,
        "gate_scale": args.gate_scale,
        "gate_temperature": args.gate_temperature,
        "parameter_matched_nonfrozen_controls": True,
        "trainable_parameters": trainable_count,
        "total_parameters": total_count,
        "trainable_fraction": trainable_count / total_count,
        "selection_metric": "0.5 * clean validation MF1 + 0.5 * deterministic mixed-local validation MF1",
        "groups": [list(range(start, end)) for start, end in config["groups"]],
        "input": {key: config[key] for key in ("channels", "samples", "patch", "classes")},
        "args": vars(args),
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8",
    )
    print("=== TASK-RELEVANCE-GUIDED LOCAL ADAPTER PILOT ===", flush=True)
    print(
        f"mode={args.mode} seed={args.seed} device={device} checksum={checksum}",
        flush=True,
    )
    print(
        f"parameters total={total_count:,} trainable={trainable_count:,} "
        f"fraction={trainable_count / total_count:.4%} test_accessed=False",
        flush=True,
    )

    best_score, best_epoch, best_state, best_validation = -1.0, 0, None, None
    stale, history = 0, []
    started = time.time()
    if args.mode == "frozen":
        best_state = copy.deepcopy(model.state_dict())
        clean = evaluate_condition(
            model, loaders["val"], criterion, device, config,
            max_batches=args.max_eval_batches,
        )
        mixed = evaluate_condition(
            model, loaders["val"], criterion, device, config, condition="mixed",
            seed=args.seed + 81000, relevance_temperature=args.relevance_temperature,
            max_batches=args.max_eval_batches,
        )
        best_validation = {"complete": clean, "mixed_local": mixed}
        torch.save(
            {"model": best_state, "protocol": protocol, "args": vars(args)},
            output / "best.pt",
        )
    else:
        corruption_generator = torch.Generator(device=device).manual_seed(args.seed + 80000)
        for epoch in range(1, args.epochs + 1):
            model.train()
            totals = {
                "loss": 0.0, "task": 0.0, "reliability": 0.0,
                "relevance": 0.0, "identity": 0.0, "sparsity": 0.0,
            }
            count = 0
            for step, (clean_x, clean_quality, target) in enumerate(loaders["train"], 1):
                if args.max_train_batches and step > args.max_train_batches:
                    break
                clean_x = clean_x.to(device, non_blocking=True)
                clean_quality = clean_quality.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                train_x, train_quality, severity = corrupt_group_time(
                    clean_x, clean_quality, config["groups"], config["patch"],
                    probability=args.local_probability, full_span_probability=0.10,
                    generator=corruption_generator,
                )
                if args.mode in ("relevance", "joint", "oracle"):
                    relevance, _, supervised = task_relevance_target(
                        model, clean_x, clean_quality, target, severity,
                        config["groups"], config["patch"], criterion,
                        args.relevance_temperature,
                    )
                else:
                    relevance = torch.zeros_like(severity)
                    supervised = torch.zeros_like(severity, dtype=torch.bool)

                optimizer.zero_grad(set_to_none=True)
                logits, auxiliary = model(
                    train_x, train_quality, return_aux=True,
                    **oracle_kwargs(args.mode, severity, relevance),
                )
                task_term = criterion(logits, target)
                reliability_term = task_term.new_zeros(())
                if args.mode in ("reliability", "joint", "oracle"):
                    reliability_term = balanced_reliability_loss(
                        auxiliary["local_severity"], severity,
                    )
                relevance_term = sampled_relevance_loss(
                    auxiliary["local_relevance"], relevance, supervised,
                )
                clean_override = None
                clean_relevance = None
                if args.mode == "oracle":
                    clean_override = torch.zeros_like(severity)
                    clean_relevance = torch.zeros_like(relevance)
                clean_logits, clean_auxiliary = model(
                    clean_x, clean_quality, return_aux=True,
                    severity_override=clean_override,
                    relevance_override=clean_relevance,
                    force_gate=True if args.mode == "oracle" else None,
                )
                with torch.no_grad():
                    base_logits = model.base(clean_x, clean_quality)
                identity_term = clean_identity_loss(
                    clean_logits, base_logits, clean_auxiliary,
                )
                if args.mode in ("plain", "availability", "oracle"):
                    sparse_activation = torch.zeros_like(
                        clean_auxiliary["local_severity"],
                    )
                elif args.mode == "reliability":
                    sparse_activation = clean_auxiliary["local_severity"]
                elif args.mode == "relevance":
                    sparse_activation = clean_auxiliary["local_relevance"]
                else:
                    sparse_activation = combine_joint_gate(
                        clean_auxiliary["local_severity"],
                        clean_auxiliary["local_relevance"],
                        args.gate_rule, args.gate_scale,
                        args.gate_temperature,
                    )
                sparsity_term = sparse_activation.mean()
                loss = (
                    task_term
                    + args.reliability_weight * reliability_term
                    + args.relevance_weight * relevance_term
                    + args.identity_weight * identity_term
                    + args.sparsity_weight * sparsity_term
                )
                loss.backward()
                nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                batch_count = len(target)
                count += batch_count
                for name, value in (
                    ("loss", loss), ("task", task_term),
                    ("reliability", reliability_term),
                    ("relevance", relevance_term),
                    ("identity", identity_term),
                    ("sparsity", sparsity_term),
                ):
                    totals[name] += value.item() * batch_count
                if step == 1 or step % 100 == 0:
                    print(
                        f"epoch {epoch}/{args.epochs} step {step}/{len(loaders['train'])} "
                        f"loss={loss.item():.5f} task={task_term.item():.5f} "
                        f"rel={reliability_term.item():.5f} "
                        f"relevance={relevance_term.item():.5f} "
                        f"identity={identity_term.item():.5f}", flush=True,
                    )
            clean = evaluate_condition(
                model, loaders["val"], criterion, device, config,
                max_batches=args.max_eval_batches,
            )
            mixed = evaluate_condition(
                model, loaders["val"], criterion, device, config,
                condition="mixed", seed=args.seed + 81000,
                relevance_temperature=args.relevance_temperature,
                max_batches=args.max_eval_batches,
            )
            score = 0.5 * clean["macro_f1"] + 0.5 * mixed["macro_f1"]
            record = {
                "epoch": epoch,
                **{name: value / max(count, 1) for name, value in totals.items()},
                "selection_score": score,
                "complete": clean,
                "mixed_local": mixed,
            }
            history.append(record)
            print(
                f"EPOCH {epoch:02d}/{args.epochs} train_loss={record['loss']:.6f} "
                f"clean_mf1={clean['macro_f1']:.5f} mixed_mf1={mixed['macro_f1']:.5f} "
                f"selection={score:.5f}", flush=True,
            )
            if score > best_score:
                best_score, best_epoch, stale = score, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
                best_validation = {"complete": clean, "mixed_local": mixed}
                torch.save(
                    {"model": best_state, "protocol": protocol, "args": vars(args)},
                    output / "best.pt",
                )
            else:
                stale += 1
                if stale >= args.patience:
                    print(
                        f"EARLY_STOPPING epoch={epoch} patience={args.patience}",
                        flush=True,
                    )
                    break

    if best_state is None:
        raise RuntimeError("pilot training produced no checkpoint")
    model.load_state_dict(best_state)
    complete = evaluate_condition(
        model, loaders["val"], criterion, device, config,
        max_batches=args.max_eval_batches,
    )
    missing = {
        name: evaluate_condition(
            model, loaders["val"], criterion, device, config,
            missing_group=index, seed=args.seed + 82000 + index,
            relevance_temperature=args.relevance_temperature,
            max_batches=args.max_eval_batches,
        )
        for index, name in enumerate(config["group_names"])
    }
    local = {
        kind: evaluate_condition(
            model, loaders["val"], criterion, device, config,
            condition=kind, seed=args.seed + 83000 + index,
            relevance_temperature=args.relevance_temperature,
            max_batches=args.max_eval_batches,
        )
        for index, kind in enumerate(SEEN_CORRUPTIONS + UNSEEN_CORRUPTIONS)
    }
    result = {
        "protocol": protocol,
        "best_epoch": best_epoch,
        "best_validation": best_validation,
        "history": history,
        "validation_complete": complete,
        "validation_missing": missing,
        "validation_local_corruptions": local,
        "mean_missing_macro_f1": mean_macro_f1(missing),
        "mean_seen_local_macro_f1": mean_macro_f1({
            key: local[key] for key in SEEN_CORRUPTIONS
        }),
        "mean_unseen_local_macro_f1": mean_macro_f1({
            key: local[key] for key in UNSEEN_CORRUPTIONS
        }),
        "elapsed_seconds": time.time() - started,
    }
    (output / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8",
    )
    print(
        f"FINAL_LOCAL_PILOT mode={args.mode} seed={args.seed} "
        f"complete={complete['macro_f1']:.5f} "
        f"missing={result['mean_missing_macro_f1']:.5f} "
        f"seen={result['mean_seen_local_macro_f1']:.5f} "
        f"unseen={result['mean_unseen_local_macro_f1']:.5f} "
        f"clean_gate={complete['adapter_gate_mean']:.5f}", flush=True,
    )


if __name__ == "__main__":
    main()
