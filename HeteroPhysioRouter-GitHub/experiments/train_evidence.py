import argparse
import copy
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from heterophysiorouter import QualityAwareRouter
from experiments.train import basic_metrics, load_pamap2, load_sleep


GROUPING_ARMS = (
    "physical_aligned",
    "random_aligned",
    "random_repr_physical_failure",
    "physical_repr_random_corruption",
)
ROBUST_ARMS = (
    "clean",
    "group_moddrop",
    "channel_dropout",
    "sensor_token_mask",
    "embracenet_style",
    "mixed_group",
)
COUNTERFACTUAL_ARMS = (
    "proxy_mixed",
    "cf_utility",
    "cf_integrity",
    "cf_joint",
)
ALL_ARMS = GROUPING_ARMS + ROBUST_ARMS + COUNTERFACTUAL_ARMS


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def physical_groups(config):
    return tuple(tuple(range(start, end)) for start, end in config["groups"])


def random_size_matched_groups(groups, seed=20261107):
    generator = np.random.default_rng(seed)
    channels = np.asarray([value for group in groups for value in group], dtype=np.int64)
    generator.shuffle(channels)
    result, offset = [], 0
    for group in groups:
        result.append(tuple(sorted(channels[offset:offset + len(group)].tolist())))
        offset += len(group)
    return tuple(result)


def mask_groups(x, quality_x, selected, groups):
    x = x.clone()
    quality_x = quality_x.clone()
    for group_index, indices in enumerate(groups):
        rows = selected[:, group_index]
        if rows.any():
            for value in (x, quality_x):
                subset = value[rows].clone()
                subset[:, :, list(indices)] = 0
                value[rows] = subset
    return x, quality_x


def corrupt_explicit_groups(x, quality_x, probability, groups, mode, generator=None):
    if probability <= 0:
        severity = torch.zeros(len(x), len(groups), device=x.device, dtype=x.dtype)
        return x, quality_x, severity
    x = x.clone()
    quality_x = quality_x.clone()
    selected = torch.rand(
        (len(x), len(groups)), device=x.device, generator=generator,
    ) < probability
    all_selected = selected.all(1)
    if all_selected.any():
        rows = all_selected.nonzero(as_tuple=False).squeeze(1)
        keep = torch.randint(
            len(groups), (len(rows),), device=x.device, generator=generator,
        )
        selected[rows, keep] = False
    kinds_for_mode = {
        "dropout": (0,),
        "mixed": (0, 1, 2),
    }
    kinds = kinds_for_mode[mode]
    severity = torch.zeros(len(x), len(groups), device=x.device, dtype=x.dtype)
    for group_index, indices in enumerate(groups):
        rows = selected[:, group_index].nonzero(as_tuple=False).squeeze(1)
        if not len(rows):
            continue
        choices = torch.as_tensor(kinds, device=x.device)
        degradation = choices[torch.randint(
            len(choices), (len(rows),), device=x.device, generator=generator,
        )]
        for kind in kinds:
            kind_rows = rows[degradation == kind]
            if not len(kind_rows):
                continue
            index = list(indices)
            if kind == 0:
                for value in (x, quality_x):
                    subset = value[kind_rows].clone()
                    subset[:, :, index] = 0
                    value[kind_rows] = subset
                severity[kind_rows, group_index] = 1
            elif kind == 1:
                amount = torch.empty(
                    (len(kind_rows), 1, 1), device=x.device, dtype=x.dtype,
                ).uniform_(0.5, 1.5, generator=generator)
                for value in (x, quality_x):
                    subset = value[kind_rows].clone()
                    segment = subset[:, :, index]
                    scale = segment.std(1, keepdim=True, unbiased=False).clamp_min(1e-4)
                    noise = torch.randn(
                        segment.shape, device=x.device, dtype=x.dtype, generator=generator,
                    )
                    subset[:, :, index] = segment + noise * scale * amount
                    value[kind_rows] = subset
                severity[kind_rows, group_index] = amount[:, 0, 0] / 1.5
            else:
                for value in (x, quality_x):
                    subset = value[kind_rows].clone()
                    segment = subset[:, :, index]
                    subset[:, :, index] = segment.mean(1, keepdim=True)
                    value[kind_rows] = subset
                severity[kind_rows, group_index] = 0.9
    return x, quality_x, severity


def corrupt_channels(x, quality_x, probability, generator=None):
    selected = torch.rand(
        (len(x), x.shape[-1]), device=x.device, generator=generator,
    ) < probability
    all_selected = selected.all(1)
    if all_selected.any():
        rows = all_selected.nonzero(as_tuple=False).squeeze(1)
        keep = torch.randint(
            x.shape[-1], (len(rows),), device=x.device, generator=generator,
        )
        selected[rows, keep] = False
    x = x.clone().masked_fill(selected.unsqueeze(1), 0)
    quality_x = quality_x.clone().masked_fill(selected.unsqueeze(1), 0)
    return x, quality_x


def per_sample_task_loss(logits, target, criterion):
    return nn.functional.cross_entropy(
        logits, target, weight=criterion.weight,
        label_smoothing=criterion.label_smoothing, reduction="none",
    )


@torch.no_grad()
def counterfactual_utility(model, x, quality_x, target, groups, criterion,
                           temperature=0.5):
    was_training = model.training
    model.eval()
    clean_loss = per_sample_task_loss(model(x, quality_x), target, criterion)
    damages = []
    for group_index in range(len(groups)):
        selected = torch.zeros(
            len(x), len(groups), dtype=torch.bool, device=x.device,
        )
        selected[:, group_index] = True
        masked_x, masked_quality = mask_groups(x, quality_x, selected, groups)
        masked_loss = per_sample_task_loss(
            model(masked_x, masked_quality), target, criterion,
        )
        damages.append(masked_loss - clean_loss)
    damage = torch.stack(damages, dim=1)
    utility = torch.softmax(damage / temperature, dim=1)
    utility = utility / utility.amax(1, keepdim=True).clamp_min(1e-6)
    model.train(was_training)
    return utility, damage


def reliability_loss(arm, clean_aux, corrupt_aux, utility, severity):
    clean_score = clean_aux["sensor_reliability"]
    corrupt_score = corrupt_aux["sensor_reliability"]
    integrity = (1 - severity).clamp(0, 1)
    if arm == "cf_utility":
        return nn.functional.mse_loss(clean_score, utility)
    if arm == "cf_integrity":
        return nn.functional.mse_loss(corrupt_score, integrity)
    if arm == "cf_joint":
        clean_term = nn.functional.mse_loss(clean_score, utility)
        corrupt_term = nn.functional.mse_loss(corrupt_score, utility * integrity)
        return 0.5 * clean_term + corrupt_term
    return clean_score.sum() * 0


def apply_condition(x, quality_x, missing, groups):
    if not missing:
        return x, quality_x
    selected = torch.zeros(len(x), len(groups), dtype=torch.bool, device=x.device)
    selected[:, list(missing)] = True
    return mask_groups(x, quality_x, selected, groups)


@torch.no_grad()
def evaluate(model, loader, criterion, device, classes, groups, conditions, max_batches=0):
    results = {}
    model.eval()
    for condition_name, missing in conditions:
        labels, predictions, total_loss, seen = [], [], 0.0, 0
        scores, gates = [], []
        for batch_index, (x, quality_x, target) in enumerate(loader, 1):
            if max_batches and batch_index > max_batches:
                break
            x = x.to(device, non_blocking=True)
            quality_x = quality_x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            x, quality_x = apply_condition(x, quality_x, missing, groups)
            logits, auxiliary = model(x, quality_x, return_aux=True)
            total_loss += criterion(logits, target).item() * len(target)
            seen += len(target)
            labels.extend(target.cpu().tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
            scores.append(auxiliary["sensor_reliability"].cpu())
            gates.append(auxiliary["gates"].cpu())
        result = basic_metrics(labels, predictions, total_loss / max(seen, 1), classes)
        result["sensor_reliability_mean"] = torch.cat(scores).mean(0).tolist()
        result["branch_gate_mean"] = torch.cat(gates).mean(0).tolist()
        results[condition_name] = result
    return results


def condition_list(group_names, include_pairs):
    conditions = [("complete", ())]
    conditions.extend(
        (f"missing_{name}", (index,)) for index, name in enumerate(group_names)
    )
    if include_pairs:
        for first in range(len(group_names)):
            for second in range(first + 1, len(group_names)):
                conditions.append((
                    f"missing_{group_names[first]}_{group_names[second]}",
                    (first, second),
                ))
    return conditions


def arm_plan(family, arm, physical, random_groups):
    if family == "grouping":
        plans = {
            "physical_aligned": (physical, physical),
            "random_aligned": (random_groups, random_groups),
            "random_repr_physical_failure": (random_groups, physical),
            "physical_repr_random_corruption": (physical, random_groups),
        }
        representation, corruption = plans[arm]
        return representation, corruption, "mixed", 0.0
    if family == "robust":
        token_probability = 0.25 if arm == "sensor_token_mask" else 0.0
        mode = {
            "clean": "none",
            "group_moddrop": "dropout",
            "channel_dropout": "channel",
            "sensor_token_mask": "none",
            "embracenet_style": "dropout",
            "mixed_group": "mixed",
        }[arm]
        return physical, physical, mode, token_probability
    return physical, physical, "mixed", 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("pamap2", "sleep"), required=True)
    parser.add_argument("--family", choices=("grouping", "robust", "counterfactual"), required=True)
    parser.add_argument("--arm", choices=ALL_ARMS, required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask_probability", type=float, default=0.25)
    parser.add_argument("--counterfactual_weight", type=float, default=0.2)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_eval_batches", type=int, default=0)
    args = parser.parse_args()

    family_arms = {
        "grouping": GROUPING_ARMS,
        "robust": ROBUST_ARMS,
        "counterfactual": COUNTERFACTUAL_ARMS,
    }
    if args.arm not in family_arms[args.family]:
        parser.error(f"arm {args.arm} does not belong to family {args.family}")
    if args.family == "grouping" and args.dataset != "pamap2":
        parser.error("size-matched random grouping is only identifiable on PAMAP2")

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader_function = load_pamap2 if args.dataset == "pamap2" else load_sleep
    loaders, checksum, counts, config = loader_function(
        args.data, args.batch_size, args.workers, splits=("train", "val"),
    )
    physical = physical_groups(config)
    random_groups = random_size_matched_groups(physical)
    representation_groups, corruption_groups, corruption_mode, token_probability = arm_plan(
        args.family, args.arm, physical, random_groups,
    )
    model = QualityAwareRouter(
        config["channels"], config["samples"], config["patch"], config["classes"],
        config["groups"], "full_no_calibration", args.hidden,
        channel_group_indices=representation_groups,
        sensor_token_mask_probability=token_probability,
        sensor_fusion="embrace" if args.arm == "embracenet_style" else "weighted",
    ).to(device)
    weights = np.minimum(
        counts.sum() / (config["classes"] * np.maximum(counts, 1)), 5.0,
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.as_tensor(weights, dtype=torch.float32, device=device),
        label_smoothing=0.05,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "version": "tpami_evidence_v1",
        "dataset": args.dataset,
        "family": args.family,
        "arm": args.arm,
        "canonical_checksum": checksum,
        "selection_metric": "validation macro_f1 on complete input",
        "test_policy": "predeclared arm; test evaluated once after validation selection",
        "representation_groups": [list(group) for group in representation_groups],
        "corruption_groups": [list(group) for group in corruption_groups],
        "physical_evaluation_groups": [list(group) for group in physical],
        "random_group_seed": 20261107,
        "corruption_mode": corruption_mode,
        "mask_probability": args.mask_probability,
        "sensor_token_mask_probability": token_probability,
        "sensor_fusion": "embrace" if args.arm == "embracenet_style" else "weighted",
        "baseline_scope": (
            "EmbraceNet-style feature-wise stochastic sensor-token embracement with group ModDrop; not an exact original-network reproduction"
            if args.arm == "embracenet_style" else None
        ),
        "counterfactual_target": (
            None if args.family != "counterfactual" else args.arm
        ),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "input": {
            key: config[key] for key in ("channels", "samples", "patch", "classes")
        },
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print("=== PREDECLARED TPAMI EVIDENCE EXPERIMENT ===", flush=True)
    print(
        f"dataset={args.dataset} family={args.family} arm={args.arm} seed={args.seed} "
        f"checksum={checksum}", flush=True,
    )
    print(
        f"representation_groups={protocol['representation_groups']} "
        f"corruption_groups={protocol['corruption_groups']}", flush=True,
    )

    corruption_generator = torch.Generator(device=device).manual_seed(args.seed + 20261107)
    best_score, best_epoch, best_state, best_validation, stale = -1.0, 0, None, None, 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, seen = 0.0, 0
        for step, (x, quality_x, target) in enumerate(loaders["train"], 1):
            if args.max_train_batches and step > args.max_train_batches:
                break
            x = x.to(device, non_blocking=True)
            quality_x = quality_x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            utility = None
            if args.family == "counterfactual" and args.arm != "proxy_mixed":
                utility, _ = counterfactual_utility(
                    model, x, quality_x, target, physical, criterion,
                )
            if corruption_mode in ("dropout", "mixed"):
                train_x, train_quality, severity = corrupt_explicit_groups(
                    x, quality_x, args.mask_probability, corruption_groups,
                    corruption_mode, generator=corruption_generator,
                )
            elif corruption_mode == "channel":
                train_x, train_quality = corrupt_channels(
                    x, quality_x, args.mask_probability, generator=corruption_generator,
                )
                severity = torch.zeros(
                    len(x), len(physical), device=device, dtype=x.dtype,
                )
            else:
                train_x, train_quality = x, quality_x
                severity = torch.zeros(
                    len(x), len(physical), device=device, dtype=x.dtype,
                )

            optimizer.zero_grad(set_to_none=True)
            logits, corrupt_aux = model(train_x, train_quality, return_aux=True)
            loss = criterion(logits, target)
            if utility is not None:
                _, clean_aux = model(x, quality_x, return_aux=True)
                cf_loss = reliability_loss(
                    args.arm, clean_aux, corrupt_aux, utility, severity,
                )
                loss = loss + args.counterfactual_weight * cf_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(target)
            seen += len(target)
            if step == 1 or step % 100 == 0:
                print(
                    f"epoch {epoch}/{args.epochs} step {step}/{len(loaders['train'])} "
                    f"loss={loss.item():.5f}", flush=True,
                )

        validation = evaluate(
            model, loaders["val"], criterion, device, config["classes"], physical,
            [("complete", ())],
            max_batches=args.max_eval_batches,
        )["complete"]
        print(
            f"EPOCH {epoch:02d}/{args.epochs} train_loss={total_loss / max(seen, 1):.6f} "
            f"val={validation}", flush=True,
        )
        if validation["macro_f1"] > best_score:
            best_score = validation["macro_f1"]
            best_epoch = epoch
            best_validation = validation
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
            torch.save(
                {"model": best_state, "protocol": protocol, "args": vars(args)},
                output / "best.pt",
            )
        else:
            stale += 1
            if stale >= args.patience:
                print(f"EARLY_STOPPING epoch={epoch} patience={args.patience}", flush=True)
                break
    if best_state is None:
        raise RuntimeError("training produced no validation checkpoint")
    model.load_state_dict(best_state)

    physical_conditions = condition_list(
        config["group_names"], include_pairs=True,
    )
    validation_conditions = evaluate(
        model, loaders["val"], criterion, device, config["classes"], physical,
        physical_conditions,
        max_batches=args.max_eval_batches,
    )
    if args.family == "grouping":
        random_conditions = [
            (f"random_missing_group{index}", (index,))
            for index in range(len(random_groups))
        ]
        validation_conditions.update(evaluate(
            model, loaders["val"], criterion, device, config["classes"], random_groups,
            random_conditions, max_batches=args.max_eval_batches,
        ))
    test_loaders, test_checksum, _, test_config = loader_function(
        args.data, args.batch_size, args.workers, splits=("train", "test"),
    )
    if test_checksum != checksum:
        raise RuntimeError("canonical checksum changed before locked test evaluation")
    test_conditions = evaluate(
        model, test_loaders["test"], criterion, device, config["classes"], physical,
        physical_conditions,
        max_batches=args.max_eval_batches,
    )
    if args.family == "grouping":
        test_conditions.update(evaluate(
            model, test_loaders["test"], criterion, device, config["classes"], random_groups,
            random_conditions, max_batches=args.max_eval_batches,
        ))
    result = {
        "protocol": protocol,
        "best_epoch": best_epoch,
        "best_validation_complete": best_validation,
        "validation_conditions": validation_conditions,
        "test_conditions": test_conditions,
        "elapsed_seconds": time.time() - start,
        "args": vars(args),
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"FINAL_EVIDENCE dataset={args.dataset} family={args.family} arm={args.arm} "
        f"seed={args.seed} test_complete={test_conditions['complete']}", flush=True,
    )


if __name__ == "__main__":
    main()
