import argparse
import hashlib
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset, TensorDataset

from heterophysiorouter import (CORRUPTION_MODES, VARIANTS, QualityAwareRouter,
                                corrupt_groups, monotonic_calibration_loss)


PAMAP2_GROUPS = ((0, 1), (1, 18), (18, 35), (35, 52))


SLEEP_RE = re.compile(r"^(?:\d{2}_){3}\d+_(SC\d+E\d+)_(\d+)\.hdf5$")


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def sleep_records(directory):
    records = defaultdict(list)
    for path in Path(directory).glob("*.hdf5"):
        match = SLEEP_RE.match(path.name)
        if match: records[match.group(1)].append((int(match.group(2)), path))
    for record in records: records[record].sort(key=lambda item: item[0])
    return dict(sorted(records.items()))


class SleepDataset(Dataset):
    def __init__(self, directory):
        self.records = sleep_records(directory)
        self.items = [path for values in self.records.values() for _, path in values]

    def __len__(self): return len(self.items)

    def __getitem__(self, index):
        with h5py.File(self.items[index], "r") as h5:
            eeg = np.asarray(h5["EEG"], dtype=np.float32).reshape(3000, -1)[:, :2]
            eog = np.asarray(h5["EOG"], dtype=np.float32).reshape(3000, -1)[:, :1]
            label = int(np.asarray(h5["stages"]).reshape(-1)[0])
        signal = np.concatenate((eeg, eog), axis=1)
        raw_signal = signal.copy()
        signal = (signal - signal.mean(0, keepdims=True)) / (signal.std(0, keepdims=True) + 1e-6)
        return torch.from_numpy(signal), torch.from_numpy(raw_signal), label


def pamap2_patch_length(window):
    """Keep six temporal patches across the declared window sensitivities."""
    if window % 6 == 0:
        return window // 6
    for candidate in range(min(window, 32), 0, -1):
        if window % candidate == 0:
            return candidate
    return 1


def load_pamap2(root, batch_size, workers, splits=("train", "val")):
    root = Path(root); manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    arrays = {}
    for split in splits:
        parts = [np.load(root / f"subject{s}.npz") for s in manifest["split"][split]]
        arrays[split] = (np.concatenate([p["x"] for p in parts]), np.concatenate([p["y"] for p in parts]))
    if "train" not in arrays:
        parts = [np.load(root / f"subject{s}.npz") for s in manifest["split"]["train"]]
        train_x = np.concatenate([p["x"] for p in parts])
        train_y = np.concatenate([p["y"] for p in parts])
    else:
        train_x, train_y = arrays["train"]
    mean = train_x.mean((0, 1), keepdims=True); std = train_x.std((0, 1), keepdims=True) + 1e-6
    loaders = {}
    for split, (x, y) in arrays.items():
        raw_x = x.astype(np.float32); x = ((x - mean) / std).astype(np.float32)
        loaders[split] = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(raw_x), torch.from_numpy(y.astype(np.int64))),
                                    batch_size=batch_size, shuffle=split == "train", num_workers=workers,
                                    pin_memory=True, persistent_workers=workers > 0)
        print(f"samples.{split}={len(y)} labels={np.bincount(y, minlength=12).tolist()}", flush=True)
    window = int(manifest["window"])
    config = {"channels": 52, "samples": window, "patch": pamap2_patch_length(window), "classes": 12,
              "groups": PAMAP2_GROUPS, "group_names": ("heart_rate", "hand", "chest", "ankle")}
    return loaders, manifest["checksum"], np.bincount(train_y, minlength=12), config


def load_sleep(root, batch_size, workers, splits=("train", "val")):
    root = Path(root); audit = json.loads((root / "audit.json").read_text(encoding="utf-8"))
    loaders = {}
    for split in splits:
        dataset = SleepDataset(root / split)
        loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=split == "train", num_workers=workers,
                                    pin_memory=True, persistent_workers=workers > 0)
        print(f"samples.{split}={len(dataset)} records={len(dataset.records)} labels={audit[split]['labels']}", flush=True)
    config = {"channels": 3, "samples": 3000, "patch": 100, "classes": 5,
              "groups": ((0, 1), (1, 2), (2, 3)), "group_names": ("eeg1", "eeg2", "eog")}
    return loaders, audit["checksum"], np.asarray(audit["train"]["labels"]), config


def basic_metrics(labels, predictions, loss, classes):
    return {"loss": loss, "accuracy": accuracy_score(labels, predictions),
            "balanced_accuracy": balanced_accuracy_score(labels, predictions),
            "macro_f1": f1_score(labels, predictions, labels=range(classes), average="macro", zero_division=0),
            "kappa": cohen_kappa_score(labels, predictions),
            "f1_per_class": f1_score(labels, predictions, labels=range(classes), average=None, zero_division=0).tolist(),
            "pred_dist": np.bincount(predictions, minlength=classes).tolist(),
            "confusion": confusion_matrix(labels, predictions, labels=range(classes)).tolist()}


def missing_conditions(groups, names, include_pairs=False):
    yield "complete", ()
    for index, name in enumerate(names): yield f"missing_{name}", (index,)
    if include_pairs:
        for first in range(len(groups)):
            for second in range(first + 1, len(groups)):
                yield f"missing_{names[first]}_{names[second]}", (first, second)


def apply_missing(x, quality_x, missing, groups):
    if not missing: return x, quality_x
    x = x.clone(); quality_x = quality_x.clone()
    for index in missing:
        start, end = groups[index]; x[:, :, start:end] = 0; quality_x[:, :, start:end] = 0
    return x, quality_x


@torch.no_grad()
def evaluate(model, loader, criterion, device, classes, groups, missing=(), max_batches=0):
    model.eval(); labels=[]; predictions=[]; total=0.; count=0; gates=[]; sensor_quality=[]; temporal_quality=[]; class_gates=defaultdict(list)
    for step, (x, quality_x, y) in enumerate(loader, 1):
        if max_batches and step > max_batches: break
        x=x.to(device,non_blocking=True); quality_x=quality_x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
        x,quality_x=apply_missing(x,quality_x,missing,groups)
        logits,aux=model(x,quality_x,return_aux=True); total+=criterion(logits,y).item()*len(x); count+=len(x)
        pred=logits.argmax(1); labels.extend(y.cpu().tolist()); predictions.extend(pred.cpu().tolist()); gates.append(aux["gates"].cpu())
        sensor_quality.append(aux["sensor_reliability"].mean(1).cpu()); temporal_quality.append(aux["temporal_reliability"].mean(1).cpu())
        for label,gate in zip(y.cpu().tolist(),aux["gates"].cpu().tolist()): class_gates[label].append(gate)
    result=basic_metrics(labels,predictions,total/count,classes); gate_tensor=torch.cat(gates)
    result["gate_mean"]={name:value for name,value in zip(("sensor","temporal","global"),gate_tensor.mean(0).tolist())}
    result["gate_std"]={name:value for name,value in zip(("sensor","temporal","global"),gate_tensor.std(0,unbiased=False).tolist())}
    result["gate_by_class"]={str(label):np.mean(values,axis=0).tolist() for label,values in class_gates.items()}
    result["mean_sensor_reliability"]=torch.cat(sensor_quality).mean().item(); result["mean_temporal_reliability"]=torch.cat(temporal_quality).mean().item()
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument("--dataset",choices=("pamap2","sleep"),required=True)
    p.add_argument("--variant",choices=VARIANTS,required=True); p.add_argument("--data",required=True); p.add_argument("--output",required=True)
    p.add_argument("--epochs",type=int,default=20); p.add_argument("--patience",type=int,default=5); p.add_argument("--batch_size",type=int,default=256)
    p.add_argument("--workers",type=int,default=4); p.add_argument("--lr",type=float,default=5e-4); p.add_argument("--hidden",type=int,default=128)
    p.add_argument("--seed",type=int,default=42); p.add_argument("--mask_probability",type=float,default=.25)
    p.add_argument("--corruption_mode", choices=("auto",) + CORRUPTION_MODES, default="auto")
    p.add_argument("--isolate_corruption_rng", action="store_true")
    p.add_argument("--max_train_batches",type=int,default=0); p.add_argument("--max_val_batches",type=int,default=0)
    a=p.parse_args(); seed_all(a.seed); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders,checksum,counts,config=(load_pamap2 if a.dataset=="pamap2" else load_sleep)(a.data,a.batch_size,a.workers)
    model=QualityAwareRouter(config["channels"],config["samples"],config["patch"],config["classes"],
                             config["groups"],a.variant,a.hidden).to(device)
    weights=np.minimum(counts.sum()/(config["classes"]*np.maximum(counts,1)),5.)
    criterion=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device),label_smoothing=.05)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01); output=Path(a.output); output.mkdir(parents=True,exist_ok=True)
    protocol={"dataset":a.dataset,"variant":a.variant,"canonical_checksum":checksum,"test_locked":True,"selection_metric":"validation macro_f1",**config}
    protocol["training_corruption"] = {
        "mode": a.corruption_mode,
        "mask_probability": a.mask_probability,
        "rng_isolated_from_model_dropout": a.isolate_corruption_rng,
    }
    protocol["groups"]=[list(g) for g in protocol["groups"]]
    (output/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    print(f"DEVELOPMENT_ONLY test_locked=True dataset={a.dataset} variant={a.variant} checksum={checksum}",flush=True)
    print(f"input={config['samples']}x{config['channels']} patch={config['patch']} trainable={sum(p.numel() for p in model.parameters()):,}",flush=True)
    best=-1.; stale=0; state=None; best_val=None; best_epoch=0; start=time.time()
    for epoch in range(1,a.epochs+1):
        model.train(); total=0.; count=0
        for step,(x,quality_x,y) in enumerate(loaders["train"],1):
            if a.max_train_batches and step>a.max_train_batches: break
            x=x.to(device,non_blocking=True); quality_x=quality_x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            calibrated=a.variant in ("quality_calibrated","cross_quality","full","full_no_cross")
            final_ablation=a.variant.startswith("degrade_")
            paired_views=(
                a.variant in ("full","full_no_cross","full_no_calibration","cross_quality","no_degradation")
                or final_ablation
            )
            corrupted_classification=(
                a.variant in ("full","full_no_cross","full_no_calibration")
                or (final_ablation and a.variant != "no_degradation")
            )
            if paired_views:
                corruption_mode={
                    "degrade_dropout_only":"dropout",
                    "degrade_noise_only":"noise",
                    "degrade_flatline_only":"flatline",
                }.get(a.variant,"mixed")
                if a.corruption_mode != "auto":
                    corruption_mode = a.corruption_mode
                isolate_corruption_rng=a.isolate_corruption_rng or a.variant in (
                    "degrade_mixed_reference","degrade_dropout_only",
                    "degrade_noise_only","degrade_flatline_only",
                )
                if isolate_corruption_rng:
                    cpu_rng=torch.random.get_rng_state()
                    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                train_x,train_quality,severity=corrupt_groups(
                    x,quality_x,a.mask_probability,config["groups"],mode=corruption_mode
                )
                if isolate_corruption_rng:
                    torch.random.set_rng_state(cpu_rng)
                    if cuda_rng is not None:
                        torch.cuda.set_rng_state_all(cuda_rng)
                corrupt_logits,corrupt_aux=model(train_x,train_quality,return_aux=True)
                clean_logits,clean_aux=model(x,quality_x,return_aux=True)
                logits=corrupt_logits if corrupted_classification else clean_logits
            else:
                if calibrated:
                    logits,clean_aux=model(x,quality_x,return_aux=True)
                else:
                    logits=model(x,quality_x)
                if calibrated:
                    corrupt_x,corrupt_quality,severity=corrupt_groups(x,quality_x,a.mask_probability,config["groups"])
                    _,corrupt_aux=model(corrupt_x,corrupt_quality,return_aux=True)
            loss=criterion(logits,y)
            if calibrated:
                loss=loss+.2*monotonic_calibration_loss(clean_aux,corrupt_aux,severity)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step(); total+=loss.item()*len(x); count+=len(x)
            if step==1 or step%100==0: print(f"epoch {epoch}/{a.epochs} step {step}/{len(loaders['train'])} loss={loss.item():.5f}",flush=True)
        val=evaluate(model,loaders["val"],criterion,device,config["classes"],config["groups"],max_batches=a.max_val_batches)
        print(f"epoch {epoch} train_loss={total/count:.6f} val_complete={val}",flush=True)
        if val["macro_f1"]>best:
            best=val["macro_f1"]; stale=0; best_val=val; best_epoch=epoch; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            torch.save({"model":state,"protocol":protocol,"args":vars(a)},output/"best.pt")
        else:
            stale+=1
            if stale>=a.patience: print(f"early stopping at epoch {epoch}",flush=True); break
    model.load_state_dict(state); conditions={}
    include_pairs=a.dataset=="sleep"
    for name,missing in missing_conditions(config["groups"],config["group_names"],include_pairs):
        conditions[name]=evaluate(model,loaders["val"],criterion,device,config["classes"],config["groups"],missing,a.max_val_batches)
        print(f"VALIDATION condition={name} metrics={conditions[name]}",flush=True)
    result={"protocol":protocol,"best_epoch":best_epoch,"best_validation_complete":best_val,
            "validation_conditions":conditions,"parameters":sum(p.numel() for p in model.parameters()),
            "elapsed_seconds":time.time()-start,"args":vars(a)}
    (output/"validation_metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(f"FINAL_VALIDATION variant={a.variant} best_epoch={best_epoch} complete={conditions['complete']}",flush=True)


if __name__=="__main__": main()
