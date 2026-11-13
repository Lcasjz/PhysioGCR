import argparse
import json
import random
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

from benchmarks.sleep_edf.data import audit_canonical, canonical_records

VARIANTS = {"global": (0, 0, 1), "temporal": (1, 0, 0), "sensor": (0, 1, 0),
            "temporal_sensor": (1, 1, 0), "full": (1, 1, 1)}


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class SleepEpochDataset(Dataset):
    def __init__(self, directory):
        self.items = [p for values in canonical_records(directory).values() for _, p in values]

    def __len__(self): return len(self.items)

    def __getitem__(self, index):
        with h5py.File(self.items[index], "r") as h5:
            eeg = np.asarray(h5["EEG"], dtype=np.float32).reshape(3000, -1)[:, :2]
            eog = np.asarray(h5["EOG"], dtype=np.float32).reshape(3000, -1)[:, :1]
            y = int(np.asarray(h5["stages"]).reshape(-1)[0])
        x = np.concatenate([eeg, eog], axis=1)
        x = (x - x.mean(0, keepdims=True)) / (x.std(0, keepdims=True) + 1e-6)
        return torch.from_numpy(x), y


class Pool(nn.Module):
    def __init__(self, hidden): super().__init__(); self.score = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
    def forward(self, x, dim):
        w = torch.softmax(self.score(x).squeeze(-1), dim=dim)
        return (w.unsqueeze(-1) * x).sum(dim)


class SleepTokenizer(nn.Module):
    def __init__(self, hidden=128, patch=100, samples=3000, channels=3):
        super().__init__(); self.patch = patch; self.patches = samples // patch
        self.norm = nn.LayerNorm(patch)
        self.project = nn.Sequential(nn.Linear(patch, hidden), nn.GELU(), nn.Dropout(.2))
        self.channel_embedding = nn.Parameter(torch.randn(1, 1, channels, hidden) * .02)
        self.time_embedding = nn.Parameter(torch.randn(1, self.patches, 1, hidden) * .02)
        self.temporal_pool = Pool(hidden); self.sensor_pool = Pool(hidden); self.global_pool = Pool(hidden)
        self.types = nn.Parameter(torch.randn(3, 1, hidden) * .02)

    def forward(self, x):
        local = x.transpose(1, 2).unfold(2, self.patch, self.patch).permute(0, 2, 1, 3)
        local = self.project(self.norm(local)) + self.channel_embedding + self.time_embedding
        temporal = self.temporal_pool(local, 2) + self.types[0]
        sensor = self.sensor_pool(local, 1) + self.types[1]
        global_token = self.global_pool(temporal, 1).unsqueeze(1) + self.types[2]
        return temporal, sensor, global_token


class Model(nn.Module):
    def __init__(self, variant, hidden=128):
        super().__init__(); self.variant = variant; self.tokenizer = SleepTokenizer(hidden)
        layer = nn.TransformerEncoderLayer(hidden, 4, hidden * 4, .2, batch_first=True, norm_first=True)
        self.fusion = nn.TransformerEncoder(layer, 2, norm=nn.LayerNorm(hidden))
        self.query = nn.Parameter(torch.randn(1, 1, hidden) * .02)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(.2), nn.Linear(hidden, 5))

    def forward(self, x, return_embedding=False):
        groups = self.tokenizer(x); flags = VARIANTS[self.variant]
        tokens = torch.cat([g for g, flag in zip(groups, flags) if flag], 1)
        fused = self.fusion(torch.cat([self.query.expand(len(x), -1, -1), tokens], 1))
        embedding = fused[:, 0]
        logits = self.head(embedding)
        return (logits, embedding) if return_embedding else logits


def mask_modalities(x, probability):
    if probability <= 0:
        return x
    masked = x.clone()
    channel_mask = torch.rand(len(x), 3, device=x.device) < probability
    all_masked = channel_mask.all(dim=1)
    if all_masked.any():
        rows = all_masked.nonzero(as_tuple=False).squeeze(1)
        keep = torch.randint(0, 3, (len(rows),), device=x.device)
        channel_mask[rows, keep] = False
    return masked * (~channel_mask).unsqueeze(1)


def metrics(y, p, loss):
    return {"loss": loss, "accuracy": accuracy_score(y, p), "balanced_accuracy": balanced_accuracy_score(y, p),
            "macro_f1": f1_score(y, p, average="macro", zero_division=0), "kappa": cohen_kappa_score(y, p),
            "f1_per_class": f1_score(y, p, labels=range(5), average=None, zero_division=0).tolist(),
            "pred_dist": np.bincount(p, minlength=5).tolist(), "confusion": confusion_matrix(y, p, labels=range(5)).tolist()}


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval(); ys=[]; ps=[]; total=0.; n=0
    for x, y in loader:
        x=x.to(device, non_blocking=True); y=y.to(device, non_blocking=True); logits=model(x)
        total += criterion(logits, y).item()*len(x); n += len(x); ys.extend(y.cpu().tolist()); ps.extend(logits.argmax(1).cpu().tolist())
    return metrics(ys, ps, total/n)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--variant", choices=VARIANTS, required=True)
    p.add_argument("--data", default="code/MHET-main/sleepEDF_canonical"); p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=20); p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=128); p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4); p.add_argument("--hidden", type=int, default=128); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--modality_mask_prob", type=float, default=0.)
    p.add_argument("--consistency_weight", type=float, default=0.)
    a=p.parse_args(); seed_all(a.seed); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audit=audit_canonical(a.data); print(f"CANONICAL_AUDIT={audit}", flush=True)
    loaders={};
    for split in ("train", "val", "test"):
        ds=SleepEpochDataset(Path(a.data)/split); loaders[split]=DataLoader(ds, a.batch_size, shuffle=split=="train", num_workers=a.workers, pin_memory=True, persistent_workers=a.workers>0)
    counts=np.asarray(audit["train"]["labels"]); weights=np.minimum(counts.sum()/(5*np.maximum(counts,1)), 5.)
    model=Model(a.variant,a.hidden).to(device); criterion=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device),label_smoothing=.05)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01); output=Path(a.output); output.mkdir(parents=True,exist_ok=True)
    print(f"variant={a.variant} input=3000x3 patch=100 classes=5 trainable={sum(x.numel() for x in model.parameters()):,}",flush=True)
    best=-1.; stale=0; state=None; best_val=None; best_epoch=0; start=time.time()
    for epoch in range(1,a.epochs+1):
        model.train(); total=0.; n=0
        for step,(x,y) in enumerate(loaders["train"],1):
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True); optimizer.zero_grad(set_to_none=True)
            if a.modality_mask_prob > 0:
                masked_x = mask_modalities(x, a.modality_mask_prob)
                logits, masked_embedding = model(masked_x, return_embedding=True)
                loss = criterion(logits, y)
                if a.consistency_weight > 0:
                    with torch.no_grad():
                        _, clean_embedding = model(x, return_embedding=True)
                    consistency = 1 - nn.functional.cosine_similarity(masked_embedding, clean_embedding, dim=1).mean()
                    loss = loss + a.consistency_weight * consistency
            else:
                loss=criterion(model(x),y)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step(); total+=loss.item()*len(x); n+=len(x)
            if step==1 or step%100==0: print(f"epoch {epoch}/{a.epochs} step {step}/{len(loaders['train'])} loss={loss.item():.5f}",flush=True)
        val=evaluate(model,loaders["val"],criterion,device); print(f"epoch {epoch} train_loss={total/n:.6f} val={val}",flush=True)
        if val["macro_f1"]>best:
            best=val["macro_f1"]; stale=0; best_val=val; best_epoch=epoch; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; torch.save({"model":state,"variant":a.variant,"checksum":audit["checksum"]},output/"best.pt")
        else:
            stale+=1
            if stale>=a.patience: print(f"early stopping at epoch {epoch}",flush=True); break
    model.load_state_dict(state); test=evaluate(model,loaders["test"],criterion,device)
    result={"model":"sleep_multigranular_tokenizer","variant":a.variant,"canonical_checksum":audit["checksum"],"best_epoch":best_epoch,"best_val":best_val,"test":test,"elapsed_seconds":time.time()-start,"args":vars(a)}
    (output/"metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); print(f"FINAL variant={a.variant} best_val={best_val} test={test}",flush=True)


if __name__ == "__main__": main()
