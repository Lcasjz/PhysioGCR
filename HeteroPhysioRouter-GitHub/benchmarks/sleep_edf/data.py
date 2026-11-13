import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


FILE_RE = re.compile(r"^(?:\d{2}_){3}\d+_(SC\d+E\d+)_(\d+)\.hdf5$")


def canonical_records(directory):
    records = defaultdict(list)
    for path in Path(directory).glob("*.hdf5"):
        match = FILE_RE.match(path.name)
        if match:
            records[match.group(1)].append((int(match.group(2)), path))
    for record in records:
        records[record].sort(key=lambda item: item[0])
    return dict(sorted(records.items()))


def read_hdf_epoch(path, eeg_channel=0):
    with h5py.File(path, "r") as h5:
        eeg = np.asarray(h5["EEG"], dtype=np.float32)
        label = int(np.asarray(h5["stages"]).reshape(-1)[0])
    if eeg.ndim == 3:
        eeg = eeg[:, :, 0]
    if eeg.ndim == 1:
        signal = eeg
    else:
        signal = eeg[:, eeg_channel]
    signal = (signal - signal.mean()) / (signal.std() + 1e-6)
    return signal.astype(np.float32), label


def audit_canonical(root):
    root = Path(root)
    cache = root / "audit.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    digest = hashlib.sha256()
    result = {}
    for split in ("train", "val", "test"):
        records = canonical_records(root / split)
        counts = Counter()
        epochs = 0
        for record, items in records.items():
            for index, path in items:
                with h5py.File(path, "r") as h5:
                    label = int(np.asarray(h5["stages"]).reshape(-1)[0])
                counts[label] += 1
                epochs += 1
                digest.update(f"{split}|{record}|{index}|{label}\n".encode("ascii"))
        result[split] = {
            "records": len(records), "epochs": epochs,
            "labels": [counts[i] for i in range(5)],
        }
    result["checksum"] = digest.hexdigest()
    cache.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


class CanonicalEpochDataset(Dataset):
    def __init__(self, directory):
        self.records = canonical_records(directory)
        self.items = [item for items in self.records.values() for item in items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        _, path = self.items[index]
        signal, label = read_hdf_epoch(path)
        return torch.from_numpy(signal).unsqueeze(0), label


class CanonicalSequenceDataset(Dataset):
    def __init__(self, directory, seq_len=20):
        self.seq_len = seq_len
        self.items = []
        self.records = canonical_records(directory)
        for record, epochs in self.records.items():
            for start in range(0, len(epochs), seq_len):
                self.items.append((record, epochs[start:start + seq_len]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        _, items = self.items[index]
        signals, labels = zip(*(read_hdf_epoch(path) for _, path in items))
        return torch.from_numpy(np.stack(signals)).unsqueeze(1), torch.tensor(labels), len(items)


def collate_sequences(batch):
    seq_len = max(x.shape[0] for x, _, _ in batch)
    x = torch.zeros(len(batch), seq_len, 1, 3000)
    y = torch.full((len(batch), seq_len), -100, dtype=torch.long)
    mask = torch.zeros(len(batch), seq_len, dtype=torch.bool)
    for i, (signals, labels, length) in enumerate(batch):
        x[i, :length] = signals
        y[i, :length] = labels
        mask[i, :length] = True
    return x, y, mask
