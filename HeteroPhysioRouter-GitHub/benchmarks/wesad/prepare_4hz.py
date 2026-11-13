import argparse
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np


CLASS_NAMES = ("baseline", "stress", "amusement")
LABEL_MAP = {1: 0, 2: 1, 3: 2}
TARGET_HZ = 4
WINDOW_SECONDS = 60
STRIDE_SECONDS = 30
WINDOW = TARGET_HZ * WINDOW_SECONDS
STRIDE = TARGET_HZ * STRIDE_SECONDS


def find_subjects(root):
    paths = {}
    for path in Path(root).rglob("S*.pkl"):
        name = path.stem.upper()
        if len(name) >= 2 and name[1:].isdigit():
            paths[int(name[1:])] = path
    if not paths:
        raise FileNotFoundError(
            f"No WESAD subject pickle files found under {root}. "
            "Download and extract WESAD.zip first."
        )
    return dict(sorted(paths.items()))


def as_channels(value):
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        return array[:, None]
    if array.ndim == 2:
        return array
    raise ValueError(f"unexpected signal shape: {array.shape}")


def downsample_mean(value, source_hz, target_hz=TARGET_HZ):
    value = as_channels(value)
    block = source_hz // target_hz
    if source_hz % target_hz:
        raise ValueError(f"source rate {source_hz} is not divisible by target rate {target_hz}")
    length = len(value) // block * block
    if length == 0:
        return np.empty((0, value.shape[1]), dtype=np.float32)
    value = value[:length].reshape(-1, block, value.shape[1])
    with np.errstate(invalid="ignore"):
        result = np.nanmean(value, axis=1)
    return np.asarray(result, dtype=np.float32)


def majority_labels(labels, source_hz, target_hz=TARGET_HZ):
    labels = np.asarray(labels).reshape(-1)
    block = source_hz // target_hz
    length = len(labels) // block * block
    output = []
    for group in labels[:length].reshape(-1, block):
        valid = [int(value) for value in group if int(value) >= 0]
        output.append(Counter(valid).most_common(1)[0][0] if valid else -1)
    return np.asarray(output, dtype=np.int64)


def load_subject(path):
    with path.open("rb") as handle:
        data = pickle.load(handle, encoding="latin1")
    if "signal" not in data or "label" not in data:
        raise ValueError(f"missing signal/label keys in {path}")
    chest = data["signal"]["chest"]
    wrist = data["signal"]["wrist"]

    # All streams are represented at 4 Hz. Chest signals are averaged from
    # 700 Hz; wrist signals are averaged/interpreted from their native rates.
    chest_parts = [
        downsample_mean(chest["ACC"], 700),
        downsample_mean(chest["ECG"], 700),
        downsample_mean(chest["EDA"], 700),
        downsample_mean(chest["EMG"], 700),
        downsample_mean(chest["Resp"], 700),
        downsample_mean(chest["Temp"], 700),
    ]
    wrist_parts = [
        downsample_mean(wrist["ACC"], 32),
        downsample_mean(wrist["BVP"], 64),
        downsample_mean(wrist["EDA"], 4),
        downsample_mean(wrist["TEMP"], 4),
    ]
    lengths = [len(part) for part in chest_parts + wrist_parts]
    labels = majority_labels(data["label"], 700)
    length = min(lengths + [len(labels)])
    features = np.concatenate([part[:length] for part in chest_parts + wrist_parts], axis=1)
    labels = labels[:length]
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    windows = []
    targets = []
    for start in range(0, length - WINDOW + 1, STRIDE):
        raw_labels = labels[start:start + WINDOW]
        mapped = [LABEL_MAP[int(value)] for value in raw_labels if int(value) in LABEL_MAP]
        if not mapped:
            continue
        # Keep windows dominated by one target state and never silently mix
        # transient/meditation labels into a target class.
        counts = Counter(mapped)
        target, count = counts.most_common(1)[0]
        if count / len(raw_labels) < 0.90:
            continue
        if any(int(value) not in LABEL_MAP for value in raw_labels):
            continue
        windows.append(features[start:start + WINDOW])
        targets.append(target)
    x = np.asarray(windows, dtype=np.float32)
    y = np.asarray(targets, dtype=np.int64)
    return x, y, features.shape[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--output", default="data/canonical")
    args = parser.parse_args()
    raw_root = Path(args.raw_root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    subjects = find_subjects(raw_root)
    subject_ids = list(subjects)
    if len(subject_ids) < 10:
        raise RuntimeError(f"expected at least 10 WESAD subjects, found {subject_ids}")

    # Fixed, deterministic subject split. The exact IDs are stored in the
    # manifest and therefore become part of the benchmark protocol.
    split = {
        "train": subject_ids[:10],
        "val": subject_ids[10:12],
        "test": subject_ids[12:],
    }
    digest = hashlib.sha256()
    manifest = {
        "version": 1,
        "dataset": "WESAD",
        "task": "baseline_stress_amusement",
        "label_map": {str(key): value for key, value in LABEL_MAP.items()},
        "class_names": list(CLASS_NAMES),
        "target_hz": TARGET_HZ,
        "window_seconds": WINDOW_SECONDS,
        "stride_seconds": STRIDE_SECONDS,
        "window": WINDOW,
        "stride": STRIDE,
        "features": 14,
        "feature_names": [
            "chest_acc_x", "chest_acc_y", "chest_acc_z", "chest_ecg",
            "chest_eda", "chest_emg", "chest_resp", "chest_temp",
            "wrist_acc_x", "wrist_acc_y", "wrist_acc_z", "wrist_bvp",
            "wrist_eda", "wrist_temp",
        ],
        "split": split,
        "subjects": {},
        "source": "WESAD official Siegen/Sciebo release",
    }
    for subject, path in subjects.items():
        x, y, features = load_subject(path)
        np.savez_compressed(output / f"subject{subject}.npz", x=x, y=y)
        counts = np.bincount(y, minlength=len(CLASS_NAMES))
        manifest["subjects"][str(subject)] = {
            "source": str(path),
            "windows": int(len(y)),
            "features": int(features),
            "labels": counts.tolist(),
        }
        digest.update(f"{subject}|{len(y)}|{counts.tolist()}\n".encode("ascii"))
        print(f"subject{subject}: windows={len(y)} features={features} labels={counts.tolist()}", flush=True)
    split_stats = {}
    for split_name, split_subjects in split.items():
        split_counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
        for subject in split_subjects:
            split_counts += np.asarray(manifest["subjects"][str(subject)]["labels"], dtype=np.int64)
        if np.any(split_counts == 0):
            raise RuntimeError(f"split {split_name} has no samples for at least one class: {split_counts.tolist()}")
        split_stats[split_name] = {
            "subjects": split_subjects,
            "windows": int(split_counts.sum()),
            "labels": split_counts.tolist(),
        }
    manifest["split_stats"] = split_stats
    manifest["checksum"] = digest.hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"split": split, "checksum": manifest["checksum"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
