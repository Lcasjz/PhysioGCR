import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


CLASS_NAMES = [
    "rope_jumping", "lying", "sitting", "standing", "walking", "running",
    "cycling", "nordic_walking", "ascending_stairs", "descending_stairs",
    "vacuum_cleaning", "ironing",
]

# PAMAP2 activity IDs used by the published Attend-and-Discriminate protocol.
ACTIVITY_MAP = {
    24: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,
    12: 8, 13: 9, 16: 10, 17: 11,
}


def read_subject(path):
    rows = []
    with path.open("r", encoding="ascii", errors="replace") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 4:
                continue
            try:
                values = np.asarray([float(x) if x.lower() != "nan" else np.nan for x in fields], dtype=np.float32)
            except ValueError:
                continue
            rows.append(values)
    if not rows:
        raise ValueError(f"No numeric rows in {path}")
    width = max(len(row) for row in rows)
    data = np.full((len(rows), width), np.nan, dtype=np.float32)
    for i, row in enumerate(rows):
        data[i, :len(row)] = row
    return data


def fill_missing(x):
    x = x.copy()
    for col in range(x.shape[1]):
        values = x[:, col]
        good = np.flatnonzero(np.isfinite(values))
        if len(good) == 0:
            values[:] = 0
        else:
            values[:] = np.interp(np.arange(len(values)), good, values[good])
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", required=True)
    ap.add_argument("--output", default="data/canonical")
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--stride", type=int, default=12)
    args = ap.parse_args()

    raw = Path(args.raw_root)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    subjects = sorted(raw.glob("subject*.dat"))
    if not subjects:
        raise FileNotFoundError(f"No subject*.dat files found under {raw}")

    split = {"train": [], "val": [], "test": [], "excluded": []}
    for path in subjects:
        sid = int(path.stem.replace("subject", ""))
        split["excluded" if sid == 109 else "test" if sid == 108 else "val" if sid == 105 else "train"].append(sid)

    manifest = {
        "version": 1,
        "window": args.window,
        "stride": args.stride,
        "feature_columns": "all columns after timestamp and activityID",
        "class_names": CLASS_NAMES,
        "activity_map": {str(k): v for k, v in ACTIVITY_MAP.items()},
        "split": split,
        "subjects": {},
    }
    digest = hashlib.sha256()
    for path in subjects:
        sid = int(path.stem.replace("subject", ""))
        raw_data = read_subject(path)
        activity = raw_data[:, 1].astype(np.int64)
        features = fill_missing(raw_data[:, 2:])
        windows, labels = [], []
        start = 0
        while start < len(activity):
            raw_label = int(activity[start])
            end = start + 1
            while end < len(activity) and int(activity[end]) == raw_label:
                end += 1
            if raw_label in ACTIVITY_MAP:
                label = ACTIVITY_MAP[raw_label]
                for offset in range(start, end - args.window + 1, args.stride):
                    windows.append(features[offset:offset + args.window])
                    labels.append(label)
                    digest.update(f"{sid}|{offset}|{label}\n".encode("ascii"))
            start = end
        x = np.asarray(windows, dtype=np.float32)
        y = np.asarray(labels, dtype=np.int64)
        np.savez_compressed(out / f"subject{sid}.npz", x=x, y=y)
        manifest["subjects"][str(sid)] = {"windows": int(len(y)), "features": int(features.shape[1])}
        print(f"subject{sid}: rows={len(features)} windows={len(y)} features={features.shape[1]}", flush=True)
    manifest["checksum"] = digest.hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"split": split, "checksum": manifest["checksum"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
