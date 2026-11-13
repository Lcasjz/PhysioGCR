"""Build the public Husformer-style WESAD four-modality input cache.

Every sample is one label-pure second and contains the exact public-code
reshape convention: Resp/ECG 700 -> (50, 14), BVP 64 -> (16, 4), and
EDA 4 -> (1, 4). The benchmark keeps baseline/stress/amusement and applies
subject-level LOSO later; it does not use the public random sample split.
"""

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from benchmarks.wesad.prepare_4hz import LABEL_MAP, as_channels


MODALITIES = (
    ("resp", "chest", "Resp", 700, (50, 14)),
    ("bvp", "wrist", "BVP", 64, (16, 4)),
    ("ecg", "chest", "ECG", 700, (50, 14)),
    ("eda", "wrist", "EDA", 4, (1, 4)),
)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repair_signal(value):
    value = as_channels(value).copy()
    for channel in range(value.shape[1]):
        good = np.isfinite(value[:, channel])
        if not good.any():
            raise ValueError(f"signal channel {channel} has no finite observations")
        if not good.all():
            index = np.arange(len(value))
            value[:, channel] = np.interp(index, index[good], value[good, channel])
    return value.astype(np.float32)


def extract_subject(raw):
    labels = np.asarray(raw["label"]).reshape(-1)
    streams = {
        name: repair_signal(raw["signal"][device][signal])
        for name, device, signal, _, _ in MODALITIES
    }
    seconds = min(
        len(labels) // 700,
        *(len(streams[name]) // rate for name, _, _, rate, _ in MODALITIES),
    )
    output = {name: [] for name, *_ in MODALITIES}
    targets, starts = [], []
    for second in range(seconds):
        label_block = labels[second * 700:(second + 1) * 700]
        if len(label_block) != 700 or not np.all(label_block == label_block[0]):
            continue
        raw_label = int(label_block[0])
        if raw_label not in LABEL_MAP:
            continue
        for name, _, _, rate, shape in MODALITIES:
            block = streams[name][second * rate:(second + 1) * rate]
            if block.shape != (rate, 1):
                raise ValueError(f"unexpected {name} block shape {block.shape}")
            output[name].append(block.reshape(shape))
        targets.append(LABEL_MAP[raw_label])
        starts.append(second)
    arrays = {name: np.asarray(values, dtype=np.float32) for name, values in output.items()}
    arrays["y"] = np.asarray(targets, dtype=np.int64)
    arrays["start_seconds"] = np.asarray(starts, dtype=np.int64)
    return arrays


def prepare(reference_root, output):
    reference_root, output = Path(reference_root), Path(output)
    reference = json.loads((reference_root / "manifest.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("preprocessing_version") != 1:
            raise ValueError("existing Husformer-input cache uses another protocol version")
        for subject, entry in manifest["subjects"].items():
            path = output / f"subject{subject}.npz"
            if not path.exists() or file_hash(path) != entry["cache_sha256"]:
                raise ValueError(f"missing or altered cache: {path}")
        print(
            f"CACHE_VERIFIED protocol=husformer_public_input subjects={len(manifest['subjects'])} "
            f"checksum={manifest['checksum']}", flush=True,
        )
        return manifest

    manifest = {
        "version": 1,
        "preprocessing_version": 1,
        "dataset": "WESAD",
        "task": "baseline_stress_amusement",
        "protocol": "husformer_public_four_modality_one_second",
        "claim": "public Husformer input geometry with strict subject-level evaluation",
        "label_map": {str(key): value for key, value in LABEL_MAP.items()},
        "class_names": reference["class_names"],
        "sample_seconds": 1,
        "stride_seconds": 1,
        "modalities": {
            name: {"device": device, "signal": signal, "native_hz": rate, "shape": list(shape)}
            for name, device, signal, rate, shape in MODALITIES
        },
        "normalization": "fit per modality feature dimension on train subjects only",
        "split_policy": "LOSO test subject; next sorted subject validation; remaining subjects training",
        "subjects": {},
        "source": reference["source"],
    }
    identity = hashlib.sha256()
    for subject, entry in reference["subjects"].items():
        source = Path(entry["source"])
        print(f"PREPARE_HUSFORMER_INPUT subject=S{subject} source={source}", flush=True)
        with source.open("rb") as handle:
            raw = pickle.load(handle, encoding="latin1")
        arrays = extract_subject(raw)
        cache = output / f"subject{subject}.npz"
        np.savez_compressed(cache, **arrays)
        counts = np.bincount(arrays["y"], minlength=3)
        identity.update(subject.encode("ascii"))
        identity.update(arrays["start_seconds"].tobytes())
        identity.update(arrays["y"].tobytes())
        manifest["subjects"][subject] = {
            "source": str(source),
            "source_sha256": file_hash(source),
            "cache_sha256": file_hash(cache),
            "samples": int(len(arrays["y"])),
            "labels": counts.tolist(),
        }
        shapes = {name: list(arrays[name].shape) for name, *_ in MODALITIES}
        print(f"AUDIT S{subject} samples={len(arrays['y'])} labels={counts.tolist()} shapes={shapes}", flush=True)
    manifest["sample_identity_checksum"] = identity.hexdigest()
    manifest["checksum"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"HUSFORMER_INPUT_CACHE_COMPLETE checksum={manifest['checksum']} saved={manifest_path}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="data/canonical")
    parser.add_argument("--output", default="data/husformer_input")
    args = parser.parse_args()
    prepare(args.reference, args.output)


if __name__ == "__main__":
    main()
