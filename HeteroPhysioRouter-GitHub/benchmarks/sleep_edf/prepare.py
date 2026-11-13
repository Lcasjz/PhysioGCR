"""Convert Sleep-EDF Expanded PSG/Hypnogram pairs to the canonical HDF5 cache."""

import argparse
import json
from pathlib import Path

import h5py
import mne
import numpy as np

from benchmarks.sleep_edf.data import audit_canonical


STAGE_TO_ID = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}


def pair_files(data_dir):
    data_dir = Path(data_dir)
    hypnograms = {path.name[:6]: path for path in data_dir.glob("*-Hypnogram.edf")}
    return [
        (psg, hypnograms[psg.name[:6]])
        for psg in sorted(data_dir.glob("*-PSG.edf"))
        if psg.name[:6] in hypnograms
    ]


def subject_id(path):
    if path.name.startswith(("SC", "ST")) and len(path.name) >= 5:
        return path.name[3:5]
    return path.name[:5]


def split_subjects(subjects, seed=0, train_fraction=0.7, val_fraction=0.15):
    generator = np.random.default_rng(seed)
    subjects = np.asarray(sorted(set(subjects)))
    generator.shuffle(subjects)
    train_count = int(len(subjects) * train_fraction)
    val_count = max(1, int(len(subjects) * val_fraction)) if len(subjects) >= 3 else 0
    if train_count + val_count >= len(subjects):
        train_count = max(1, len(subjects) - val_count - 1)
    return {
        "train": set(subjects[:train_count].tolist()),
        "val": set(subjects[train_count:train_count + val_count].tolist()),
        "test": set(subjects[train_count + val_count:].tolist()),
    }


def write_record(psg_path, hypnogram_path, output_root, split_name, channels, sample_rate):
    raw = mne.io.read_raw_edf(psg_path, preload=True, stim_channel=None, verbose="ERROR")
    missing = [channel for channel in channels if channel not in raw.ch_names]
    if missing:
        raise ValueError(f"{psg_path.name} is missing channels: {missing}")
    raw.pick(list(channels))
    if int(round(raw.info["sfreq"])) != sample_rate:
        raw.resample(sample_rate, verbose="ERROR")
    raw.set_annotations(mne.read_annotations(hypnogram_path))
    events, _ = mne.events_from_annotations(
        raw, event_id=STAGE_TO_ID, chunk_duration=30.0, verbose="ERROR"
    )
    if not len(events):
        return 0
    epochs = mne.Epochs(
        raw,
        events,
        event_id=list(STAGE_TO_ID.values()),
        tmin=0,
        tmax=30.0 - 1.0 / sample_rate,
        baseline=None,
        preload=True,
        on_missing="ignore",
        verbose="ERROR",
    )
    signals = epochs.get_data().astype(np.float32)
    labels = epochs.events[:, 2].astype(np.int64)
    signals = (signals - signals.mean(-1, keepdims=True)) / (
        signals.std(-1, keepdims=True) + 1e-6
    )
    output = Path(output_root) / split_name
    output.mkdir(parents=True, exist_ok=True)
    record = psg_path.stem.replace("-PSG", "")
    for index, (signal, label) in enumerate(zip(signals, labels)):
        seconds = index * 30
        name = (
            f"{seconds // 3600 % 24:02d}_{seconds % 3600 // 60:02d}_"
            f"{seconds % 60:02d}_{int(label)}_{record}_{index:04d}.hdf5"
        )
        with h5py.File(output / name, "w") as handle:
            handle.create_dataset("EEG", data=signal[:2].T[:, :, None], compression="gzip")
            handle.create_dataset("EOG", data=signal[2].T[:, None, None], compression="gzip")
            handle.create_dataset(
                "stages", data=np.asarray([label], dtype=np.int64), compression="gzip"
            )
    return len(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--output", default="data/sleep_edf")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample_rate", type=int, default=100)
    parser.add_argument(
        "--channels",
        nargs=3,
        default=("EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"),
        metavar=("EEG1", "EEG2", "EOG"),
    )
    args = parser.parse_args()
    pairs = pair_files(args.raw_root)
    if not pairs:
        raise FileNotFoundError(f"no paired PSG/Hypnogram EDF files under {args.raw_root}")
    split = split_subjects([subject_id(psg) for psg, _ in pairs], seed=args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    split_json = {name: sorted(subjects) for name, subjects in split.items()}
    (output / "split.json").write_text(json.dumps(split_json, indent=2), encoding="utf-8")
    totals = {name: 0 for name in split}
    for psg, hypnogram in pairs:
        subject = subject_id(psg)
        split_name = next(name for name, members in split.items() if subject in members)
        count = write_record(
            psg, hypnogram, output, split_name, args.channels, args.sample_rate
        )
        totals[split_name] += count
        print(f"record={psg.name} split={split_name} epochs={count}", flush=True)
    audit = audit_canonical(output)
    print(json.dumps({"epochs": totals, "audit": audit}, indent=2), flush=True)


if __name__ == "__main__":
    main()
