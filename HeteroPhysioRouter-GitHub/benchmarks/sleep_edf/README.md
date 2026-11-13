# Sleep-EDF benchmark

Download Sleep-EDF Expanded from [PhysioNet](https://physionet.org/content/sleep-edfx/1.0.0/).
The default protocol expects the Sleep Cassette PSG and Hypnogram EDF pairs.

```bash
python -m benchmarks.sleep_edf.prepare --raw_root /path/to/sleep-edfx/sleep-cassette --output data/sleep_edf
```

The converter extracts 30-second epochs at 100 Hz from `EEG Fpz-Cz`,
`EEG Pz-Oz`, and `EOG horizontal`, merges stages N3/N4, and creates a
subject-disjoint 70/15/15 split with seed 0. It writes `split.json`,
`audit.json`, and per-epoch HDF5 files.

Run the proposed router:

```bash
python -m experiments.train --dataset sleep --variant full_no_calibration --data data/sleep_edf --output runs/sleep_edf/router --batch_size 128
```

Run the in-repository multi-granular tokenizer benchmark:

```bash
python -m benchmarks.sleep_edf.train_tokenizer_baseline --variant temporal_sensor --data data/sleep_edf --output runs/sleep_edf/tokenizer
```

The paper benchmark also evaluates SleepDIFFormer, MHFNet, SleepFocalNet,
LogicSleep, and TinySleepNet. Their source trees are not redistributed here.
Use the upstream repositories listed in `THIRD_PARTY.md` and point their local
training adapters at the exact same canonical cache. This distinction prevents
unlicensed third-party code from being presented as part of this repository.
