# PAMAP2 benchmark

Download PAMAP2 from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring).
Extract the `Protocol` directory and prepare the common cache from the repository
root:

```bash
python -m benchmarks.pamap2.prepare --raw_root /path/to/PAMAP2/Protocol --output data/pamap2
```

The fixed protocol uses subjects 101-104, 106, and 107 for training, 105 for
validation, and 108 for testing. Subject 109 is excluded from this fixed split
because the Protocol recording is severely incomplete. The cache uses 12
activities, 24-sample windows, stride 12, and all 52 post-label features.
Windows never cross activity boundaries.

Run a literature-architecture adaptation:

```bash
python -m benchmarks.pamap2.train_baselines --model deepconvlstm --data data/pamap2 --output runs/pamap2/deepconvlstm
```

Available names are `deepconvlstm`, `deepsense`, `attend_discriminate`,
`tinyhar`, `mcnn`, and `sahar`. These rows share samples, split, normalization,
and metrics while retaining documented architecture-specific training choices.
They are common-input adaptations, not claims of exact paper-score reproduction.

Run PhysioGCR on the fixed split:

```bash
python -m experiments.train --dataset pamap2 --variant full_no_calibration --data data/pamap2 --output runs/pamap2/router
```

Run one LOSO fold, leaving subject 101 for testing:

```bash
python -m benchmarks.pamap2.train_loso --test_subject 101 --data data/pamap2 --output runs/pamap2/loso/S101
```
