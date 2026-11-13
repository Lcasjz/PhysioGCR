# WESAD benchmark

Download WESAD from the [dataset project page](https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/)
or its [author-hosted archive](https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download).
The extracted tree should contain files such as `S2/S2.pkl`.

Create the reference manifest and public-Husformer geometry:

```bash
python -m benchmarks.wesad.prepare_4hz --raw_root /path/to/WESAD --output data/wesad_4hz
python -m benchmarks.wesad.prepare_husformer --reference data/wesad_4hz --output data/wesad_husformer
```

The comparison task retains baseline, stress, and amusement. Each sample is one
label-pure second with Resp `50x14`, BVP `16x4`, ECG `50x14`, and EDA `1x4`.
Evaluation is strict LOSO: the requested subject is test, the next sorted
subject is validation, and all remaining subjects train.

Run one model/fold:

```bash
python -m benchmarks.wesad.train_loso --model husformer --data data/wesad_husformer --loso_test_subject 2 --output runs/wesad/husformer/S2
```

Available architecture adaptations are `cocoa`, `semipfl`,
`stress_calibration_2026`, `fg_dualnet`, and `husformer`. Add the proposed
quality module with `--quality_module`. For the parameter-efficient setting,
first train the backbone, then pass its checkpoint using `--initial_checkpoint`
and `--adaptation_mode parameter_efficient`.

All rows use identical LOSO samples and metrics. They preserve the central
published mechanism but are not exact reproductions of original paper scores.

After a complete suite, aggregate folds with:

```bash
python -m benchmarks.wesad.aggregate --runs runs/wesad_suite --data data/wesad_husformer
python -m benchmarks.wesad.aggregate_quality --runs runs/wesad_suite --data data/wesad_husformer
```
