# PhysioGCR

Official implementation of **PhysioGCR: Physiological Group-Consistent
Recognition**, with common-input benchmarks for robust physiological
time-series learning.

**Paper:** *PhysioGCR: Robust Physiological Time-Series Recognition via
Multi-Granular Sensor-Temporal Representations and Sensor-Group Corruption
Augmentation* (manuscript in preparation).

[Model architecture](docs/model_architecture.md) |
[Benchmark protocols](benchmarks/README.md) |
[GitHub publication guide](docs/github_release_guide.md) |
[中文发布指南](docs/github_release_guide_zh.md)

PhysioGCR aligns three units that are often treated separately: the physical
sensor group used to construct representations, the group exposed to simulated
degradation during training, and the group removed or corrupted at evaluation.
It combines sensor-group, temporal, and global tokens with quality-aware
cross-granular routing. The same evaluation framework covers sleep staging,
wearable activity recognition, and affective-state recognition.

> This release contains no raw data, checkpoints, participant identifiers, or
> paper result logs. Download each public dataset from its owner and build a
> local cache using the commands below.

## Repository layout

```text
src/physiogcr/            Public PhysioGCR API
src/heterophysiorouter/   Core implementation and compatibility namespace
experiments/              Main training, locked evaluation, mechanism diagnostics
benchmarks/sleep_edf/     Sleep-EDF cache builder and tokenizer benchmark
benchmarks/pamap2/        PAMAP2 cache builder, baselines, and LOSO runner
benchmarks/wesad/         WESAD cache builders, literature adapters, LOSO runner
configs/                  Human-readable protocol declarations
docs/                     Architecture and reproducibility notes
tests/                    Shape and metric smoke tests
```

## Installation

Python 3.10 or newer and a recent PyTorch build are recommended. Install a
CUDA-compatible PyTorch wheel first when GPU training is required, then run:

```bash
python -m pip install -e ".[sleep,dev]"
```

All commands below are run from the repository root. Editable installation
makes `physiogcr` and the compatibility namespace `heterophysiorouter`
importable while retaining `python -m` experiment entry points.

```python
from physiogcr import PhysioGCR

model = PhysioGCR(
    channels=52, samples=24, patch=4, classes=12,
    channel_groups=((0, 1), (1, 18), (18, 35), (35, 52)),
    variant="full_no_calibration",
)
```

## Method at a glance

1. **Grouped multi-granular tokenization:** local patches are aggregated into
   physical sensor-group tokens, temporal tokens, and a global token.
2. **Quality-aware cross-granular routing:** scale-invariant signal-quality
   proxies, bidirectional sensor-temporal attention, and a sample-conditioned
   router fuse the three representations.
3. **Group-degradation training:** complete physical groups are dropped,
   noised, or flattened so training degradation matches the missing-group
   evaluation unit.
4. **Optional local correction diagnostics:** group-by-time reliability and
   task relevance support experiments on selective local adaptation.

See [model architecture](docs/model_architecture.md) for implementation detail.

## Why group consistency?

Physiological systems fail in physical units: an electrode, body location, or
acquisition stream can disappear or degrade together. PhysioGCR therefore uses
the same sensor-group partition to define:

- what the model represents as a source;
- what training corrupts as one failure unit; and
- what evaluation removes or perturbs as one condition.

This is the repository's central claim. Quality-aware routing and local
correction are evaluated mechanisms within that broader principle, not the
definition of PhysioGCR itself.

## Data and benchmark commands

### Sleep-EDF Expanded

Data: [PhysioNet Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/1.0.0/)

```bash
python -m benchmarks.sleep_edf.prepare --raw_root /path/to/sleep-cassette --output data/sleep_edf
python -m experiments.train --dataset sleep --variant full_no_calibration --data data/sleep_edf --output runs/sleep_edf/router --batch_size 128
```

### PAMAP2

Data: [UCI PAMAP2](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring)

```bash
python -m benchmarks.pamap2.prepare --raw_root /path/to/PAMAP2/Protocol --output data/pamap2
python -m experiments.train --dataset pamap2 --variant full_no_calibration --data data/pamap2 --output runs/pamap2/router
python -m benchmarks.pamap2.train_baselines --model deepconvlstm --data data/pamap2 --output runs/pamap2/deepconvlstm
```

### WESAD

Data: [WESAD project page](https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/)

```bash
python -m benchmarks.wesad.prepare_4hz --raw_root /path/to/WESAD --output data/wesad_4hz
python -m benchmarks.wesad.prepare_husformer --reference data/wesad_4hz --output data/wesad_husformer
python -m benchmarks.wesad.train_loso --model husformer --data data/wesad_husformer --loso_test_subject 2 --output runs/wesad/husformer/S2
```

Dataset-specific protocol details are in the README inside each benchmark
directory. The machine-readable overview is [protocols.json](configs/protocols.json).

Run an entire suite sequentially with resumable completion checks:

```bash
python scripts/run_suite.py --dataset pamap2 --data data/pamap2 --output runs/pamap2_suite --loso
python scripts/run_suite.py --dataset wesad --data data/wesad_husformer --output runs/wesad_suite --quality
```

Use `--dry_run` to inspect every generated command before starting a long run.

## Output contract

Runners report accuracy, balanced accuracy, macro-F1, Cohen's kappa,
class-wise statistics, prediction distribution, and confusion matrix where
applicable. Each run records its split, cache checksum, configuration, selected
validation checkpoint, and final metrics under the requested output directory.

Use `experiments.evaluate` for a locked test evaluation after development:

```bash
python -m experiments.evaluate --dataset pamap2 --data data/pamap2 --checkpoint runs/pamap2/router/best.pt --output runs/pamap2/router/test_metrics.json
```

## Benchmark scope

The benchmark includes in-repository common-input adapters and documented hooks
for upstream sleep-staging implementations. An architecture adaptation is not
an exact reproduction of the score in its source paper. Read
[reproducibility notes](docs/reproducibility.md) and [third-party sources](THIRD_PARTY.md)
before reporting results.

Qwen/LLM experiments from early exploration are intentionally excluded: they
are not part of the final method represented by this repository.

## Tests

```bash
python -m compileall src experiments benchmarks tests
python -m pytest -q
```

## Citation

Software citation metadata is provided in `CITATION.cff`. Replace its author
and repository-owner placeholders before a public release. A paper citation
and link will be added when a public manuscript is available.

## License

Original code in this repository is released under the MIT License. Dataset
terms and third-party repositories remain governed by their respective owners
and licenses.
