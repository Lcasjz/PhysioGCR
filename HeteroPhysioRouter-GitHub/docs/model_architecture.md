# Model architecture

PhysioGCR (Physiological Group-Consistent Recognition) treats a physical sensor
or physiological modality as the
same unit in representation construction, synthetic degradation, and
robustness evaluation. The reference implementation is
`src/heterophysiorouter/model.py`.

## 1. Multi-granular tokenizer

For an input `x` with shape `[batch, time, channels]`, channels are partitioned
into declared physical groups. The tokenizer creates three views:

- **Sensor-group tokens** aggregate local patch embeddings within each physical
  group.
- **Temporal tokens** aggregate all groups at each temporal patch.
- **Global token** summarizes the complete sample.

Learned channel, group, and temporal embeddings retain sensor identity. The
group declarations are dataset-specific and are never inferred from feature
column order at evaluation time.

## 2. Quality-aware cross-granular routing

Scale-invariant proxies measure roughness, flatness, outlier rate, crest factor,
spectral entropy, and availability without mixing channels with incompatible
physical units. Separate Transformer encoders model sensor and temporal tokens.
Bidirectional cross-attention exchanges information between the two views.

Sensor, temporal, and global representations are pooled independently. A
sample-conditioned router then predicts three normalized branch weights from
the representations and reliability summaries. The weighted representation is
fed to an MLP classifier.

## 3. Group-degradation training

`corrupt_groups` applies dropout, scale-relative noise, or flatline corruption
to whole physical groups. The `full_no_calibration` configuration trains the
classifier on these degraded samples. The `full` configuration additionally
uses a monotonic ranking loss so affected groups should receive lower relative
reliability while unaffected groups remain stable.

## 4. Local correction research extension

`local_corruption.py` and `local_correction.py` contain the group-by-time
integrity diagnostics and the task-relevance local adapter used in the
mechanistic experiments. These modules test whether selective correction has a
useful performance ceiling. They are research extensions, not required for the
main router baseline.

## Input conventions

| Dataset | Tensor shape | Physical groups | Classes |
|---|---:|---|---:|
| Sleep-EDF | `B x 3000 x 3` | EEG1, EEG2, EOG | 5 |
| PAMAP2 | `B x 24 x 52` | heart rate, hand, chest, ankle | 12 |
| WESAD | four native-shape streams | Resp, BVP, ECG, EDA | 3 |

WESAD uses the same design through `benchmarks/wesad/quality_module.py`, where
the quality representation is integrated into each literature backbone by a
zero-initialized residual FiLM adapter.
