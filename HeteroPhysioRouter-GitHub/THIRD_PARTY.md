# Third-party methods and data

The repository license applies only to files distributed in this repository.
It does not relicense datasets, papers, checkpoints, or source code maintained
by other authors.

## Datasets

| Dataset | Source | Distribution |
|---|---|---|
| Sleep-EDF Expanded | [PhysioNet](https://physionet.org/content/sleep-edfx/1.0.0/) | Download under the terms shown by PhysioNet |
| PAMAP2 | [UCI repository](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring) | Download under the terms shown by UCI |
| WESAD | [Project page](https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/) | Download under the terms shown by the authors |

No raw data or derived subject cache is included here.

## Sleep-EDF benchmark lineage

The common cache can be consumed by local adapters for the following upstream
projects. Their source code is deliberately not copied into this repository.

| Method | Upstream source or paper |
|---|---|
| SleepDIFFormer | [Project page](https://lanxin1105.github.io/SleepDIFFormer-Page/), [arXiv:2508.15215](https://arxiv.org/abs/2508.15215) |
| MHFNet | Paper: *MHFNet: A Multimodal Hybrid-Embedding Fusion Network for Automatic Sleep Staging*; obtain code from the paper's author release |
| SleepFocalNet | [Author repository](https://github.com/hasanzan/SleepFocalNet), [DOI](https://doi.org/10.1007/s10044-025-01475-0) |
| LogicSleep | Paper: *LogicSleep: a neurosymbolic-guided framework for explainable and personalized sleep staging*; obtain code from its official release |
| TinySleepNet | [Author repository](https://github.com/akaraspt/tinysleepnet), [IEEE record](https://ieeexplore.ieee.org/document/9176741) |

Results from these projects must be labeled as common-input adaptations unless
the unmodified author protocol is used end to end.

## PAMAP2 benchmark lineage

`benchmarks/pamap2` contains compact PyTorch architecture adaptations for
DeepConvLSTM, DeepSense, Attend-and-Discriminate, TinyHAR, MC-CNN, and SA-HAR.
The implementations preserve the documented central operations but use the
repository's common PAMAP2 cache, subject split, and output metrics. They are
not byte-for-byte copies of upstream repositories and must not be described as
exact reproductions of source-paper scores.

## WESAD benchmark lineage

| Benchmark name | Source | Local status |
|---|---|---|
| COCOA | [cruiseresearchgroup/COCOA](https://github.com/cruiseresearchgroup/COCOA), DOI `10.1145/3550316` | Common-input architecture adaptation |
| SemiPFL | [arvintashakori/SemiPFL](https://github.com/arvintashakori/SemiPFL), DOI `10.1109/JIOT.2022.3233599` | Common-input architecture adaptation; upstream is GPL-3.0 |
| Stress Calibration | [KMDR82/wearable-stress-calibration](https://github.com/KMDR82/wearable-stress-calibration) | Common-input method adaptation |
| FG-DualNet | [rashidkamal-ai/poisoning-resilient-stress-detection](https://github.com/rashidkamal-ai/poisoning-resilient-stress-detection) | Common-input architecture adaptation |
| Husformer | [SMARTlab-Purdue/Husformer](https://github.com/SMARTlab-Purdue/Husformer), [arXiv:2209.15182](https://arxiv.org/abs/2209.15182) | Public-geometry architecture adaptation |

The local implementations were written as interoperable adapters; the
upstream source trees and their licenses remain separate. In particular, code
from a GPL or unlicensed upstream repository has not been copied into the MIT
package.
