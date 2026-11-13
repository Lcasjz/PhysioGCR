# Benchmark overview

All rows within a dataset use a shared sample cache, subject protocol, label
mapping, train-only normalization, and metric contract. Model-specific signal
operations and losses are retained when they are part of the documented method.

| Dataset | Included in-repository rows | Evaluation |
|---|---|---|
| Sleep-EDF | Multi-granular tokenizer; PhysioGCR | Fixed subject-disjoint train/validation/test |
| PAMAP2 | DeepConvLSTM, DeepSense, Attend-and-Discriminate, TinyHAR, MC-CNN, SA-HAR, PhysioGCR | Fixed split and PhysioGCR LOSO |
| WESAD | COCOA, SemiPFL, Stress Calibration, FG-DualNet, Husformer, each with optional quality adapter | Strict LOSO |

The SleepDIFFormer, MHFNet, SleepFocalNet, LogicSleep, and TinySleepNet source
trees used in the paper experiments are maintained by their respective authors
and are not redistributed. The Sleep-EDF cache contract in this repository is
the interoperability boundary for those local adapters. See `THIRD_PARTY.md`.

The PAMAP2 and WESAD literature rows are common-input architecture adaptations.
They support controlled comparison under one protocol, but are not presented as
exact reproductions of source-paper scores.
