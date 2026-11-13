# Reproducibility notes

## Claims and labels

This repository contains two different types of code:

1. The PhysioGCR implementation and its ablations.
2. Common-input adaptations of published benchmark architectures.

An adapted row is not an exact reproduction of the score reported in the
source paper. Dataset splitting, labels, metrics, and input samples are fixed
within each benchmark so architecture comparisons use the same evaluation
units. Architecture-specific operations and losses are retained where the
available paper or source code specifies them.

## Checkpoints and selection

- Development uses validation macro-F1 for checkpoint selection.
- Test sets are read once after validation-only selection is complete.
- Cache manifests contain protocol metadata and checksums.
- PAMAP2 and WESAD additionally provide subject-level LOSO entry points.
- Report mean and standard deviation across declared seeds or LOSO folds; do
  not select a seed using test performance.

## Expected artifacts

Training commands write a protocol JSON, a best checkpoint, and metrics JSON
under the requested output directory. Generated caches, checkpoints, logs, and
raw datasets are excluded from Git by default.

## Known boundaries

- Sleep literature models live in their upstream repositories because several
  do not provide a redistribution license. This repository supplies the shared
  cache and an in-repository tokenizer benchmark; see `THIRD_PARTY.md`.
- The common-input WESAD rows preserve the named mechanisms but are explicitly
  architecture adaptations, not byte-for-byte source reproductions.
- Natural device-failure annotations and external clinical cohorts are not
  distributed with these datasets. Synthetic group degradation therefore does
  not establish performance under every real hardware failure.
