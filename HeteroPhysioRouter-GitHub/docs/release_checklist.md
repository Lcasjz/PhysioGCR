# GitHub release checklist

- Replace `Anonymous authors` and `OWNER` in `CITATION.cff` and `pyproject.toml`.
- Add the final paper title, DOI/arXiv identifier, and BibTeX after publication.
- Confirm that every copied or adapted file is authored by the project team or
  is compatible with the repository license.
- Do not force-add files ignored under `data/`, `runs/`, `logs/`, or checkpoint
  extensions.
- Run `python -m compileall src experiments benchmarks scripts tests`.
- Install the development extra and run `python -m pytest -q`.
- Dry-run each long queue with `python scripts/run_suite.py ... --dry_run`.
- Publish exact cache checksums, seeds, split declarations, and software
  versions with the paper release.
