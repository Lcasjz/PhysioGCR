# GitHub publication guide

## 1. Final identity

- **Repository name:** `PhysioGCR`
- **Method name:** PhysioGCR
- **Expansion:** Physiological Group-Consistent Recognition
- **One-line description:** Group-consistent multi-granular learning for robust physiological time-series recognition.
- **Suggested topics:** `physiological-signals`, `time-series`, `multimodal-learning`, `wearable-sensing`, `sleep-staging`, `human-activity-recognition`, `missing-modality`, `pytorch`

Use `PhysioGCR` consistently in the paper, repository title, figure captions,
release names, and citation. `HeteroPhysioRouter` may appear only as the legacy
Python namespace retained for checkpoint and script compatibility.

## 2. Before creating the repository

1. Replace `OWNER` and `Anonymous authors` in `CITATION.cff` and
   `pyproject.toml` if the review policy permits identification.
2. Decide whether the venue permits a public repository during double-blind
   review. If not, create a private or anonymous mirror and keep author names
   removed.
3. Run the checks in `docs/release_checklist.md`.
4. Inspect the staged files before the first commit. Never force-add ignored
   data, checkpoints, logs, caches, or subject-level artifacts.

## 3. Create and upload

Install Git for Windows and GitHub CLI, then open a terminal in the repository:

```bat
cd /d D:\HeteroPhysioRouter-GitHub
git init
git branch -M main
git add .
git status
git commit -m "Initial public release of PhysioGCR"
gh auth login
gh repo create PhysioGCR --public --source . --remote origin --push
```

The commands above use Windows CMD. The local folder name does not need to
match the GitHub repository name. GitHub CLI is optional: the Chinese guide
also describes creating an empty repository in the browser and pushing with
Git alone.

For a review-anonymous release, use a neutral account and omit personal links,
commit identities, acknowledgments, and institution-specific paths. Follow the
venue's policy rather than assuming that anonymized GitHub is always allowed.

## 4. Configure the GitHub page

Set the repository description to:

> Group-consistent multi-granular learning for robust physiological time-series recognition across Sleep-EDF, PAMAP2, and WESAD.

Add the suggested topics above. Enable Issues, keep Discussions optional, and
do not enable Git LFS for datasets because raw data must remain with the data
owners. Add links to the paper and project page only when they are public.

## 5. First release

Create tag `v0.1.0` after reproducing the documented smoke tests:

```bash
git tag -a v0.1.0 -m "PhysioGCR v0.1.0"
git push origin v0.1.0
gh release create v0.1.0 --title "PhysioGCR v0.1.0" --notes "Initial research release with three common-input benchmarks."
```

Attach no raw data. Checkpoints should be released separately only if their
training data terms permit redistribution; publish checksum and protocol
metadata beside every checkpoint.

## 6. Recommended README order

1. Method name and one-sentence claim.
2. Framework figure after the final Fig. 1 is available.
3. News and paper links.
4. Method summary and supported claims.
5. Installation.
6. Dataset preparation with owner download links.
7. Minimal training and evaluation examples.
8. Full benchmark commands and expected artifacts.
9. Reproduction boundaries and third-party sources.
10. Citation, license, and contact.

## 7. Suggested launch text

### Short GitHub/social description

> PhysioGCR aligns the physical sensor groups used for representation,
> training-time corruption, and robustness evaluation. The release includes a
> multi-granular PyTorch implementation and common-input protocols for
> Sleep-EDF, PAMAP2, and WESAD.

### Longer project introduction

> Physiological recognition models combine heterogeneous sensors, but the unit
> represented by a model often differs from the physical unit that becomes
> unavailable or corrupted in deployment. PhysioGCR addresses this mismatch
> through sensor-group consistency: one physical partition organizes
> multi-granular representation, structured degradation, and condition-specific
> evaluation. Sensor-group, temporal-patch, and global evidence are retained in
> parallel and can interact before prediction. The repository provides the
> method, controlled robustness experiments, locked evaluation utilities, and
> common-input benchmark adapters for three physiological recognition tasks.

Avoid describing all adapted benchmark rows as exact original-paper
reproductions, and avoid claiming that dynamic routing is uniformly responsible
for the robustness gain. The evidence most strongly supports physical group
alignment and structured group-degradation exposure.
