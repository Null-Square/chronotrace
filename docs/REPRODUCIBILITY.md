# ChronoTrace Reproducibility Guide

ChronoTrace separates **release verification**, **proof/code verification**, **paper compilation**, and **model-scale scientific replay**. Reviewers do not need to download Pythia weights to verify the frozen result ledger and certificate implementation.

## Level 0 — frozen release audit

Purpose: verify the paper-facing result and provenance package without model downloads.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
make audit
```

This checks:

- final selection version and scientific-run provenance;
- canonical hashes of the frozen confirmation/methodology/source locks;
- fresh seed set and balanced target schedule;
- per-seed to aggregate arithmetic;
- 27/32 complete-history coverage and 182/192 pair coverage;
- five full-history abstentions and ten ambiguous pair decisions;
- zero contradictions, zero double-exclusions, and non-invalid suite state;
- exact terminal/soundness flags;
- artifact digest formatting and stage-call/freeze accounting;
- consistency of headline result copy in README, reviewer guide, results freeze, and manuscript.

Expected result:

```text
ChronoTrace release audit: PASS
```

## Level 1 — code, proof, and generated-asset checks

Purpose: verify the implementation and synthetic/terminal proof gates.

```bash
make reviewer
```

This runs:

```text
release audit
release asset synchronization
ruff check .
pytest
```

The test suite covers ordered interaction identities, exact-prefix streaming, projected interactions, local-order hierarchy construction, proof-safe dual correction, multi-witness certificates, pairwise label-blind decisions, information-barrier constructions, confirmation protocol drift, and release/package invariants.

## Level 2 — manuscript compilation

Purpose: verify that the journal-neutral paper builds from committed source.

With a TeX distribution containing `latexmk`, standard LaTeX packages, BibTeX, and TikZ:

```bash
make reviewer-full
```

Or from `paper/`:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The GitHub Actions `Paper` workflow performs the same compile on changes to the paper workspace, fails on unresolved citations/references, and uploads the resulting PDF.

## Level 3 — inspect immutable scientific evidence

The frozen scientific run is:

```text
GitHub Actions run: 33418210637
scientific head:    7107221c16a001a7974ca1b436d9cacd26145fe2
```

Fresh seed artifacts:

```text
2186192236  artifact 9768220564  sha256:78d9abc998364b5686bfdcb194ea44e2c8e514fa5623c95a1317559cfad59dcc
1368008047  artifact 9768257657  sha256:0df6ab12047ff36a2c54ef4e1cb7966fb372cf27423e3761347e097e00e0eb96
92712904    artifact 9768112808  sha256:1830a159945aa50b5f4c4e71fe15bbf047cb9dba4e3c96c09fabc81498d4e668
1944430236  artifact 9768224175  sha256:fd6a5a0126507d7f7448ad4effafb5d202a988175725bb9e3c83e612530a0006
```

Raw result SHA-256 values and all aggregate checks are recorded in:

```text
configs/chronotrace_pairwise_multi_witness_confirmation_v3.selection.json
```

Use the artifact digests when downloading evidence through GitHub Actions or the GitHub CLI.

## Level 4 — model-scale replay or independent replication

The original fresh v3 seed jobs should be treated as immutable confirmation evidence. Re-executing them later is a **replication**, not a continuation of the one-shot confirmation protocol.

For an independent replication or extension:

1. use the pinned model/tokenizer/training settings from the v3 locks and workflow;
2. use a new protocol/version and a new seed derivation rule if the goal is a new confirmation claim;
3. run all target histories for the new seed set without adaptation between seeds;
4. preserve exact stage-call accounting and witness-freeze boundaries;
5. compare the new result to v3 without modifying the v3 selection.

The model-scale stack is pinned in the relevant GitHub Actions workflow and protocol locks. Large model/data files are deliberately not stored in the repository.

## Generated publication assets

Reviewer SVGs, paper result macros, and the case-level confirmation matrix are generated from the frozen selection:

```bash
python scripts/generate_release_assets.py --write
python scripts/generate_release_assets.py --check
```

The sync test prevents manual drift between the selection JSON and generated result material.

## What cannot be reconstructed from the repository alone

The repository intentionally does not commit:

- model checkpoints;
- generated training datasets/codebook output directories;
- GitHub Actions artifact ZIPs;
- external package caches.

Instead it records exact model revisions, deterministic generation rules, protocol hashes, run IDs, artifact IDs, and SHA-256 digests. This keeps the repository small enough to review while preserving a verifiable path to the external evidence.

## Scientific interpretation boundary

Successful reproduction of the terminal `N=K=4` experiment does not establish an assumption-free exact fixed-depth decoder for `N>K`. See `docs/K_LOCAL_INFORMATION_BARRIER.md` and `docs/RESULTS_FREEZE.md` before extending the claim.
