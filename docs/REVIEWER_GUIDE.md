# Reviewer guide: current bidirectional manuscript

Start with `paper/bidirectional/main.tex`, `docs/PUBLICATION_EVIDENCE.md`, the native `src/chronotrace/geometry/bidirectional.py`, and `tests/test_bidirectional.py`.

The output is a conservative live set under a known-recipe model. A unique replay-consistent survivor is not an unconditional provenance proof. The crucial assumptions are global regularity, invertible GD updates, exact candidate coverage, and valid numerical allowances. Fixed guards are not interval arithmetic.

## Evidence checks

1. Read all six primary rows, including 0/24 strong-FP16 verification and 8/24 ranked retrieval. The primary statistical unit is the seed cluster; 72 endpoints share 12 sampled permutations.
2. Distinguish 8.74x gradient-work reduction from 7.33x measured paired timing speedup. Both are versus exhaustive prefix-cached replay; ranked replay has only a scored-work comparator.
3. Inspect Supplement S1's 72 complete candidate-distance references, the 144 output records, and the set-retention checks. The verifier checks stored artifacts; rerunning the frozen driver independently recomputes distances.
4. Treat the 170-parameter frozen-encoder head experiment separately. It is not full-network/LLM validation, and only one of six targets has a complete reference.

`python scripts/audit_publication.py` checks current paper-facing aggregation. `python -m pytest -q` and `python -O -m pytest -q` exercise the native repository. `make paper` builds the current Elsevier-class manuscript. See `research/bidirectional_v1/REPRODUCE.md` for the separate pinned experimental workspace.

## Historical record

The earlier Pythia manuscript and its 27/32 histories, 182/192 pairs, and zero contradictory inferred pairs remain in `paper/legacy_pythia/` and `docs/RESULTS_FREEZE.md`. They are not the present study. `make audit` and `make assets-check` remain historical Pythia audit commands, not bidirectional result verification.

## Scope of review still needed

Specialist theorem review, external replication, broader models and strong temporal baselines are not claimed to have occurred. The author must approve the final manuscript and complete administrative declarations before journal submission.
