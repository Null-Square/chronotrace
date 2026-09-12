# ChronoTrace

**Conditional training-order verification by bidirectional replay.**

Author: **Omar Al-Tawil**. Current manuscript target: **Neurocomputing** (Elsevier), original research article. This is a publication-preparation repository, not an accepted paper or a production forensic service.

> Given a known base, known deterministic training stages, and an exported final checkpoint, which stage orders remain compatible—and can alternatives be excluded more cheaply than complete replay?

## Current paper and evidence

Start with [the manuscript source](paper/bidirectional/main.tex), [the reviewer guide](docs/REVIEWER_GUIDE.md), and [the publication evidence ledger](docs/PUBLICATION_EVIDENCE.md). `make paper` builds the current manuscript; previous manuscript material is archived in `paper/legacy_pythia/`.

| Observation setting | Bidirectional unique verified | Equal-gradient-work ranked first match |
|---|---:|---:|
| FP32 export, all durations | **72/72** | 9/72 |
| FP16 export, durations 5 and 10 | **48/48** | 6/48 |
| FP16 export, duration 20 | **0/24** | **8/24** |

The primary study has **72 problem-specific endpoints, 144 export observations, 12 seed clusters, and 12 sampled permutations reused across task/duration settings**. It uses eight known stages, 32 full-batch GD updates per stage, and two 64-dimensional digit-classification tasks. Training and replay are FP64; export precision is not training precision.

All 40,320 endpoints were independently enumerated for every primary problem after predictions were fixed. Every reference-compatible order was retained. Across the 144 observations, 120 full histories and 3,360 pair relations were verified, with zero observed wrong outputs. Zero observed errors is not a universal zero-error guarantee. All 24 strong-FP16 failures have unique exhaustive solutions: they are bound/budget failures, not proven information loss.

**Measured value:** median FP32 gradient-work reduction is **8.74x** versus complete prefix-cached replay. Eight spent-input serial timing pairs yield a **7.33x median paired runtime speedup** versus complete replay (1.824 s versus 13.214 s marginal medians). This is not a runtime comparison against ranked replay; its work was scored using the exhaustive reference. The original timing artifact lacks a CPU-model identifier.

A separate frozen-encoder bridge verifies 12 observations of six **170-parameter classifier heads**, after 72.31–72.46% head-loss reduction. It does not audit the encoder or a complete deep network; only the first bridge target has a complete endpoint reference.

## Method and boundaries

The native engine is [`bidirectional.py`](src/chronotrace/geometry/bidirectional.py). It joins forward prefix states with residual-controlled inverse suffixes and selectively replays survivors against coordinate-wise checkpoint rounding cells. Failed inverse iterations enlarge uncertainty. Replay-budget exhaustion retains untested candidates.

Guarantees require a known base, immutable deterministic GD recipes, known gradients, correct global Lipschitz bounds with `eta * L < 1`, and valid numerical allowances. The code uses fixed numerical guards, **not formally validated interval arithmetic**. A `verified_unique` result means replay consistency and conditional exclusion under the declared candidate model, not unconditional provenance or legal ownership.

Current middle joins still compare `N!` vector pairs; memory and worst-case verification remain combinatorial. Unknown batches, unknown recipes, Adam/momentum state, missing/repeated stages, and arbitrary LLM pipelines are not validated.

## Reproduce

Native implementation and full repository tests:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
python -O -m pytest -q
python scripts/audit_publication.py
make paper
```

The **complete frozen experimental workspace and raw candidate-distance evidence are supplied as Supplement S1**, not duplicated into this repository's native scripts directory. See [reproduction and artifact identity](research/bidirectional_v1/REPRODUCE.md). Use its pinned environment for numerical reproduction; the native repository development dependencies are broader.

From the extracted Supplement S1 workspace:

```bash
pip install -r requirements-bidirectional.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:scripts
python scripts/verify_bidirectional_artifacts.py --root .
python -m pytest -q tests
python scripts/run_bidirectional_confirmation.py \
  --config configs/bidirectional_v1.lock.json \
  --output artifacts/independent_rerun --workers 1
```

Stored-artifact verification is not a new training run or external replication. The standalone workspace has 1,073 test items; native repository tests are a different collection. Publication CI tests native normal/optimized execution separately.

## Historical evidence, not the current headline

The earlier Pythia-14M terminal confirmation remains **27 / 32** full histories and **182 / 192** pairs, with the original **STRONG** outcome tier. It used `N=K=4`, one update per stage, and complete terminal candidate acquisition. Its result and scientific locks are unchanged. Logical closure (28/32) and stored discrete distances (32/32) are post-hoc audits, not replacement confirmation outcomes. See [historical freeze](docs/RESULTS_FREEZE.md).

Historical negative/common-tail results and the later nonterminal secant study remain visible; none is pooled into the current sample. Reversible optimization, temporal traces, and meet-in-the-middle search are established ideas; the paper claims a particular assumption-explicit audit construction and measured work boundary.

## AI-script disclosure and security scope

The Methods explicitly disclose ChatGPT's substantive role in writing and revising the experimental Python scripts, tests, analysis and plotting code; the manuscript separately discloses AI-assisted text drafting. The [revision note](docs/AI_ASSISTANCE_AND_DUAL_USE_2026_09_12.md) records the scope and remaining author-review requirements.

The new dual-use discussion considers whether chronology information could help an observer who already has checkpoint and recipe access. It does **not** demonstrate LLM safeguard removal, a detachable safety layer, hidden-weight extraction, or semantic understanding of individual weights. The [defensive evaluation protocol](docs/security/CHRONOLOGY_INFORMATION_RISK_PROTOCOL.md) is proposed and unexecuted. No original result or scientific code is changed by this discussion.

## Publication status

[Submission checklist](paper/SUBMISSION_CHECKLIST.md) records the remaining author approvals and journal checks. Affiliation, corresponding email, funding/conflicts, contribution declarations, rights/licensing, and final author approval must not be inferred from repository ownership. No acceptance, published DOI, external replication, or software license is asserted.

The author is responsible for the final submitted work. The manuscript discloses AI assistance in methods, analysis, coding, and writing. Cite the exact software commit and the manuscript as unpublished until a publication identifier exists; see [CITATION.cff](CITATION.cff).
