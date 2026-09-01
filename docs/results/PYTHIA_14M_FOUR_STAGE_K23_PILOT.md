# Pythia-14M Four-Stage FP64 K2/K3 Methodology Pilot

Date: 2026-08-29

Status: **the pre-frozen comparative hierarchy rule passes on the non-confirmatory pilot codebook, but absolute four-stage reconstruction remains weak. Confirmation codebooks remain unobserved at the time of this record.**

## Role

This run is methodology development only. It uses Pythia-14M, FP64, one update per stage, the previously used T2 codebook seed `1011473075`, and all 24 A/B/C/D permutations. It is not held-out confirmation evidence.

The four confirmation seeds `4294917749`, `3885207466`, `402469483`, and `2000073798` were prohibited by the pilot runner and were not observed.

## Frozen inputs

- Model: `EleutherAI/pythia-14m-deduped@step143000`
- Precision: FP64 portable CPU path
- Learning rate: `1e-4`, selected by the chronology-blind singleton calibration run `33265789914`
- Updates per stage: `1`
- Pilot protocol: `configs/pythia_14m_four_stage_k23_pilot.lock.json`
- Pilot interpretation: `configs/pythia_14m_four_stage_k23_pilot_interpretation.lock.json`
- Candidate objective: exact full-parameter Euclidean endpoint error
- Candidate implementation: exact quadratic-form scoring, validated against direct full-vector L2 on the Pythia run

## Evidence

- Workflow run: `33267599644`
- Artifact ID: `9719284325`
- Artifact digest: `sha256:62e2b5e75e8ba4b6168471a78e40d0f4bd0e8ad7289d041dc25439f618b0615a`
- Raw result JSON SHA256: `3c4f5986175c9b4ae1af9ce4fd8ebaa72e7c63fa91e72a29a8fa670bf236c02c`
- Canonical result JSON SHA256: `6dba413a9ff6c622803cd165fe78cd9eb314d865ff126472554c2cd57c56eaf2`
- Numerical execution fingerprint: `deaad55af513e78d0c4c1d5636836bcb7d7325be64d8df0b196cd6a66b262d42`
- Quadratic/direct spot checks: `4`
- Maximum quadratic/direct absolute error: `9.411915691259765e-14`

## Result

| Metric | K=2 | K=3 |
|---|---:|---:|
| Full order | `0/24` | `3/24` |
| Prefix depth 1 | `6/24` | `9/24` |
| Prefix depth 2 | `2/24` | `5/24` |
| Prefix depth 3 | `0/24` | `3/24` |
| Pairwise precedence | `77/144` | `79/144` |
| Mean true-candidate endpoint error | `0.1980420` | `0.1333009` |
| Minimum full-order margin | `1.4725e-5` | `1.7315e-5` |

Comparison:

- K2 wrong -> K3 correct: `3`
- K2 correct -> K3 wrong: `0`
- K3 has lower true-candidate endpoint error: `24/24`
- mean true-error ratio K3/K2: `0.6738115`

The three exact K3 successes are `BADC`, `CBAD`, and `DBAC`. K3 predictions remain concentrated on a small set of candidate templates, so this is not evidence that four-stage chronology is solved.

## Frozen interpretation-rule adjudication

All five pre-frozen hierarchy-support conditions pass:

1. K3 full-order correct is strictly greater than K2: `3 > 0` — PASS.
2. K2-wrong/K3-correct exceeds K2-correct/K3-wrong: `3 > 0` — PASS.
3. Mean true-error ratio K3/K2 is below 1: `0.6738115 < 1` — PASS.
4. K3 depth-3 prefix recovery is not lower: `3 >= 0` — PASS.
5. K3 pairwise-precedence recovery is not lower: `79 >= 77` — PASS.

The pairwise-sufficient exception does not apply because K2 is `0/24`.

## Interpretation

The defensible pilot conclusion is narrow:

> On this non-confirmatory Pythia-14M four-stage instance, exact degree-3 training interactions contain additional chronology information beyond the degree-2 truncation under the same endpoint-distance decoder.

The pilot does **not** establish robust four-stage reconstruction, generalization across codebooks, superiority to ordinary recency/capability baselines, or a paper-level result.

## Decision

The frozen pilot rule permits a held-out confirmation protocol to be frozen. Before exposing the four confirmation codebooks, the confirmation design must preserve K2-vs-K3 as the primary comparative test, report the full partial-order hierarchy and fixed-terminal strata, and include a marginal stage-loss/recency baseline as a secondary falsifier of trivial temporal-capability explanations.

31M remains blocked.
