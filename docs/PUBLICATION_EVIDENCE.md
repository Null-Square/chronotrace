# Publication evidence ledger — 12 September 2026

Current article: **ChronoTrace: Conditional training-order verification by bidirectional replay**. Author: **Omar Al-Tawil**. Target journal: **Neurocomputing**, original research (not accepted/submitted).

| Claim | Evidence anchor | Boundary |
|---|---|---|
| No-loss conditional middle join and budget-safe replay | `docs/theory/BIDIRECTIONAL_JOIN.md`; current manuscript theorems | Exact arithmetic and justified allowances; no interval proof |
| FP32 unique verification 72/72 versus ranked matches 9/72 | `research/bidirectional_v1/utility_summary.json`; S1 `artifacts/confirmation_v1/records.json` | 72 endpoints but 12 seed clusters and 12 sampled orders; shared dataset |
| Full primary verification 120/144 and 3,360 pairs, zero observed wrong | Same records; 72 full candidate-distance references in S1 | Exports paired; strong-FP16 failures retained; no universal error-rate claim |
| FP32 median exhaustive/new gradient ratio 8.7381596064 | Same records, exact work ledger | Cold prefixes, all inverse checks, replay included; not total-complexity improvement |
| Median paired timing speedup 7.3273558773x | S1 `artifacts/serial_timing_v1.json` | Eight spent-input same-environment pairs, not ranked-runtime; CPU model not recorded |
| Frozen-encoder head 12/12 observations verified | S1 `artifacts/neural_head_bridge_v1/records.json` | Six heads; shared encoder; only first target fully enumerated |
| 1,073 standalone test items | S1 tests; publication-preparation rerun | Different from the native repository suite; not scientific sample size |

## Frozen identity

Native scientific engine commit: `1d9f222739c097291a6b843f020413fd311053ef`.
Primary protocol SHA-256: `931b5e2caa42a9d707ade143b9da9a22360ee0f10e9255dfe21a9c508ac1a0e5`.
Primary records SHA-256: `0da6ba2e6b06e65d1d640bfa79ea8cce187a2079cce3dbc9bbd26e336efb5181`.

The frozen runner does not pass generating labels into the decoder. Complete endpoint-reference construction occurs after primary decisions. Ranked costs use that reference to score a fixed ranking, not to select it. Numerical reproducibility, independent computational paths, and external replication are different claims.

The primary figure/table aggregates are checked by `scripts/audit_publication.py`. Full raw-data checks require Supplement S1. Historical Pythia and secant outputs remain historical; no source lock, seed, threshold, or scientific result is rewritten during publication preparation.
