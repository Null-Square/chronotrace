# Pythia Scale Learning-Rate Gate

Date: 2026-08-29

Status: **complete; chronology-blind protocol freeze**.

The scale gate selected a common plain-SGD learning rate before any multi-stage chronology endpoint was trained or scored.

## Frozen controls

- Models: `EleutherAI/pythia-14m-deduped`, `31m-deduped`, `70m-deduped`.
- Revision: `step143000`.
- Precision: FP32.
- Optimizer: plain SGD, momentum `0`, weight decay `0`.
- Stability stage: A only.
- Updates: 8 fixed full-batch updates.
- Candidate LRs: `1e-4`, `3e-4`, `1e-3`, `3e-3`, `1e-2`.
- Selection: largest LR passing the same frozen rule on all three models.
- No chronology labels, AB/BA endpoints, or permutation scores were inputs to selection.

The selector artifact explicitly records `chronology_data_observed=false`.

## Result

Selected common LR: **`1e-4`**.

| Model | LR | Loss ratio | Relative displacement | Pass |
| --- | ---: | ---: | ---: | :---: |
| 14M | 1e-4 | 0.488756 | 0.000168582 | yes |
| 14M | 3e-4 | 4.243227 | 0.000498301 | no |
| 14M | 1e-3 | 2.891250 | 0.001717841 | no |
| 14M | 3e-3 | 11.324709 | 0.005112370 | no |
| 14M | 1e-2 | 13.254548 | 1.295430119 | no |
| 31M | 1e-4 | 0.538356 | 0.000038068 | yes |
| 31M | 3e-4 | 0.871201 | 0.000104306 | yes |
| 31M | 1e-3 | 4.687045 | 0.001432118 | no |
| 31M | 3e-3 | 6.238227 | 0.003230235 | no |
| 31M | 1e-2 | 13.124671 | 0.007790925 | no |
| 70M | 1e-4 | 0.025320 | 0.000152433 | yes |
| 70M | 3e-4 | 0.028598 | 0.000511174 | yes |
| 70M | 1e-3 | 0.066948 | 0.000834681 | yes |
| 70M | 3e-3 | 0.288428 | 0.001861033 | yes |
| 70M | 1e-2 | 0.600458 | 0.006889934 | yes |

The common rate is constrained by the 14M model; this is desirable because the rate was not selected to maximize chronology recovery.

## Tokenizer/data lock

All 15 probes independently regenerated the same tokenizer/codebook/data identity:

- tokenizer fingerprint: `c08d95c22b6ea19a127c2e46b72a5229543eee2d35949b80334388b5ff89e51a`
- codebook SHA-256: `569a03e93bf337286bbda7b5b22255f5b03491a1e270517a2de395776c90894b`
- worlds: `978b673e604867f296c62039236c8bc02c7aeb7c140552da90933556f54c8053`
- stage A: `69d833a008b78a7ce270d1c37ef3cd6c31ac3793e7a860cbc3fc75ea69c797e2`
- stage B: `6aec67c875e9077b6ce634a62bb85b9be58269292dacbfa70eef57a7bfe90ac5`
- stage C: `7e900c0b4a677900f22f68470104625d8e10038bf84c816f2ca0fa5b2aa6022b`

Every alias/entity/signal/zone identifier is exactly two existing Pythia tokenizer IDs, and all 128 identifier token IDs are globally disjoint. The exact codebook is frozen in `configs/pythia_scale_codebook.json`.

## Evidence provenance

- workflow run: `33216943852`
- selection artifact: `9703869554`
- artifact digest: `31deaf9786d7a9a77da1a01ba77d863543a037ada60dce3797d06f10de82add8`

## Observation, not a tuning signal

The 70M checkpoint had a much larger initial completion loss (~314.49) on stage A than 14M/31M (~9). This was not used to modify the protocol because the predeclared selection rule is relative and chronology-blind. It should be investigated separately only if it materially affects interpretation of the later scale result.

## Next locked gate

The first chronology test is Pythia-14M only, 16 updates per stage, all six A/B/C permutations. The finite-pair interaction table is constructed from exactly `N^2 = 9` singleton/directed-pair stage executions. Full three-stage histories are used only as held-out validation endpoints.
