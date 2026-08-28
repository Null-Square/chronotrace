# Phase-0 v1 discovery result

## Status

**Discovery completed; confirmation intentionally not run.**

Phase-0 v1 asked whether a Pythia-70M endpoint trained on the same synthetic facts in opposite macro orders (`A -> B` versus `B -> A`) retained an order-dependent signature that could be detected from fixed post-training probes without merely reading out ordinary task capability.

The implementation was run under the frozen protocol fingerprint:

`7bf6a5714a95f3b14780892552777abd45ad99c9830c5b9bb1a88a5c2354a220`

The clean FP32 discovery workflow was GitHub Actions run `33201301575` on branch `experiment/phase0-discovery-fp32`. All 16 discovery endpoints (8 paired seeds x 2 histories) completed with finite FP32 training metrics and finite feature extraction.

## Discovery readout

| Readout | Balanced accuracy | AUROC |
| --- | ---: | ---: |
| Forensic feature detector | 1.000 | 1.000 |
| Capability-only baseline | 1.000 | 1.000 |

The perfect forensic classification is therefore **not evidence of a nontrivial training-path memory**. Ordinary A/B capability alone also recovers the macro order perfectly.

The predeclared capability-matching gate required, for every matched seed, both the A-control and B-control mean-margin differences between histories to be <= `1.0`. All eight seed pairs failed this gate.

| Seed | A-control gap | B-control gap | Max gap | Gate |
| ---: | ---: | ---: | ---: | --- |
| 11 | 8.0913 | 7.9811 | 8.0913 | fail |
| 13 | 3.3939 | 13.8596 | 13.8596 | fail |
| 17 | 7.9084 | 9.5033 | 9.5033 | fail |
| 19 | 6.2987 | 13.5211 | 13.5211 | fail |
| 23 | 0.2134 | 13.9531 | 13.9531 | fail |
| 29 | 3.2459 | 5.9420 | 5.9420 | fail |
| 31 | 9.7041 | 8.5290 | 9.7041 | fail |
| 37 | 10.0165 | 6.9914 | 10.0165 | fail |

Maximum observed matched control-margin gap: `13.953095368496685`.

## Interpretation

Phase-0 v1 is a **clean confounded/negative result** for the intended claim. The endpoint contains enough information to identify which stage was most recent, but v1 does not isolate a historical/path signature beyond that ordinary recency/forgetting effect.

The confirmation gate therefore remains closed. Confirmation seeds `101`, `103`, `107`, and `109` were not trained, sealed, or inspected.

## Consequence for Phase-0b

The next design must remove the trivial terminal-capability cue before asking whether any order information survives. Phase-0b introduces an identical balanced terminal rehearsal stage `C` after both histories:

- `A -> B -> C`
- `B -> A -> C`

Stage C contains an equal mixture of A and B examples and uses the same deterministic stage-specific shuffle rule regardless of the preceding order. Its length is selected using only already-consumed v1 discovery seeds. Only after a fixed C design satisfies the capability-matching requirement will a new protocol be frozen with fresh discovery seeds. The original confirmation seeds remain untouched.
