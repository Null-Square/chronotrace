# Finite Pair Interaction Gate

Date: 2026-08-29

Status: **positive controlled result; preferred white-box decoder for the first large-model scale gate**.

CI source: PR #10, workflow run `33214025787`.

## Question

Can exact finite pairwise stage interactions reconstruct a full multi-stage chronology after both the one-step HVP approximation and the differential macro-stage approximation leave perfect recovery?

The finite-pair construction uses

`I_{j<-i} = F_j(F_i(theta_0)) - theta_0 - Delta_i - Delta_j`

for every ordered stage pair. After caching the N singleton endpoints, the complete ordered-pair interaction table needs `N(N-1)` additional stage executions: exactly `N^2` total stage executions.

There is no Hessian-vector product, double backward, or finite-difference epsilon. For three or more stages, singleton and pairwise probes are used to predict full histories without replaying every complete chronology.

## Fixed experiment

- model: same deterministic 1,032-parameter causal transformer used by the prior gates;
- optimizer: plain SGD, no momentum;
- per-update learning rate: `0.01`;
- stages: A, B, C;
- updates per stage: `{1,2,4,8,16,32,64,128,256}`;
- ground truth: all `3! = 6` full histories;
- finite-difference macro comparator epsilon: `1e-4`;
- finite-pair decoder epsilon: none.

## Recovery result

| Updates/stage | Micro HVP | Differential macro | Finite pair | Max singleton displacement |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 6/6 | 6/6 | 6/6 | `0.01085` |
| 2 | 4/6 | 6/6 | 6/6 | `0.02158` |
| 4 | 2/6 | 6/6 | 6/6 | `0.04271` |
| 8 | 2/6 | 6/6 | 6/6 | `0.08384` |
| 16 | 3/6 | 6/6 | 6/6 | `0.16262` |
| 32 | 5/6 | 6/6 | 6/6 | `0.31768` |
| 64 | 4/6 | 6/6 | 6/6 | `0.61171` |
| 128 | 2/6 | 4/6 | 6/6 | `1.15237` |
| 256 | 1/6 | 3/6 | 6/6 | `2.08180` |

Failure boundaries:

- micro HVP decoder first loses perfection at **2 updates/stage**;
- differential macro decoder first loses perfection at **128 updates/stage**;
- finite-pair decoder has no failure in the fixed sweep through **256 updates/stage**.

## Pairwise margins and higher-order residual

| Updates/stage | Finite-pair min signature separation | Finite-pair min decode margin | Max triple+ remainder norm | `2*remainder/separation` |
| ---: | ---: | ---: | ---: | ---: |
| 1 | `4.8898e-05` | `4.7250e-05` | `1.4270e-06` | `0.0584` |
| 2 | `1.9373e-04` | `1.8157e-04` | `1.0695e-05` | `0.1104` |
| 4 | `7.6086e-04` | `6.7783e-04` | `7.5198e-05` | `0.1977` |
| 8 | `2.9427e-03` | `2.4499e-03` | `4.7402e-04` | `0.3222` |
| 16 | `1.1112e-02` | `8.5240e-03` | `2.6350e-03` | `0.4743` |
| 32 | `4.0685e-02` | `2.7829e-02` | `1.4815e-02` | `0.7283` |
| 64 | `1.4585e-01` | `6.4228e-02` | `1.0420e-01` | `1.4289` |
| 128 | `5.3561e-01` | `1.8630e-01` | `5.3628e-01` | `2.0025` |
| 256 | `1.8476` | `1.6685e-01` | `1.8636` | `2.0173` |

The simple sufficient nearest-signature certificate

`2 ||r_high|| / delta_min < 1`

holds through 32 updates/stage. It becomes inconclusive from 64 onward, but empirical decoding remains 6/6 through 256. Therefore the global norm bound is conservative: the higher-order residual is not aligned adversarially with the closest wrong chronology signature in this controlled system.

This distinction must be preserved. We can claim successful recovery outside the sufficient certificate; we cannot claim the certificate guarantees those later points.

## Interpretation

The result supports three increasingly strong statements in the controlled setting:

1. chronology information is an antisymmetric interaction effect, not merely first-order learning displacement;
2. complete finite stage maps extend recovery far beyond one-step local geometry;
3. exact finite pair interactions extend recovery beyond a local derivative of the stage map while removing the perturbation-scale hyperparameter.

The finite-pair method is therefore the preferred candidate for the first Pythia-scale white-box experiment.

## New research object: training-history interaction order

The finite-pair endpoint is a second-order interaction truncation. Its residual contains genuine three-stage and higher interactions. This suggests a hierarchy:

- singleton effects: `O(N)` probes;
- pair interactions: `O(N^2)` probes;
- triple interactions: `O(N^3)` probes;
- ...
- exhaustive chronology replay: factorial histories.

A potentially useful quantity is the **minimum interaction order required to identify a training chronology** under a target error/confidence level. This is not yet implemented or claimed as a contribution; it is a research direction motivated by the measured pairwise residual.

## Limits

This remains a deterministic small-model result with known base checkpoint, known exact stages, full weights, and plain SGD. It does not establish robustness to stochastic batches, persistent Adam state, unknown stage recipes, distillation, merging, or black-box access.
