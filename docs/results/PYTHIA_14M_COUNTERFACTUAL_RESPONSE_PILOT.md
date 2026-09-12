# Pythia-14M counterfactual first-order response pilot

Status: complete non-confirmatory methodology pilot. This result uses only the already-spent pilot codebook and does not observe any held-out confirmation codebook.

## Evidence identity

- workflow run: `33269532005`
- artifact: `9719904212`, `pythia-14m-counterfactual-response-pilot`
- artifact digest: `sha256:109e351bbcd4529a4c141f77878b0d7f62add059cd78206aed5880c8e43f7a7d`
- raw result JSON SHA-256: `95033d740119445896c6d48fa4abbc2c6fbafd500a7ba74f6d21a826fa335207`
- canonical result JSON SHA-256: `20c807a82c48f560e6317bc937fd635950dc30b49199b4e78c3781cce10575d8`
- model: `EleutherAI/pythia-14m-deduped`, revision `step143000`
- precision: FP64
- learning rate: `1e-4`
- pilot codebook seed: `1011473075`
- confirmation codebooks observed: **false**
- stage executions: `136`
- gradient response evaluations: `288`

## Integrity replay

The runner completed all frozen guards. It reproduced the previously frozen 24 endpoint hashes and reproduced the old endpoint metrics exactly:

| decoder | full order | depth-3 prefix | pairwise precedence |
|---|---:|---:|---:|
| K2 endpoint | 0/24 | 0/24 | 77/144 |
| K3 endpoint | 3/24 | 3/24 | 79/144 |

Therefore the response result is compared against the same endpoint experiment rather than a numerically drifting rerun.

## Frozen primary response

The primary response family was fixed before any susceptibility output:

`[loss_A, loss_B, loss_C, loss_D, S_AB, S_AC, S_AD, S_BC, S_BD, S_CD]`

where

`S_AB(theta) = <grad L_A(theta), grad L_B(theta)>`.

Candidate response references were generated only from the K2/K3 interaction-predicted candidate states. Reference standardization was fit from candidate references only; target responses were not used to construct or normalize candidate references.

## Result

The first-order active-response decoder failed.

For K3 candidate states, every tested response family produced only:

- full order: **1/24**
- depth-3 prefix: **1/24**
- pairwise precedence: **72/144**

This includes loss-only, self-susceptibility-only, cross-susceptibility-only, loss plus cross susceptibility, and the full loss-plus-susceptibility family.

For the frozen primary K3 family, the five checks were:

- active full order beats K3 endpoint: **false**
- active full order beats recency: **false**
- active depth-3 prefix is not worse than K3 endpoint: **false**
- active precedence is not worse than K3 endpoint: **false**
- candidate reference separation is positive: **true**

Thus `primary_active_adds_information_all=false` and `strong_low_order_rescue=false`.

## Failure diagnosis

The failure is not caused by candidate references collapsing together. The K3 primary reference family has positive standardized separation (`minimum candidate separation = 0.3204957790796499`). However, the target responses lie far from the simulated candidate-response manifold: the primary K3 target-to-best-reference distance has median approximately `70.25`, while candidate-reference separations are sub-unit after the same reference-only standardization. The primary decoder predicts `BACD` for all 24 target histories.

This indicates severe local-response model mismatch: the K3 endpoint approximation is useful under direct parameter-space distance but is not accurate enough locally for its gradients/susceptibilities to approximate those of the true endpoint. First-order differential response amplifies the endpoint approximation error.

There is also a structural limitation: `S_AB = S_BA` by construction. The tested first-order Gram response is symmetric in the challenged stage pair, whereas chronology is an oriented/noncommutative property. The probe is therefore poorly matched to the mechanism being inferred.

## Interpretation

This pilot **falsifies the tested first-order susceptibility decoder**. It is not evidence for a breakthrough active-observability result and should not be rescued by post-hoc feature additions.

It does not falsify active observability in general. The next methodology test, if pursued, must use an explicitly oriented observable fixed before output. The chosen next hypothesis is a finite future-commutator probe

`C_AB(s) = T_B(T_A(s)) - T_A(T_B(s))`,

whose sign reverses exactly under A/B exchange and whose small-step leading term is the local Lie bracket. This redesign remains restricted to the already-spent pilot codebook until it passes its own algebraic and empirical gates.
