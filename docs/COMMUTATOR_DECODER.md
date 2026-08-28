# Commutator Decoder

This document specifies the ChronoTrace inverse-geometry research direction that replaced the old classifier/washout approach.

## Research question

Given:

- a known base checkpoint `theta_0`,
- candidate training stages and their losses/data,
- and a finished endpoint `theta_*`,

can we infer the **order in which the candidate stages were applied** from the endpoint geometry, without observing the original training transcript?

The first experiments are deliberately white-box. They establish identifiability before optimizer-state persistence, stochastic training, approximate candidate stages, language-model scale, or black-box query search are introduced.

## Two-stage local expansion

Let one training step on stage `D` be

`U_D(theta) = theta - eta g_D(theta)`,

with `g_D = grad L_D` and Hessian `H_D` evaluated at the common base point unless noted otherwise.

For A then B,

`theta_AB = theta_0 - eta(g_A + g_B) + eta^2 H_B g_A + O(eta^3)`.

For B then A,

`theta_BA = theta_0 - eta(g_A + g_B) + eta^2 H_A g_B + O(eta^3)`.

Define the directed second-order terms

`q_AB = H_B g_A`,

`q_BA = H_A g_B`,

and the antisymmetric bracket

`b_AB = q_AB - q_BA = H_B g_A - H_A g_B`.

Then

`theta_AB - theta_BA = eta^2 b_AB + O(eta^3)`.

The important experimental feature is that both histories share the same entire first-order displacement `-eta(g_A + g_B)`. We therefore do not need a third washout stage to manufacture first-order equivalence.

## Order-independent midpoint

Define

`theta_mid = theta_0 - eta(g_A + g_B) + 0.5 eta^2(q_AB + q_BA)`.

Then

`theta_AB - theta_mid = +0.5 eta^2 b_AB + O(eta^3)`,

`theta_BA - theta_mid = -0.5 eta^2 b_AB + O(eta^3)`.

This gives the normalized endpoint score

`ChronoScore(theta_*) = 2 <theta_* - theta_mid, b_AB> / (eta^2 ||b_AB||^2)`.

In the local limit the score approaches `+1` for A->B and `-1` for B->A. A near-zero bracket norm means the pair is locally non-identifiable under this decoder and must be reported as such rather than forced into a classification.

## Unknown step size

For a shared unknown local SGD step size, first-order displacement yields the estimator

`eta_hat = - <theta_* - theta_0, sum_i g_i> / ||sum_i g_i||^2`.

Its order-dependent bias is higher order. The first benchmark tests this estimator separately rather than assuming the learning rate must always be supplied to the forensic auditor.

## N-stage chronology

For a permutation `pi = (pi_1, ..., pi_N)` with one local step per stage,

`theta_pi = theta_0 - eta sum_i g_i + eta^2 sum_{a<b} H_{pi_b} g_{pi_a} + O(eta^3)`.

For every unordered pair `{i,j}`, define

`b_ij = H_j g_i - H_i g_j`.

Define the permutation-independent symmetric reference

`theta_sym = theta_0 - eta sum_i g_i + 0.5 eta^2 sum_{i<j}(H_j g_i + H_i g_j)`.

The residual becomes

`theta_pi - theta_sym = 0.5 eta^2 sum_{i<j} s_pi(i,j) b_ij + O(eta^3)`,

where `s_pi(i,j)=+1` when `i` precedes `j` and `-1` otherwise.

Thus a chronology is represented by a signed combination of pairwise bracket vectors. The first implementation exhaustively scores all permutations for small `N`. A scalable decoder will need to estimate pairwise orientations and project them onto the nearest transitive ranking rather than enumerate `N!` candidates.

## Controlled positive results

The local mechanism gate and the causal-transformer gate both passed. On the 1,032-parameter causal transformer, the measured scaling exponents were approximately:

- commutator remainder: `3.0029`;
- held-out behavior difference: `1.9897`;
- shared displacement: `0.9988`.

All six A/B/C permutations were recovered across the fixed local step-size sweep.

The later finite macro-stage gate also passed. Treating a complete multi-update stage as an operator extended perfect three-stage recovery through 64 updates/stage even though the one-step HVP decoder lost perfect recovery at two updates. Exact recorded values are in `docs/results/commutator_macro_gate.md`.

## Finite stage operators

A realistic stage is a training map

`F_D(theta) = theta + Delta_D(theta)`.

For near-identity finite stages,

`F_B(F_A(theta_0)) - F_A(F_B(theta_0))`

is governed at leading interaction order by

`J Delta_B Delta_A - J Delta_A Delta_B`.

The repository estimates those terms with centered finite differences of ordinary stage runs. This avoids Hessian materialization and double-backward requirements. The full formulation and fixed stress test live in `docs/MACRO_OPERATOR_DECODER.md`.

## Replay complexity

If the base checkpoint and exact stage procedures are known, a trivial forensic baseline is exhaustive replay of every candidate chronology. ChronoTrace is only interesting if it avoids that factorial cost.

The pairwise geometry needs `O(N^2)` stage probes to characterize candidate interactions. Exact small-N enumeration is currently used only to validate the geometry against ground truth. A later scalable ranking decoder is required before claiming end-to-end polynomial-time chronology reconstruction.

## Relation to earlier ChronoTrace phases

- **Phase-0 v1:** AB/BA was perfectly classifiable, but capability-only classification was also perfect. Rejected as recency/forgetting.
- **Phase-0b:** common shuffled A+B terminal training failed to achieve the capability-equivalence gate; at the longest tested washout the contextual Order-Witness reached chance before capability differences disappeared.
- **Phase-0c/BJW:** implemented and smoke-tested as a stronger equalization operator, but archived before Pythia compute because post-hoc equalization may erase the very second-order history residual of interest.
- **Commutator Decoder:** constructs first-order-equivalent histories by design and recovers chronology from the predicted antisymmetric residual.
- **Macro-Operator Decoder:** lifts the same idea from one gradient update to complete deterministic multi-update training stages.

## Novelty boundary

The mechanism is not claimed as new. Existing optimization work uses noncommutativity/Lie-bracket corrections to understand, reverse, or choose training order. Other work studies known example-order fingerprints, data provenance, membership, and model lineage.

The provisional ChronoTrace contribution is the **inverse endpoint problem**:

> infer an unknown candidate macro-stage chronology from a finished model using pairwise noncommutative stage interactions, without exhaustively replaying every candidate full history.

Before publication, this claim must be re-audited against new and existing literature. Do not write "first ever." Prefer: "After targeted literature search, we found close forward-order and provenance work but no direct method centered on this inverse endpoint reconstruction task."

## Next gates

The next work should isolate one unresolved dimension at a time. Before adding optimizer-state persistence or stochasticity, the candidate method should be made more practical by replacing epsilon-based differential interaction estimates with exact finite pairwise stage interactions where possible. Then the first large-model experiment should isolate **model scale** using deterministic stages and a simple optimizer.

Longer-term hard problems include:

- stage procedures that are only approximately known;
- stochastic data ordering and dropout;
- persistent Adam/AdamW state across macro stages;
- unknown stage intensity and schedules;
- low-dimensional parameter projections;
- distillation, merging, quantization, unlearning, or later common training;
- black-box Order Witnesses aligned with the relevant history interactions;
- and partial-order recovery when exact permutation signatures are not identifiable.
