# Commutator Decoder

This document specifies the next ChronoTrace research gate after the negative Phase-0 and Phase-0b results.

## Research question

Given:

- a known base checkpoint `theta_0`,
- candidate training stages and their losses/data,
- and a finished endpoint `theta_*`,

can we infer the **order in which the candidate stages were applied** from the endpoint geometry, without observing the original training transcript?

The first experiment is deliberately white-box and local. It exists to establish identifiability before we add optimizer state, long horizons, language-model scale, or black-box query search.

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

Thus a chronology is represented by a signed combination of pairwise bracket vectors. The first implementation exhaustively scores all permutations for `N<=5`. If the concept survives, later work can estimate pairwise orientations in the bracket basis and project them onto the nearest transitive ranking.

## Required Phase-1 gates

Before any new Pythia experiment, a float64 nonlinear toy system must satisfy all of the following across a fixed step-size sweep:

1. `||(theta_AB-theta_BA) - eta^2 b_AB||` scales approximately as `eta^3`.
2. A smooth held-out behavioral difference between AB and BA scales approximately as `eta^2`.
3. Shared endpoint displacement from the base scales approximately as `eta`.
4. Pairwise ChronoScore has the correct sign and approaches `+/-1` as `eta` shrinks.
5. The three-stage decoder recovers all six permutations of A/B/C.
6. The unknown-step-size estimator converges to the supplied local SGD step size.

Failure of these gates blocks language-model compute. They are mechanism checks, not tunable benchmark targets.

## Next model gate

If the analytic smoke passes, the next experiment is a locally created tiny GPT-NeoX model with deterministic full-batch or fixed-batch **plain SGD** updates. AdamW is intentionally excluded from the theorem-validation stage because its momentum/variance state creates an augmented dynamical system. Optimizer-state chronology is a later extension, not something to hide inside the base claim.

Only after the tiny transformer reproduces the predicted scaling do we move to Pythia-70M. For Pythia, Hessian-vector products may initially be restricted to a declared parameter subspace (for example the final transformer block or output projection) to keep the white-box calculation tractable. Any projection must be fixed before endpoint results are read.

## Relation to earlier ChronoTrace phases

- **Phase-0 v1:** AB/BA was perfectly classifiable, but capability-only classification was also perfect. Rejected as recency/forgetting.
- **Phase-0b:** common shuffled A+B terminal training failed to achieve the capability-equivalence gate; at the longest tested washout the contextual Order-Witness reached chance before capability differences disappeared.
- **Phase-0c/BJW:** implemented and smoke-tested as a stronger equalization operator, but archived before Pythia compute because post-hoc equalization may erase the very second-order history residual of interest.
- **Commutator Decoder:** constructs first-order-equivalent histories by design and tries to recover chronology from the predicted second-order residual itself.

## Novelty boundary

The mechanism is not claimed as new: contemporary sequential-learning work already derives and uses Lie-bracket/noncommutative terms to predict forward order effects. Other work studies known example-order fingerprints, data provenance, and model lineage.

The provisional ChronoTrace contribution is the **inverse endpoint problem**:

> infer an unknown candidate macro-stage chronology from a finished model by decoding the signed commutator residual after removing the permutation-independent endpoint terms.

Before publication, this claim must be re-audited against new and existing literature. Do not write "first ever." Prefer: "After targeted literature search, we found close forward-order and provenance work but no direct method centered on this inverse endpoint reconstruction task."

## Longer-term questions

If local chronology is identifiable, the important hard problems become:

- how the bracket decoder degrades with multiple updates per stage;
- how to reconstruct chronology with unknown stage intensity and learning-rate schedules;
- how to lift the state to optimizers such as Adam `(theta, m, v)`;
- whether projected parameter subspaces preserve enough chronology information;
- whether distillation, merging, quantization, unlearning, or later common training erase the geometric trace;
- how to synthesize black-box **Order Witnesses** whose output gradients align with the relevant history bracket;
- and what partial-order information remains identifiable when exact permutation recovery is impossible.
