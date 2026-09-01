# Training-History Theory

Status: canonical working theory document as of 2026-08-29. Earlier hypotheses are retained through the append-only research journal; corrections here reflect the current best interpretation rather than silently rewriting experimental history.

## 1. The inverse problem

Let a candidate training stage `D` be a complete update operator. In the simplest reset-SGD setting it acts on weights:

`F_D : theta -> theta'`.

For a candidate chronology `pi = (pi_1, ..., pi_N)`, the endpoint is

`E_pi = F_{pi_N} o ... o F_{pi_1}(theta_0)`.

ChronoTrace asks the inverse question:

> Given an observed finished model, candidate training stages, and an explicitly stated access/observation regime, what information about the unknown chronology `pi` is identifiable?

The broad fact that sequential updates need not commute is not the contribution:

`F_B(F_A(theta)) != F_A(F_B(theta))`

in general.

Forward curriculum/data-order work asks how order changes learning. ChronoTrace asks whether the order-sensitive residue can be inverted into an unknown total order or partial order after training.

The current mechanism experiments operate in the **white-box simulator regime**: the base checkpoint, candidate stages, training rule, and replay procedure are known; final weights are observable; chronology is unknown. This is not yet a black-box provenance claim. See `docs/THEORY_ACCESS_AND_TRACE_CHANNELS.md`.

## 2. Real training is an extended-state dynamical system

A realistic optimizer is not a map over weights alone. A more faithful trainer state is

`z_t = (theta_t, m_t, v_t, q_t, rng_t, data_t, scaler_t, ...)`,

where, depending on the stack:

- `theta`: model weights;
- `m,v`: momentum / Adam moment state;
- `q`: scheduler and global-step state;
- `rng`: dropout, sampling, augmentation, distributed randomness;
- `data`: stream position / shuffle state;
- `scaler`: mixed-precision loss-scale state.

A macro stage is a composition of many updates on this extended state. A released checkpoint normally exposes only a projection such as

`P(z)=theta`.

The realistic endpoint is therefore

`theta_final = P(F_{pi_N} o ... o F_{pi_1}(z_0))`.

This immediately separates three questions:

1. **full-state chronology:** are histories distinct in complete trainer state?
2. **weight chronology:** does enough history survive projection into weights?
3. **behavioral chronology:** does enough survive a further projection from weights into query responses?

Information cannot increase under these deterministic projections. Schematically,

`I(history; behavior) <= I(history; weights) <= I(history; full trainer state)`.

The present reset/plain-SGD experiments deliberately remove optimizer/schedule/stochastic carry-over so that recovered chronology is attributable to weight geometry rather than hidden Adam state or an explicit training clock.

### Representative Pythia pretraining is much richer than the bridge

The public Pythia-14M recipe uses Adam (`beta1=0.9`, `beta2=0.95`, `eps=1e-8`), LR `1e-3`, weight decay `0.1`, clipping `1.0`, cosine decay, `1%` warmup, FP16, and `143000` training iterations with a roughly 2M-token global batch. The Pythia suite saw roughly 300B tokens.

Sources:

- https://github.com/EleutherAI/pythia
- https://github.com/EleutherAI/pythia/blob/main/models/14M/pythia-14m.yml

The current bridge instead uses FP32, deterministic full batches, constant-rate plain SGD, no momentum/weight decay, and short controlled stages. This is intentional mechanism isolation, not a realistic pretraining simulation.

## 3. Distinct chronology channels have different leading orders

A central correction to the early theory is that “training chronology” is not one mechanism. Real training mixes several channels with different leading-order behavior.

### 3.1 Geometric noncommutativity — second order under reset constant-step SGD

For one small gradient update per stage,

`F_A(theta) = theta - eta g_A(theta)`

and similarly for B. Expanding at `theta_0`,

`theta_AB = theta_0 - eta(g_A + g_B) + eta^2 H_B g_A + O(eta^3)`

`theta_BA = theta_0 - eta(g_A + g_B) + eta^2 H_A g_B + O(eta^3)`.

Therefore

`theta_AB - theta_BA = eta^2(H_B g_A - H_A g_B) + O(eta^3)`.

Under these controls the first-order displacement is order-independent. Chronology begins at `O(eta^2)` through an antisymmetric Lie-bracket-like term.

This is the channel isolated by the current mechanism program.

### 3.2 Training-clock / schedule asymmetry — chronology can appear at first order

Suppose position 1 and position 2 use different learning rates `eta_1` and `eta_2`. Ignoring state-dependence beyond first order,

`theta_AB = theta_0 - eta_1 g_A - eta_2 g_B + O(eta^2)`

`theta_BA = theta_0 - eta_1 g_B - eta_2 g_A + O(eta^2)`.

Thus

`theta_AB - theta_BA = (eta_2-eta_1)(g_A-g_B) + O(eta^2)`.

Warmup, cosine decay, stage-position regularization, or any global-step-dependent rule can therefore create an `O(eta)` time-stamp-like chronology signal.

### 3.3 Optimizer memory — persistent momentum can also create first-order chronology

For simple momentum,

`v_t = mu v_{t-1} + g_t`

`theta_t = theta_{t-1} - eta v_t`,

with `v_0=0`, a first-order expansion gives

`theta_AB - theta_BA = -eta mu(g_A-g_B) + O(eta^2)`.

So optimizer memory can write chronology into the weights themselves even if optimizer state is discarded before forensic inspection.

Adam is more nonlinear because it carries first and second moments, normalization and bias corrections, but the same qualitative issue holds. Consequently, a positive realistic-Adam chronology result must not be mislabeled as pure geometric path memory.

### 3.4 Stochastic-stream state

Batch order, shuffling, dropout, distributed reduction order, mixed-precision state, and other randomness can either:

- be independently reset/matched across histories, acting mainly as noise; or
- be carried through stage boundaries, becoming a genuine chronology channel.

These cases require separate interventions.

### 3.5 Mechanism taxonomy

A useful working hierarchy is:

| Mechanism | Representative source | Simplified leading chronology order |
|---|---|---|
| training clock | LR/global-step differs by position | `O(eta)` |
| optimizer memory | persistent momentum/Adam | `O(eta)` in simple momentum example |
| geometric noncommutativity | constant-step reset SGD | `O(eta^2)` |
| prefix-conditioned interaction | earlier stage changes later pair geometry | degree 3 |
| deeper path interaction | nested state-conditioned effects | degree 4+ |

The order here is mechanistic/asymptotic, not a guaranteed ranking of empirical magnitude.

## 4. Local mechanism: Lie brackets

The small-step relation

`theta_AB - theta_BA = eta^2(H_B g_A - H_A g_B) + O(eta^3)`

is consistent with Baker-Campbell-Hausdorff, splitting-method, Magnus and chronological-calculus theory, where ordered compositions of noncommuting flows contain commutators and nested commutators.

The mathematics itself is classical, and recent sequential-learning work also uses Lie-bracket geometry in the forward direction. The Lie bracket is therefore not the ChronoTrace novelty claim.

What matters here is the **inverse use** of order-sensitive interactions to distinguish an unknown chronology.

## 5. Exact finite interaction hierarchy

The local expansion is insufficient for long finite stages. ChronoTrace therefore uses a nonperturbative interaction decomposition.

Fix a total order `pi` over a stage set `U`. For subset `S subseteq U`, let `E_pi(S)` be the endpoint obtained by executing only stages in `S` while preserving their relative order in `pi`, with `E_pi(empty)=theta_0`.

Define by Boolean-lattice Möbius inversion

`Phi_pi(S) = sum_{T subseteq S} (-1)^(|S|-|T|) E_pi(T)`.

Then exactly,

`E_pi(U) = sum_{S subseteq U} Phi_pi(S)`.

No approximation occurs until the interaction expansion is truncated by degree.

### Degree 1 — singleton effects

`Phi({A}) = F_A(theta_0)-theta_0 = Delta_A`.

### Degree 2 — directed finite pair interactions

If A precedes B,

`Phi(A,B) = F_B(F_A(theta_0)) - F_A(theta_0) - F_B(theta_0) + theta_0`.

This is the finite directed pair interaction used by the current static pair decoder.

### Degree 3 — exact three-stage interaction

For exactly three stages A/B/C, the residual after subtracting the base, all singleton effects, and all chronology-selected pair effects is the **exact degree-3 interaction**. It is not “third and higher”; there are no higher stage-subset degrees in a three-stage system.

This correction matters because the portable Pythia-14M failure can be analyzed exactly rather than attributed to an unspecified approximation error.

## 6. Training-History Interaction Order

This hierarchy motivates the working quantity

`K*(pi, epsilon)`

= the smallest interaction degree K whose truncated representation distinguishes the relevant chronology to the required robustness/error tolerance.

Possible regimes:

- `K=2`: static pair interactions are sufficient;
- `K=3`: later pair order depends materially on the earlier prefix;
- large K: chronology is strongly nonlocal and low-order inversion may be impractical;
- unidentifiable after observation projection: no retained degree separates the candidate histories through the permitted interface.

The current paper direction is stronger if it characterizes this boundary rather than simply reporting one chronology-classification accuracy.

## 7. Prefix-conditioned interactions

Define the exact finite commutator of B and C at arbitrary state `theta`:

`C_BC(theta) = F_C(F_B(theta)) - F_B(F_C(theta))`.

For two histories sharing first stage A,

`ABC = F_C(F_B(F_A(theta_0)))`

`ACB = F_B(F_C(F_A(theta_0)))`,

so exactly

`E_ABC - E_ACB = C_BC(F_A(theta_0))`.

A base-anchored static pair decoder instead uses `C_BC(theta_0)`.

Define the prefix-conditioned commutator drift

`T_{A;BC} = C_BC(F_A(theta_0)) - C_BC(theta_0)`.

This drift is an order-3 object: earlier learning changes the geometry of later order.

The portable Pythia-14M result first suggested this because every 3/6 error preserved the first stage and swapped only positions 2 and 3. T1 then directly measured that the relevant conditioned tail commutators were dramatically smaller and nearly orthogonal to their base versions.

That observation is mechanistically useful, but the sharper decomposition below explains the asymmetric one-tail-survives pattern.

## 8. Exact shared-prefix midpoint/separation decomposition

Consider two histories sharing a prefix `p` and differing only in tail order `ij` versus `ji`.

Let

`P_pij`, `P_pji`

be the static finite-pair predictions and

`E_pij`, `E_pji`

be the actual endpoints.

Define the static pair direction

`d0 = P_pij - P_pji`,

the actual conditioned separation

`dc = E_pij - E_pji`,

and the common midpoint shift

`b = (E_pij+E_pji)/2 - (P_pij+P_pji)/2`.

Normalize their projections onto the static pair axis:

`alignment = <dc,d0> / ||d0||^2`

`midpoint_bias = 2<b,d0> / ||d0||^2`.

Then exactly:

- forward history `pij` is closer to its own static-pair prediction than `pji` iff
  `alignment + midpoint_bias > 0`;
- reverse history `pji` is closer to its own prediction iff
  `alignment - midpoint_bias > 0`.

Therefore both tail orders are simultaneously recoverable against each other iff

`alignment > |midpoint_bias|`.

Define

`tail_robustness = alignment - |midpoint_bias|`.

This separates two distinct ways a lower-order model can fail:

1. the conditioned order-sensitive separation can rotate/shrink/reverse;
2. a common higher-order midpoint drift can push both histories toward the same lower-order candidate.

## 9. What T1 actually showed on the frozen Pythia-14M instance

Portable Pythia-14M at 16 updates/stage reproducibly produced:

- `ABC -> ACB`
- `ACB -> ACB`
- `BAC -> BAC`
- `BCA -> BAC`
- `CAB -> CAB`
- `CBA -> CAB`.

Descriptively:

- exact order: `3/6`;
- first stage: `6/6`;
- pairwise precedence: `15/18 = 83.3%`;
- mean Kendall tau: `2/3`.

### 9.1 Conditioned commutator collapse/rotation

For prefixes A/B/C respectively:

| Prefix | base norm | conditioned norm | base/conditioned cosine | relative drift |
|---|---:|---:|---:|---:|
| A | 0.205614 | 0.037957 | 0.103456 | 0.997954 |
| B | 0.259722 | 0.046565 | 0.030969 | 1.010468 |
| C | 0.244604 | 0.062985 | 0.052139 | 1.019547 |

Thus the tail interaction measured at the conditioned state is much smaller and almost orthogonal to its base-state version.

### 9.2 Midpoint bias dominates the remaining aligned signal

The exact midpoint replay measured:

| Prefix | Tail | alignment | midpoint bias | forward score | reverse score |
|---|---|---:|---:|---:|---:|
| A | BC | 0.019095 | -0.137284 | -0.118189 | +0.156379 |
| B | AC | 0.005551 | +0.177508 | +0.183059 | -0.171957 |
| C | AB | 0.013423 | +0.245139 | +0.258562 | -0.231716 |

The actual conditioned separation retains a tiny **positive** projection on the original pair direction; it does not simply reverse. But the common third-order midpoint bias is roughly 7.2x, 32.0x and 18.3x larger than that aligned signal.

Its sign selects exactly which member of each prefix pair survives:

- A prefix: `ACB` survives, `ABC` loses;
- B prefix: `BAC` survives, `BCA` loses;
- C prefix: `CAB` survives, `CBA` loses.

The best current description of this frozen instance is therefore:

> useful conditioned tail-order separation nearly collapses, while a much larger exact third-order midpoint drift pushes both endpoints toward one static-pair candidate.

This is an exact geometric diagnosis of one motivating instance, not independent generalization evidence.

## 10. Correction: directional contamination is diagnostic, not independent evidence

For true lower-order signature `s_pi`, competitor `s_sigma`, and omitted residual `r_pi`, let

`d = s_pi - s_sigma`.

Nearest-signature decoding prefers the true history iff

`2<r_pi,d> + ||d||^2 > 0`.

Equivalently,

`chi_(pi,sigma) = -2<r_pi,d> / ||d||^2 < 1`.

Earlier interpretation treated the fact that failed histories had `chi > 1` as strong support. This was too strong: `chi < 1` is algebraically equivalent to the two-candidate nearest-signature decision itself.

Thus `chi` is useful for **directional accounting**—which competitor direction the omitted term contaminates—but its match to the observed winner is not independent empirical validation.

The nontrivial independent questions are whether fresh instances repeatedly show structured same-prefix errors, coarse chronology surviving exact-order failure, and systematic state-conditioned interaction changes.

## 11. Norm certificates versus directional geometry

A sufficient but conservative robustness condition is

`2||r_pi|| / ||d|| < 1`.

It can fail even when decoding succeeds because a large residual may be mostly orthogonal to the competing chronology direction. This occurred in the controlled tiny transformer: the norm certificate failed at long stages while finite-pair reconstruction remained 6/6.

Consequently result files should distinguish:

- higher-order residual norm;
- chronology-signature separation;
- residual direction;
- exact or decomposed decision geometry.

Global relative parameter displacement is also a poor locality proxy in high dimensions: the 14M endpoint moved only about `1e-4` of the base parameter norm while degree-3 residuals were already comparable to chronology-signature separation.

## 12. Probe complexity is not inference complexity

If all ordered interactions through fixed degree K are measured, stage-map probe count scales as

`sum_{r=1..K} P(N,r) = O(N^K)`

for fixed K.

Examples:

- K=1: `O(N)`;
- K=2: `O(N^2)`;
- K=3: `O(N^3)`.

This is **probe complexity**, not automatically chronology-decoding complexity.

There are still `N!` total orders, and finding the best globally consistent permutation from pair/higher-order evidence can remain combinatorial. ChronoTrace must therefore report separately:

1. cost to construct the interaction representation;
2. cost to infer a consistent total/partial order from it.

A future practical decoder may use prefix beams, branch-and-bound, constrained ranking, partial-order output, or higher-order probes only on ambiguous branches. No polynomial worst-case decoding claim is currently justified.

Three stages are especially weak for efficiency claims: measuring conditioned degree-3 interactions nearly touches complete histories. Meaningful low-order-versus-factorial scaling should use `N >= 4`.

## 13. Behavioral observability and Order Witnesses

Weight-space identifiability does not imply black-box identifiability.

For a scalar behavioral statistic `f_q(theta)` and a small chronology direction

`Delta_order = theta_pi - theta_sigma`,

locally

`f_q(theta_pi)-f_q(theta_sigma) ~= grad f_q(theta)^T Delta_order`.

This gives a mechanistic interpretation of an Order Witness:

> a query whose behavioral sensitivity has a large projection onto a chronology-sensitive parameter direction relative to noise.

A schematic witness score is

`|grad f_q^T Delta_order| / sigma_q`.

Ordinary benchmarks can be chronology-blind even when weights are identifiable because their observation gradients may lie nearly orthogonal to the chronology subspace.

This is the bridge from the current R1 white-box simulator experiments toward eventual black-box provenance.

## 14. Literature boundary

Several neighboring claims are already occupied and must be cited rather than presented as ChronoTrace novelty:

- data/task order affects optimization;
- training order can leave persistent traces;
- training can exhibit memory/non-Markovian behavior;
- optimizer state carries prior gradients;
- future training can transform the behavioral value of earlier state;
- training can be modeled as a multi-time/tomographic process.

Especially relevant 2026 work includes:

- Sevetlidis & Pavlidis, **Process-Tensor Tomography of SGD** (AISTATS 2026; arXiv:2601.16563);
- Sevetlidis & Pavlidis, **Training Memory in Deep Neural Networks** (arXiv:2601.21624);
- Xu, **Stored in Optimizer State, Valued by Later Training** (arXiv:2608.20442);
- Guo, **Delayed Optimizer-State Transport Shapes Short-Horizon Training Decisions** (arXiv:2608.24593);
- Kuditipudi et al., **Blackbox Model Provenance via Palimpsestic Membership Inference** (NeurIPS 2025).

Accordingly, **“Training-History Tomography” should not be treated as a distinctive project umbrella**.

The still-defensible candidate gap is narrower:

> **post-hoc reconstruction or partial-order identification of an unknown semantic macro-stage chronology from a finished model, together with the interaction degree and access/observation regime required for that inverse problem.**

This remains a candidate novelty boundary, not proof of firstness.

## 15. Current theory-driven experiment program

### T1 — complete: diagnose the frozen 14M failure

T1 and the midpoint replay are complete. They exactly reproduced the portable 14M basis/endpoints and established the instance-specific geometry described above.

Evidence:

- original T1 run `33243747235`;
- midpoint replay `33245010517`;
- `docs/results/pythia_14m_theory_diagnostic.md`.

T1 is explanatory evidence on the same instance that generated the hypothesis, not confirmation.

### T2 — frozen independent Pythia-14M interaction map

T2 is the current independent falsifier.

Frozen before outcomes:

- four mechanically derived tokenizer-safe codebooks;
- stage lengths `{1,2,4,8,16,32}`;
- all six A/B/C permutations;
- same portable 14M checkpoint and `1e-4` chronology-blind plain-SGD rate;
- no performance-based condition selection;
- full-order, first-stage, pairwise-precedence, Kendall, commutator and midpoint/separation metrics reported for every condition.

Key independent question:

> Does coarse chronology systematically survive after exact total-order recovery degrades, with failures concentrated in same-prefix tail ambiguity and accompanied by state-conditioned interaction collapse/midpoint dominance?

See `configs/pythia_14m_t2.lock.json` and `docs/experiments/PYTHIA_14M_T2_PROTOCOL.md`.

### T3 — contingent: four-stage interaction-order test

Only if T2 supports a repeatable hierarchy should the next mechanism experiment use at least four stages.

The purpose is to test whether a degree-3 representation can constrain a `4!` chronology space without merely replaying the complete three-stage histories that define the degree-3 terms.

T3 must report both probe cost and inference cost and should include partial-order metrics, not only exact permutation accuracy.

### T4 — later: factor realistic chronology channels

Only after the geometric hierarchy is understood should the program introduce realistic training channels factorially:

- optimizer reset vs persistent momentum/Adam;
- constant LR vs warmup/decay;
- matched/reset vs continuously carried stochastic state;
- weights-only vs behavioral observation.

The scientific question is not whether those channels contain memory—the literature already establishes that they can—but **how each channel changes inverse chronology identifiability and provenance survivability**.

## 16. Current falsifiable research claim

The strongest claim justified as a hypothesis today is:

> In controlled sequential language-model training, chronology information is represented hierarchically: low-order interactions can preserve coarse precedence while finer order requires state-conditioned higher-order interactions. The minimum required interaction degree depends on how nonlocal the training stages become and on the observation interface.

T2 is designed to falsify the first independent part of that statement.

If T2 does not reproduce structured coarse-to-fine chronology across fresh codebooks, the state-conditioned hierarchy must be revised before any model-size scale-up.

Pythia-31M remains blocked until T2 is interpreted.
