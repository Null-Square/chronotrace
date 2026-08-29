# Training-History Theory

Status: working theory document. This file should evolve by explicit additions/corrections, not by silently deleting failed mechanisms.

## 1. The inverse problem

Let a training stage `D` be a complete update operator. In the simplest reset-SGD setting it acts on weights:

`F_D : theta -> theta'`.

For a candidate chronology `pi = (pi_1, ..., pi_N)`, the endpoint is

`E_pi = F_{pi_N} o ... o F_{pi_1}(theta_0)`.

ChronoTrace asks the inverse question:

> Given an observed endpoint and candidate stage procedures, what information about `pi` is identifiable?

The key fact is noncommutativity:

`F_B(F_A(theta)) != F_A(F_B(theta))`

in general.

Forward sequential-learning work asks how this changes performance or which order should be chosen. ChronoTrace asks whether the noncommutative residue can be inverted after training.

## 2. Real training is an extended-state dynamical system

A realistic optimizer is not a map over weights alone. A more faithful state is

`z_t = (theta_t, m_t, v_t, q_t, rng_t, data_t, scaler_t, ...)`,

where, depending on the training stack:

- `theta`: model weights;
- `m,v`: optimizer moment state;
- `q`: scheduler / global-step state;
- `rng`: dropout, sampling, augmentation and distributed randomness;
- `data`: data-stream position / shuffle state;
- `scaler`: mixed-precision loss-scale state.

A single update is therefore `U_t(z_t, batch_t)`, and a macro stage is a composition of many such updates.

The checkpoint normally exposes only a projection `P(z)=theta`. Thus the realistic inverse problem is

`theta_final = P(F_{pi_N} o ... o F_{pi_1}(z_0))`.

This gives two distinct questions:

1. **state chronology:** is the full extended state path identifiable?
2. **weight chronology:** after projecting away optimizer/RNG/scheduler state, is enough history still encoded in weights?

The current ChronoTrace mechanism experiments deliberately reset optimizer state and use deterministic SGD so that any recovered chronology comes from the geometry of the weights, not hidden Adam memory.

### Representative Pythia pretraining is much richer than the current bridge

The official Pythia-14M training config uses Adam with `beta1=0.9`, `beta2=0.95`, `eps=1e-8`, learning rate `1e-3`, weight decay `0.1`, gradient clipping `1.0`, cosine decay, `1%` warmup, FP16, and `143000` training iterations with a roughly 2M-token global batch. The public Pythia suite saw about 300B training tokens.

Sources:

- EleutherAI Pythia repository: https://github.com/EleutherAI/pythia
- Pythia-14M config: https://github.com/EleutherAI/pythia/blob/main/models/14M/pythia-14m.yml

By contrast, the first ChronoTrace scale bridge uses FP32, deterministic full batches, plain SGD, no momentum/weight decay, learning rate `1e-4`, and only 16 updates per stage. This is intentional mechanism isolation, but it is not yet a realistic pretraining pipeline.

For Adam, persistent moments create an additional chronology channel. With Pythia's `beta1=0.9`, the contribution of an old gradient to the first moment halves after about 6.6 optimizer steps. With `beta2=0.95`, the corresponding second-moment half-life is about 13.5 steps. If moments persist across a stage boundary, the early updates of the next stage directly depend on the previous stage even at identical weights. ChronoTrace must eventually separate this **optimizer-memory trace** from the **weight-geometry trace** studied now.

## 3. Local mechanism: Lie brackets

For one small gradient update per stage,

`F_A(theta) = theta - eta g_A(theta)`

and similarly for B. Expanding at `theta_0`,

`theta_AB = theta_0 - eta(g_A + g_B) + eta^2 H_B g_A + O(eta^3)`

`theta_BA = theta_0 - eta(g_A + g_B) + eta^2 H_A g_B + O(eta^3)`.

Therefore

`theta_AB - theta_BA = eta^2 (H_B g_A - H_A g_B) + O(eta^3)`.

The first-order learning displacement is order-independent. Chronology begins in an antisymmetric second-order term.

This is consistent with standard Baker-Campbell-Hausdorff / splitting-method theory, where compositions of noncommuting flows contain pair commutators and then nested higher-order commutators. It is also adjacent to Sweeney (2026), who uses Lie-bracket geometry in the **forward** direction to predict beneficial sequential-learning order:

- https://arxiv.org/abs/2606.24993

The Lie bracket itself is therefore not the novelty claim.

## 4. Exact finite interaction hierarchy

The local BCH view suggests a stronger nonperturbative decomposition for complete finite training stages.

Fix a total order `pi` over a stage set `U`. For a subset `S subseteq U`, let `E_pi(S)` be the endpoint obtained by executing only the stages in S, preserving their relative order in pi. Define `E_pi(empty)=theta_0`.

Define the interaction associated with S by Boolean-lattice Mobius inversion:

`Phi_pi(S) = sum_{T subseteq S} (-1)^(|S|-|T|) E_pi(T)`.

Then exactly,

`E_pi(U) = sum_{S subseteq U} Phi_pi(S)`.

This is not an approximation. The approximation appears only when the expansion is truncated by interaction order.

### Order 1: singleton stage effects

`Phi({A}) = F_A(theta_0) - theta_0 = Delta_A`.

### Order 2: directed finite pair interactions

If A occurs before B,

`Phi(A,B) = F_B(F_A(theta_0)) - F_A(theta_0) - F_B(theta_0) + theta_0`.

This is exactly the current finite-pair interaction `I_{B<-A}`.

### Order 3: prefix-conditioned / triple interaction

For order `A,B,C`,

`Phi(A,B,C)`

is the exact residual left after subtracting the base checkpoint, all singleton effects, and all three directed pair interactions selected by that chronology.

Thus the current Pythia failure is not an unspecified modeling error. It is direct evidence that the omitted third-order term is comparable to the separation among pairwise chronology signatures.

### Probe complexity

If all ordered interactions through fixed order K are measured and lower-order prefixes are cached, the number of stage-map extensions scales as

`sum_{r=1..K} P(N,r) = O(N^K)`

for fixed K, where `P(N,r)=N!/(N-r)!`.

Examples:

- K=1: `O(N)` singleton probes;
- K=2: `N + N(N-1) = N^2` stage executions;
- K=3: `N + N(N-1) + N(N-1)(N-2) = O(N^3)`;
- exhaustive full-history validation: `N * N!` stage executions if every complete chronology is replayed independently.

This motivates **training-history interaction order**: the minimum K for which an order-K interaction model can reliably identify the chronology.

## 5. The key missing object: prefix-conditioned commutators

Define the exact finite commutator of stages B and C at an arbitrary model state theta:

`C_BC(theta) = F_C(F_B(theta)) - F_B(F_C(theta))`.

Now compare two histories that share first stage A:

`ABC = F_C(F_B(F_A(theta_0)))`

`ACB = F_B(F_C(F_A(theta_0)))`.

Their difference is exactly

`E_ABC - E_ACB = C_BC(F_A(theta_0))`.

This identity is central.

The current finite-pair decoder measures `C_BC(theta_0)`, not `C_BC(F_A(theta_0))`.

Define the **prefix-conditioned commutator drift**

`T_{A;BC} = C_BC(F_A(theta_0)) - C_BC(theta_0)`.

`T_{A;BC}` is a third-order interaction: it measures how learning A changes the later B/C order effect.

### Why this matches the 14M failure pattern

The portable Pythia-14M decoder produced:

- `ABC -> ACB`
- `ACB -> ACB`
- `BAC -> BAC`
- `BCA -> BAC`
- `CAB -> CAB`
- `CBA -> CAB`.

Every error preserves stage 1 and swaps only stages 2 and 3.

For a pair such as `ABC` versus `ACB`, the two histories have the same A/B and A/C precedence edges. They differ only in B/C precedence. Therefore a decoder that gets the earliest stage right but substitutes the base-state `C_BC(theta_0)` for the conditioned `C_BC(F_A(theta_0))` has exactly the observed failure mode.

This is currently a mechanistic hypothesis generated by one locked instance. It must be tested directly before being treated as an explanation.

## 6. Static tournament versus state-dependent tournament

A base pairwise decoder treats the candidate stages as a static directed tournament: each pair has one order vector measured at `theta_0`.

The more realistic object is a **state-dependent chronology tournament**:

`C_ij(theta)`.

As training proceeds, theta changes, so the preferred / identifiable orientation of later pairs can rotate, shrink, grow, or reverse.

Chronology is therefore a path through a field of pair interactions, not a ranking read from one static matrix.

A useful drift quantity is

`rho_{A;BC} = ||C_BC(F_A(theta_0)) - C_BC(theta_0)|| / ||C_BC(theta_0)||`.

If the commutator field is locally Lipschitz with constant `L_BC`, then

`||T_{A;BC}|| <= L_BC ||F_A(theta_0)-theta_0||`.

This predicts a principled transition: base-anchored pairwise reconstruction should degrade as prefix displacement grows relative to pair-signature separation and commutator stability.

## 7. Decoder correctness depends on residual direction, not only residual norm

Let the order-K prediction for the true chronology pi be

`theta_hat_pi = reference + s_pi`

and the real endpoint be

`theta_pi = theta_hat_pi + r_pi`,

where `r_pi` is the omitted higher-order residual.

For a competing chronology sigma, define

`d_{pi,sigma} = s_pi - s_sigma`.

Nearest-signature decoding prefers pi over sigma exactly when

`||r_pi||^2 < ||r_pi + d_{pi,sigma}||^2`,

which simplifies to

`2 <r_pi, d_{pi,sigma}> + ||d_{pi,sigma}||^2 > 0`.

Equivalently define the **directional contamination ratio**

`chi_{pi,sigma} = -2 <r_pi, d_{pi,sigma}> / ||d_{pi,sigma}||^2`.

The exact pairwise decision remains correct against sigma iff

`chi_{pi,sigma} < 1`.

This is stronger than the current norm-only sufficient bound

`2 ||r_pi|| / ||d_{pi,sigma}|| < 1`,

which follows from Cauchy-Schwarz but can be very conservative.

This explains an important earlier observation: on the tiny transformer the norm bound exceeded 1 at long stages while finite-pair decoding still achieved 6/6. The omitted residual was large but not sufficiently aligned with the nearest wrong chronology direction. On Pythia-14M, the residual is apparently aligned with specific tail-swap directions strongly enough to cross their decision boundaries.

Future result files should report both:

- norm-based residual/separation ratio;
- exact directional contamination against every competing chronology.

## 8. What the current numbers say

Portable Pythia-14M, 16 updates/stage:

- base parameter norm: `1234.92688`
- singleton displacement norms:
  - A `0.2062331`
  - B `0.1697814`
  - C `0.1847850`
- finite-pair minimum signature separation: `0.2056145`
- maximum triple+ remainder norm: `0.2054421`
- maximum norm certificate ratio: `1.99832`
- full-order recovery: `3/6`
- descriptive pairwise precedence accuracy: `15/18 = 83.3%`
- descriptive first-stage recovery: `6/6`
- descriptive mean Kendall tau: `2/3`.

The relative parameter movement is tiny (`~1e-4` of the base norm), yet the triple+ remainder is approximately the same absolute scale as the minimum chronology-signature separation. This warns against using global relative weight displacement as a proxy for interaction locality in a high-dimensional model.

The relevant dimensionless quantities should compare **interaction order to chronology separation**, not displacement to total parameter norm.

## 9. Training-history trace channels

The eventual theory should distinguish at least four mechanisms:

### A. Geometric trace

Noncommutativity of weight-update fields / finite stage maps. This is the channel isolated by the current reset-SGD experiments.

### B. Optimizer-memory trace

Persistent momentum / Adam first- and second-moment state makes the next stage depend explicitly on previous gradients, even at identical weights.

### C. Schedule/time trace

Learning-rate schedules, weight-decay schedules, curriculum position, and global step make the same data stage a different operator depending on when it occurs.

### D. Stochastic trace

Data shuffling, sampling, dropout, mixed-precision rounding, distributed reductions, and other randomness create a distribution over stage operators rather than one deterministic map.

A scientifically clean program should add these channels one at a time. A positive result under persistent Adam state should not be mislabeled as a pure weight-geometry result.

## 10. Theory-driven next experiments

Do not move directly to 31M.

### T1 — Diagnostic replay of the same frozen 14M instance

Purpose: test the mechanism, not improve accuracy.

Record the vectors necessary to compute:

- exact triple Mobius interaction for each A/B/C order;
- `C_BC(theta_0)` and `C_BC(F_A(theta_0))`, plus corresponding conditional commutators for every first-stage choice;
- prefix-conditioned commutator drift norms and angles;
- exact `chi_{pi,sigma}` directional contamination for every true/competing order;
- per-edge precedence margins.

Prediction: the three failed histories should cross the decision boundary primarily on the single tail-pair edge, and the corresponding conditioned commutator should differ materially from the base commutator.

This replay uses the same frozen instance and is **diagnostic only**; it cannot establish generalization.

### T2 — Independent 14M interaction-order map

Before generating results, freeze several independent world/codebook seeds and a stage-length sweep chosen to map a transition rather than maximize accuracy.

Suggested stage lengths: `{1,2,4,8,16,32}` with the already frozen chronology-blind SGD rate unless a separate stability-only rule requires otherwise.

Report separately:

- exact full-order recovery;
- first-stage recovery;
- pairwise precedence accuracy;
- Kendall tau;
- minimum pair-signature separation;
- prefix-conditioned commutator drift;
- norm residual ratio;
- directional contamination ratio.

Prediction: lower interaction order should be sufficient in the small-displacement regime, while prefix-conditioned / third-order effects should become necessary as the stage operator moves farther from the base state.

### T3 — Four-stage test of polynomial interaction order

Three stages are insufficient to demonstrate an efficient third-order decoder, because measuring all ordered triples already touches all `3!` complete histories.

The first meaningful test of an order-3 reconstruction basis should use at least four stages:

- full histories: `4! = 24` for held-out ground truth;
- interactions through order 2: `4 + 12 = 16` stage-map extensions;
- through order 3: `4 + 12 + 24 = 40` stage-map extensions;
- naive full replay: `4 * 24 = 96` stage executions.

The point is not the modest 4-stage savings. It is to test whether a fixed interaction order can reconstruct a factorial chronology space and to measure where the hierarchy breaks.

### T4 — Only then add realistic optimizer state

If the geometric hierarchy generalizes, introduce persistent Adam state as a separate experiment. The training operator should then act on the extended state `z=(theta,m,v,q,...)`, while evaluation can compare:

- full-state access;
- weights-only access;
- optimizer-reset versus optimizer-persistent stage boundaries.

This will reveal whether realistic training history is encoded mainly through weights, optimizer memory, or both.

## 11. Current novelty boundary

Known / adjacent:

- training order affects optimization and stability;
- curriculum / task ordering can be optimized;
- local Lie brackets predict forward order effects;
- model lineage and example-order fingerprints can be detected in other provenance settings.

Representative nearby work:

- Sweeney, 2026, *The Geometry of Sequential Learning: Lie-Bracket Prediction of Transfer Order*: https://arxiv.org/abs/2606.24993
- Dherin et al., 2025, *Training in reverse: How iteration order influences convergence and stability in deep learning*: https://research.google/pubs/training-in-reverse-how-iteration-order-influences-convergence-and-stability-in-deep-learning/
- Li & Hiratani, 2025, *Optimal Task Order for Continual Learning of Multiple Tasks*: https://proceedings.mlr.press/v267/li25z.html

The candidate ChronoTrace contribution is narrower:

> **inverse training-history reconstruction from a finished endpoint, together with an interaction-order theory that characterizes when singleton, pairwise, prefix-conditioned, and higher-order stage effects are sufficient for chronology identifiability.**

This remains a provisional novelty statement. A publication-stage literature audit is still required.
