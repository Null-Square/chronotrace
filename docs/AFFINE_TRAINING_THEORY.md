# Exact Affine Theory for Sequential Training Chronology

Date: 2026-08-29

Status: theory note derived while the independent T2 Pythia-14M map was already frozen and running. It does not change T2.

Purpose: replace vague appeals to “path dependence” with an exactly solvable finite-stage model that separates ordinary recency, noncommuting curvature, common-tail decay, observation loss, and optimizer-state memory.

## 1. Quadratic stage losses give exact affine training operators

Consider a stage `i` with quadratic loss

`L_i(theta) = 1/2 (theta-a_i)^T H_i (theta-a_i)`,

where `H_i` is symmetric positive semidefinite and `a_i` is the stage optimum.

One full-batch gradient-descent step with learning rate `eta_i` is

`theta' = theta - eta_i H_i(theta-a_i)`

so

`theta' = M_i theta + (I-M_i)a_i`,

with

`M_i = I - eta_i H_i`.

If the same stage is trained for `k_i` identical full-batch steps, its **complete finite stage operator** is exactly

`F_i(theta) = A_i theta + c_i`,

where

`A_i = M_i^(k_i)`

and

`c_i = (I-A_i)a_i`.

Thus repeated training on a quadratic task is not merely locally affine; the whole macro stage is exactly affine.

This gives a solvable model for the same finite-stage object used by ChronoTrace.

## 2. Exact endpoint of an arbitrary chronology

For chronology

`pi = (pi_1, ..., pi_N)`,

the endpoint is

`E_pi = A_{pi_N} ... A_{pi_1} theta_0`

`      + sum_(r=1..N) A_{pi_N} ... A_{pi_(r+1)} c_{pi_r}`.

The empty product after the final stage is the identity.

This expression already reveals two distinct sources of order information:

1. **multiplicative order:** the matrices `A_i` are multiplied in chronology order;
2. **translation/recency order:** each stage optimum contribution `c_i` is transformed by every stage that follows it.

These mechanisms should not be conflated.

## 3. Exact two-stage order difference

For stages i and j,

`F_j(F_i(theta)) - F_i(F_j(theta))`

is exactly

`(A_j A_i - A_i A_j) theta`

`+ (A_j-I)c_i - (A_i-I)c_j`.

Using `c_i=(I-A_i)a_i`,

`Delta_ji(theta)`

`= (A_j A_i - A_i A_j) theta`

`  + (I-A_i)(I-A_j)a_j`

`  - (I-A_j)(I-A_i)a_i`.

This equation cleanly separates a matrix-order term from an optimum/translation term.

## 4. Two mechanisms: curvature commutator versus recency translation

### 4.1 Common optimum: pure noncommuting geometry

If `a_i=a_j=a`, then each stage acts around the same fixed point:

`F_i(theta)-a = A_i(theta-a)`.

Therefore

`F_j(F_i(theta)) - F_i(F_j(theta))`

`= (A_j A_i - A_i A_j)(theta-a)`.

So with a shared optimum, chronology is encoded only through the noncommutativity of the finite contraction maps.

If `A_i` and `A_j` commute, the two orders are **exactly indistinguishable** for every initial theta in this model.

This is a useful impossibility statement:

> With common optimum and mutually commuting quadratic stage maps, endpoint-only chronology is not identifiable.

### 4.2 Different optima: order can be visible even when curvature commutes

If `A_i A_j = A_j A_i`, then the matrix commutator vanishes, but the translation term becomes

`Delta_ji = (I-A_i)(I-A_j)(a_j-a_i)`

when the commuting matrices also commute with the indicated products, as they do here.

Therefore different stage optima can create an order signal **without any noncommuting curvature at all**.

This is the exact affine analogue of ordinary recency/forgetting.

Most starkly, in one dimension all matrices commute automatically, yet

`Delta_ji = (1-A_i)(1-A_j)(a_j-a_i)`.

So a scalar learner can reveal which task came later simply because the tasks pull toward different optima.

This formalizes why “order can be classified” is far too weak a scientific claim: a perfect classifier can be reading nothing deeper than endpoint bias toward the most recent task.

The original Phase-0 v1 failure is conceptually in this family: capability differences already exposed the last-stage identity.

## 5. Exact scalar recency model

Suppose every stage has the same scalar contraction `0 < alpha < 1` but a different optimum `a_i`:

`F_i(theta) = alpha theta + (1-alpha)a_i`.

For an N-stage chronology,

`E_pi = alpha^N theta_0`

`      + (1-alpha) sum_(r=1..N) alpha^(N-r) a_{pi_r}`.

The most recent stage receives weight `(1-alpha)`, the previous stage `(1-alpha)alpha`, and so on.

Thus chronology is encoded as a geometrically weighted recency code even though the dynamics are completely commutative in their linear part.

This model predicts a simple failure mode for naive provenance:

- if stage capabilities/optima remain distinguishable, recent stages dominate the endpoint;
- a chronology detector can perform perfectly for a trivial reason;
- common continuation suppresses older contributions geometrically.

It is therefore a useful null model for any more sophisticated ChronoTrace claim.

## 6. Common continuation gives an exact history-decay law

Let two histories produce endpoints `theta_pi` and `theta_sigma`. Apply the **same** affine continuation stage W to both:

`F_W(theta) = A_W theta + c_W`.

Their difference after one common continuation is

`Delta_1 = F_W(theta_pi)-F_W(theta_sigma) = A_W Delta_0`,

where

`Delta_0 = theta_pi-theta_sigma`.

After k repeated common continuations,

`Delta_k = A_W^k Delta_0`.

The translation `c_W` cancels exactly.

This gives an exact affine theory of **provenance survivability**.

If `A_W` is diagonalizable with eigenvalues `lambda_r`, each history component along eigenmode r decays as

`|lambda_r|^k`.

A mode with `0<|lambda|<1` has half-life

`k_1/2 = log(1/2) / log(|lambda|)`.

More generally,

`||Delta_k|| <= ||A_W^k|| ||Delta_0||`.

This sharpens the earlier qualitative “Training-History Half-Life” idea:

> common continuation does not erase all history at one universal rate; each chronology direction has a survivability spectrum determined by the continuation dynamics.

In a nonlinear model, the analogous local object is the product of continuation Jacobians along the common trajectory.

## 7. Why common washout can erase a forensic witness before capability differences

Suppose capability and chronology are different projections of the endpoint difference:

`capability_gap_k = u^T A_W^k Delta_0`

`forensic_gap_k   = v^T A_W^k Delta_0`.

Even under the same common continuation, these two observables can decay at very different rates because `u` and `v` project onto different eigenmodes of `A_W`.

Therefore there is no general reason for a balanced common tail to eliminate ordinary capability evidence before eliminating a particular chronology witness.

This gives a clean theoretical explanation for the failed Phase-0b washout strategy: the desired ordering of decay rates was an empirical assumption, not a theorem.

The interesting regime for temporal provenance is instead one where the continuation dynamics happen to satisfy

`|u^T A_W^k Delta_0| ~ 0`

while

`|v^T A_W^k Delta_0| > 0`.

Finding such an observation `v` is exactly an Order-Witness problem.

## 8. Observation maps define identifiability

Let the auditor observe a linearized measurement

`y = Q theta`.

Two candidate histories are distinguishable iff

`Q(E_pi-E_sigma) != 0`.

For a finite candidate set `Pi`, define the observed separation

`delta_Q = min_(pi != sigma) ||Q(E_pi-E_sigma)||`.

Exact candidate-history identifiability requires `delta_Q > 0`.

If measurement/model error has norm at most `epsilon`, nearest-candidate decoding is robust whenever

`delta_Q > 2 epsilon`.

This makes the access hierarchy concrete:

- `Q=I`: full weight-space observation;
- low-dimensional `Q`: selected parameter features;
- local behavioral Jacobian: black-box query observation.

A history can be perfectly distinct in weights and completely invisible under a particular benchmark projection.

## 9. Affine collision and impossibility conditions

Even when some pair commutators are nonzero, exact total-order identification is not guaranteed.

For a candidate set of permutations, chronology is identifiable only if

`E_pi != E_sigma`

for every distinct pair after the chosen observation map.

Collisions can arise from:

- commuting maps with a shared optimum;
- symmetries among stage matrices/optima;
- cancellation of multiple order-sensitive terms;
- projection into an observation null space;
- strong common continuation contracting the distinguishing modes below noise.

Therefore noncommutativity is a **source** of chronology information, not a sufficient theorem of global identifiability.

## 10. Three-stage interaction in the affine model

For three affine stages, the full endpoint is still exact and can be decomposed by the same Möbius interaction hierarchy used by ChronoTrace.

The degree-3 residual measures what cannot be represented by base-anchored singleton and pair terms.

In affine dynamics this higher-order term arises because pair effects themselves are transformed by preceding/following affine maps. Thus prefix-conditioned pair geometry is not peculiar to neural-network nonlinearities; it already appears in the composition algebra of finite affine stages.

The Pythia-14M midpoint/separation decomposition can therefore be viewed as the nonlinear high-dimensional analogue of a general operator-composition phenomenon rather than an ad hoc artifact of transformers.

## 11. Extended-state affine operators unify optimizer memory

For momentum or linearized Adam-like dynamics, augment the state:

`z = [theta; v]`

or more generally

`z = [theta; m; v; ...]`.

A quadratic loss plus linear momentum update is an affine map in this larger state:

`z' = B_i z + d_i`.

Chronology can then be analyzed with exactly the same matrix-product machinery as above.

Crucially, the auditor may observe only weights:

`theta = P z`.

Two histories can therefore have:

- large separation in optimizer state but little current weight separation;
- later common training that transports the hidden optimizer-state difference back into weights;
- or optimizer differences that decay before becoming behaviorally visible.

This is the precise mathematical reason future realistic ChronoTrace experiments must distinguish **full-state memory** from **weight-observable provenance**.

## 12. Testable predictions from the affine theory

The affine model produces several falsifiable predictions for later experiments.

### P1 — common-optimum commuting control should erase geometry-only chronology

Construct stages with matched optimum and commuting Hessians. Under deterministic reset GD, AB and BA should be identical up to numerical error.

This is a strong negative control for implementation and for claims of mysterious order fingerprints.

### P2 — different-optimum scalar control should produce trivial recency chronology

A one-dimensional or jointly diagonal stage system with different optima should allow order discrimination despite zero matrix commutator.

This is a positive control for the capability/recency confound.

### P3 — common continuation should obey mode-specific exponential decay in the quadratic regime

For a fixed common continuation W, measured history differences should follow the spectrum of `A_W`.

This can validate a quantitative provenance-survivability / half-life framework before applying it to nonlinear LLMs.

### P4 — observation-specific half-lives should differ

Capability and forensic projections should decay at different rates when they align with different continuation eigenmodes.

This directly tests why common washout need not solve capability matching.

### P5 — persistent optimizer state should change the leading chronology channel

A controlled momentum experiment should show first-order chronology that disappears or changes form when optimizer state is reset at stage boundaries.

This would validate the mechanism decomposition, not claim optimizer memory as novel.

## 13. Relation to the current Pythia program

The affine theory does **not** replace the nonlinear finite-interaction experiments.

Its role is to provide exact null models and impossibility/control cases:

- trivial different-optimum recency;
- pure noncommuting common-optimum geometry;
- exact common-tail decay;
- observation projection;
- extended-state optimizer transport.

The currently running T2 experiment remains an independent test of whether the structured low-order/partial-order phenomenon repeats in Pythia-14M synthetic training.

No T2 parameter or interpretation threshold is changed by this note.

## 14. Scientific implication

The most useful conceptual decomposition is now:

`observed chronology signal`

`= recency/translation component`

`+ geometric noncommutative component`

`+ prefix-conditioned higher-order component`

`+ optimizer/time/stochastic components`

`all passed through an observation map and later continuation dynamics`.

A convincing ChronoTrace paper should not merely show that one of these components exists. It should identify which component supports the recovered chronology, state the access regime, and characterize when the relevant history directions survive or become unobservable.
