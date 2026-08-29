# Path-Signature / Chronological-Calculus Framing

Status: mathematical framing for ChronoTrace. The classical mathematics described here is **not** a novelty claim.

## Why this connection matters

ChronoTrace studies ordered compositions of noncommuting training operators. Classical control and differential-equation theory has several closely related ways to represent ordered paths:

- Chen / path signatures: iterated integrals indexed by words;
- Chen-Fliess series: nonlinear input-output expansions built from iterated integrals;
- Magnus expansion: the logarithm of a time-ordered exponential expressed as nested commutators;
- free Lie / log-signature representations: compressed coordinates for genuinely order-sensitive information.

These frameworks suggest that the chronology signal should not be thought of as one mysterious fingerprint. It should appear in a **graded hierarchy**:

- degree 1: individual stage effects;
- degree 2: pair order / commutator effects;
- degree 3: prefix-conditioned and nested three-stage effects;
- higher degree: progressively finer path information.

ChronoTrace's empirical `K=1,2,3,...` interaction hierarchy is therefore conceptually analogous to truncating an ordered path signature or log-signature.

## A discrete training path

Let the candidate stage alphabet be

`A = {A_1, ..., A_N}`.

A training chronology is a word

`pi = A_{i_1} A_{i_2} ... A_{i_m}`

acting on an initial training state `z_0` through stage maps `F_i`.

The endpoint is

`E_pi = P(F_{i_m} ... F_{i_1}(z_0))`,

where `P` projects the extended training state to whatever is observable, usually model weights.

The central inverse question is whether `pi` can be recovered, wholly or partially, from `E_pi` plus controlled probes of the candidate stage maps.

## Local continuous-time analogy

If training is approximated by a controlled differential equation

`d theta / dt = sum_i u_i(t) V_i(theta)`,

then the control path `u(t)` specifies which training source/stage is active over time.

The endpoint expansion contains iterated integrals of the control path multiplied by ordered compositions of vector fields. The logarithmic / Magnus form reorganizes these into Lie brackets and nested Lie brackets such as

`[V_A,V_B]`,

`[V_A,[V_B,V_C]]`,

and higher terms.

This matches the ChronoTrace mechanism already observed experimentally:

- the AB/BA difference is second order and governed locally by a Lie bracket;
- once stages become finite, the relevant pair interaction depends on the state reached by the preceding prefix;
- three-stage residuals correspond to missing degree-3 information.

## Discrete finite-stage analogue

ChronoTrace currently uses an exact finite interaction decomposition rather than assuming infinitesimal training.

For chronology `pi` and stage subset `S`, let `E_pi(S)` be the endpoint of the ordered subsequence induced by `S`. Möbius inversion gives exact interactions

`Phi_pi(S) = sum_{T subseteq S} (-1)^(|S|-|T|) E_pi(T)`.

The full endpoint is exactly

`E_pi(U) = sum_{S subseteq U} Phi_pi(S)`.

This is not literally the classical continuous path signature, but the structural analogy is strong:

- both are graded by interaction/order degree;
- both preserve ordering information through noncommutative terms;
- truncation produces a controllable information loss;
- higher degree becomes necessary when lower-order terms cannot separate candidate paths.

## What may become the real scientific quantity

A useful definition is the **Training-History Interaction Order**

`K*(pi, epsilon)`

= the smallest degree K such that a K-truncated chronology representation identifies the true history with the required robustness / error tolerance.

This reframes the research from

> Can ChronoTrace decode the order?

into

> How much noncommutative interaction depth is required to identify this training path from the observable endpoint?

That is potentially much more informative.

Possible regimes:

1. **K=1:** order is irrelevant or ordinary singleton/capability effects dominate.
2. **K=2:** pairwise chronology is sufficient; a static tournament works.
3. **K=3:** pair effects are state-dependent; prefix-conditioned interactions are needed.
4. **large K:** the training path is strongly nonlocal and low-order forensic reconstruction may not be computationally useful.
5. **unidentifiable after projection:** distinct extended-state histories collapse to the same observable weight endpoint.

## Truncation should be judged geometrically

A large higher-order residual norm does not automatically imply decoding failure. The residual matters only insofar as it points toward a competing chronology.

For true signature `s_pi`, competitor `s_sigma`, and omitted residual `r_pi`, define

`d = s_pi - s_sigma`.

The exact nearest-signature boundary is crossed when

`chi_{pi,sigma} = -2 <r_pi,d> / ||d||^2 >= 1`.

This directional criterion is analogous to asking whether omitted signature degrees perturb the observable in an order-confusing direction, not merely whether their norm is large.

This is important because the controlled tiny transformer retained perfect finite-pair decoding even after a conservative norm bound failed, whereas Pythia-14M appears to fail along specific tail-swap directions.

## State dependence and Chen-style prefix structure

For finite maps define

`C_BC(theta) = F_C(F_B(theta)) - F_B(F_C(theta))`.

Then

`E_ABC - E_ACB = C_BC(F_A(theta_0))`.

This shows that later pair order is conditioned on the earlier path prefix. It is the finite-stage analogue of a higher-degree ordered interaction: the effect of the B/C word depends on having already traversed A.

Therefore a realistic chronology representation should probably be **prefix-adaptive**, rather than one static pair table measured at `theta_0`.

## What is classical and what might be new

Classical / not novel:

- ordered paths admit iterated-integral representations;
- Magnus expansions contain nested commutators;
- log-signatures live in a free Lie algebra;
- truncation degree controls how much path information is retained;
- noncommuting vector fields make order matter.

Relevant references include work on path signatures, Chen-Fliess expansions, and Magnus expansions, as well as modern sequential-learning Lie-bracket work.

Candidate ChronoTrace contribution:

> Treat an LLM training pipeline as an unknown ordered path through training-state space, construct low-order empirical interaction coordinates from candidate training stages, and study when those coordinates are sufficient to invert or constrain the hidden chronology of a finished model.

A stronger eventual contribution would connect empirical interaction order to a predictive identifiability boundary and show useful reconstruction with fixed low K as the number of candidate stages grows.

## Consequences for experiment design

1. Do not optimize a decoder until the required interaction degree is understood.
2. Measure K=1/K=2/K=3 contributions separately.
3. Report partial order, not only exact-permutation accuracy.
4. Test whether the required degree changes with stage duration / displacement.
5. Distinguish weight-geometry path information from optimizer-state and stochastic-state channels.
6. Use N>=4 before claiming that K=3 is computationally preferable to exhaustive chronology replay.
7. Treat failure of a low-order truncation as information about path complexity, not automatically as failure of the broad chronology hypothesis.
