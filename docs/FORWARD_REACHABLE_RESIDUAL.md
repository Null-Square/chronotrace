# Finite forward-reachable residual decoding

## Motivation

The reverse-peeling experiments exposed a numerical boundary: directly solving

\[
F_j(x;y)=x-y-\eta\nabla L_j(x)=0
\]

can be unstable even when the predecessor geometry is informative. If a finite predecessor codebook is already available, solving this nonlinear equation is unnecessary.

For a candidate predecessor state \(z\) and candidate final stage \(j\), evaluate the residual directly:

\[
\rho(z,j;y)=\|z-y-\eta\nabla L_j(z)\|_2
            =\|T_j(z)-y\|_2.
\]

This is a finite hypothesis test rather than an inverse optimization problem.

## Exact finite-codebook theorem

Let \(\mathcal C=\{(z_i,j_i)\}_{i=1}^M\) be a finite set of predecessor/final-stage hypotheses and let

\[
q_i=T_{j_i}(z_i).
\]

Assume the observed endpoint is \(y=q_{i^*}\). Define

\[
\hat i=\arg\min_i \|q_i-y\|_2.
\]

If \(q_{i^*}\neq q_i\) for every \(i\neq i^*\), then \(\hat i=i^*\).

**Proof.** The true hypothesis has residual zero. Every competing hypothesis has strictly positive residual by the no-collision assumption. Therefore the true hypothesis is the unique minimizer. \(\square\)

This theorem does not require \(T_j\) to be invertible, contractive, or locally well conditioned.

## Robust margin certificate

Let the best and runner-up residuals be \(r_1<r_2\), with margin \(m=r_2-r_1\). If only the observed target is perturbed by \(e\), then each candidate distance changes by at most \(\|e\|_2\). Hence the winning hypothesis is unchanged whenever

\[
\|e\|_2 < m/2.
\]

The quantity \(m/2\) is therefore a target-noise certificate for the fixed finite codebook.

## Why degree three is enough for four-stage predecessors

An ordered interaction basis measured through degree three contains every interaction term needed to reconstruct any word of length at most three. Therefore, for four distinct stages \(A,B,C,D\), every three-stage predecessor can be reconstructed from the K3 basis even though a four-stage K3 endpoint prediction omits the degree-four interaction.

This creates a useful asymmetry:

- passive K3 endpoint decoding can fail on the four-stage target because the four-way interaction is missing;
- the three-stage predecessor codebook is still complete under the same K3 basis;
- one actual candidate final-stage gradient evaluation converts each predecessor into a reachable four-stage endpoint;
- chronology can then be decoded by finite transition consistency.

The affine regression test in `tests/test_peeling.py` deliberately uses a system where direct degree-three endpoint decoding recovers 18/24 histories, while finite forward-reachable residual decoding recovers 24/24 without any inverse solve.

## Pythia-14M spent-history gate

The first Pythia test remains on the already-observed pilot codebook seed `1011473075` and the already-observed history `ABCD`. In the frozen K3 pilot, direct degree-three endpoint decoding predicts `BACD`, so this history is a genuine pre-existing K3 miss rather than a history selected after seeing the new method.

The forward-reachable smoke will:

1. reproduce the frozen FP64 base, tokenizer, dataset, and `ABCD` target endpoint;
2. measure the same degree-three ordered interaction basis;
3. reproduce the known K3 baseline prediction `BACD`;
4. enumerate the 24 predecessor/final-stage hypotheses;
5. evaluate exactly one candidate-stage gradient at each three-stage K3 predecessor;
6. score \(\rho(z,j;y)\) and decode the unique minimum;
7. report the global residual margin and its target-noise radius \(m/2\).

No inverse iteration, line search, Hessian solve, held-out codebook, or confirmation output is permitted in this smoke.

## Scope boundary

This is not yet a scalability result. Enumerating all predecessor/final-stage hypotheses is factorial in the number of distinct stages. A successful four-stage smoke would establish a mechanism: **the missing highest-order chronology information can be exposed by finite transition consistency even when passive K3 truncation and explicit inverse recovery fail.** A separate experiment is required to replace factorial enumeration with a polynomial or K-local relaxation before making a scalability claim.
