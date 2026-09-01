# Reverse-Operator Peeling for Training-History Reconstruction

Status: theory/methodology note. This is not a paper claim. No held-out confirmation codebook output is used here.

## 1. Why this direction exists

The four-stage Pythia pilot exposed a specific failure mode of direct low-order decoding. The exact ordered interaction basis through degree 3 represents every history of length at most 3 exactly, but a four-stage endpoint contains an unmeasured degree-4 interaction. Direct K3 decoding therefore compares the final four-stage state against candidates that all omit the same interaction degree.

Reverse peeling changes the problem before decoding it. Instead of approximating a four-stage endpoint with a degree-3 model, hypothesize the final stage, invert that one training operator, and decode the resulting three-stage predecessor where K3 is exact.

This is a synthesis of known ingredients, not a claim that gradient-step invertibility or switched-system mode reconstruction is new mathematics.

## 2. Training operators

Let each deterministic one-step training stage be

`T_j(x) = x - eta * grad L_j(x)`.

A history `w=(w_1,...,w_N)` produces

`y = T_{w_N} ... T_{w_1}(theta0)`.

For a candidate final stage `j`, define a candidate predecessor

`q_j(y) = T_j^{-1}(y)`

whenever the inverse exists in the relevant region.

For gradient descent, if `grad L_j` is globally L_j-Lipschitz and `eta L_j < 1`, then

`x -> y + eta grad L_j(x)`

is a contraction, so `T_j` has a unique inverse obtained by fixed-point iteration.

The Pythia experiment will treat fixed-point convergence and forward reconstruction as empirical numerical checks; it will not infer a global Hessian bound from finite samples.

## 3. Exact point-set peeling

For candidate final stage `j`, let `R_j` be the set of all predecessor states reachable by permutations of the remaining stages.

Define

`r_j(y) = dist(q_j(y), R_j)`.

If the true final stage is `j*`, then `q_{j*}(y)` is the true predecessor and therefore `r_{j*}(y)=0` under exact inversion and an exact predecessor set.

A wrong final stage can also have zero residual only if its inverse image lies on the reachable set for the remaining stages. Such a collision is a genuine non-identifiability case and must not be broken by implementation order.

### Proposition 1 — exact last-stage recovery

Assume all candidate stage maps are invertible at `y`, the true predecessor is represented exactly, and

`dist(q_j(y), R_j) > 0`

for every wrong stage `j != j*`. Then minimum-residual peeling uniquely recovers the true final stage.

The proof is immediate: the true score is zero and every wrong score is positive.

## 4. Low-order predecessor approximation

Suppose instead that only an approximate predecessor set `Rhat_j` is available and its Hausdorff error from `R_j` is bounded by `E_j`.

Let

`delta_j = dist(q_j(y), R_j)`.

Then

- true stage: `rhat_{j*} <= E_{j*}`;
- wrong stage: `rhat_j >= delta_j - E_j`.

Therefore the true final stage is uniquely selected whenever

`min_{j != j*} (delta_j - E_j) > E_{j*}`.

If every approximation error is bounded by a common `E`, the sufficient condition simplifies to

`min_{j != j*} delta_j > 2E`.

This gives a clean separation-vs-truncation-error interpretation instead of relying only on empirical accuracy.

## 5. Exact K+1 corollary

With an exact ordered-interaction basis through degree K, every distinct-stage history of length at most K is reconstructed exactly by the interaction expansion.

Therefore for an `(K+1)`-stage chronology, one correct reverse peel reduces the predecessor to K stages and removes the truncation error completely.

For the current Pythia N=4,K=3 setup, the correct predecessor after peeling is exactly representable by the already-measured K3 basis. This is why N=4 is a particularly sharp falsification test of reverse peeling.

## 6. Executable toy result

`tests/test_peeling.py` fixes a deterministic noncommuting quadratic-gradient system where direct degree-3 endpoint decoding recovers exactly 18/24 four-stage histories. Under the same degree-3 interaction information, exact reverse-stage peeling recovers 24/24.

The test also verifies fixed-point inversion of the gradient step and requires exact residual collisions to be declared non-identifiable.

This proves that reverse peeling can strictly dominate direct truncated decoding. It does not establish that Pythia has the required inverse margins.

## 7. Computational caveat

Exact point-set peeling is not a scalable algorithm if `R_j` is constructed by enumerating every predecessor permutation. Recursively enumerating suffixes remains factorial in the worst case.

A successful four-stage point-set experiment would therefore establish only the mechanism: reverse removal of one known training operator can convert an inaccurate low-order endpoint approximation into an exact shorter-history problem.

It would not establish scalable chronology reconstruction.

## 8. Scalable relaxation hypothesis

For a remaining stage set `S`, a K-truncated prediction can be written as the base plus interaction contributions associated with ordered subsets of size at most K.

Instead of enumerating permutations, define a polynomial-size local-order relaxation over these interaction coordinates. At minimum, for each unordered subset of size r<=K, the coefficients over its `r!` possible local orders lie on a simplex. Stronger relaxations additionally enforce marginal consistency between overlapping pair and triple order variables.

Every genuine K-truncated permutation prediction lies in this feasible set, but the relaxation may contain states that no global permutation realizes.

A scalable last-stage score is then

`dist(T_j^{-1}(y), C_{S,K})`,

where `C_{S,K}` is the chosen K-local reachable relaxation. For fixed K, the number of interaction variables and local consistency constraints is polynomial in N.

The central future question is whether the relaxation stays tight enough to discriminate the true peeled stage on real training dynamics.

## 9. Falsification ladder

1. **Toy theorem gate:** fixed-point inverse, 18/24 -> 24/24 rescue, and exact-collision non-identifiability must pass CI.
2. **One-history Pythia gate:** on the already-spent pilot codebook and a preselected `ABCD` target, all four candidate inverse solves must be numerically stable, and the true final stage D must have a unique predecessor-residual minimum.
3. **All-24 spent gate:** freeze the complete decoder and interpretation before evaluating all 24 histories. Reverse peeling must beat both frozen K3 endpoint decoding (3/24) and frozen marginal recency (1/24).
4. **Relaxation gate:** before claiming scalability, replace factorial predecessor enumeration with a frozen polynomial-size K-local reachable relaxation and show that it preserves most or all of the peeling gain.
5. **Stage-count scaling:** test N=5 with fixed K=3 before increasing model size. This tests the claimed algorithmic structure more directly than moving immediately to Pythia-31M.
6. **Held-out confirmation:** only after the method and aggregate rule are frozen may the four clean confirmation codebooks be exposed.

Failure at any early gate should stop or redesign this branch on already-spent data only.
