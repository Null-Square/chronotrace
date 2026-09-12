# K-local information barrier and projected interaction lifts

ChronoTrace measures exact ordered interactions through a chosen degree `K`. The natural
question is whether those measurements alone can certify a nontrivial bound on an unseen
`K+1` interaction. Without an additional regularity assumption, the answer is no even for
smooth deterministic one-step SGD.

## Finite-query polynomial perturbation

Let one stage update be

\[
F_j(x)=x-\eta\nabla L_j(x).
\]

Assume every observation collected so far invokes stage `j` only at the finite state set
\(Q=\{q_1,\ldots,q_m\}\). Let \(x_*\notin Q\) be an unqueried predecessor state and let
\(v\) be any desired gradient change. Define

\[
P(x)=\prod_{q\in Q}\|x-q\|^2,
\qquad
\psi(x)=\frac{P(x)}{P(x_*)}\langle v,x-x_*\rangle.
\]

Every factor \(\|x-q\|^2\) has a double zero at its corresponding query. Consequently

\[
\nabla\psi(q)=0\quad(q\in Q),
\qquad
\nabla\psi(x_*)=v.
\]

Replacing `L_j` by `L_j + psi` therefore leaves every previously queried SGD transition
unchanged while changing the unseen transition at `x_*` by `-eta v`. Choosing `v` along
any fixed certificate direction changes that unseen directional interaction by an
arbitrary scalar.

**Consequence.** No finite universal bound on an unseen `K+1` directional interaction can
be derived from `K`-local transition observations alone for arbitrary smooth stage losses.
One must either acquire some `K+1` information or impose an explicit regularity class.

This theorem is an information requirement, not a computational lower bound: it explains
why the hierarchy must occasionally lift interaction degree.

## Witness-projected K+1 lift

For a fixed unit separating direction `u`, linearity of Möbius inversion gives

\[
\phi_u(w)=\langle u,\Phi(w)\rangle.
\]

When an exact endpoint `E(w)` of length `K+1` is measured, its new interaction can be
computed directly in scalar space:

\[
\phi_u(w)=
\langle u,E(w)-\theta_0\rangle
-
\sum_{s\subsetneq w,\,s\ne\varnothing}\phi_u(s).
\]

The full `K+1` interaction tensor never needs to be retained. Measuring all ordered words
of length `K+1` costs `P(N,K+1)=O(N^{K+1})` stage transitions for fixed `K`, rather than
`N!` complete-history enumeration.

## Local-order marginal hierarchy

For every unordered stage subset `S` with `|S| <= K`, introduce one nonnegative weight
`mu[S,sigma]` for each ordering `sigma` of `S`. Enforce

1. one simplex per subset; and
2. consistency of every subset distribution with each one-element deletion.

Every global chronology maps to a one-hot feasible point. Hence minimizing a linear
projected-interaction objective over the local polytope gives a conservative lower bound
on minimization over actual global chronologies.

The coordinate count is

\[
\sum_{r=1}^{K} P(N,r),
\]

which is polynomial in `N` for fixed `K`. At `K=N`, the top-level simplex is a
distribution over complete permutations and all lower marginals are induced from it, so
the hierarchy becomes exact. This terminal exact level is factorial, as expected from
generic permutation-CSP hardness.

For the current four-stage spent Pythia experiment, `K=4=N`. A witness-projected K4
measurement therefore has **no omitted interaction tail**: it can directly test whether
the frozen K3 separating directions remain valid certificates for the true four-stage
chronology classes.
