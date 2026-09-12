# Residual-controlled bidirectional chronology joins

## Scope

This is a conditional construction for unknown stage order, not an invention of meet-in-the-middle search or reversible optimization. It replaces a loose full-trajectory approximation with explicit prefix states and inverse-suffix enclosures. Known immutable deterministic GD recipes, known base, N distinct stages used exactly once, global gradient regularity, and valid numerical allowances are assumptions. Momentum/Adam, unknown batches, and recipe mismatch are not established by this experiment.

## Step inverse lemma

Let T(x)=x-eta*g(x), where g is globally L-Lipschitz on R^d and q=eta*L<1. For any z, x -> z+eta*g(x) is a contraction, so T has a unique inverse at z. The reverse triangle inequality gives

    ||T(x)-T(v)|| >= (1-q)||x-v||.

For an approximate inverse xhat of an approximate right-hand side zhat,

    ||xhat-T^{-1}(z)|| <= (||T(xhat)-zhat||+||zhat-z||)/(1-q).

If incoming error is E, measured residual is r, and delta is a justified computational allowance, propagate

    E_previous = (E+r+delta)/(1-q).

Induction over reversed updates and stages encloses every inverse image of the observation set. This bound depends on residuals, not a solver's convergence flag. Iteration exhaustion cannot justify discarding a suffix; it enlarges its radius.

## Observation and target error

FP16/32 observations define a closed Cartesian rounding cell C, with each interval bounded by midpoints to the previous and next representable values. Both ties are included. Overflow-neighbor boundaries are explicitly rejected. FP64 is a point observation with separate computation allowances.

Inverse transport begins from the center and Euclidean half-diagonal of C, plus the declared error in the numerically trained target. For nonexpansive GD with m updates per stage and per-update error delta, target error is bounded by N*m*delta. General forward Lipschitz maps use accumulated gain bounds. Computed prefix states also carry their own errors. The target computation is not silently assumed exact.

## No-loss middle join

Every chronology has a unique split p+s, where p has k stages and s has N-k stages and their stage sets are complementary. Let a_p approximate its forward prefix with radius e_p, and b_s enclose the inverse suffix with radius E_s. Any compatible chronology has a middle state in both balls. Therefore a necessary condition is

    ||a_p-b_s|| <= e_p+E_s.

Construct both catalogues completely and compare every complementary pair. Rejecting only pairs that violate this condition preserves every compatible chronology. A declared join allowance enlarges the right side rather than causing optimistic borderline exclusions.

## Budget-safe final verification

Replay each joined suffix from its cached prefix. Distance to the closed observation cell is 1-Lipschitz, so a candidate can be excluded only when its distance exceeds its accumulated forward error, target error, and stated guard. Retain every untested joined candidate when the replay cap is exhausted. The live set therefore still contains every compatible candidate, conditional on the assumptions.

A single verified live chronology is unique within the declared candidate model. An empty live set signals inconsistent assumptions; it does not identify the incorrect assumption. A nonempty set cannot prove that an unknown out-of-model process did not coincidentally produce the same endpoint.

Final verification uses the observation box itself, not just the circumscribed ball used during inverse transport. This is ordinary set-membership reasoning, not a new theorem about quantization.

## Work and limitations

Let A(N,k)=sum_{r=1}^k P(N,r). Prefix work is m*A(N,k) gradients. All actual inverse fixed-point and residual-check gradients are counted. A target-blind estimate c selects k minimizing A(N,k)+c*A(N,N-k). N=8,c=8 chooses k=5: 8,800 prefix stages and 400 inverse stages, versus 109,600 stages for complete forward prefix-cached replay.

The reference join still checks N! middle-vector distances. Memory, join computation, and worst-case replay remain combinatorial. This is a reduction in expensive stage/gradient work under the measured conditions, not subfactorial total computation or a solution for arbitrary N.

The exact-arithmetic results are conditional on correct global L and numerical bounds. Experimentally L is numerically computed with declared inflation, and fixed allowances are used. Neither is a formal interval-arithmetic proof. The code's metadata signature detects ordinary recipe metadata changes but cannot authenticate arbitrary hidden mutation inside a gradient callback closure.

## Related work

Maclaurin, Duvenaud, Adams (ICML 2015), Gradient-based Hyperparameter Optimization through Reversible Learning, reverses known optimization trajectories for hyperparameter gradients: https://proceedings.mlr.press/v37/maclaurin15.html

Krasheninnikov, Turner, Krueger (2025), Fresh in memory, already establishes temporal information in language-model activations: https://arxiv.org/abs/2509.14223

Set-membership identification of switched linear systems is another relevant sequence-inference setting: https://www.sciencedirect.com/science/article/pii/S0005109814004828

The intended distinction is unknown-order endpoint auditing with explicit access assumptions, residual error transport, quantization-aware alternative exclusion, and measured work savings. Priority and publication quality require specialist review, not a broad claim that training chronology, inversion, or sequence inference is new.
