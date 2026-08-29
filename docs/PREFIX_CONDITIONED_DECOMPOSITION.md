# Prefix-Conditioned Failure Decomposition

Date frozen: 2026-08-29

Purpose: pre-result derivation for the Pythia-14M T1 theory diagnostic. This note was written before reading T1 output so that midpoint/separation explanations are not selected post hoc.

## 1. Base finite-pair approximation

For three stages A, B, C, write the exact endpoint of a chronology as a sum of:

- base checkpoint `theta_0`;
- singleton effects `Delta_A, Delta_B, Delta_C`;
- directed pair interactions chosen by the chronology;
- an exact order-3 residual `Phi_pi(ABC)`.

For histories sharing prefix A:

`E_ABC = P_ABC + R_ABC`

`E_ACB = P_ACB + R_ACB`,

where `P_pi` is the current pair-truncated prediction and `R_pi` is the exact third-order interaction for that chronology.

Because `ABC` and `ACB` share A-before-B and A-before-C, their pair predictions differ only by the B/C pair orientation.

## 2. Exact midpoint/separation coordinates

Define the pair-model midpoint

`M0_A;BC = (P_ABC + P_ACB)/2`

and base pair separation

`S0_A;BC = P_ABC - P_ACB = C_BC(theta_0)`.

Define third-order midpoint bias

`B_A;BC = (R_ABC + R_ACB)/2`

and third-order separation drift

`D_A;BC = R_ABC - R_ACB`.

Then exactly:

`(E_ABC + E_ACB)/2 = M0_A;BC + B_A;BC`

and

`E_ABC - E_ACB = S0_A;BC + D_A;BC`.

But the exact endpoint difference is also

`E_ABC - E_ACB = C_BC(F_A(theta_0))`.

Therefore

`D_A;BC = C_BC(F_A(theta_0)) - C_BC(theta_0)`.

So the difference of the two order-3 residuals is exactly the **prefix-conditioned commutator drift**.

## 3. Two distinct failure modes

A base pair decoder can fail even if the two actual endpoints remain distinct.

### Failure mode A — separation drift

`D_A;BC` rotates, shrinks, enlarges, or reverses the vector separating the two tail orders.

Useful diagnostics:

- `||D|| / ||S0||`;
- cosine `cos(S0, S0 + D)`;
- true conditioned separation norm `||S0+D||`;
- projection of D onto the base decision axis.

### Failure mode B — midpoint bias

`B_A;BC` translates both actual endpoints together relative to the pair-model midpoint.

Even if the true separation direction remains mostly correct, a sufficiently large component of B along the pairwise decision axis can make both actual endpoints closer to the same pair-model candidate.

Useful diagnostics:

- `||B|| / ||S0||`;
- projection `2 <B,S0> / ||S0||^2`;
- midpoint-bias angle to `S0`;
- midpoint displacement orthogonal to `S0`.

The observed Pythia pattern, where both members of a prefix pair map to the same prediction (for example `ABC -> ACB` and `ACB -> ACB`), is compatible with either a strong midpoint bias, a strong asymmetric separation drift, or both. T1 should not assume one in advance.

## 4. Relationship to directional contamination

For any true chronology pi and competitor sigma, the current pair decoder uses signatures `s_pi` and `s_sigma`. Let

`d = s_pi - s_sigma`

and exact omitted residual `r_pi = E_pi - P_pi`.

The exact nearest-signature decision flips against sigma iff

`chi_pi,sigma = -2 <r_pi,d> / ||d||^2 >= 1`.

For a tail swap with shared prefix, `d` lies on the base pair-order separation axis. Therefore the chi values for the two members of the pair can be rewritten in terms of midpoint bias B and separation drift D. This connects the global nearest-signature diagnostic directly to the prefix-conditioned decomposition above.

## 5. Falsifiable predictions for T1

Without changing the decoder, model, stage data, learning rate, or stage length:

1. The three previously misdecoded histories should have at least one competitor with `chi >= 1`.
2. If the structured tail-swap explanation is correct, the dominant boundary-crossing competitor should usually be the same-prefix tail swap.
3. For each prefix A/B/C, `R_prefixXY - R_prefixYX` should match the corresponding conditioned-commutator drift to numerical precision.
4. If commutator drift is the main mechanism, the conditioned tail-pair commutator should differ materially in norm and/or direction from the base commutator.
5. If midpoint bias is important, the two residuals sharing a prefix should have a large common component along the base tail-pair separation axis.

Failure of predictions 2/4 would weaken the state-conditioned-tail explanation even if generic higher-order contamination is present.

## 6. Why this matters for the next decoder

A successful diagnosis would suggest a **prefix-adaptive** reconstruction scheme:

1. infer a coarse early prefix from low-order evidence;
2. evaluate later pair interactions at the inferred prefix state rather than only at `theta_0`;
3. continue recursively while tracking uncertainty.

This would be fundamentally different from a static global pair tournament. It would also connect naturally to a path-signature / chronological-calculus view in which later interaction coefficients are conditioned by the earlier path.

This is only a design implication. No adaptive decoder should be implemented until independent-seed experiments confirm that the structured partial-order phenomenon generalizes.
