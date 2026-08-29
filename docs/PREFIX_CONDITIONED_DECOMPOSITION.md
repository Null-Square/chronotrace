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

---

## 7. Post-result addendum — observed decomposition on the frozen 14M instance

This section was appended **after** the pre-result derivation above. The original derivation is retained so the hypothesis trail remains visible.

Midpoint-decomposition replay: workflow `33245010517`, artifact `9712666884`.

The replay exactly reproduced the previously frozen finite-pair basis and all six history endpoint hashes.

For each shared-prefix tail pair:

| Prefix | Tail pair | Alignment | Midpoint bias | Forward score | Reverse score | Both tails recoverable? |
|---|---|---:|---:|---:|---:|---:|
| A | BC | 0.019095 | -0.137284 | -0.118189 | 0.156379 | no |
| B | AC | 0.005551 | +0.177508 | 0.183059 | -0.171957 | no |
| C | AB | 0.013423 | +0.245139 | 0.258562 | -0.231716 | no |

The result resolves the ambiguity posed in Section 3.

### The conditioned separation mostly collapses rather than reverses

The normalized projection of the actual conditioned tail separation onto the static pair direction remains positive in all three cases, but is tiny:

- prefix A: `0.019095`;
- prefix B: `0.005551`;
- prefix C: `0.013423`.

So the useful same-direction tail-order signal has nearly vanished by the time the first stage has conditioned the state.

### The common midpoint bias dominates that residual signal

The absolute normalized midpoint biases are:

- prefix A: `0.137284` — about `7.2x` the alignment;
- prefix B: `0.177508` — about `32.0x` the alignment;
- prefix C: `0.245139` — about `18.3x` the alignment.

Thus `alignment - |midpoint_bias|` is negative for every prefix. The midpoint term is strong enough to make both actual endpoints in each shared-prefix pair favor the same static-pair candidate.

The sign predicts the observed asymmetric collapse:

- A/BC: negative midpoint bias -> `ABC` loses, `ACB` survives;
- B/AC: positive midpoint bias -> `BAC` survives, `BCA` loses;
- C/AB: positive midpoint bias -> `CAB` survives, `CBA` loses.

### Interpretation correction on chi

Section 5 was written before the result and listed `chi >= 1` as a diagnostic prediction. After deriving the exact inequality more carefully, we recognize that `chi < 1` is algebraically equivalent to the two-candidate nearest-signature decision itself. Therefore its match to correct/incorrect cases is **not independent evidence** for the mechanism. It remains useful only for identifying the contaminating chronology direction.

The stronger descriptive mechanism on this instance is the combination of:

1. all full-order errors being same-prefix tail swaps;
2. dramatic prefix-conditioned commutator shrinkage/rotation;
3. tiny positive tail alignment;
4. a much larger common third-order midpoint bias whose sign selects exactly the surviving tail order.

This remains explanatory evidence on one motivating instance. T2 is the independent test of whether the structured pattern repeats.
