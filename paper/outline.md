# Final Paper Outline

## Title

**ChronoTrace: Certified Reconstruction of Training Order from Noncommutative Learning Interactions**

## Central claim

Under replay-capable weight-space access to a known base checkpoint and known candidate stage operators, unknown training chronology can be treated as an inverse certification problem. ChronoTrace uses exact ordered interactions, a witness bank frozen before higher-order candidate output, and proof-safe local-order LP certificates to exclude impossible precedence classes. In fresh Pythia-14M `N=K=4` confirmation, the frozen label-blind method certifies 27/32 complete histories and 182/192 pairwise precedences with zero contradictory certified pairs.

## 1. Introduction

- Distinguish inverse chronology from forward curriculum/order optimization, membership, influence, and lineage.
- Explain why exhaustive replay is scientifically useful but factorial.
- Introduce certificate/abstention framing rather than forced permutation classification.
- State five contributions:
  1. inverse chronology formulation;
  2. exact ordered interaction representation;
  3. proof-safe multi-witness precedence certificates;
  4. finite-query information barrier for omitted interactions;
  5. fresh label-blind Pythia confirmation.

**Figure 1:** pipeline and access regime.

## 2. Problem Setting and Access Regime

- Known base state `theta_0`.
- Known deterministic candidate stage maps `F_i`.
- Hidden chronology `pi`.
- Observed final weights `y = E(pi)`.
- Replay-capable white-box access.
- Outputs: certified pair relation, ambiguity/abstention, total order only if complete and transitive.

## 3. Exact Ordered Interaction Geometry

### 3.1 Ordered Möbius interactions

Define recursively:

`Phi(w) = E(w) - theta_0 - sum_{proper ordered subwords u} Phi(u)`.

State exact endpoint decomposition and degree-K exactness for words of length `<=K`.

### 3.2 Local commutator connection

Use reset-SGD expansion only as mechanism/background; explicitly distinguish from forward Lie-bracket ordering literature.

### 3.3 Projection

Show that frozen linear witness projection commutes with the ordered interaction decomposition.

## 4. From Witnesses to Certified Precedence

### 4.1 Frozen witness bank

Witnesses are frozen before higher-order candidate output.

### 4.2 L1-safe multi-witness theorem

For unit `u_j` and `||alpha||_1 <= 1`, `v=sum alpha_j u_j` satisfies `||v||_2<=1`, yielding a conservative Euclidean class-distance lower bound.

**Figure 4:** conceptual multi-witness geometry.

### 4.3 K-local order marginals

- simplex constraints;
- marginal consistency;
- precedence property constraints;
- fixed-K polynomial representation size.

### 4.4 Proof-safe numerical dual correction

Explain why raw floating-point LP dual values are not accepted as certificates and how reduced-cost correction restores a conservative bound.

### 4.5 Label-blind pair rule

Test both orientations independently; infer only when exactly one is excluded; abstain if neither; invalidate if both.

## 5. Exactness and Fixed-Depth Information Barrier

### 5.1 Terminal exactness

At `K=N=4`, independently compare local-order primals to complete-permutation convex hulls for every orientation class.

### 5.2 Information barrier

Give the finite-query smooth-perturbation construction showing that degree-`<=K` observations cannot universally control an unseen `K+1` directional contribution for arbitrary smooth one-step SGD.

**Figure 5:** fixed-K polynomial representation versus terminal exactness and omitted-tail barrier.

## 6. Experimental Program

- Pythia-14M as controlled mechanism bridge.
- Deterministic one-update stage operators.
- Numerical reproducibility/pinned CPU path.
- Evidence roles: spent development, preregistered negative, fresh confirmation.

**Figure 2:** development/falsification ladder.

## 7. Development Results and Falsifications

### 7.1 Static low-order decoder failure

Structured late-stage ambiguity shows local/base pair interactions are insufficient at finite stage sizes.

### 7.2 Exact reachable geometry

24/24 spent histories separable, establishing endpoint identifiability but requiring factorial full-history enumeration.

### 7.3 K3 convex pruning

A/B eliminated; C/D survive on spent ABCD.

### 7.4 Preregistered single-witness K4 negative

C survives despite nonzero exact Euclidean separation; all exactness checks pass. Interpret as witness-direction failure.

### 7.5 Post-hoc multi-witness methodology development

Using only witnesses already frozen before K4 output, safe combinations separate the remaining wrong class and all six wrong pair orientations on spent ABCD. No new Pythia calls, but explicitly post-hoc.

## 8. Fresh Label-Blind Confirmation

### 8.1 Protocol/provenance freeze

- Explain discovery that old v1 seeds had already been consumed.
- Mark old seeds spent.
- Mechanically derive new v3 seeds.
- Launch all four together from immutable marker.
- No intermediate adaptation.

### 8.2 Main result

**Table 3 / Figure 3:**

```text
2186192236  6/8 histories  43/48 pairs
1368008047  7/8            47/48
92712904    6/8            44/48
1944430236  8/8            48/48
TOTAL       27/32          182/192
```

Also report:

```text
5 full-history abstentions
10 ambiguous pair decisions
0 contradictions
0 double-exclusions
STRONG preregistered tier
```

### 8.3 Numerical validity

Report terminal convex-hull agreement, projected reconstruction residual, corrected-bound soundness, and exact active-lift replay.

### 8.4 Aggregation plumbing correction

Document the sorted-JSON-key issue transparently; emphasize that all scientific seed jobs had already completed and no measurements/method/thresholds changed.

## 9. Related Work and Novelty Boundary

Compare by problem/access rather than by claiming standard mathematical tools are new:

- curriculum/data ordering;
- forward Lie-bracket transfer-order planning;
- membership inference;
- influence/data attribution;
- model lineage;
- palimpsestic known-transcript provenance;
- training-memory/process-tensor work;
- permutation representations/local marginal hierarchies.

**Table 1:** problem/access comparison.

## 10. Discussion

- What fresh confirmation establishes.
- Why the preregistered negative is scientifically important.
- Certificate coverage versus forced prediction.
- Main open boundary: exact fixed-depth scalability for `N>K`.

## 11. Limitations

Keep concise and explicit:

- Pythia-14M, four stages;
- terminal `K=N=4` confirmation;
- replay-capable white-box access;
- controlled plain-SGD stage channel;
- synthetic codebooks;
- four fresh seeds/eight histories each.

## 12. Conclusion

Conclude narrowly: trained endpoints can support certified inverse chronology under controlled replay-capable access; terminal confirmation is strong; universal fixed-depth exactness remains open and is constrained by the information barrier.

## Appendix

- exact confirmation run/artifact identifiers;
- abstention matrix;
- preregistered outcome tiers;
- provenance correction;
- build/reproducibility details.

## Manuscript guardrails

See `CLAIMS_AND_EVIDENCE.md`. In particular, do not claim standalone novelty for Möbius inversion, Lie brackets, Sherali–Adams/RLT, convex duality, or L1 witness combination, and do not claim arbitrary-N polynomial exact reconstruction.