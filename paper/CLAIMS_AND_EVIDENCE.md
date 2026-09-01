# ChronoTrace Paper Claim-to-Evidence Ledger

This file is the manuscript guardrail. Every strong sentence in the paper should map to one of the supported claims below. If a sentence does not map cleanly, either weaken it, cite it as related work/background, or move it to future work.

## Core claims

| ID | Paper claim | Evidence | Status |
| --- | --- | --- | --- |
| C1 | Unknown macro-stage chronology is a distinct inverse problem from forward curriculum selection, membership, ancestry, and influence. | Problem definition + related-work comparison. | **Supported framing** |
| C2 | Deterministic sequential stage endpoints admit an exact ordered Möbius interaction decomposition. | `src/chronotrace/geometry/interactions.py`; exact decomposition tests. | **Theorem / exact identity** |
| C3 | For words of length `<=K`, the degree-K truncation is exact. | Interaction recursion; streaming/direct-prefix equality tests. | **Theorem / exact identity** |
| C4 | Projection onto frozen linear witnesses commutes with the ordered interaction decomposition. | `projected_interactions.py`; tests. | **Theorem / exact identity** |
| C5 | An L1 combination of unit witnesses remains norm-bounded and therefore yields a valid Euclidean residual lower bound. | Triangle inequality + independent support recomputation; `multi_witness.py`. | **Theorem / exact inequality** |
| C6 | Chronology properties can be certified by minimizing a combined witness score over K-local order marginals and using a corrected LP dual lower bound. | `local_order_hierarchy.py`, `local_order_lp.py`, `multi_witness_local_order.py`, tests. | **Certified algorithm** |
| C7 | At terminal `K=N`, the local-order hierarchy equals the permutation convexification for the tested N=4 setting. | Terminal exactness construction + independent convex-hull checks on every confirmation orientation class. | **Exact terminal validation** |
| C8 | The final decision rule is label-blind: it tests both orientations for each pair before consulting the generating chronology for evaluation. | v2 scientific engine source-order gate + pairwise certificate tests + v3 wrapper/protocol. | **Protocol invariant** |
| C9 | On fresh Pythia-14M confirmation, ChronoTrace certifies 27/32 full histories and 182/192 pair precedences, with 0 contradictory inferred pairs. | Scientific run `33418210637`; frozen v3 selection. | **Fresh confirmatory result** |
| C10 | Abstention is conservative: unresolved pairs are left ambiguous rather than assigned a guessed orientation. | Decision rule; 10 ambiguous pairs across 5 cases; 0 contradictions/double-exclusions. | **Fresh confirmatory behavior** |
| C11 | Exact reachable endpoint geometry on the spent N=4 system separates all 24 histories. | Frozen all24 reachable selection. | **Spent mechanism result** |
| C12 | A preregistered single-witness K4 certificate can fail despite nonzero exact Euclidean class separation. | Valid repaired projected-K4 run; class C exact Euclidean distance nonzero but witness score nonpositive. | **Preregistered negative** |
| C13 | Multi-witness aggregation can resolve that witness-direction failure using only witnesses frozen before K4 output. | Spent post-hoc multi-witness analysis; no new model calls. | **Post-hoc methodology development** |
| C14 | Finite degree-`<=K` observations do not universally determine unseen `K+1` directional behavior for arbitrary smooth one-step SGD. | `docs/K_LOCAL_INFORMATION_BARRIER.md`; constructive perturbation proof. | **Information-barrier theorem** |
| C15 | Fixed-K representation size is polynomial in N, but the final terminal K=N experiment does not establish subfactorial exact chronology reconstruction for arbitrary N. | Coordinate count + terminal acquisition cost + barrier theorem. | **Complexity boundary** |

## Claims that require careful wording

### Novelty

Allowed:

> We introduce a proof-oriented inverse chronology framework that combines ordered interaction geometry, frozen witness banks, and local-order property certificates to reconstruct an unknown training order from a final model under replay-capable access.

Avoid:

> We discover that training order matters.

Avoid:

> We introduce Möbius inversion / Sherali-Adams / convex duality / L1 witness combination.

These mathematical ingredients are established. The contribution is their formulation and integration for inverse training chronology, plus the exact/certified empirical pipeline.

### Scalability

Allowed:

> For fixed K, the interaction/local-order representation has polynomial size in N, and pairwise property queries require O(N^2) certificate calls.

Required continuation:

> Exact endpoint certification below terminal depth additionally requires control of omitted interactions; the paper gives an information barrier against universal assumption-free tail inference.

Avoid:

> ChronoTrace reconstructs arbitrary histories in polynomial time.

### Confirmation

Allowed:

> The frozen fresh v3 suite confirms the label-blind terminal certificate at N=4 on Pythia-14M.

Avoid:

> The multi-witness idea was independently discovered on held-out data.

It was developed post-hoc on the spent ABCD instance, then frozen and evaluated on new deterministic seeds.

### Error/abstention

Allowed:

> ChronoTrace returned a complete certified history in 27/32 cases and abstained in five; no certified pair contradicted the generating chronology.

Avoid:

> ChronoTrace achieved 100% accuracy.

The correct statement is 84.375% complete-certificate coverage with zero contradictory certified pair relations in the fresh suite.

## Main-table numbers

### Fresh confirmation

| Seed | Full histories | Pair relations | Ambiguous pairs |
| --- | ---: | ---: | ---: |
| 2186192236 | 6/8 | 43/48 | 5 |
| 1368008047 | 7/8 | 47/48 | 1 |
| 92712904 | 6/8 | 44/48 | 4 |
| 1944430236 | 8/8 | 48/48 | 0 |
| **Total** | **27/32** | **182/192** | **10** |

### Certificate integrity

| Check | Maximum / result |
| --- | ---: |
| projected reconstruction residual | `6.765421556309548e-17` |
| terminal primal exactness error | `4.063047641389428e-17` |
| terminal witness-hull checks | all pass |
| corrected witness-geometry soundness | all pass |
| Euclidean-vertex soundness | all pass |
| exact active-lift replay | all pass |

### Historical baselines / development

| Method / result | Full chronology result | Role |
| --- | ---: | --- |
| static K3 on N=4 pilot | 3/24 | spent baseline |
| exact forward-reachable candidate endpoints | 24/24 | spent identifiability mechanism; factorial enumeration |
| K3 convex last-stage | eliminates A/B; C/D survive | spent certificate development |
| preregistered single-witness K4 | C survives; D survives | **scientific negative** |
| spent multi-witness pairwise | all six wrong ABCD orientations excluded | post-hoc methodology development |
| fresh label-blind multi-witness terminal certificate | 27/32 | **confirmation** |

## Figures and what each is allowed to imply

### Figure 1 — Problem and certificate pipeline

May show:

`hidden chronology -> final model -> ordered interaction probes -> frozen witness projections -> pair-class certificates -> partial/total order`

Must visibly label the access regime as replay-capable white-box.

### Figure 2 — Development ladder

Show the scientific progression:

`static low-order failure -> exact reachable identifiability -> K3 pruning -> single-witness K4 negative -> multi-witness fix -> fresh label-blind confirmation`

The single-witness node should be explicitly marked **negative**.

### Figure 3 — Fresh confirmation coverage

Show per-seed full-history and pairwise coverage. Do not plot abstentions as errors; distinguish `certified`, `ambiguous/abstain`, and `contradiction` (zero).

### Figure 4 — Certificate geometry

Illustrate a target point, a wrong-class convex hull, several frozen witness directions, and an L1-safe combined direction. This is conceptual, not a literal projection of model weights unless generated from the frozen artifact.

### Figure 5 — Complexity/exactness boundary

Show fixed-K polynomial representation on the left, terminal K=N exactness on the right, and the omitted-tail/information barrier between them.

## Related-work comparison axes

The paper should compare by **problem and access**, not by claiming mathematical ingredients are absent elsewhere.

| Work family | Known training transcript? | Forward vs inverse | Output | Typical access |
| --- | --- | --- | --- | --- |
| curriculum / data-order optimization | candidate order chosen by method | forward | preferred schedule | gradients/loss/HVPs |
| Lie-bracket transfer-order planning | candidate domains known | forward | preferred pair/order | model derivatives |
| membership inference | examples known | inverse inclusion | used/not used | black-box or white-box |
| model lineage/provenance | candidate ancestor/reference known | inverse ancestry | dependence/lineage | black-box or weights |
| palimpsestic order provenance | disclosed randomized training order used as watermark/statistic | inverse dependence | evidence of derivation | query/text |
| **ChronoTrace** | candidate stage identities known; chronology hidden | **inverse chronology** | certified precedence / total order / abstention | replay-capable weights |

## Submission-facing limitations

Keep these concise but explicit:

1. The decisive confirmation is Pythia-14M, N=4, terminal K=N=4.
2. Candidate stage identities and replay procedure are known.
3. The method currently operates in weight space under deterministic plain-SGD-style controlled stages.
4. Exact fixed-K certification for N>K needs additional omitted-tail control.
5. Confirmation covers four fresh codebooks and eight balanced histories per codebook, not arbitrary natural post-training pipelines.

These limitations narrow the claim; they do not erase the contribution.
