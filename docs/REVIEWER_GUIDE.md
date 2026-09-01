# ChronoTrace Reviewer Guide

This guide is the shortest path through the repository for a scientific reviewer. The repository contains a long append-only development record; this file distinguishes the **final frozen claim** from historical experiments that motivated it.

## 1. Paper claim in one sentence

ChronoTrace formulates unknown training chronology as an inverse problem over ordered training operators and constructs **proof-safe precedence certificates** from ordered Möbius interactions; in a frozen fresh Pythia-14M `N=4,K=4` confirmation, it certifies 27/32 complete histories and 182/192 pairwise precedences, with five conservative abstentions and zero contradictory pair inferences.

## 2. Access regime

The confirmed result is **replay-capable white-box**. The auditor knows:

- the base Pythia checkpoint;
- the four candidate stage datasets/operators;
- the one-update plain-SGD rule;
- the numerical execution protocol;
- the final model weights;
- and can replay controlled stage-map extensions.

The chronology is hidden from the certificate decision rule. This is not a black-box provenance or ownership result.

## 3. Final frozen evidence

Primary selection:

- `configs/chronotrace_pairwise_multi_witness_confirmation_v3.selection.json`
- selection commit: `8ed5c7deda81080200d5ca5b2de01ed7f31b94d7`
- scientific run: `33418210637`
- scientific head: `7107221c16a001a7974ca1b436d9cacd26145fe2`

Headline metrics:

| Quantity | Result |
| --- | ---: |
| fresh seed jobs | 4 / 4 successful |
| complete-history cases | 32 |
| complete histories certified | **27 / 32** |
| pair decisions | 192 |
| pairwise precedences certified | **182 / 192** |
| complete-history abstentions | 5 |
| ambiguous pair decisions | 10 |
| contradictory pair inferences | **0** |
| both orientations excluded | **0** |
| invalid suite | **false** |
| preregistered tier | **strong** |

Per-seed complete-history coverage:

```text
2186192236  6/8
1368008047  7/8
92712904    6/8
1944430236  8/8
```

The ten ambiguous pair decisions are concentrated in five cases. They are abstentions, not incorrect certified relations.

## 4. Fresh-seed provenance

An earlier v1 workflow had already executed the four seeds originally reserved as held-out. That was discovered before the v3 confirmation claim was made. Those seeds were formally marked spent and **their numerical outputs were not used to tune v3**.

The final v3 seeds were mechanically generated using the frozen SHA-256 rule recorded in:

- `configs/chronotrace_pairwise_multi_witness_confirmation_v3.lock.json`
- `configs/chronotrace_pairwise_multi_witness_methodology_v3.lock.json`
- `configs/chronotrace_pairwise_multi_witness_confirmation_v1_spent.provenance.json`

Fresh v3 seeds:

```text
2186192236
1368008047
92712904
1944430236
```

All four were launched together from one immutable marker. No method, threshold, target set, or decision rule was changed after the fresh run began.

## 5. Artifact integrity

The final scientific artifacts are:

| Seed | Artifact | ZIP digest |
| --- | ---: | --- |
| 2186192236 | 9768220564 | `sha256:78d9abc998364b5686bfdcb194ea44e2c8e514fa5623c95a1317559cfad59dcc` |
| 1368008047 | 9768257657 | `sha256:0df6ab12047ff36a2c54ef4e1cb7966fb372cf27423e3761347e097e00e0eb96` |
| 92712904 | 9768112808 | `sha256:1830a159945aa50b5f4c4e71fe15bbf047cb9dba4e3c96c09fabc81498d4e668` |
| 1944430236 | 9768224175 | `sha256:fd6a5a0126507d7f7448ad4effafb5d202a988175725bb9e3c83e612530a0006` |

Raw result SHA-256 values are included in the frozen selection file.

## 6. Why the first aggregate job failed

All four scientific seed jobs completed successfully and uploaded evidence. The first aggregate job then failed with:

```text
fresh confirmation target order/coverage drift
```

The seed JSON files are emitted with sorted keys, but the v3 aggregator incorrectly compared dictionary iteration order with the preregistered target order. Object-key order is semantically irrelevant.

Correction:

- code fix: `23f58008d856d4cdecfb8a6777e4d8fdf302741c`
- regression test: `d38bceb55d2148980d3130d1228f35e5f4b26f97`
- CI: `33475012691`, success

The correction requires the exact target **key set and count**, and changes no seed result, scientific method, threshold, target history, or classification tier.

## 7. Certificate construction

### 7.1 Ordered interaction basis

For a deterministic stage word `w`, let `E(w)` be the endpoint from the common base `theta_0`. The exact ordered interaction is recursively

```text
Phi(w) = E(w) - theta_0 - sum_{u proper ordered subword of w, u nonempty} Phi(u).
```

Thus

```text
E(w) = theta_0 + sum_{u ordered subword of w, u nonempty} Phi(u).
```

For `|w| <= K`, the degree-`K` truncation is exact.

Implementation:

- `src/chronotrace/geometry/interactions.py`
- streaming exact-prefix measurement tests: `tests/test_streaming_exact_interactions.py`

### 7.2 Low-degree witness bank

Before any K4 candidate output is observed, the runner freezes four unit K3 witnesses, one per final-stage class. The witness bank is therefore pre-output with respect to the higher-order active lifts used in the final certificate.

### 7.3 Multi-witness lower bound

For unit witnesses `u_j` and coefficients `alpha` with `||alpha||_1 <= 1`,

```text
v = sum_j alpha_j u_j
```

satisfies `||v||_2 <= 1`. Therefore for every residual `r`,

```text
||r||_2 >= <v,r>.
```

The coefficient vector can be optimized numerically, but final certification independently recomputes the support value after L1 normalization.

Implementation:

- `src/chronotrace/geometry/multi_witness.py`
- `src/chronotrace/geometry/multi_witness_local_order.py`
- `tests/test_multi_witness.py`
- `tests/test_multi_witness_local_order.py`

### 7.4 Local-order relaxation and proof-safe dual

For fixed `K`, ChronoTrace uses nonnegative K-local order marginals with simplex and adjacent-level marginal-consistency constraints. A combined witness objective is minimized over the candidate precedence class. The LP solver proposes a dual, then ChronoTrace corrects the numerical dual with subset-wise reduced-cost minima so the reported bound remains conservative even if the solver dual has small feasibility violations.

Implementation:

- `src/chronotrace/geometry/local_order_hierarchy.py`
- `src/chronotrace/geometry/local_order_lp.py`
- `src/chronotrace/geometry/pairwise_certificate.py`

### 7.5 Label-blind pair decision

For each unordered pair `{i,j}`, the method independently certifies both classes:

```text
i < j
j < i
```

without using the generating chronology. Exactly one excluded class implies the opposite precedence. Neither excluded means **abstain**. Both excluded invalidates the result. A complete history is returned only when all six pair orientations are inferred and form a transitive total order.

## 8. Exactness checks at N=4,K=4

The final confirmation is terminal: `K=N=4`. At this level the local-order hierarchy is exact for the permutation convex hull. Every orientation-class LP primal is checked against an independent convex-hull solve over the complete permutations in that class.

The frozen suite reports:

```text
maximum projected reconstruction residual = 6.765421556309548e-17
maximum terminal primal exactness error    = 4.063047641389428e-17
all terminal witness-hull checks           = true
all corrected witness-space bounds sound   = true
all Euclidean-vertex checks sound           = true
all target active lifts replay exactly      = true
```

## 9. Important negative results

A reviewer should not interpret the final method as a sequence of only successful experiments.

- The original behavioral detector was confounded by capability/recency.
- Static low-order finite-pair decoding failed on finite Pythia stages.
- Reverse/inverse peeling suffered inversion failures.
- The first K3 affine certificate did not prune candidate final stages strongly enough.
- A frozen K3 convex certificate pruned A/B but left C/D.
- The preregistered one-witness K4 diagnostic was a **scientific negative**: C survived despite having nonzero exact Euclidean separation.
- Post-hoc analysis showed the failure was witness direction adequacy, not missing K4 information; a combination of already-frozen witnesses separated C.

The single-witness negative is retained as negative. Multi-witness development is labeled post-hoc on the spent seed.

## 10. Scalability boundary

The fixed-K local hierarchy uses

```text
sum_{r=1}^K P(N,r)
```

coordinates, polynomial in `N` for fixed `K`. Pair property queries are `O(N^2)`.

However, the final exact Pythia confirmation uses `K=N=4`, so interaction acquisition is terminal/factorial. For `N>K`, exact endpoint certification requires controlling omitted interactions above K.

`docs/K_LOCAL_INFORMATION_BARRIER.md` proves a finite-query obstruction for arbitrary smooth one-step SGD: degree-`<=K` observations alone cannot universally determine an unseen `K+1` directional tail without additional assumptions or information.

The paper therefore claims a **certificate hierarchy with an exact terminal level**, not an assumption-free polynomial exact full-history decoder for arbitrary N.

## 11. Minimal code review path

Recommended order:

1. `src/chronotrace/geometry/interactions.py`
2. `src/chronotrace/geometry/projected_interactions.py`
3. `src/chronotrace/geometry/local_order_hierarchy.py`
4. `src/chronotrace/geometry/local_order_lp.py`
5. `src/chronotrace/geometry/multi_witness.py`
6. `src/chronotrace/geometry/multi_witness_local_order.py`
7. `src/chronotrace/geometry/pairwise_certificate.py`
8. `scripts/pythia_14m_pairwise_multi_witness_confirmation_v2.py` — unchanged scientific engine used by v3
9. `scripts/pythia_14m_pairwise_multi_witness_confirmation_v3.py` — provenance/fresh-seed wrapper
10. `scripts/analyze_pairwise_multi_witness_confirmation_suite_v3.py`

Then inspect the corresponding tests.

## 12. Local test path

For the package/proof logic:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
ruff check .
pytest
```

The model-scale confirmation itself depends on the pinned CPU PyTorch/Transformers stack and cached/downloadable model assets recorded in `.github/workflows/pythia-14m-pairwise-multi-witness-confirmation.yml` and the protocol locks.

## 13. Paper-facing files

- `paper/main.tex` — final manuscript source
- `paper/references.bib` — bibliography
- `paper/CLAIMS_AND_EVIDENCE.md` — exact claim ledger
- `paper/FIGURE_PLAN.md` — figure definitions and source data
- `paper/outline.md` — compact current outline; historical pre-result outline has been superseded

## 14. Historical record

The project intentionally keeps historical protocols, selections, invalid-attempt records, and research journal entries. They are evidence of methodology development and falsification, not dead code to be hidden from review.

Use `docs/RESEARCH_JOURNAL.md` when auditing the complete development chronology.