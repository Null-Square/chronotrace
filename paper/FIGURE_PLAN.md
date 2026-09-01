# ChronoTrace Final Figure Plan

The final paper should use a small number of figures, each with a distinct scientific job. Avoid decorative figures that repeat the text.

Paper TikZ sources live under `paper/figures/`. Browser-visible reviewer versions live under `assets/`. Frozen-data-derived assets are regenerated with:

```bash
python scripts/generate_release_assets.py --write
python scripts/generate_release_assets.py --check
```

## Figure 1 — Inverse chronology certificate pipeline

**Purpose:** establish the problem/access regime and show what is new operationally.

Panels:

1. hidden permutation of known stage operators from a known base checkpoint;
2. released/final weight endpoint;
3. exact low-degree ordered interaction measurement and witness freezing;
4. higher-order witness-projected active lifts;
5. proof-safe pair-class LP certificates;
6. output: certified precedence edges, total order if complete, otherwise abstention.

Required visual labels:

- `known: base + candidate stages + training rule`
- `hidden: chronology`
- `observed: final weights`
- `decision is label-blind`
- `certificate may abstain`

Do **not** imply black-box access.

Sources:

- paper: `paper/figures/pipeline.tex`
- browser/README: `assets/chronotrace-pipeline.svg`

## Figure 2 — Scientific development ladder

**Purpose:** make the falsification history a strength rather than burying it.

Sequence:

```text
behavioral detector
  -> confounded negative
static finite-pair/K3
  -> structured failure
exact reachable all24
  -> identifiability, factorial
K3 convex certificate
  -> A/B pruned, C/D remain
single-witness K4 (preregistered)
  -> scientific negative
multi-witness on spent seed
  -> post-hoc method development
label-blind freeze + fresh v3
  -> 27/32 strong confirmation
```

Use explicit markers `negative`, `mechanism`, `post-hoc`, `confirmation`.

Source: `paper/figures/development_ladder.tex`.

## Figure 3 — Fresh confirmation coverage

**Purpose:** headline empirical evidence.

Per-seed bars:

```text
seed          histories   pairs
2186192236       6/8      43/48
1368008047       7/8      47/48
92712904         6/8      44/48
1944430236       8/8      48/48
```

Aggregate annotation:

```text
27/32 complete histories
182/192 pair relations
10 ambiguous pairs
0 contradictions
0 double-exclusions
```

The visualization must distinguish abstention from incorrect certification.

Sources:

- paper: `paper/figures/confirmation_coverage.tex`
- browser/README: `assets/chronotrace-results.svg`

## Figure 4 — Multi-witness certificate geometry

**Purpose:** explain why one witness can fail while a safe combination succeeds.

Conceptual geometry:

- target endpoint `y`;
- wrong-order convex hull `C_wrong`;
- two frozen unit witness directions `u_1,u_2` individually with nonpositive minimum support;
- L1-safe combination `v = alpha_1 u_1 + alpha_2 u_2`, `||alpha||_1 <= 1`;
- positive support gap `min_{q in C_wrong} <v,y-q> > 0`;
- conclude `dist(y,C_wrong)` is at least that gap.

This figure is conceptual unless generated from artifact projections. Caption must say so.

Source: `paper/figures/multi_witness_geometry.tex`.

## Figure 5 — Exactness/scalability boundary

**Purpose:** prevent overclaiming and turn the information barrier into a theorem-level contribution.

Horizontal hierarchy:

```text
K=1 -> K=2 -> ... -> fixed K<N -> ... -> K=N
```

Above:

- representation size `sum_{r<=K} P(N,r) = O(N^K)` for fixed K;
- pair property queries `O(N^2)`.

Below fixed `K<N`:

- omitted degree `>K` interactions;
- information barrier: no universal finite-query tail bound for arbitrary smooth one-step SGD.

At `K=N`:

- terminal exact permutation convexification;
- exact but factorial interaction acquisition.

Source: `paper/figures/complexity_boundary.tex`.

## Appendix Figure A1 — Case-level confirmation matrix

**Purpose:** show exactly where full-history abstentions occurred without collapsing them into an aggregate rate.

Rows are the four fresh seeds and columns are the eight frozen target histories. `C` denotes a complete certified chronology; `A` denotes a conservative full-history abstention. No cell represents an incorrect certified history.

Source: `paper/figures/confirmation_matrix.tex`.

This file is generated from the frozen selection and must not be edited by hand.

## Tables

### Table 1 — Related-work problem/access comparison

Columns: work family, known transcript/stage identities, forward/inverse, target question, access, certificate/score.

### Table 2 — Development ladder numerical results

Include static K3, exact all24, K3 convex, single-witness K4 negative, post-hoc spent multi-witness, fresh v3.

### Table 3 — Fresh v3 per-seed confirmation

The four seed rows and aggregate.

### Table 4 — Numerical validity checks

Maximum residual/exactness errors and all-pass booleans.

## Generated paper values

`paper/generated/results_macros.tex` contains frozen result macros generated from the v3 selection. It is intended for venue-template conversion so repeated headline values can be sourced from one file rather than retyped.

## Appendix figures

- case-level confirmation matrix: `paper/figures/confirmation_matrix.tex`;
- protocol/provenance timeline including spent v1 correction;
- exact terminal hull-vs-local-LP agreement diagnostics;
- synthetic K<N relaxation-gap examples.
