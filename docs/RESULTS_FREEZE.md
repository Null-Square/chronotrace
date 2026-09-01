# ChronoTrace Final Results Freeze

**Freeze date:** 2026-09-01  
**Status:** research method frozen; manuscript phase  
**Primary selection:** `configs/chronotrace_pairwise_multi_witness_confirmation_v3.selection.json`

This document is the paper-facing result ledger. It is intentionally shorter than the append-only research journal and should be updated only to correct factual/provenance errors, never to retune scientific thresholds after confirmation.

## A. Final confirmatory result

Scientific run:

```text
run id:          33418210637
scientific head: 7107221c16a001a7974ca1b436d9cacd26145fe2
seed jobs:       4/4 success
```

Fresh seed set:

```text
2186192236
1368008047
92712904
1944430236
```

Frozen targets per seed:

```text
ABCD
BCDA
CDAB
DABC
DCBA
ADCB
BADC
CBAD
```

Aggregate:

```text
complete histories certified: 27 / 32 = 84.375%
pairwise precedences certified: 182 / 192 = 94.7916667%
full-history abstentions: 5
ambiguous pair decisions: 10
contradictory inferred pairs: 0
both orientations excluded: 0
invalid suite: false
preregistered tier: strong
```

Per seed:

| Seed | Full histories | Pair precedences | Ambiguous pairs |
| --- | ---: | ---: | ---: |
| 2186192236 | 6/8 | 43/48 | 5 |
| 1368008047 | 7/8 | 47/48 | 1 |
| 92712904 | 6/8 | 44/48 | 4 |
| 1944430236 | 8/8 | 48/48 | 0 |

Abstaining cases:

```text
2186192236: DABC -> AC, BC ambiguous
2186192236: CBAD -> AB, AC, BD ambiguous
1368008047: BCDA -> AD ambiguous
92712904:   BCDA -> AB ambiguous
92712904:   CBAD -> AB, AC, BC ambiguous
```

No abstaining case produced a contradictory certified precedence.

## B. Numerical/certificate integrity

Frozen aggregate checks:

```text
minimum excluded-orientation margin over guard = 6.015076349674094e-06
maximum projected reconstruction residual      = 6.765421556309548e-17
maximum terminal primal exactness error         = 4.063047641389428e-17
all terminal witness-hull exactness passed      = true
all corrected bounds sound in witness geometry  = true
all corrected bounds sound vs Euclidean vertices = true
all target active lifts replayed exactly        = true
```

The decision threshold itself remained `1e-6`; the reported minimum field is the excess margin over that guard.

## C. Artifact digests

| Seed | Artifact ID | ZIP SHA-256 | Raw result SHA-256 |
| --- | ---: | --- | --- |
| 2186192236 | 9768220564 | `78d9abc998364b5686bfdcb194ea44e2c8e514fa5623c95a1317559cfad59dcc` | `5e3c45a5a15e9aca7d359f0106919771b49b15d9d10d07a9150d6a4c5752610e` |
| 1368008047 | 9768257657 | `0df6ab12047ff36a2c54ef4e1cb7966fb372cf27423e3761347e097e00e0eb96` | `1ec1370652c86386959e27b0300d9c526cce55401ff6e5a51c043e9611553306` |
| 92712904 | 9768112808 | `1830a159945aa50b5f4c4e71fe15bbf047cb9dba4e3c96c09fabc81498d4e668` | `c4d5e5b16fb2c02f5dddf46f74b25ee487f4032405b45adfe825b5bc63034228` |
| 1944430236 | 9768224175 | `fd6a5a0126507d7f7448ad4effafb5d202a988175725bb9e3c83e612530a0006` | `69dd387ace07ecf71d6f2f0b73206dc282d635ec36bebe52c7da82165ec3e37e` |

## D. Aggregation correction provenance

The initial post-science aggregator failed only because it compared JSON object iteration order with the frozen target order. The runner emits JSON using sorted keys, so dictionary order is not the experimental target order.

```text
failed aggregate job: 99577612088
correction commit:     23f58008d856d4cdecfb8a6777e4d8fdf302741c
regression commit:     d38bceb55d2148980d3130d1228f35e5f4b26f97
regression CI:         33475012691 success
```

The correction requires exact target-key set/count and does not alter any scientific seed output, threshold, target, witness, LP, or classification boundary.

## E. Development result hierarchy

### E1. Controlled theory/mechanism

Established before the final confirmation:

- affine/quadratic chronology controls;
- local reset-SGD commutator scaling;
- exact ordered finite interaction decomposition;
- projected interaction commutation;
- proof-safe local-order LP correction;
- directional-tail certificate theorem;
- finite-query K-local information barrier.

### E2. Exact finite reachable mechanism

On the spent Pythia-14M four-stage system, enumerating all complete reachable histories recovered **24/24** histories with positive separation. This demonstrates endpoint identifiability in the controlled system but is factorial full-history enumeration and is not the scalable method claim.

### E3. K3 convex pruning

On spent target ABCD, the frozen K3 convex last-stage diagnostic eliminated final-stage classes A and B while C and D survived. No true chronology certificate was claimed at this point.

### E4. Preregistered single-witness K4 diagnostic

The repaired, valid K4 run was a **scientific negative**:

```text
C eliminated: false
D survives: true
strong success: false
scientific negative: true
```

The exact Euclidean distance to class C was nonzero, but the one frozen C witness had negative support on a C vertex. This isolated witness-direction adequacy as the bottleneck.

### E5. Post-hoc multi-witness development

Using only witnesses already frozen before K4 output and no additional Pythia calls, a safe L1 combination produced positive wrong-class bounds on the spent target. Extending the idea to pairwise precedence classes showed all six wrong orientations separated on the spent ABCD target. This was explicitly methodology development, not confirmation.

### E6. Label-blind freeze and fresh confirmation

Before fresh v3 execution, the interface was frozen to test **both** orientations for every pair without reference to the generating target. Fresh v3 then produced the final 27/32 and 182/192 result.

## F. Provenance correction: spent v1 seeds

The four seeds originally intended for confirmation were discovered to have been executed by an earlier v1 workflow. They are therefore **spent**:

```text
4294917749
3885207466
402469483
2000073798
```

The v1 numerical outputs were not used to tune the final v3 scientific method. Fresh v3 seeds were mechanically SHA-256-derived and frozen before execution.

See:

- `configs/chronotrace_pairwise_multi_witness_confirmation_v1_spent.provenance.json`
- `configs/chronotrace_pairwise_multi_witness_confirmation_v3.lock.json`
- `configs/chronotrace_pairwise_multi_witness_methodology_v3.lock.json`

## G. Claims permitted by this freeze

The paper may claim:

1. Chronology reconstruction can be formulated as an inverse problem over known candidate stage operators.
2. Exact ordered Möbius interactions give a finite graded representation of deterministic stage compositions.
3. Unit-witness banks can be combined with L1-safe coefficients to obtain conservative Euclidean separation lower bounds.
4. A proof-safe local-order LP can certify that a precedence class is impossible, with explicit abstention when it cannot.
5. At terminal `K=N=4`, the hierarchy is checked against exact permutation convex hulls.
6. On the frozen fresh Pythia-14M suite, ChronoTrace certifies 27/32 complete histories and 182/192 pair relations with zero contradictory certified relations.
7. Degree-`<=K` information alone cannot universally control unseen `K+1` behavior for arbitrary smooth one-step SGD without additional assumptions/information.

## H. Claims prohibited by this freeze

The paper must **not** claim:

- black-box chronology recovery;
- arbitrary unknown stage discovery;
- legal model ownership/provenance proof;
- universal exact polynomial-time chronology reconstruction;
- that Möbius inversion, Sherali-Adams/local marginals, convex duality, or L1 witness combination are individually novel;
- that the post-hoc multi-witness spent result is confirmatory;
- that the preregistered single-witness K4 run was positive;
- that fixed `K<N` is exact without an omitted-tail assumption or additional measurements.

## I. Research freeze rule

After this document, new experiments are not to be used to retune the final method for the current paper. Presentation-only analyses of already-frozen artifacts are permitted if they do not alter decisions or thresholds. Any genuinely new scientific experiment belongs to a separately named follow-up protocol/paper.