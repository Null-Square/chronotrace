# Post-hoc audit: logical closure and terminal discrete baselines

Date: 2026-09-12. Status: **post-hoc reanalysis, not a new model-training confirmation**.

Audited branch commit: `4e71d4f8257293c250a9fbe10390a976525a7520`.
Original scientific run: `33418210637`, commit `7107221c16a001a7974ca1b436d9cacd26145fe2`.

The four original Actions artifact ZIP digests and result JSON digests matched the frozen v3 selection. The added script verifies the JSON hashes again and reproduces the original aggregate before evaluating new post-processing. No frozen protocol, selection, manuscript headline, or scientific engine is changed by this audit.

## Results

| Procedure | Complete histories / 32 | Decided or entailed pairs / 192 | Wrong inferred pairs |
|---|---:|---:|---:|
| Original frozen multi-witness certificate | 27 | 182 | 0 |
| Logical closure of original certified edges | 28 | 183 | 0 |
| Stored discrete witness-distance baseline | 32 | 192 | 0 |
| Stored exact Euclidean vertex-distance oracle | 32 | 192 | 0 |

The first row remains the original confirmation. All other rows are newly computed post-hoc analyses. They require zero additional training-stage calls and zero additional LP solves. The stored discrete minima were not independently recomputed from endpoint tensors; the downloadable JSONs do not retain those tensors. This is a stored-statistic reanalysis, not an independent rerun of training or projection measurements.

## Logical closure repair

The frozen `_reconstruct_total_order` requires all six pairs to be directly certified. In seed `92712904`, case `BCDA`, the certified edges include `B<C`, `C<D`, and `D<A`. Therefore `B<A` follows by transitivity even though the pair-specific LP abstained. There is exactly one compatible full order.

`close_certified_precedences` now returns all entailed relations, proof paths consisting only of original edges, and a full order only when the topological ordering is unique. Global cycles are explicitly invalid. It performs graph traversal rather than factorial enumeration. Soundness remains conditional on the input edges being sound.

This is a correctness/completeness improvement, not a new mathematical novelty claim. It rescues one full history and one pair without changing a certificate threshold. Do not relabel 28/32 as the old protocol's preregistered outcome tier; the original 27/32 result stays immutable.

## Why the discrete baseline matters

For a chronology class H and fixed unit witnesses u_j, define

`d_vertex(H) = min_{pi in H} max_j |<u_j, y-theta_pi>|`.

In exact arithmetic,

`d_hull(H) <= d_vertex(H) <= min_{pi in H} ||y-theta_pi||_2`.

Thus discrete witness distance is a valid class-distance lower bound and dominates its convex-hull relaxation once complete candidate endpoints have already been acquired. This does not negate the usefulness of relaxations before enumeration; it identifies the correct same-information terminal baseline.

The frozen engine already saves `left_before_right_nearest_vertex_proxy` and `right_before_left_nearest_vertex_proxy` in every pair result. The audit tests both orientations using the frozen `1e-6` elimination threshold, subtracts `1e-10`, and normalizes by the largest recorded witness norm when greater than one. Ground-truth labels and evaluation fields are excluded from the decision-only interface and consulted only after all predictions have been fixed.

All ten original ambiguous pair decisions have zero stored wrong-class hull distance but nonzero wrong-class discrete separation. Across all cases, the minimum normalized wrong-class discrete distance is approximately `3.5391e-4`; stored true-class discrete distances are zero. The stored Euclidean vertex comparator also resolves all 32 histories.

A fixed floating-point guard is not a formal interval-arithmetic error bound. The numbers establish a clear stored-statistic comparison on this frozen dataset, not a universal zero-error guarantee or a validated tolerance to model export noise.

## Replay accounting and research implication

Per codebook: 40 calls acquire ordered prefixes through K=3, 32 calls generate eight independent benchmark targets, and 24 calls acquire all terminal extensions. Total: 96. Candidate/probe acquisition alone is **64**, exactly the count for exhaustive N=4 prefix-cached acquisition. Target generation should be reported separately for every method.

The current terminal N=K=4 result validates a controlled certificate mechanism; it does not demonstrate a probe-efficiency advantage over exact replay. Its confirmation also uses Pythia-14M, FP64 deterministic SGD, a known base and stage recipes, and **one update per stage** at learning rate `1e-4`. Four codebooks are not 32 independent model initializations or 192 independent pairwise statistical trials.

The next decisive experiment should fix K=2 or K=3 while increasing N, forbid uncharged terminal candidate information, and compare coverage at equal stage-call/optimizer-update budgets. Add finite-length training stages and natural task families before broad deployment claims. The existing omitted-interaction information barrier still applies: low-degree prediction bounds are not endpoint certificates without justified tail control.

A promising proposed extension is adaptive disjunctive refinement: close existing edges, split only unresolved chronology classes, tighten child bounds, and stop at a budget with explicit ambiguity. Terminal singleton classes recover the discrete baseline. This direction has not been implemented in this audit and has factorial worst-case complexity.

## Prior-art update

The novelty claim must account for **Fresh in memory: Training-order recency is linearly encoded in language model activations** (Krasheninnikov, Turner, Krueger; arXiv:2509.14223, September 2025). It is a close inverse-temporal neighbor, including activation probes for acquisition time; it is not merely forward curriculum optimization.

Also distinguish ChronoTrace from **Blackbox Model Provenance via Palimpsestic Membership Inference** (arXiv:2510.19796), **Commute Your Domains** (arXiv:2501.15556), and **The Geometry of Sequential Learning: Lie-Bracket Prediction of Transfer Order** (arXiv:2606.24993). Training-order traces, temporal probes, and commutator geometry cannot independently be the novelty. The stronger target is budgeted unknown-order reconstruction with explicit ambiguity, access assumptions, and nonterminal empirical advantage.

## Reproduce

```bash
python -m pytest -q tests/test_certificate_postprocess.py
python scripts/audit_chronology_postprocess.py \
  --raw-root /path/to/extracted/original/artifacts \
  --output posthoc_reanalysis.json
```

Expected original artifacts: IDs `9768220564`, `9768257657`, `9768112808`, `9768224175`, one per v3 seed. JSON SHA-256 values are pinned in the script and match the original frozen selection.

The new unit suite passed **762 tests**, including all 729 four-stage pair assignments checked against an exhaustive independent permutation oracle, cycle rejection, proof paths, malformed inputs, and nonfinite scalar rejection. The full existing repository suite and model training were not rerun during this audit.
