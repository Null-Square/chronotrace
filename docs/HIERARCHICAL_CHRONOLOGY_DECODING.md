# Hierarchical Chronology Decoding

Status: theory/design note. This does **not** change the frozen T2b experiment.

## Motivation

T2 produced an unusually structured result over 144 independent three-stage endpoints:

- exact full order: `72/144`;
- first stage: `144/144`;
- all `72/72` errors preserve the true first stage and swap only positions 2 and 3.

A binary metric such as exact-permutation accuracy calls half of these endpoints wrong even though five sixths of the pairwise precedence information and the entire first stage are recovered. For training-history forensics, that throws away useful information.

The inverse problem should therefore produce a **hierarchy of claims**:

1. which stage or stage-set is identifiable as early;
2. which pairwise precedence relations are robust;
3. how deep a prefix can be recovered;
4. only then whether the complete permutation is identifiable.

## Candidate endpoint model

Let `Pi_N` be all permutations of N candidate stages. Let an order-K interaction model predict endpoint

`theta_hat_pi^(K)`

for each candidate chronology `pi`.

For an observed endpoint `theta*`, define candidate error

`e(pi) = ||theta* - theta_hat_pi^(K)||`.

The ordinary decoder returns

`argmin_pi e(pi)`.

This requires making a total-order claim even when only a prefix is supported.

## Prefix groups

For a prefix `p=(p_1,...,p_r)`, define

`G(p) = { pi in Pi_N : pi begins with p }`.

Define the prefix-group error

`E(p) = min_{pi in G(p)} e(pi)`.

For all prefixes of the same length r, rank them by `E(p)`.

The best prefix of depth r is

`p_r* = argmin_{|p|=r} E(p)`.

This is simply grouped nearest-signature decoding. It requires no new scientific information and makes explicit how much chronology the current representation actually supports.

## Prefix margin

Let `p_r*` be the best depth-r prefix and let `q_r*` be the second-best distinct prefix. Define

`M_r = E(q_r*) - E(p_r*)`.

A positive margin identifies the best prefix within the candidate model. A robust forensic claim additionally needs the omitted-interaction / numerical uncertainty to be small relative to this margin.

This suggests a **recoverable prefix depth**

`D*(theta*) = max r such that prefix p_r* remains certified/robust`.

For three stages:

- `D*=1` means the earliest stage is recoverable but the tail order is not;
- `D*=2` is already equivalent to the complete permutation because the final stage is forced;
- `D*=0` means even the first stage is not supported.

T2 is therefore naturally described as many endpoints with `D*=1`, not simply as failed full-order decoding.

## Pairwise partial order

Prefix recovery is only one kind of partial chronology. For candidate stages i and j, compare the best candidate consistent with `i<j` against the best candidate consistent with `j<i`:

`E(i<j) = min_{pi: i before j} e(pi)`

`E(j<i) = min_{pi: j before i} e(pi)`.

The edge margin

`M_ij = |E(i<j) - E(j<i)|`

measures how strongly the current representation supports one precedence relation over the other.

A set of robust directed edges forms a partial order. This is preferable to forcing a total order when some late-stage relations have collapsed.

## Hierarchical reconstruction algorithm

A prefix-adaptive decoder can proceed recursively:

1. score all one-stage prefix groups;
2. retain only prefixes whose error is within a predefined uncertainty/beam rule;
3. for each retained prefix, evaluate or approximate the interactions of remaining stages at the state induced by that prefix;
4. score the next prefix extension;
5. continue until the remaining order is not identifiable or all stages are placed.

This is conceptually different from the static base-checkpoint tournament. Later pair interactions are evaluated conditionally on the inferred earlier path.

## Why this may help computationally

Measuring all order-K interactions costs polynomially many stage-map probes for fixed K, but naive scoring of all N! permutations is still factorial. Hierarchical decoding creates a route to practical inference:

- prefix groups allow early pruning;
- beam search retains uncertainty rather than committing to one wrong total order;
- pairwise edge constraints can feed a topological-sort or ranking solver;
- conditioned probes can be allocated only to ambiguous branches.

This does **not** prove polynomial-time inference. Probe complexity and inference complexity remain separate claims.

A future scalable algorithm should report both:

- number of training-stage probe executions;
- number of chronology hypotheses scored/expanded.

## Connection to interaction order

Define `K*` as the minimum interaction degree required for a desired chronology claim.

A more informative object is

`K*(r, epsilon)`

= the minimum interaction degree needed to recover a prefix/partial order of depth r at robustness epsilon.

This allows a model to have, for example:

- reliable first-stage identity at K=2;
- reliable second-stage identity only at K=3;
- no stable complete ordering under the available observation projection.

That structure matches the T2 finding much better than a single yes/no notion of provenance.

## Falsifiable future tests

Conditional on T2b validating the local-to-finite asymptotic bridge, the four-stage experiment should report:

1. exact 4-stage permutation recovery;
2. depth-1, depth-2, and depth-3 prefix recovery separately;
3. robust pairwise precedence edges;
4. interaction degree needed for each depth;
5. probe executions and chronology hypotheses expanded;
6. whether prefix-conditioned degree-3 information resolves ambiguities that remain at degree 2.

If K=2 already gives exact four-stage order on every condition, the hierarchical claim is unnecessary in that regime. If even depth-1 prefixes fail under fresh instances, the T2 structure does not generalize beyond three stages. If K=3 improves deeper-prefix recovery while K=2 retains only coarse chronology, that would be direct evidence for a graded training-history interaction hierarchy.
