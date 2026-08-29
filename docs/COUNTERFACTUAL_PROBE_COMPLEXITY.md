# Counterfactual Probe Complexity — Rank Is Not Minimum Probe Count

Status: methodology correction frozen before completion of the first Pythia susceptibility pilot.

The response-matrix rank and the minimum number of physical forensic probes answer different questions.

Let `H` have candidate histories as rows and available physical response coordinates as columns.

## Linear-span dimension

If `rank(H)=r`, then some `r` columns span every response column. Equality of two candidate rows on those basis columns therefore implies equality on the entire response family.

This is a valid sufficient upper bound on a distinction-preserving physical column set.

It is **not** the minimum physical probe count.

A rank-2 matrix such as

`[[0,0], [1,1], [2,4]]`

already has one physical column `[0,1,2]` that distinguishes every candidate. Thus rank can strictly exceed the smallest number of physical probes needed for finite identification.

If arbitrary linear combinations of response coordinates are allowed as measurements, finite separation can be compressed still further; therefore rank must not be marketed as the number of forensic questions required.

## Exact finite physical-probe problem

For a fixed finite physical probe family, define the full indistinguishability relation

`E(H) = {(i,k): ||H_i-H_k|| <= tau}`.

The exact finite minimum physical-probe set is

`min |J|  subject to  E(H[:,J]) = E(H)`.

For exact arithmetic (`tau=0`), each physical coordinate separates some subset of candidate pairs, so this is a pair-separation set-cover problem. For positive Euclidean tolerance, combined coordinate energy matters and the repository implementation checks subsets directly.

`minimum_distinguishing_probe_subset` performs exhaustive search in increasing subset size for small physical response families and returns a mathematically exact minimum under the supplied tolerance.

## Reporting rule

Future response experiments should distinguish:

1. response-matrix rank — linear response-span dimension;
2. minimum physical distinguishing subset — exact finite number of available coordinates required to preserve all distinctions;
3. minimum pair separation — numerical robustness of those distinctions;
4. decoder accuracy against approximate candidate models — whether the identifiable candidate geometry actually matches real target histories.

None of these quantities alone establishes a breakthrough. The stronger target remains behavioral active observability that improves chronology inference beyond passive behavior and survives finite-continuation validation.