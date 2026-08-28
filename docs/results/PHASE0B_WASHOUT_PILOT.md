# Phase-0b Shuffled-Union Washout Pilot

Status: **design-only negative result**

Workflow run: `33207229999`

Model: `EleutherAI/pythia-70m-deduped` at `step143000`

Histories: `ABC` versus `BAC`

Design seeds only: `13, 23, 29`

Confirmation seeds touched: **no**

## Question

Can an identical balanced terminal rehearsal stage C remove the direct A/B capability differences that confounded Phase-0 v1 while leaving a recoverable training-order signal?

For this pilot, C was a deterministic shuffled union containing exactly one copy of every A and every B training example. The candidate durations were fixed before the result: `50`, `150`, and `300` optimizer updates.

A candidate qualified only if **every** matched seed satisfied both:

- absolute ABC-vs-BAC A-control mean-margin gap `<= 1.0`;
- absolute ABC-vs-BAC B-control mean-margin gap `<= 1.0`.

Forensic performance was not allowed to select the candidate.

## Aggregate result

| C steps | Capability gate | Worst paired gap | Capability-only BA | Capability-only AUROC | Order-Witness BA | Order-Witness AUROC |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 50 | fail | 12.8486 | 1.000 | 1.000 | 1.000 | 1.000 |
| 150 | fail | 5.4859 | 0.833 | 1.000 | 0.833 | 0.889 |
| 300 | fail | 9.4641 | 0.833 | 0.778 | 0.500 | 0.444 |

Selected C: **none**.

## Paired capability gaps

### C = 50

| Seed | A-control gap | B-control gap | Pass |
| ---: | ---: | ---: | --- |
| 13 | 0.4832 | 9.5435 | no |
| 23 | 6.0746 | 12.8486 | no |
| 29 | 0.7171 | 6.2722 | no |

### C = 150

| Seed | A-control gap | B-control gap | Pass |
| ---: | ---: | ---: | --- |
| 13 | 1.5487 | 5.4859 | no |
| 23 | 5.4441 | 0.5820 | no |
| 29 | 2.5897 | 1.9447 | no |

### C = 300

| Seed | A-control gap | B-control gap | Pass |
| ---: | ---: | ---: | --- |
| 13 | 4.4397 | 9.4641 | no |
| 23 | 2.3425 | 4.1213 | no |
| 29 | 0.5533 | 1.2718 | no |

## Interpretation

This pilot does **not** establish path identifiability under endpoint equivalence.

The most informative observation is the separation at C=300:

- the contextual Order-Witness detector is at chance (`0.500` balanced accuracy);
- the capability-only baseline still classifies chronology above chance (`0.833` balanced accuracy);
- the frozen capability-equivalence gate still fails.

Therefore the shuffled-union common tail can erase the current interaction-specific forensic feature **before** it produces robust A/B endpoint equivalence.

The capability gaps are also non-monotonic in C. The worst gap improves from C=50 to C=150 and then worsens at C=300. This means that simply extrapolating to a longer shuffled terminal stage is not a principled next experiment.

## Next design

Phase-0c uses **Balanced Joint Washout (BJW)** instead of shuffled-union sampling.

Every terminal optimizer step will contain equal numbers of matched A and B examples from the same synthetic worlds/templates. Both histories receive the exact same sequence of joint minibatches. This targets local terminal-gradient asymmetry directly rather than relying on aggregate corpus balance.

Phase-0c remains design-only until the same frozen `<= 1.0` capability gate is satisfied. Fresh discovery and confirmation seeds remain untouched.
