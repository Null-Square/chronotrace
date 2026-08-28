# Literature Map

This is a working map, not a complete systematic review. Verify bibliographic details before paper submission.

## 1. Sequential learning and training order

**Question:** Does order change the endpoint?

Relevant lines of work include curriculum learning, continual learning, catastrophic forgetting, task-order effects, and recent geometric analyses of sequential updates.

Key distinction for ChronoTrace: these works mainly study the forward effect of order. ChronoTrace studies inference of order from the endpoint.

## 2. Membership inference

**Question:** Was example or document X in training?

Methods such as Min-K%, document-level membership inference, ReCaLL, and related contamination detectors show that training exposure can leave post-hoc behavioral signals.

Key distinction: membership is about inclusion. ChronoTrace holds inclusion fixed and changes ordering.

## 3. Model provenance and lineage

**Question:** Is model B descended from model A, or did a target use a particular randomized training run?

Relevant work includes black-box palimpsestic membership inference and white-box checkpoint-lineage signatures.

Key distinction: lineage establishes ancestry. ChronoTrace aims to reconstruct latent events within a training trajectory.

### Strongest current novelty threat: palimpsestic membership inference

Kuditipudi et al., **"Blackbox Model Provenance via Palimpsestic Membership Inference"** (NeurIPS 2025; arXiv:2510.19796) is especially close and must be treated as a primary baseline rather than a passing citation.

It demonstrates that language models retain statistically detectable correlations with the **known example-level order** of a randomized training transcript. The signal can survive later fine-tuning and, in controlled TinyStories experiments, can remain detectable across multiple later epochs. This is direct evidence that later training does not necessarily erase earlier temporal information.

ChronoTrace therefore must **not** claim that it is the first work to show that training order leaves a persistent endpoint trace.

The intended distinction is sharper:

1. Palimpsestic provenance starts from Alice's known ordered transcript and tests whether a target artifact is statistically dependent on that particular randomized run.
2. ChronoTrace treats the macro training-stage chronology itself as the unknown variable to infer.
3. ChronoTrace's core benchmark holds the stage data multiset fixed and asks whether `A->B` versus `B->A` can be recovered after an identical common terminal stage.
4. The strongest ChronoTrace regime requires ordinary A/B endpoint capabilities to be matched before chronology decoding is considered valid.
5. ChronoTrace Order Witnesses target cross-stage interaction structure rather than correlation between likelihood and disclosed per-example training positions.

A successful paper must empirically compare or adapt the palimpsestic idea where possible. A result that can be explained entirely by ordinary recency, per-example likelihood gradients, or known-transcript correlation is not sufficient for the stronger ChronoTrace claim.

## 4. Training-data attribution

**Question:** Which data influenced an output or capability?

Influence functions, TRAK-like methods, fact tracing, counterfactual memorization, and newer training-data provenance methods study causal or supporting data.

Key distinction: ChronoTrace asks about the temporal/compositional path by which learning occurred, including histories with the same data multiset.

## 5. Unlearning fingerprints

**Question:** Does an intervention such as unlearning leave a detectable trace?

Recent results indicate that model interventions can remain detectable in behavior or representations after the target behavior changes.

This supports the broader path-persistence hypothesis: endpoints may preserve information about optimization events, not only current capabilities.

## 6. Current claim boundary

The broad claim **"training order leaves a detectable trace"** is already occupied and should not appear as the main novelty claim.

The current candidate contribution is:

> **Post-hoc identification of unknown semantic macro-stage chronology under endpoint capability equivalence.**

A stronger extension is to measure how this identifiability decays under increasing common terminal training, producing a **training-history half-life / provenance-survivability curve**.

If the common terminal stage removes both capability differences and chronology signal, that is evidence for an identifiability boundary rather than a positive provenance result. If chronology remains decodable after capability matching, that is the result to validate on fresh seeds, additional model sizes, and alternative synthetic task families.

## 7. Closest novelty threats to monitor

Search continuously for work using terms such as:

- training-history reconstruction;
- training-order inference;
- inverse sequential learning;
- optimization trajectory forensics;
- curriculum inference from model weights;
- model archaeology;
- post-hoc training pipeline reconstruction;
- learning-path provenance;
- task-order identification;
- stage-order fingerprinting;
- temporal model provenance;
- chronology inference after continued training;
- training-order identification under matched endpoint behavior.

Any paper that infers an *unknown ordinary training-stage order from only the endpoint*, especially after common continued training or capability matching, is a direct novelty threat and must be reviewed immediately.
