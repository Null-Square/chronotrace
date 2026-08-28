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

## 4. Training-data attribution

**Question:** Which data influenced an output or capability?

Influence functions, TRAK-like methods, fact tracing, counterfactual memorization, and newer training-data provenance methods study causal or supporting data.

Key distinction: ChronoTrace asks about the temporal/compositional path by which learning occurred, including histories with the same data multiset.

## 5. Unlearning fingerprints

**Question:** Does an intervention such as unlearning leave a detectable trace?

Recent results indicate that model interventions can remain detectable in behavior or representations after the target behavior changes.

This supports the broader path-persistence hypothesis: endpoints may preserve information about optimization events, not only current capabilities.

## 6. Closest novelty threats to monitor

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
- stage-order fingerprinting.

Any paper that infers an *unknown ordinary training-stage order from only the endpoint* is a direct novelty threat and must be reviewed immediately.
