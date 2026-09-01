# Literature Map

This is a working map, not a complete systematic review. Verify bibliographic details before paper submission.

## 1. Sequential learning and training order

**Question:** Does order change the endpoint?

Relevant lines of work include curriculum learning, continual learning, catastrophic forgetting, task-order effects, data-order studies, and recent geometric analyses of sequential updates.

Key distinction for ChronoTrace: these works mainly study the forward effect of order. ChronoTrace studies inference of an unknown order from the endpoint.

Examples already reviewed include:

- optimal task-order work in continual learning;
- structured / anticipatory recovery from interference;
- fine-grained LLM data-order scheduling;
- pretraining-data-order effects on language-model learning dynamics;
- Lie-bracket / noncommutative analyses of sequential update order.

None of those forward results makes “order matters” a ChronoTrace novelty claim.

## 2. Membership inference

**Question:** Was example or document X in training?

Methods such as Min-K%, document-level membership inference, ReCaLL, and related contamination detectors show that training exposure can leave post-hoc behavioral signals.

Key distinction: membership is about inclusion. ChronoTrace holds inclusion fixed and changes ordering.

## 3. Model provenance and lineage

**Question:** Is model B descended from model A, or did a target use a particular randomized training run?

Relevant work includes black-box palimpsestic membership inference and white-box checkpoint-lineage signatures.

Key distinction: lineage establishes ancestry. ChronoTrace aims to reconstruct latent events within a training trajectory.

### Strong novelty threat: palimpsestic membership inference

Kuditipudi et al., **“Blackbox Model Provenance via Palimpsestic Membership Inference”** (NeurIPS 2025; arXiv:2510.19796) is especially close and must be treated as a primary baseline rather than a passing citation.

It demonstrates that language models retain statistically detectable correlations with the **known example-level order** of a randomized training transcript. The signal can survive later fine-tuning and, in controlled TinyStories experiments, can remain detectable across multiple later epochs. This is direct evidence that later training does not necessarily erase earlier temporal information.

ChronoTrace therefore must **not** claim that it is the first work to show that training order leaves a persistent endpoint trace.

The intended distinction is sharper:

1. Palimpsestic provenance starts from a known ordered transcript and tests whether a target artifact is statistically dependent on that particular randomized run.
2. ChronoTrace treats the macro training-stage chronology itself as the unknown variable to infer.
3. ChronoTrace holds the candidate stage-data multiset fixed.
4. The current mechanism program asks what interaction order is required to reconstruct or constrain that hidden chronology.
5. A later black-box regime should target chronology-sensitive cross-stage interactions rather than disclosed per-example training-position correlation.

A successful paper must empirically compare or adapt the palimpsestic idea where possible. A result explained entirely by recency, per-example likelihood gradients, or known-transcript correlation is not sufficient for the stronger ChronoTrace claim.

## 4. Training-data attribution

**Question:** Which data influenced an output or capability?

Influence functions, TRAK-like methods, fact tracing, counterfactual memorization, and newer training-data provenance methods study causal or supporting data.

Key distinction: ChronoTrace asks about the temporal/compositional path by which learning occurred, including histories with the same data multiset.

## 5. Training memory, process tensors, and optimizer-state transport

This area is now a **primary neighboring literature** and narrows several ChronoTrace claims.

### Process-Tensor Tomography of SGD

Sevetlidis & Pavlidis, **“Process-Tensor Tomography of SGD: Measuring Non-Markovian Memory via Back-Flow of Distinguishability”** (AISTATS 2026; arXiv:2601.16563) models training as a multi-time process mapping controlled instruments—batch choices, augmentations, optimizer micro-steps—to observables. It introduces a back-flow witness of observable-level non-Markovian memory and a causal break that resets optimizer buffers.

URL: https://arxiv.org/abs/2601.16563

Important consequences for ChronoTrace:

1. **“Training-History Tomography” is not a safe distinctive umbrella phrase.** Tomography is already explicitly used for training memory.
2. Treating training as a multi-time dynamical process over a latent state and observation channel is not novel by itself.
3. Showing that previous interventions affect later observables is not novel by itself.
4. Resetting optimizer state as a causal mechanism test is not novel by itself.

The current distinction remains material: the process-tensor paper asks whether controlled histories exhibit operational memory / non-Markovianity. Its stated contribution is a measurement witness under known interventions. It does not center post-hoc recovery of an **unknown macro-stage permutation** from a finished endpoint. Searches of the available full text for “recover”, “infer”, “unknown”, and “chronology” did not reveal such an inverse reconstruction objective. This is supportive evidence for a distinction, not proof of uniqueness.

### Training Memory in Deep Neural Networks survey

Sevetlidis & Pavlidis, **“Training Memory in Deep Neural Networks: Mechanisms, Evidence, and Measurement Gaps”** (arXiv:2601.21624) organizes training memory by optimizer moments/averaging, data-order policies, nonconvex path, and auxiliary state, and advocates paired interventions and explicit audit artifacts.

URL: https://arxiv.org/abs/2601.21624

Consequences:

- the taxonomy “optimizer memory + data order + path geometry + auxiliary state” is **not** a ChronoTrace novelty;
- carrying/resetting optimizer state and order-window swaps are established measurement primitives;
- ChronoTrace should cite this survey when motivating the need to distinguish chronology channels.

### Optimizer-state transport

Xu, **“Stored in Optimizer State, Valued by Later Training: A Causal Account of Subliminal Trait Transfer”** (arXiv:2608.20442, Aug. 20 2026) treats parameters and Adam moments as one trainer state, identifies first-moment state as a causal carrier, and shows that later training routes can assign different behavioral value to the same stored ancestry.

URL: https://arxiv.org/abs/2608.20442

Guo, **“Delayed Optimizer-State Transport Shapes Short-Horizon Training Decisions”** (arXiv:2608.24593, Aug. 25 2026) studies complete model-plus-optimizer-state transport through committed future minibatch paths and shows that optimizer memory can change short-horizon schedule decisions.

URL: https://arxiv.org/abs/2608.24593

Consequences:

- “optimizer state stores training history” is definitely not novel;
- “future training conditions the value of an earlier perturbation” is also occupied;
- a future ChronoTrace Adam experiment must be framed as **how optimizer memory changes inverse chronology identifiability**, not as discovery of optimizer memory itself.

### What remains different

These training-memory works strengthen the dynamical-state motivation but also force a narrower contribution boundary:

> **Given candidate semantic training stages and a finished model, infer the previously unknown macro-stage chronology or partial order, and characterize the interaction degree / observation access required for that inverse problem.**

The strongest current theory-specific distinction is **Training-History Interaction Order**: whether degree-2, prefix-conditioned degree-3, or deeper interactions are required to distinguish candidate histories. This remains a candidate contribution and must continue to be searched explicitly.

## 6. Unlearning fingerprints

**Question:** Does an intervention such as unlearning leave a detectable trace?

Recent results indicate that model interventions can remain detectable in behavior or representations after the target behavior changes.

This supports the broader path-persistence hypothesis: endpoints may preserve information about optimization events, not only current capabilities.

However, intervention detectability is not the same as reconstruction of the relative chronology among multiple ordinary training stages.

## 7. Current claim boundary

The following claims are **already occupied or too broad** and should not be used as the main novelty statement:

- “training order matters”;
- “training order leaves an endpoint trace”;
- “training has memory”;
- “optimizer state carries history”;
- “later training can transform the effect of earlier training”;
- “training can be viewed as a multi-time dynamical process / tomography problem.”

The current candidate contribution is instead:

> **Post-hoc identification or partial-order reconstruction of unknown semantic macro-stage chronology from a finished model, using a graded hierarchy of order-sensitive training interactions.**

Important access qualifiers must accompany that statement. The current Pythia mechanism experiments are in a white-box simulator regime with a known base checkpoint, known candidate stages, known training rule, replay access, and final weights. They are not yet black-box provenance.

A stronger eventual contribution would show that low-order interaction coordinates can reconstruct a factorial chronology space using polynomially many training probes, while reporting probe complexity separately from combinatorial inference complexity.

## 8. Closest novelty threats to monitor

Search continuously for work using terms such as:

- training-history reconstruction;
- training-order inference;
- inverse sequential learning;
- inverse process tensor training;
- optimization trajectory forensics;
- curriculum inference from model weights;
- model archaeology;
- post-hoc training pipeline reconstruction;
- learning-path provenance;
- task-order identification;
- stage-order fingerprinting;
- temporal model provenance;
- chronology inference after continued training;
- training-order identification under matched endpoint behavior;
- interaction-order reconstruction;
- prefix-conditioned training-history inference;
- path-signature inversion of optimization trajectories.

Any paper that infers an *unknown ordinary training-stage order from only the endpoint*, especially after common continued training or under matched endpoint behavior, is a direct novelty threat and must be reviewed immediately.
