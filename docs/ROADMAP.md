# Roadmap

## R0 — Research scaffold

Status: **active / initialization**

- define the inverse sequential-learning problem;
- freeze terminology;
- define MVP falsification criteria;
- create reproducibility contracts;
- create paper skeleton.

## R1 — MVP path-persistence validation

- choose base model and stage construction;
- implement deterministic synthetic world generator;
- implement AB/BA trainer;
- train discovery endpoints;
- create fixed probe bank;
- extract simple behavioral features;
- fit seed-held-out history detector;
- train confirmation endpoints;
- run capability controls;
- publish the first validation report.

**Exit:** decide whether H1 survives.

## R2 — Mechanistic localization

- collect layer-wise representations;
- compare A-only, B-only, and A×B probes;
- test commutator-motivated features;
- identify where history information concentrates;
- test checkpoint-metadata and weight-space shortcut controls.

## R3 — Training-history half-life

- append matched continuation stage C;
- sweep continuation length;
- estimate history-signal decay;
- test whether different layers decay at different rates.

## R4 — Multi-stage tomography

- move from two stages to 3–5 stages;
- recover pairwise precedence;
- reconstruct total orders;
- extend to partial-order graphs and interleaving.

## R5 — Acquisition mechanisms

- direct exposure;
- distributed/mosaic exposure;
- rule acquisition;
- distillation;
- later fine-tuning;
- unlearning and relearning.

Study whether matched endpoint behavior retains mechanism-specific traces.

## R6 — Black-box transfer

- discover order witnesses on shadow models;
- remove access to target weights/logits progressively;
- test transfer across architectures and scales;
- characterize query complexity and calibration.

## R7 — Frontier-lab utility studies

Potential authorized applications:

- post-training pipeline audit;
- unlearning verification;
- capability-origin analysis;
- checkpoint governance;
- safety-stage ordering analysis;
- forensic comparison of internal experimental branches.
