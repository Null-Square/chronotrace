# Experiment Protocol

## Purpose

ChronoTrace results are unusually vulnerable to confounds. Random seed, checkpoint metadata, capability imbalance, probe reuse, and researcher iteration can all create false confidence. This protocol is mandatory for results intended for the paper.

## 1. Configuration is immutable per run

Each run starts from a versioned configuration. The run manifest stores the resolved configuration and repository commit SHA.

Do not edit a completed run in place.

## 2. Separate discovery and confirmation

Use two disjoint model-seed sets.

**Discovery set** may be used to:

- choose probes;
- select features;
- tune a classifier;
- inspect failure modes;
- form new hypotheses.

**Confirmation set** may be used only after the analysis pipeline and exclusion rules are frozen.

If the confirmation analysis changes after results are inspected, record the change and treat the old confirmation set as discovery data.

## 3. Record every run

Required manifest fields:

- run ID;
- UTC start/end time;
- git commit;
- Python and package environment;
- hardware summary;
- base model and revision;
- training history;
- all seeds;
- stage dataset identifiers and hashes;
- optimizer and scheduler;
- batch/token counts;
- precision;
- checkpoint hashes;
- output artifact hashes;
- status and failure reason.

## 4. Use deterministic data definitions

Data generation must be controlled by explicit seeds and versioned generator parameters.

Store hashes for generated stage files. A and B histories must point to the same immutable stage artifacts in matched experiments.

## 5. Prevent label shortcuts

Before fitting a detector, audit the feature table for direct or indirect labels:

- path names;
- file names;
- timestamps;
- run numbers that encode class;
- trainer metadata;
- different tensor serialization settings;
- hardware allocation patterns;
- differing checkpoint frequency.

The history label should be joined only after feature extraction when practical.

## 6. Capability matching

Evaluate A-only, B-only, and cross-stage tasks for every endpoint.

Report:

- mean and distribution by history;
- standardized effect sizes;
- a simple history classifier using only ordinary capability metrics.

A forensic detector must be compared against this capability-only baseline.

## 7. Statistics

The primary unit is the independently trained model, not an individual prompt. Thousands of prompts from one model do not create thousands of independent model samples.

Report at minimum:

- number of independent model runs;
- balanced accuracy;
- AUROC;
- uncertainty across model runs;
- bootstrap or exact confidence interval appropriate to the statistic;
- all exclusions and failed runs.

Avoid significance tests that treat prompt-level observations as independent model replicates.

## 8. Seed robustness

Whenever computationally possible, use multiple:

- model initialization or training seeds;
- data-generation seeds;
- detector seeds.

The confirmation split must hold out model-training seeds.

## 9. Artifact policy

Do not commit large artifacts to Git.

The repository stores code, configuration, manifests, small tables, and derived summaries. Checkpoints and raw activations belong in external artifact storage when that is introduced.

## 10. Negative results

Negative results are part of PathBench. Record conditions where history becomes indistinguishable, including model scale, stage similarity, learning rate, continuation length, and probe family.

These boundaries may be more scientifically important than maximizing classifier accuracy.
