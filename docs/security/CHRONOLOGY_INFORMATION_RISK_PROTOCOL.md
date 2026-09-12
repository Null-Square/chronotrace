# Proposed defensive protocol: incremental risk from chronology information

**Status:** proposed, not executed; not an external preregistration. No new success rate, vulnerability, or defense claim follows from this document. Original scientific locks and results remain unchanged.

## Research question

Does inferred training chronology provide additional information or reduce work for a model-state analysis or authorized benign intervention, beyond what an observer with the same checkpoint and recipe access can already do?

Order identification, state estimation, semantic explanation, and control of behaviour are separate outcomes. A study must not substitute success on the first for evidence of the last. Recovering a stage name is not proof of identifying a parameter component that implements its behaviour.

## Access conditions

| Condition | Information available | Relevance to the current guarantee |
|---|---|---|
| Hosted inference only | Inputs and outputs | Current method cannot run; no model-extraction claim |
| Checkpoint only | Final weights, perhaps architecture | Base and recipes remain missing; current guarantee does not apply |
| Approximate recipe | Weights plus partially known training | Exploratory robustness only unless model discrepancy is bounded |
| Declared complete recipe | Base, checkpoint, deterministic gradients and regularity bounds | Closest to current setting; still conditional on error allowances |

Use authorized small models and non-sensitive public or synthetic data. Stage-specific properties should be benign, such as digit-task calibration or output formatting in a harmless synthetic task. This protocol does not call for disabling actual LLM safeguards or evaluating harmful-content generation.

## Matched conditions and separation

Within each supported access condition, compare hidden-order, ChronoTrace-inferred-order (including ambiguity), and evaluator-known-order conditions. The evaluator retains the actual order and intermediate checkpoints. Those labels and checkpoints are not inputs to the hidden/inferred-order systems. Use the same recipes, initial state, export precision, resources, and analysis procedure apart from order information. Include a same-access replay baseline; possession of a base and full recipes already permits substantial counterfactual replay.

Charge the complete chronology acquisition cost, including prefix construction, inverse residual checks, joins, verification and any repeated trials. Report both gradient work and measured runtime, with machine configuration and uncertainty. Do not award an apparent advantage merely because the chronology computation is omitted from one budget.

Lock hypotheses, seeds, access partitions, stopping rules, and endpoints before fresh evaluation. Existing ChronoTrace inputs are development data for this new question. No new study lock has been made in this document.

## Measurements

Report separately: chronological coverage and wrong orders; distance of estimated intermediate states from evaluator-held states and coverage of the claimed enclosures; cost to reach a predeclared benign analysis objective; agreement with an authorized reference intervention; changes on unrelated benign held-out tasks; and invalid/ambiguous output rates under recipe mismatch, export noise, unknown optimizer state and model non-identifiability. Two different histories can be compatible with one observation; retain such cases rather than forcing one.

The primary incremental-effect comparison is inferred versus hidden order at matched total work. Evaluator-known order is an upper-information reference, not an implementable attack. A benefit that disappears when chronology acquisition is charged should be reported as no demonstrated net work advantage.

## Interpretation and controls

An intermediate-state estimate is not semantic understanding of a weight. An authorized benign intervention is not evidence of safety removal. Weight distance alone does not establish functional equivalence, and task accuracy does not measure all safety properties. Chronology consistent with an approved stage does not prove that the stage achieved its intended effect. Failure of reconstruction does not prove tamper resistance.

Any later safety-specific study needs separate authorization, controlled evaluation infrastructure, model-owner or specialist review, a release-risk assessment and coordinated disclosure where applicable. Do not release operational artifacts that enable a newly demonstrated exploit as part of this benign protocol. Proposed defensive controls include signed training records, checkpoint integrity checks, least-privilege access to recipes and optimizer state, separate deployment controls, and independent behavioural evaluations. These controls are recommendations, not measured defenses in the present paper.
