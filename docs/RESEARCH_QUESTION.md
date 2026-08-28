# Research Question

## Primary question

Can a finished language model retain recoverable information about the **order of learning stages** that produced it?

The first controlled problem compares two histories:

```text
AB: base -> stage A -> stage B
BA: base -> stage B -> stage A
```

Both histories must use the same stage data, token budget, optimizer family, hyperparameters, and model architecture. The variable of interest is stage order.

## Inverse sequential learning

Most sequential-learning work studies the forward problem:

> How does training order affect the final model?

ChronoTrace studies an inverse problem:

> Given the final model, what can we infer about the training order?

The endpoint can contain path-dependent information because sequential gradient update operators do not generally commute.

## Scope of the first paper

The first paper does not need to reconstruct an industrial pretraining pipeline. It needs to establish four things:

1. **Existence:** a history signal survives controlled AB/BA training.
2. **Generalization:** the signal transfers to random seeds not used to fit the detector.
3. **Non-triviality:** ordinary capability differences do not fully explain the signal.
4. **Persistence:** the signal can survive at least some common subsequent training.

If these results hold, later work can address multi-stage reconstruction, acquisition-mechanism inference, and black-box transfer.

## What this is not

ChronoTrace is not primarily:

- training-data membership inference;
- model lineage verification;
- document provenance ranking;
- contamination detection;
- curriculum optimization;
- a watermark inserted during training.

Those areas are relevant baselines and neighboring work. ChronoTrace targets the latent history of ordinary optimization without requiring a purpose-built provenance marker.

## Identifiability boundary

Not every pair of histories is distinguishable. If two histories induce the same distribution over observable models, no forensic method can recover the difference from the endpoint.

The empirical question is therefore not only whether history can be recovered. It is also:

> Which properties of stages, interactions, models, and subsequent training make history identifiable?

This boundary is a first-class research result, including negative results.
