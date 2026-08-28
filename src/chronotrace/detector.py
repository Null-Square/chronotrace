"""Seed-held-out history detection for Phase 0."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chronotrace.config import ExperimentConfig


@dataclass(frozen=True)
class FeatureSample:
    """One endpoint represented by a fixed feature mapping."""

    run_id: str
    history: str
    seed: int
    features: dict[str, float]
    feature_groups: dict[str, list[str]]

    @property
    def label(self) -> int:
        return 1 if self.history == "AB" else 0


def _require_stack() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import balanced_accuracy_score, roc_auc_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError('Install the MVP dependencies with: pip install -e ".[mvp]"') from exc
    pipeline_stack = (make_pipeline, StandardScaler)
    return np, LogisticRegression, balanced_accuracy_score, roc_auc_score, pipeline_stack


def load_feature_samples(root: str | Path) -> list[FeatureSample]:
    """Load all endpoint feature files under a run root."""

    samples: list[FeatureSample] = []
    for path in sorted(Path(root).glob("phase0-*/features.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        samples.append(
            FeatureSample(
                run_id=str(payload["run_id"]),
                history=str(payload["history"]),
                seed=int(payload["training_seed"]),
                features={name: float(value) for name, value in payload["features"].items()},
                feature_groups={
                    name: list(values) for name, values in payload["feature_groups"].items()
                },
            )
        )
    return samples


def _feature_names(samples: list[FeatureSample], group: str) -> list[str]:
    if not samples:
        raise ValueError("no feature samples were loaded")
    names = sorted(samples[0].feature_groups[group])
    for sample in samples[1:]:
        if sorted(sample.feature_groups[group]) != names:
            raise ValueError(f"feature schema mismatch for {sample.run_id}")
    return names


def _matrix(samples: list[FeatureSample], names: list[str], np: Any) -> tuple[Any, Any]:
    x = np.asarray([[sample.features[name] for name in names] for sample in samples], dtype=float)
    y = np.asarray([sample.label for sample in samples], dtype=int)
    if not np.isfinite(x).all():
        raise ValueError("non-finite feature value detected")
    return x, y


def _model(
    LogisticRegression: Any,
    make_pipeline: Any,
    StandardScaler: Any,
    *,
    c_value: float,
) -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            solver="liblinear",
            random_state=0,
            max_iter=2000,
        ),
    )


def _metrics(
    y_true: Any,
    y_pred: Any,
    y_prob: Any,
    balanced_accuracy_score: Any,
    roc_auc_score: Any,
) -> dict[str, float]:
    result = {"balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred))}
    if len(set(int(value) for value in y_true)) == 2:
        result["auroc"] = float(roc_auc_score(y_true, y_prob))
    return result


def _leave_one_seed_out(
    samples: list[FeatureSample],
    *,
    names: list[str],
    stack: tuple[Any, Any, Any, Any, Any],
    c_value: float,
) -> dict[str, Any]:
    np, LogisticRegression, balanced_accuracy_score, roc_auc_score, pipeline_stack = stack
    make_pipeline, StandardScaler = pipeline_stack
    seeds = sorted({sample.seed for sample in samples})
    if len(seeds) < 2:
        raise ValueError("leave-one-seed-out discovery requires at least two training seeds")
    true: list[int] = []
    pred: list[int] = []
    prob: list[float] = []
    per_seed: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        train = [sample for sample in samples if sample.seed != seed]
        test = [sample for sample in samples if sample.seed == seed]
        if {sample.history for sample in test} != {"AB", "BA"} or len(test) != 2:
            raise ValueError(f"seed {seed} must contain exactly one AB and one BA endpoint")
        x_train, y_train = _matrix(train, names, np)
        x_test, y_test = _matrix(test, names, np)
        model = _model(
            LogisticRegression,
            make_pipeline,
            StandardScaler,
            c_value=c_value,
        )
        model.fit(x_train, y_train)
        p = model.predict_proba(x_test)[:, 1]
        y_hat = (p >= 0.5).astype(int)
        true.extend(int(value) for value in y_test)
        pred.extend(int(value) for value in y_hat)
        prob.extend(float(value) for value in p)
        per_seed[str(seed)] = {
            sample.run_id: {"true": int(y), "p_ab": float(score)}
            for sample, y, score in zip(test, y_test, p, strict=True)
        }
    return {
        "metrics": _metrics(
            np.asarray(true),
            np.asarray(pred),
            np.asarray(prob),
            balanced_accuracy_score,
            roc_auc_score,
        ),
        "predictions": per_seed,
    }


def _cluster_bootstrap_balanced_accuracy(
    samples: list[FeatureSample],
    predictions: dict[str, float],
    *,
    repetitions: int = 2000,
    seed: int = 20260828,
) -> tuple[float, float]:
    """Bootstrap confirmation accuracy by resampling paired training seeds."""

    _, _, balanced_accuracy_score, _, _ = _require_stack()
    grouped: dict[int, list[FeatureSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.seed, []).append(sample)
    seeds = sorted(grouped)
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(repetitions):
        sampled_seeds = [rng.choice(seeds) for _ in seeds]
        y_true: list[int] = []
        y_pred: list[int] = []
        for sampled_seed in sampled_seeds:
            for sample in grouped[sampled_seed]:
                y_true.append(sample.label)
                y_pred.append(1 if predictions[sample.run_id] >= 0.5 else 0)
        values.append(float(balanced_accuracy_score(y_true, y_pred)))
    values.sort()
    low = values[int(0.025 * (len(values) - 1))]
    high = values[int(0.975 * (len(values) - 1))]
    return low, high


def _evaluate_group(
    discovery: list[FeatureSample],
    confirmation: list[FeatureSample],
    *,
    group: str,
    c_value: float,
) -> dict[str, Any]:
    stack = _require_stack()
    np, LogisticRegression, balanced_accuracy_score, roc_auc_score, pipeline_stack = stack
    make_pipeline, StandardScaler = pipeline_stack
    names = _feature_names(discovery + confirmation, group)

    discovery_cv = _leave_one_seed_out(
        discovery,
        names=names,
        stack=stack,
        c_value=c_value,
    )
    x_train, y_train = _matrix(discovery, names, np)
    x_test, y_test = _matrix(confirmation, names, np)
    model = _model(
        LogisticRegression,
        make_pipeline,
        StandardScaler,
        c_value=c_value,
    )
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_test)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    confirmation_metrics = _metrics(
        y_test, prediction, probability, balanced_accuracy_score, roc_auc_score
    )
    predictions = {
        sample.run_id: float(score)
        for sample, score in zip(confirmation, probability, strict=True)
    }
    low, high = _cluster_bootstrap_balanced_accuracy(confirmation, predictions)
    confirmation_metrics["balanced_accuracy_ci95_low"] = low
    confirmation_metrics["balanced_accuracy_ci95_high"] = high
    return {
        "feature_group": group,
        "feature_names": names,
        "discovery_leave_one_seed_out": discovery_cv,
        "confirmation": {
            "metrics": confirmation_metrics,
            "predictions_p_ab": predictions,
        },
    }


def _select_exact_samples(
    config: ExperimentConfig,
    samples: list[FeatureSample],
    split: str,
) -> list[FeatureSample]:
    seeds = {int(value) for value in config.seeds[split]}
    selected = [sample for sample in samples if sample.seed in seeds]
    expected = len(seeds) * 2
    if len(selected) != expected:
        raise ValueError(f"expected {expected} {split} endpoints, found {len(selected)}")
    for seed in seeds:
        pair = [sample for sample in selected if sample.seed == seed]
        if {sample.history for sample in pair} != {"AB", "BA"} or len(pair) != 2:
            raise ValueError(
                f"{split} seed {seed} does not have exactly one AB and one BA endpoint"
            )
    return selected


def discovery_only_report(config: ExperimentConfig, *, runs_root: str | Path) -> dict[str, Any]:
    """Evaluate discovery seeds only; never require or inspect confirmation endpoints."""

    samples = load_feature_samples(runs_root)
    discovery = _select_exact_samples(config, samples, "discovery")
    stack = _require_stack()
    c_value = float(config.forensics["detector_c"])
    result: dict[str, Any] = {"schema_version": 1, "split": "discovery"}
    for output_name, group in (
        ("forensic", "forensic"),
        ("capability_baseline", "capability"),
    ):
        names = _feature_names(discovery, group)
        result[output_name] = {
            "feature_group": group,
            "feature_names": names,
            "leave_one_seed_out": _leave_one_seed_out(
                discovery,
                names=names,
                stack=stack,
                c_value=c_value,
            ),
        }
    return result


def fit_and_evaluate(config: ExperimentConfig, *, runs_root: str | Path) -> dict[str, Any]:
    """Fit frozen discovery detectors and evaluate confirmation seeds exactly once."""

    samples = load_feature_samples(runs_root)
    discovery = _select_exact_samples(config, samples, "discovery")
    confirmation = _select_exact_samples(config, samples, "confirmation")
    c_value = float(config.forensics["detector_c"])

    return {
        "schema_version": 1,
        "forensic": _evaluate_group(
            discovery,
            confirmation,
            group="forensic",
            c_value=c_value,
        ),
        "capability_baseline": _evaluate_group(
            discovery,
            confirmation,
            group="capability",
            c_value=c_value,
        ),
    }
