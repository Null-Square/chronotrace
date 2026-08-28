"""Seed-held-out paired-history detection for ChronoTrace Phase-0 experiments."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chronotrace.config import ExperimentConfig


def _history_label(history: str) -> int:
    """Map an A-first/B-first paired history to the fixed binary target."""

    if history.startswith("AB"):
        return 1
    if history.startswith("BA"):
        return 0
    raise ValueError(
        f"history {history!r} does not encode the required AB-first versus BA-first contrast"
    )


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
        return _history_label(self.history)


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


def _validate_binary_pair(samples: list[FeatureSample], *, seed: int) -> None:
    if len(samples) != 2 or {sample.label for sample in samples} != {0, 1}:
        raise ValueError(
            f"seed {seed} must contain exactly one A-first and one B-first endpoint"
        )


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
        _validate_binary_pair(test, seed=seed)
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
            sample.run_id: {"true": int(y), "p_a_first": float(score)}
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


def capability_matching_report(
    config: ExperimentConfig,
    samples: list[FeatureSample],
) -> dict[str, Any]:
    """Evaluate the predeclared paired-history ordinary-capability matching gate."""

    required = bool(config.controls.get("require_capability_matching", False))
    threshold = float(config.controls["capability_matching_max_mean_margin_gap"])
    if threshold < 0:
        raise ValueError("capability matching threshold must be non-negative")

    seeds = sorted({sample.seed for sample in samples})
    if not seeds:
        raise ValueError("capability matching requires at least one matched seed")

    per_seed: dict[str, dict[str, Any]] = {}
    max_observed = 0.0
    all_pairs_pass = True
    for seed in seeds:
        pair = [sample for sample in samples if sample.seed == seed]
        _validate_binary_pair(pair, seed=seed)
        by_label = {sample.label: sample for sample in pair}
        a_first = by_label[1]
        b_first = by_label[0]
        a_gap = abs(
            a_first.features["a_control.mean"] - b_first.features["a_control.mean"]
        )
        b_gap = abs(
            a_first.features["b_control.mean"] - b_first.features["b_control.mean"]
        )
        pair_max = max(a_gap, b_gap)
        pair_pass = pair_max <= threshold
        max_observed = max(max_observed, pair_max)
        all_pairs_pass = all_pairs_pass and pair_pass
        per_seed[str(seed)] = {
            "a_control_mean_gap": a_gap,
            "b_control_mean_gap": b_gap,
            "max_mean_margin_gap": pair_max,
            "passed": pair_pass,
            "a_first_history": a_first.history,
            "b_first_history": b_first.history,
        }

    return {
        "required": required,
        "threshold_max_mean_margin_gap": threshold,
        "passed": all_pairs_pass,
        "max_observed_mean_margin_gap": max_observed,
        "per_seed": per_seed,
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
            "predictions_p_a_first": predictions,
        },
    }


def _select_exact_samples(
    config: ExperimentConfig,
    samples: list[FeatureSample],
    split: str,
) -> list[FeatureSample]:
    declared_histories = {str(value) for value in config.histories}
    if len(declared_histories) != 2:
        raise ValueError("paired-history analysis requires exactly two declared histories")
    if {_history_label(history) for history in declared_histories} != {0, 1}:
        raise ValueError("declared histories must contain one A-first and one B-first condition")

    seeds = {int(value) for value in config.seeds[split]}
    selected = [sample for sample in samples if sample.seed in seeds]
    expected = len(seeds) * len(declared_histories)
    if len(selected) != expected:
        raise ValueError(f"expected {expected} {split} endpoints, found {len(selected)}")
    for seed in seeds:
        pair = [sample for sample in selected if sample.seed == seed]
        if {sample.history for sample in pair} != declared_histories:
            raise ValueError(
                f"{split} seed {seed} does not contain the exact declared history pair"
            )
        _validate_binary_pair(pair, seed=seed)
    return selected


def discovery_only_report(config: ExperimentConfig, *, runs_root: str | Path) -> dict[str, Any]:
    """Evaluate discovery seeds only; never require or inspect confirmation endpoints."""

    samples = load_feature_samples(runs_root)
    discovery = _select_exact_samples(config, samples, "discovery")
    stack = _require_stack()
    c_value = float(config.forensics["detector_c"])
    result: dict[str, Any] = {
        "schema_version": 1,
        "split": "discovery",
        "capability_matching": capability_matching_report(config, discovery),
    }
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
    discovery_matching = capability_matching_report(config, discovery)
    confirmation_matching = capability_matching_report(config, confirmation)
    required = bool(config.controls.get("require_capability_matching", False))

    return {
        "schema_version": 1,
        "capability_matching": {
            "discovery": discovery_matching,
            "confirmation": confirmation_matching,
        },
        "primary_analysis_eligible": (
            not required
            or (discovery_matching["passed"] and confirmation_matching["passed"])
        ),
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
