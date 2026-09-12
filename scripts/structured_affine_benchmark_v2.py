#!/usr/bin/env python3
"""Chronology confirmation with strictly history-blind observation uncertainty.

Unlike v1, the target enclosure is formed ONLY from the observed endpoint and
fixed public export/numerical error budgets. A generating-history enclosure is
consulted after decisions solely to validate the benchmark's measurement
assumption. Any violated budget invalidates the job; it never changes a bound,
selects a candidate, or silently excludes a case.

The fixed numerical radius is a declared input-accuracy assumption, not a
universal roundoff bound proved for every possible optimizer or target.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from structured_affine_benchmark import (
    QuarticOracle,
    aggregate,
    build_problem,
    enumerate_point_baselines,
)

from chronotrace.validated_affine import (
    Box,
    decode_budgeted,
    identify_from_short_probes,
    subset_enclosures,
)

ROOT = Path(__file__).resolve().parents[1]


def _validate_radius(radius: float) -> None:
    if not math.isfinite(radius) or radius < 0:
        raise ValueError("error budgets must be finite and nonnegative")


def observation_box(observed: np.ndarray, export_radius: float, numerical_radius: float) -> Box:
    """Decision input: no history, oracle enclosure, or target evaluation fields."""
    _validate_radius(export_radius)
    _validate_radius(numerical_radius)
    observed = np.asarray(observed, dtype=np.float64)
    if observed.ndim != 1 or not observed.size or not np.isfinite(observed).all():
        raise ValueError("observed endpoint must be a finite nonempty vector")
    radius = float(np.nextafter(export_radius + numerical_radius, np.inf))
    return Box.point(observed).inflate(radius)


def run_codebook(family, seed, n, d, steps, budgets, targets_per_codebook,
                 export_radius=0.0, numerical_radius=1e-10):
    _validate_radius(export_radius)
    _validate_radius(numerical_radius)
    oracle, metadata = build_problem(family, seed, n, d, steps)
    base = np.zeros(d)
    single = [oracle.measure(j, base) for j in range(n)]
    pairs = {(i, j): oracle.measure(j, single[i].value)
             for i in range(n) for j in range(n) if i != j}
    probe_calls, probe_updates = oracle.calls, oracle.updates
    if probe_calls != n * n:
        raise RuntimeError("acquisition accounting violation")
    start = time.perf_counter()
    singleton_boxes = [x.enclosure for x in single]
    pair_boxes = {key: x.enclosure for key, x in pairs.items()}
    if isinstance(oracle, QuarticOracle):
        singleton_boxes = [value.inflate(oracle.reference_error(j, 0.0))
                           for j, value in enumerate(singleton_boxes)]
        pair_boxes = {key: value.inflate(oracle.reference_error(
            key[1], float(np.max(np.abs(single[key[0]].value)))))
                      for key, value in pair_boxes.items()}
    fit = identify_from_short_probes(
        base, [x.value for x in single], singleton_boxes, pair_boxes,
    )
    decoder_maps = fit.maps
    nonlinear_remainders = None
    if isinstance(oracle, QuarticOracle):
        nonlinear_remainders = oracle.context_remainders
    identification_seconds = time.perf_counter() - start
    start = time.perf_counter()
    cached = subset_enclosures(decoder_maps, remainder_bounds=nonlinear_remainders)
    subset_seconds = time.perf_counter() - start
    start = time.perf_counter()
    orders, pair_predictions, affine_predictions = enumerate_point_baselines(
        single, pairs, fit.maps
    )
    baseline_setup_seconds = time.perf_counter() - start
    # Domain-separated target RNG. Target labels are evaluation metadata only.
    target_digest = hashlib.sha256(f"targets-v1:{seed}:{n}".encode()).digest()
    target_seed = int.from_bytes(target_digest[:8], "big")
    target_rng = np.random.default_rng(target_seed)
    target_orders = []
    while len(target_orders) < targets_per_codebook:
        candidate = tuple(int(j) for j in target_rng.permutation(n))
        if candidate not in target_orders:
            target_orders.append(candidate)
    records = []
    for history in target_orders:
        measurement = oracle.target(history, base)
        noise = target_rng.uniform(-export_radius, export_radius, size=d)
        observed = measurement.value + noise
        target_box = observation_box(observed, export_radius, numerical_radius)
        outputs = []
        for budget in budgets:
            start = time.perf_counter()
            prediction = decode_budgeted(decoder_maps, target_box,
                                        max_bound_evaluations=budget, precomputed=cached,
                                        remainder_bounds=nonlinear_remainders)
            elapsed = time.perf_counter() - start
            # Evaluation occurs AFTER the label-blind decoder has returned.
            wrong = sum(history.index(i) > history.index(j) for i, j in prediction.precedences)
            true_retained = any(not node.suffix or history[-len(node.suffix):] == node.suffix
                                for node in prediction.frontier)
            outputs.append({
                **asdict(prediction), "budget": budget, "decode_seconds": elapsed,
                "full_correct": prediction.unique_order == history,
                "wrong_pairs": wrong, "true_history_retained": true_retained,
            })
        nearest_pair = int(np.argmin(np.linalg.norm(pair_predictions - observed, axis=1)))
        nearest_affine = int(np.argmin(np.linalg.norm(affine_predictions - observed, axis=1)))
        capability = None
        if oracle.capability_data is not None:
            features, labels = oracle.capability_data
            scores = features @ measurement.value
            capability = {"heldout_accuracy": float(np.mean((scores >= 0) == (labels >= 0))),
                          "heldout_mse": float(np.mean((scores - labels) ** 2))}
        # Only evaluation reads the generating-history arithmetic enclosure.
        budget_valid = bool(np.all(target_box.lo <= measurement.enclosure.lo)
                            and np.all(target_box.hi >= measurement.enclosure.hi))
        records.append({
            "measurement_budget_valid": budget_valid,
            "capability": capability,
            "history": history, "target_value": observed.tolist(),
            "target_lo": target_box.lo.tolist(), "target_hi": target_box.hi.tolist(),
            "pair_point_order": orders[nearest_pair].tolist(),
            "pair_point_correct": tuple(orders[nearest_pair]) == history,
            "affine_point_order": orders[nearest_affine].tolist(),
            "affine_point_correct": tuple(orders[nearest_affine]) == history,
            "predictions": outputs,
        })
    return {
        "family": family, "seed": seed, **metadata,
        "invalid_codebook": any(not record["measurement_budget_valid"] for record in records)
        or any(
            result["wrong_pairs"] or not result["true_history_retained"]
            for record in records for result in record["predictions"]
        ),
        "export_radius": export_radius,
        "numerical_radius": numerical_radius,
        "target_interval_access": "observed_endpoint_and_fixed_public_budgets_only",
        "probe_stage_calls": probe_calls, "probe_optimizer_updates": probe_updates,
        "target_generation_stage_calls": oracle.calls - probe_calls,
        "target_generation_optimizer_updates": oracle.updates - probe_updates,
        "roundoff_enclosure_macro_compositions": oracle.enclosure_compositions,
        "subset_preprocessing_applications": cached.applications,
        "identification_matrix_error_bounds": fit.matrix_error_bounds,
        "nonlinear_global_radius": getattr(oracle, "global_radius", None),
        "nonlinear_remainder_bounds": (
            None if nonlinear_remainders is None else nonlinear_remainders.tolist()
        ),
        "analytic_bound_scalar_updates": getattr(oracle, "analytic_bound_scalar_updates", 0),
        "identification_inverse_residual_bounds": fit.inverse_residual_bounds,
        "identification_seconds": identification_seconds,
        "subset_seconds": subset_seconds,
        "baseline_setup_seconds": baseline_setup_seconds,
        "enumerated_surrogate_candidates": len(orders),
        "exhaustive_training_prefix_calls_theoretical": sum(
            math.perm(n, k) for k in range(1, n + 1)),
        "exhaustive_training_replay_executed": False,
        "records": records,
    }



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.protocol.read_bytes()
    config = json.loads(raw)
    for path, expected in config.get("source_sha256", {}).items():
        actual = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"Frozen source hash mismatch: {path}")
    started = time.time()
    codebooks = []
    for family in config["families"]:
        for n in config["stage_counts"]:
            for steps in config["steps_per_stage"]:
                for seed in config["seeds"]:
                    try:
                        codebook = run_codebook(
                            family, seed, n, config["dimension"], steps, config["budgets"],
                            config["targets_per_codebook"], config.get("export_radius", 0.0),
                            config["numerical_radius"],
                        )
                    except (ValueError, RuntimeError, FloatingPointError) as exc:
                        codebook = {
                            "family": family, "seed": seed, "stages": n,
                            "steps_per_stage": steps, "invalid_codebook": True,
                            "error": f"{type(exc).__name__}: {exc}", "records": [],
                        }
                    codebooks.append(codebook)
                    # Preserve every completed or invalid job, even if interrupted.
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    checkpoint = args.output.with_suffix(".checkpoint.json")
                    checkpoint.write_text(json.dumps(codebooks, indent=2) + "\n")
                    full = sum(r["predictions"][-1]["full_correct"]
                               for r in codebook["records"])
                    print(f"{family} N={n} s={steps} seed={seed}: "
                          f"{full}/{len(codebook['records'])} full; "
                          f"invalid={codebook['invalid_codebook']}", flush=True)
    output = {
        "evidence_role": config["evidence_role"], "protocol": config,
        "protocol_sha256": hashlib.sha256(raw).hexdigest(),
        "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                          for path in config["source_sha256"]},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "platform": platform.platform()},
        "runtime_seconds": time.time() - started,
        "aggregate": aggregate(codebooks), "codebooks": codebooks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["aggregate"], indent=2), flush=True)
    if any(codebook.get("invalid_codebook") for codebook in codebooks):
        raise SystemExit("Invalid jobs occurred; all results were retained in the output")


if __name__ == "__main__":
    main()
