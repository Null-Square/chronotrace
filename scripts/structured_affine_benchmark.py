#!/usr/bin/env python3
"""Deterministic short-probe chronology benchmark; no model downloads/API calls.

The structural guarantee applies to finite repeated affine gradient updates, not
arbitrary neural training. Real-data mode fits a small least-squares linear head
on the bundled sklearn digits data. Targets and oracle coefficients never enter
the decoder. Source/data/protocol hashes and complete case records are emitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from itertools import permutations
from pathlib import Path

import numpy as np

from chronotrace.validated_affine import (
    AffineBox,
    Box,
    cube,
    decode_budgeted,
    identify_from_short_probes,
    infinity_norm_upper,
    multiply,
    subset_enclosures,
)

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Measurement:
    value: np.ndarray
    enclosure: Box


class StageOracle:
    """Counted literal gradient updates with a separately validated error box.

    The exact model is z -> M z + c repeated s times; stored M,c are exact
    binary64 coefficients defining this model. Macro interval composition is
    used only to enclose measurement roundoff, never supplied to the decoder.
    Actual value generation executes all s micro-updates.
    """
    def __init__(self, micro_matrices, micro_offsets, steps):
        self.matrices = tuple(micro_matrices)
        self.offsets = tuple(micro_offsets)
        self.steps = steps
        self.calls = 0
        self.updates = 0
        self.macros = []
        self.enclosure_compositions = 0
        d = self.offsets[0].size
        for matrix, offset in zip(self.matrices, self.offsets, strict=True):
            power = AffineBox(Box.point(matrix), Box.point(offset))
            result = AffineBox(Box.point(np.eye(d)), Box.point(np.zeros(d)))
            k = steps
            while k:
                if k & 1:
                    result = power.after(result)
                    self.enclosure_compositions += 1
                k >>= 1
                if k:
                    power = power.after(power)
                    self.enclosure_compositions += 1
            self.macros.append(result)

    def measure(self, stage, start):
        value = np.array(start, dtype=np.float64, copy=True)
        for _ in range(self.steps):
            value = self.matrices[stage] @ value + self.offsets[stage]
        self.calls += 1
        self.updates += self.steps
        enclosure = self.macros[stage].apply(Box.point(start))
        enclosure = Box(np.minimum(enclosure.lo, value), np.maximum(enclosure.hi, value))
        return Measurement(value, enclosure)

    def target(self, order, base):
        value = np.array(base, copy=True)
        enclosure = Box.point(base)
        for stage in order:
            measurement = self.measure(stage, value)
            value = measurement.value
            enclosure = self.macros[stage].apply(enclosure)
            enclosure = Box(np.minimum(enclosure.lo, value), np.maximum(enclosure.hi, value))
        return Measurement(value, enclosure)


def _positive_add(a, b):
    result = float(np.nextafter(float(a) + float(b), np.inf))
    if not math.isfinite(result):
        raise ValueError("analytic remainder bound overflowed")
    return result


def _positive_multiply(a, b):
    result = float(np.nextafter(float(a) * float(b), np.inf))
    if not math.isfinite(result):
        raise ValueError("analytic remainder bound overflowed")
    return result


class QuarticOracle(StageOracle):
    """Non-affine gradient maps: M w + c - gamma * w**3.

    This is gradient descent on a quadratic plus a positive coordinatewise
    quartic penalty. Analytic norm bounds are additional recipe information;
    finite short-probe data alone do not justify this remainder guarantee.
    """
    def __init__(self, micro_matrices, micro_offsets, steps, gamma):
        super().__init__(micro_matrices, micro_offsets, steps)
        self.gamma = float(gamma)
        self.norms = [infinity_norm_upper(Box.point(matrix)) for matrix in self.matrices]
        self.displacements = [float(np.max(np.abs(v))) for v in self.offsets]
        n = len(self.matrices)
        self.prefix_radii = np.zeros(1 << n)
        self.context_remainders = np.zeros((n, 1 << n))
        self.analytic_bound_scalar_updates = 0
        global_radius = 0.0
        for mask in range(1, 1 << n):
            endings = []
            for j in range(n):
                if not mask & (1 << j):
                    continue
                previous = mask ^ (1 << j)
                radius, error = float(self.prefix_radii[previous]), 0.0
                for _ in range(steps):
                    cubic = _positive_multiply(_positive_multiply(radius, radius), radius)
                    error = _positive_add(_positive_multiply(self.norms[j], error),
                                          _positive_multiply(self.gamma, cubic))
                    radius = self.radius_step(j, radius)
                    global_radius = max(global_radius, radius)
                    self.analytic_bound_scalar_updates += 1
                endings.append(radius)
                self.context_remainders[j, previous] = error
            self.prefix_radii[mask] = max(endings)
        self.global_radius = global_radius

    def radius_step(self, j, radius):
        cubic = _positive_multiply(_positive_multiply(radius, radius), radius)
        return _positive_add(
            _positive_add(_positive_multiply(self.norms[j], radius), self.displacements[j]),
            _positive_multiply(self.gamma, cubic),
        )

    def reference_error(self, j, start_radius):
        radius, error = float(start_radius), 0.0
        for _ in range(self.steps):
            cubic = _positive_multiply(_positive_multiply(radius, radius), radius)
            error = _positive_add(_positive_multiply(self.norms[j], error),
                                  _positive_multiply(self.gamma, cubic))
            radius = self.radius_step(j, radius)
        return error

    def global_reference_error(self, j):
        cubic = _positive_multiply(
            _positive_multiply(self.global_radius, self.global_radius), self.global_radius
        )
        error = 0.0
        for _ in range(self.steps):
            error = _positive_add(_positive_multiply(self.norms[j], error),
                                  _positive_multiply(self.gamma, cubic))
        return error

    def _run(self, stage, start, initial_box):
        value = np.array(start, dtype=np.float64, copy=True)
        enclosure = initial_box
        matrix = Box.point(self.matrices[stage])
        offset = Box.point(self.offsets[stage])
        gamma = Box.point(self.gamma)
        from chronotrace.validated_affine import matmul
        for _ in range(self.steps):
            value = self.matrices[stage] @ value + self.offsets[stage] - self.gamma * value ** 3
            enclosure = matmul(matrix, enclosure) + offset - multiply(gamma, cube(enclosure))
        self.calls += 1
        self.updates += self.steps
        enclosure = Box(np.minimum(enclosure.lo, value), np.maximum(enclosure.hi, value))
        return Measurement(value, enclosure)

    def measure(self, stage, start):
        return self._run(stage, start, Box.point(start))

    def target(self, order, base):
        result = Measurement(np.array(base, copy=True), Box.point(base))
        for stage in order:
            result = self._run(stage, result.value, result.enclosure)
        return result


def build_problem(family, seed, n, d, steps):
    nonlinear = family.endswith("_quartic")
    original_family = family
    if family == "synthetic_quartic":
        family = "synthetic_quadratic"
    elif family == "digits_quartic":
        family = "digits_ridge"
    rng = np.random.default_rng(seed)
    hessians, gradients = [], []
    metadata = {}
    capability_data = None
    if family == "synthetic_quadratic":
        for _ in range(n):
            q, _ = np.linalg.qr(rng.normal(size=(d, d)))
            hessian = (q * rng.uniform(0.25, 1.75, size=d)) @ q.T
            hessian = (hessian + hessian.T) * 0.5
            optimum = rng.normal(size=d)
            hessians.append(hessian)
            gradients.append(hessian @ optimum)
        learning_rate = 0.02
        metadata["dataset"] = "generated SPD quadratic objectives"
    elif family == "digits_ridge":
        import sklearn
        from sklearn.datasets import load_digits
        from sklearn.decomposition import PCA
        from sklearn.model_selection import train_test_split

        dataset = load_digits()
        x = dataset.data.astype(np.float64) / 16.0
        y = (dataset.target % 2).astype(np.float64) * 2.0 - 1.0
        train, heldout = train_test_split(
            np.arange(len(y)), test_size=0.25, random_state=1729, stratify=dataset.target
        )
        # Representation and split are fixed without observing any chronology.
        pca = PCA(n_components=d, svd_solver="full", whiten=True)
        features = pca.fit_transform(x[train])
        labels = y[train]
        capability_data = (pca.transform(x[heldout]), y[heldout])
        indices = rng.permutation(len(labels))
        for part in np.array_split(indices, n):
            xx, yy = features[part], labels[part]
            hessians.append(xx.T @ xx / len(part) + 0.05 * np.eye(d))
            gradients.append(xx.T @ yy / len(part))
        max_eigenvalue = max(float(np.linalg.eigvalsh(h)[-1]) for h in hessians)
        learning_rate = 0.05 / max_eigenvalue
        metadata = {
            "dataset": "sklearn bundled handwritten digits; parity least-squares head",
            "dataset_sha256": hashlib.sha256(
                dataset.data.tobytes() + dataset.target.tobytes()).hexdigest(),
            "representation": f"{d} PCA-whitened features; fixed train-only fit",
            "train_count": len(train), "heldout_count": len(heldout),
            "sklearn_version": sklearn.__version__,
            "ridge": 0.05,
            "task_definition": "random disjoint partitions; NOT natural domain shifts",
        }
    else:
        raise ValueError(f"unknown family: {family}")
    if nonlinear:
        learning_rate = 0.001
    micro = [np.eye(d) - learning_rate * h for h in hessians]
    offsets = [learning_rate * g for g in gradients]
    metadata.update({"learning_rate": learning_rate, "dimension": d, "stages": n,
                     "steps_per_stage": steps,
                     "max_micro_spectral_radius": max(
                         float(np.max(np.abs(np.linalg.eigvalsh(m)))) for m in micro)})
    if nonlinear:
        metadata.update({"family": original_family, "quartic_penalty": 0.05,
                         "nonlinear_bound_access": "known recipe norms; not short probes alone"})
        oracle = QuarticOracle(micro, offsets, steps, learning_rate * 0.05)
    else:
        oracle = StageOracle(micro, offsets, steps)
    oracle.capability_data = capability_data
    return oracle, metadata


def enumerate_point_baselines(single, pairs, maps):
    """Factorial algebraic baselines using the same acquired short probes.

    Both return non-certified point estimates. Neither makes training queries.
    Enumeration is explicit and separately accounted, never used by the main
    decoder or to select a target-history label.
    """
    n, d = len(single), single[0].value.size
    orders = np.asarray(list(permutations(range(n))), dtype=np.int16)
    positions = np.argsort(orders, axis=1)
    pair_predictions = np.broadcast_to(sum(x.value for x in single), (len(orders), d)).copy()
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            interaction = pairs[i, j].value - single[i].value - single[j].value
            pair_predictions[positions[:, i] < positions[:, j]] += interaction
    affine_predictions = np.zeros((len(orders), d))
    for k in range(n):
        for j, stage in enumerate(maps):
            mask = orders[:, k] == j
            affine_predictions[mask] = (
                affine_predictions[mask] @ stage.matrix.mid.T + stage.offset.mid
            )
    return orders, pair_predictions, affine_predictions


def run_codebook(family, seed, n, d, steps, budgets, targets_per_codebook,
                 export_radius=0.0):
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
        target_box = (measurement.enclosure + Box.point(noise)).inflate(export_radius)
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
        records.append({
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
        "invalid_codebook": any(
            result["wrong_pairs"] or not result["true_history_retained"]
            for record in records for result in record["predictions"]
        ),
        "export_radius": export_radius,
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


def aggregate(codebooks):
    groups = {}
    for codebook in codebooks:
        key = f"{codebook['family']}/N{codebook['stages']}/s{codebook['steps_per_stage']}"
        group = groups.setdefault(key, {"codebooks": 0, "cases": 0,
                                       "pair_point_correct": 0, "affine_point_correct": 0,
                                       "invalid_codebooks": 0, "budgets": {}})
        group["codebooks"] += 1
        group["invalid_codebooks"] += int(codebook.get("invalid_codebook", False))
        for record in codebook["records"]:
            group["cases"] += 1
            group["pair_point_correct"] += int(record["pair_point_correct"])
            group["affine_point_correct"] += int(record["affine_point_correct"])
            for result in record["predictions"]:
                item = group["budgets"].setdefault(str(result["budget"]), {
                    "full_correct": 0, "entailed_pairs": 0, "wrong_pairs": 0,
                    "true_history_retained": 0, "bound_evaluations": [],
                    "decode_seconds": [], "unexcluded_histories": [],
                })
                item["full_correct"] += int(result["full_correct"])
                item["entailed_pairs"] += len(result["precedences"])
                item["wrong_pairs"] += result["wrong_pairs"]
                item["true_history_retained"] += int(result["true_history_retained"])
                for field in ("bound_evaluations", "decode_seconds", "unexcluded_histories"):
                    item[field].append(result[field])
    return groups


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
                          for path in ("src/chronotrace/validated_affine.py",
                                       "scripts/structured_affine_benchmark.py")},
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
