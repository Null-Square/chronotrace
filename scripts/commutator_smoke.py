"""Fail-fast validation of ChronoTrace's local inverse-geometry hypothesis.

This is intentionally smaller and stricter than a language-model experiment. It asks
whether the second-order endpoint decoder obeys the Taylor scaling it claims and whether
three-stage chronology is recoverable from the signed bracket basis. If this script fails,
we should not spend model-training compute on the commutator decoder.
"""

from __future__ import annotations

import json
import math
from itertools import permutations

import torch

from chronotrace.geometry.commutator import (
    decode_permutation,
    local_stage_derivatives,
    multi_stage_symmetric_reference,
    pairwise_chrono_score,
    pairwise_endpoint_geometry,
)

THETA0 = torch.tensor([0.35, -0.45, 0.25], dtype=torch.float64)
ETAS = (0.1, 0.05, 0.025, 0.0125)
STAGES = ("A", "B", "C")


def losses(theta: torch.Tensor) -> dict[str, torch.Tensor]:
    x0, x1, x2 = theta
    return {
        "A": 0.5 * (x0 + 0.7 * x1**2 - 1.1) ** 2 + 0.2 * (x2 - x0 * x1) ** 2,
        "B": 0.5 * (x1 + 0.6 * x0 * x2 + 0.4) ** 2 + 0.3 * (x0 - x2**2) ** 2,
        "C": 0.5 * (x2 + 0.5 * x0 * x1 - 0.2) ** 2 + 0.25 * (x1 - 0.4 * x0**2) ** 2,
    }


def loss_fns(theta: torch.Tensor):
    return {stage: (lambda stage=stage: losses(theta)[stage]) for stage in STAGES}


def step(theta: torch.Tensor, stage: str, eta: float) -> torch.Tensor:
    (gradient,) = torch.autograd.grad(losses(theta)[stage], (theta,))
    return (theta - eta * gradient).detach().requires_grad_(True)


def run(history: tuple[str, ...] | str, eta: float) -> torch.Tensor:
    theta = THETA0.clone().requires_grad_(True)
    for stage in history:
        theta = step(theta, stage, eta)
    return theta.detach()


def scalar_loss(theta: torch.Tensor, stage: str) -> float:
    return float(losses(theta)[stage])


def loglog_slope(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("slope needs matched sequences with at least two points")
    if any(value <= 0 for value in xs) or any(value <= 0 for value in ys):
        raise ValueError("log-log slope requires positive values")
    lx = [math.log(value) for value in xs]
    ly = [math.log(value) for value in ys]
    x_mean = sum(lx) / len(lx)
    y_mean = sum(ly) / len(ly)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(lx, ly, strict=True))
    denominator = sum((x - x_mean) ** 2 for x in lx)
    return numerator / denominator


def main() -> int:
    theta = THETA0.clone().requires_grad_(True)
    gradients, cross = local_stage_derivatives(loss_fns(theta), (theta,))
    base = theta.detach()
    bracket = cross[("B", "A")] - cross[("A", "B")]
    bracket_norm = float(torch.linalg.vector_norm(bracket))
    if bracket_norm < 1e-8:
        raise RuntimeError("toy system has a degenerate A/B commutator")

    bracket_errors: list[float] = []
    capability_gaps: list[float] = []
    displacement_norms: list[float] = []
    pairwise_scores: dict[str, dict[str, float]] = {}
    permutation_accuracy: dict[str, float] = {}

    for eta in ETAS:
        endpoint_ab = run("AB", eta)
        endpoint_ba = run("BA", eta)
        observed_difference = endpoint_ab - endpoint_ba
        predicted_difference = eta**2 * bracket
        bracket_errors.append(
            float(torch.linalg.vector_norm(observed_difference - predicted_difference))
        )

        # C is not trained in the pairwise experiment and serves as a smooth held-out
        # behavioral observable. Its AB/BA difference should be second order.
        capability_gaps.append(abs(scalar_loss(endpoint_ab, "C") - scalar_loss(endpoint_ba, "C")))
        displacement_norms.append(float(torch.linalg.vector_norm(endpoint_ab - base)))

        pair_geometry = pairwise_endpoint_geometry(
            base,
            gradients["A"],
            gradients["B"],
            cross[("B", "A")],
            cross[("A", "B")],
            step_size=eta,
        )
        score_ab = pairwise_chrono_score(endpoint_ab, pair_geometry, step_size=eta)
        score_ba = pairwise_chrono_score(endpoint_ba, pair_geometry, step_size=eta)
        pairwise_scores[str(eta)] = {"AB": score_ab, "BA": score_ba}
        if score_ab <= 0 or score_ba >= 0:
            raise RuntimeError(f"pairwise chronology sign failed at eta={eta}")

        reference = multi_stage_symmetric_reference(
            base,
            gradients,
            cross,
            stages=STAGES,
            step_size=eta,
        )
        correct = 0
        candidates = list(permutations(STAGES))
        for candidate in candidates:
            endpoint = run(candidate, eta)
            decoded = decode_permutation(
                endpoint,
                reference,
                cross,
                stages=STAGES,
                step_size=eta,
            )
            correct += int(decoded.permutation == candidate)
        permutation_accuracy[str(eta)] = correct / len(candidates)
        if correct != len(candidates):
            raise RuntimeError(
                f"three-stage decoder recovered only {correct}/{len(candidates)} at eta={eta}"
            )

    bracket_error_slope = loglog_slope(list(ETAS), bracket_errors)
    capability_gap_slope = loglog_slope(list(ETAS), capability_gaps)
    displacement_slope = loglog_slope(list(ETAS), displacement_norms)

    # Taylor-theory gates. We use broad numerical bands: the point is to reject the
    # wrong asymptotic order, not fit an artificially precise exponent on four values.
    if not 2.7 <= bracket_error_slope <= 3.3:
        raise RuntimeError(f"commutator remainder is not cubic: slope={bracket_error_slope:.4f}")
    if not 1.7 <= capability_gap_slope <= 2.3:
        raise RuntimeError(
            f"held-out AB/BA behavior gap is not quadratic: {capability_gap_slope:.4f}"
        )
    if not 0.8 <= displacement_slope <= 1.2:
        raise RuntimeError(
            f"shared learning displacement is not first order: {displacement_slope:.4f}"
        )

    smallest = str(ETAS[-1])
    if abs(pairwise_scores[smallest]["AB"] - 1.0) > 0.03:
        raise RuntimeError("AB ChronoScore does not approach +1")
    if abs(pairwise_scores[smallest]["BA"] + 1.0) > 0.03:
        raise RuntimeError("BA ChronoScore does not approach -1")

    payload = {
        "status": "ok",
        "bracket_norm": bracket_norm,
        "etas": list(ETAS),
        "bracket_errors": bracket_errors,
        "capability_gaps": capability_gaps,
        "displacement_norms": displacement_norms,
        "scaling": {
            "commutator_remainder": bracket_error_slope,
            "held_out_behavior_gap": capability_gap_slope,
            "shared_displacement": displacement_slope,
        },
        "pairwise_scores": pairwise_scores,
        "three_stage_permutation_accuracy": permutation_accuracy,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
