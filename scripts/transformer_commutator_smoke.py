"""Validate inverse commutator chronology on a real tiny causal transformer.

The model uses explicit eager self-attention rather than fused SDPA kernels because this
experiment requires second derivatives for Hessian-vector products. Training histories
use exactly one plain-SGD update per stage so the local endpoint theorem is the object
under test, with no momentum or adaptive-optimizer state.
"""

from __future__ import annotations

import copy
import json
import math
from itertools import permutations

import torch
from torch import nn
from torch.nn import functional as F

from chronotrace.geometry.commutator import (
    decode_permutation,
    estimate_step_size,
    local_stage_derivatives,
    multi_stage_symmetric_reference,
    pairwise_chrono_score,
    pairwise_endpoint_geometry,
    parameter_vector,
)

VOCAB_SIZE = 23
SEQUENCE_LENGTH = 6
D_MODEL = 8
N_HEADS = 2
ETAS = (0.02, 0.01, 0.005, 0.0025)
STAGES = ("A", "B", "C")

STAGE_BATCHES = {
    "A": torch.tensor(
        [
            [1, 2, 3, 4, 5, 6],
            [1, 2, 7, 4, 8, 6],
            [9, 2, 3, 10, 5, 6],
            [9, 2, 7, 10, 8, 6],
        ],
        dtype=torch.long,
    ),
    "B": torch.tensor(
        [
            [1, 11, 3, 12, 5, 13],
            [1, 11, 7, 12, 8, 13],
            [9, 11, 3, 14, 5, 13],
            [9, 11, 7, 14, 8, 13],
        ],
        dtype=torch.long,
    ),
    "C": torch.tensor(
        [
            [15, 2, 16, 4, 17, 6],
            [15, 11, 16, 12, 17, 13],
            [18, 2, 19, 10, 20, 6],
            [18, 11, 19, 14, 20, 13],
        ],
        dtype=torch.long,
    ),
}


class EagerCausalSelfAttention(nn.Module):
    """Small explicit multi-head attention with second-order autograd support."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("attention width must be divisible by the head count")
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, length, _ = hidden.shape
        query, key, value = self.qkv(hidden).chunk(3, dim=-1)

        def heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch, length, self.heads, self.head_width).transpose(1, 2)

        query = heads(query)
        key = heads(key)
        value = heads(value)
        scores = query @ key.transpose(-1, -2) / math.sqrt(self.head_width)
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=hidden.device),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, -1e9)
        weights = torch.softmax(scores, dim=-1)
        attended = weights @ value
        attended = attended.transpose(1, 2).contiguous().view(batch, length, self.width)
        return self.output(attended)


class TinyCausalTransformer(nn.Module):
    """One-block causal LM small enough for exact full-parameter HVPs on CPU."""

    def __init__(self) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.position_embedding = nn.Parameter(torch.randn(SEQUENCE_LENGTH, D_MODEL) * 0.02)
        self.input_norm = nn.LayerNorm(D_MODEL)
        self.attention = EagerCausalSelfAttention(D_MODEL, N_HEADS)
        self.ff_norm = nn.LayerNorm(D_MODEL)
        self.feed_forward = nn.Sequential(
            nn.Linear(D_MODEL, 2 * D_MODEL),
            nn.GELU(),
            nn.Linear(2 * D_MODEL, D_MODEL),
        )
        self.output_norm = nn.LayerNorm(D_MODEL)
        self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        length = token_ids.shape[1]
        hidden = self.token_embedding(token_ids) + self.position_embedding[:length]
        hidden = hidden + self.attention(self.input_norm(hidden))
        hidden = hidden + self.feed_forward(self.ff_norm(hidden))
        return self.lm_head(self.output_norm(hidden))

    def stage_loss(self, token_ids: torch.Tensor) -> torch.Tensor:
        inputs = token_ids[:, :-1]
        targets = token_ids[:, 1:]
        logits = self(inputs)
        return F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))


def loglog_slope(xs: tuple[float, ...], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("slope needs matched sequences with at least two points")
    lx = [math.log(value) for value in xs]
    ly = [math.log(value) for value in ys]
    x_mean = sum(lx) / len(lx)
    y_mean = sum(ly) / len(ly)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(lx, ly, strict=True))
    denominator = sum((x - x_mean) ** 2 for x in lx)
    return numerator / denominator


def stage_loss_fns(model: TinyCausalTransformer):
    return {
        stage: (lambda stage=stage: model.stage_loss(STAGE_BATCHES[stage]))
        for stage in STAGES
    }


def run_history(
    base_state: dict[str, torch.Tensor],
    history: tuple[str, ...] | str,
    eta: float,
) -> tuple[torch.Tensor, TinyCausalTransformer]:
    model = TinyCausalTransformer()
    model.load_state_dict(base_state)
    for stage in history:
        model.zero_grad(set_to_none=True)
        loss = model.stage_loss(STAGE_BATCHES[stage])
        loss.backward()
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.grad is None:
                    raise RuntimeError("stage loss did not reach every model parameter")
                parameter.add_(parameter.grad, alpha=-eta)
    return parameter_vector(tuple(model.parameters())), model


def main() -> int:
    torch.manual_seed(123)
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float64)

    base_model = TinyCausalTransformer()
    base_state = copy.deepcopy(base_model.state_dict())
    parameters = tuple(base_model.parameters())
    theta0 = parameter_vector(parameters)
    gradients, cross = local_stage_derivatives(stage_loss_fns(base_model), parameters)

    bracket_ab = cross[("B", "A")] - cross[("A", "B")]
    bracket_norm = float(torch.linalg.vector_norm(bracket_ab))
    if bracket_norm < 1e-8:
        raise RuntimeError("tiny transformer has a degenerate A/B commutator")

    bracket_errors: list[float] = []
    held_out_loss_gaps: list[float] = []
    shared_displacements: list[float] = []
    step_size_errors: list[float] = []
    pairwise_scores: dict[str, dict[str, float]] = {}
    permutation_accuracy: dict[str, float] = {}
    minimum_decode_margins: dict[str, float] = {}

    for eta in ETAS:
        endpoint_ab, model_ab = run_history(base_state, "AB", eta)
        endpoint_ba, model_ba = run_history(base_state, "BA", eta)
        observed_difference = endpoint_ab - endpoint_ba
        bracket_errors.append(
            float(torch.linalg.vector_norm(observed_difference - eta**2 * bracket_ab))
        )
        held_out_loss_gaps.append(
            abs(
                float(model_ab.stage_loss(STAGE_BATCHES["C"]))
                - float(model_ba.stage_loss(STAGE_BATCHES["C"]))
            )
        )
        shared_displacements.append(float(torch.linalg.vector_norm(endpoint_ab - theta0)))

        geometry = pairwise_endpoint_geometry(
            theta0,
            gradients["A"],
            gradients["B"],
            cross[("B", "A")],
            cross[("A", "B")],
            step_size=eta,
        )
        score_ab = pairwise_chrono_score(endpoint_ab, geometry, step_size=eta)
        score_ba = pairwise_chrono_score(endpoint_ba, geometry, step_size=eta)
        pairwise_scores[str(eta)] = {"AB": score_ab, "BA": score_ba}
        if score_ab <= 0 or score_ba >= 0:
            raise RuntimeError(f"tiny-transformer pairwise decode failed at eta={eta}")

        estimated_eta = estimate_step_size(
            theta0,
            endpoint_ab,
            [gradients["A"], gradients["B"]],
        )
        step_size_errors.append(abs(estimated_eta - eta) / eta)

        reference = multi_stage_symmetric_reference(
            theta0,
            gradients,
            cross,
            stages=STAGES,
            step_size=eta,
        )
        candidates = list(permutations(STAGES))
        correct = 0
        margins: list[float] = []
        for candidate in candidates:
            endpoint, _ = run_history(base_state, candidate, eta)
            decoded = decode_permutation(
                endpoint,
                reference,
                cross,
                stages=STAGES,
                step_size=eta,
            )
            correct += int(decoded.permutation == candidate)
            margins.append(decoded.margin)
        permutation_accuracy[str(eta)] = correct / len(candidates)
        minimum_decode_margins[str(eta)] = min(margins)
        if correct != len(candidates):
            raise RuntimeError(
                f"tiny transformer decoded only {correct}/{len(candidates)} permutations "
                f"at eta={eta}"
            )

    remainder_slope = loglog_slope(ETAS, bracket_errors)
    behavior_slope = loglog_slope(ETAS, held_out_loss_gaps)
    displacement_slope = loglog_slope(ETAS, shared_displacements)
    if not 2.7 <= remainder_slope <= 3.3:
        raise RuntimeError(f"transformer commutator remainder slope={remainder_slope:.4f}")
    if not 1.7 <= behavior_slope <= 2.3:
        raise RuntimeError(f"transformer behavior-gap slope={behavior_slope:.4f}")
    if not 0.8 <= displacement_slope <= 1.2:
        raise RuntimeError(f"transformer displacement slope={displacement_slope:.4f}")

    smallest = str(ETAS[-1])
    if abs(pairwise_scores[smallest]["AB"] - 1.0) > 0.03:
        raise RuntimeError("tiny-transformer AB ChronoScore does not approach +1")
    if abs(pairwise_scores[smallest]["BA"] + 1.0) > 0.03:
        raise RuntimeError("tiny-transformer BA ChronoScore does not approach -1")
    if step_size_errors[-1] > 0.05:
        raise RuntimeError("local step-size estimator is not converging on the transformer")

    payload = {
        "status": "ok",
        "model_parameters": theta0.numel(),
        "bracket_norm": bracket_norm,
        "etas": list(ETAS),
        "scaling": {
            "commutator_remainder": remainder_slope,
            "held_out_behavior_gap": behavior_slope,
            "shared_displacement": displacement_slope,
        },
        "bracket_errors": bracket_errors,
        "held_out_loss_gaps": held_out_loss_gaps,
        "shared_displacements": shared_displacements,
        "relative_step_size_errors": step_size_errors,
        "pairwise_scores": pairwise_scores,
        "three_stage_permutation_accuracy": permutation_accuracy,
        "minimum_decode_margins": minimum_decode_margins,
        "attention_backend": "explicit_eager",
        "optimizer": "plain_sgd_no_momentum",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
