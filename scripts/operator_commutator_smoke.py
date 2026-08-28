"""Compare one-step and macro-operator chronology decoders on longer training stages.

The local commutator decoder treats a k-update stage as one effective gradient step of
size k*lr. The macro decoder instead treats the complete k-update training procedure as
one near-identity map and estimates cross-stage Jacobian-vector products by centered
finite differences. The experiment is intentionally fixed before larger-model compute:
we require the macro decoder to preserve three-stage chronology after the micro decoder
has left its local regime.
"""

from __future__ import annotations

import copy
import json
from itertools import permutations

import torch
from transformer_commutator_smoke import STAGE_BATCHES, STAGES, TinyCausalTransformer

from chronotrace.geometry.commutator import (
    decode_permutation,
    local_stage_derivatives,
    multi_stage_symmetric_reference,
    parameter_vector,
)
from chronotrace.geometry.operators import (
    decode_operator_permutation,
    local_stage_map_derivatives,
    operator_pair_geometry,
    operator_pair_score,
    operator_symmetric_reference,
)

UPDATE_LR = 0.01
STAGE_LENGTHS = (1, 2, 4, 8, 16, 32, 64)
FINITE_DIFFERENCE_EPSILON = 1e-4


def set_parameter_vector(model: TinyCausalTransformer, vector: torch.Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for parameter in model.parameters():
            width = parameter.numel()
            parameter.copy_(vector[offset : offset + width].reshape_as(parameter))
            offset += width
    if offset != vector.numel():
        raise ValueError("parameter vector does not match the tiny transformer")


def run_stage(
    base_state: dict[str, torch.Tensor],
    theta: torch.Tensor,
    stage: str,
    *,
    updates: int,
) -> torch.Tensor:
    model = TinyCausalTransformer()
    model.load_state_dict(base_state)
    set_parameter_vector(model, theta)
    for _ in range(updates):
        model.zero_grad(set_to_none=True)
        loss = model.stage_loss(STAGE_BATCHES[stage])
        loss.backward()
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.grad is None:
                    raise RuntimeError("stage loss did not reach every model parameter")
                parameter.add_(parameter.grad, alpha=-UPDATE_LR)
    return parameter_vector(tuple(model.parameters()))


def stage_maps(base_state: dict[str, torch.Tensor], updates: int):
    return {
        stage: (
            lambda theta, stage=stage: run_stage(
                base_state,
                theta,
                stage,
                updates=updates,
            )
        )
        for stage in STAGES
    }


def run_history(theta0, maps, history):
    theta = theta0
    for stage in history:
        theta = maps[stage](theta)
    return theta


def micro_local_geometry(base_model: TinyCausalTransformer):
    parameters = tuple(base_model.parameters())
    loss_fns = {
        stage: (lambda stage=stage: base_model.stage_loss(STAGE_BATCHES[stage]))
        for stage in STAGES
    }
    return local_stage_derivatives(loss_fns, parameters)


def main() -> int:
    torch.manual_seed(123)
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float64)

    base_model = TinyCausalTransformer()
    base_state = copy.deepcopy(base_model.state_dict())
    theta0 = parameter_vector(tuple(base_model.parameters()))
    micro_gradients, micro_cross = micro_local_geometry(base_model)
    candidates = list(permutations(STAGES))

    results: dict[str, dict[str, float | int]] = {}
    first_micro_failure: int | None = None

    for updates in STAGE_LENGTHS:
        maps = stage_maps(base_state, updates)
        deltas, operator_cross = local_stage_map_derivatives(
            maps,
            theta0,
            epsilon=FINITE_DIFFERENCE_EPSILON,
        )
        operator_reference = operator_symmetric_reference(
            theta0,
            deltas,
            operator_cross,
            stages=STAGES,
        )
        micro_reference = multi_stage_symmetric_reference(
            theta0,
            micro_gradients,
            micro_cross,
            stages=STAGES,
            step_size=updates * UPDATE_LR,
        )

        micro_correct = 0
        operator_correct = 0
        micro_min_margin = float("inf")
        operator_min_margin = float("inf")
        for history in candidates:
            endpoint = run_history(theta0, maps, history)
            micro = decode_permutation(
                endpoint,
                micro_reference,
                micro_cross,
                stages=STAGES,
                step_size=updates * UPDATE_LR,
            )
            operator = decode_operator_permutation(
                endpoint,
                operator_reference,
                operator_cross,
                stages=STAGES,
            )
            micro_correct += int(micro.permutation == history)
            operator_correct += int(operator.permutation == history)
            micro_min_margin = min(micro_min_margin, micro.margin)
            operator_min_margin = min(operator_min_margin, operator.margin)

        pair = operator_pair_geometry(
            theta0,
            deltas["A"],
            deltas["B"],
            operator_cross[("B", "A")],
            operator_cross[("A", "B")],
        )
        endpoint_ab = run_history(theta0, maps, "AB")
        endpoint_ba = run_history(theta0, maps, "BA")
        score_ab = operator_pair_score(endpoint_ab, pair)
        score_ba = operator_pair_score(endpoint_ba, pair)

        if micro_correct < len(candidates) and first_micro_failure is None:
            first_micro_failure = updates
        results[str(updates)] = {
            "micro_correct": micro_correct,
            "macro_correct": operator_correct,
            "micro_accuracy": micro_correct / len(candidates),
            "macro_accuracy": operator_correct / len(candidates),
            "micro_min_margin": micro_min_margin,
            "macro_min_margin": operator_min_margin,
            "macro_pair_score_ab": score_ab,
            "macro_pair_score_ba": score_ba,
            "max_stage_displacement_norm": max(
                float(torch.linalg.vector_norm(delta)) for delta in deltas.values()
            ),
        }

        if score_ab <= 0 or score_ba >= 0:
            raise RuntimeError(f"macro pairwise chronology sign failed at {updates} updates")
        if operator_correct != len(candidates):
            raise RuntimeError(
                f"macro decoder recovered only {operator_correct}/{len(candidates)} "
                f"three-stage histories at {updates} updates"
            )

    if results["1"]["micro_correct"] != len(candidates):
        raise RuntimeError("micro decoder must recover the one-update control")
    if first_micro_failure is None:
        raise RuntimeError("micro decoder never left its local regime in the fixed sweep")
    if first_micro_failure > 4:
        raise RuntimeError(
            "micro decoder remained perfect longer than the frozen stress-test boundary"
        )
    if results[str(STAGE_LENGTHS[-1])]["macro_correct"] != len(candidates):
        raise RuntimeError("macro decoder did not outlive the micro decoder through 64 updates")

    payload = {
        "status": "ok",
        "model_parameters": theta0.numel(),
        "optimizer": "plain_sgd_no_momentum",
        "per_update_learning_rate": UPDATE_LR,
        "finite_difference_epsilon": FINITE_DIFFERENCE_EPSILON,
        "stage_lengths": list(STAGE_LENGTHS),
        "first_micro_failure_updates": first_micro_failure,
        "results": results,
        "claim": "macro_operator_decoder_outlives_one_step_local_decoder",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
