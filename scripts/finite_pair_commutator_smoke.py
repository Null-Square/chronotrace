"""Stress-test finite pair chronology beyond differential macro-stage locality.

The fixed experiment compares three decoders on the same tiny causal transformer:

1. micro HVP geometry from the base checkpoint;
2. centered finite-difference macro-stage Jacobian geometry;
3. exact finite singleton + ordered-pair interaction geometry.

The finite-pair method has no perturbation epsilon. Its approximation residual consists
of triple-and-higher stage interactions. It only earns a larger-model test if it keeps
recovering all six A/B/C orders after the differential macro decoder loses perfection.
"""

from __future__ import annotations

import copy
import json
from itertools import permutations

import torch
from operator_commutator_smoke import (
    FINITE_DIFFERENCE_EPSILON,
    UPDATE_LR,
    micro_local_geometry,
    run_history,
    stage_maps,
)
from transformer_commutator_smoke import STAGES, TinyCausalTransformer

from chronotrace.geometry.commutator import (
    decode_permutation,
    multi_stage_symmetric_reference,
    parameter_vector,
)
from chronotrace.geometry.operators import (
    decode_operator_permutation,
    local_stage_map_derivatives,
    operator_symmetric_reference,
)
from chronotrace.geometry.secant import (
    decode_finite_pair_permutation,
    finite_pair_identifiability,
    finite_pair_interactions,
    finite_pair_predicted_endpoint,
    finite_pair_symmetric_reference,
    higher_order_remainder_ratio,
)

STAGE_LENGTHS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


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
    first_differential_macro_failure: int | None = None

    for updates in STAGE_LENGTHS:
        maps = stage_maps(base_state, updates)

        differential_deltas, differential_cross = local_stage_map_derivatives(
            maps,
            theta0,
            epsilon=FINITE_DIFFERENCE_EPSILON,
        )
        differential_reference = operator_symmetric_reference(
            theta0,
            differential_deltas,
            differential_cross,
            stages=STAGES,
        )

        finite_deltas, finite_interactions = finite_pair_interactions(maps, theta0)
        finite_reference = finite_pair_symmetric_reference(
            theta0,
            finite_deltas,
            finite_interactions,
            stages=STAGES,
        )
        finite_identifiability = finite_pair_identifiability(
            finite_interactions,
            stages=STAGES,
        )
        if not finite_identifiability.identifiable:
            raise RuntimeError(f"finite-pair signatures collapse at {updates} updates")

        micro_reference = multi_stage_symmetric_reference(
            theta0,
            micro_gradients,
            micro_cross,
            stages=STAGES,
            step_size=updates * UPDATE_LR,
        )

        micro_correct = 0
        differential_correct = 0
        finite_correct = 0
        finite_min_margin = float("inf")
        finite_max_higher_order_ratio = 0.0
        finite_max_remainder_norm = 0.0

        for history in candidates:
            endpoint = run_history(theta0, maps, history)

            micro = decode_permutation(
                endpoint,
                micro_reference,
                micro_cross,
                stages=STAGES,
                step_size=updates * UPDATE_LR,
            )
            differential = decode_operator_permutation(
                endpoint,
                differential_reference,
                differential_cross,
                stages=STAGES,
            )
            finite = decode_finite_pair_permutation(
                endpoint,
                finite_reference,
                finite_interactions,
                stages=STAGES,
            )
            finite_prediction = finite_pair_predicted_endpoint(
                history,
                finite_reference,
                finite_interactions,
                stages=STAGES,
            )
            higher_order_ratio = higher_order_remainder_ratio(
                endpoint,
                finite_prediction,
                minimum_signature_separation=(
                    finite_identifiability.minimum_signature_separation
                ),
            )
            remainder_norm = float(torch.linalg.vector_norm(endpoint - finite_prediction))

            micro_correct += int(micro.permutation == history)
            differential_correct += int(differential.permutation == history)
            finite_correct += int(finite.permutation == history)
            finite_min_margin = min(finite_min_margin, finite.margin)
            finite_max_higher_order_ratio = max(
                finite_max_higher_order_ratio,
                higher_order_ratio,
            )
            finite_max_remainder_norm = max(finite_max_remainder_norm, remainder_norm)

        if micro_correct < len(candidates) and first_micro_failure is None:
            first_micro_failure = updates
        if (
            differential_correct < len(candidates)
            and first_differential_macro_failure is None
        ):
            first_differential_macro_failure = updates

        results[str(updates)] = {
            "micro_correct": micro_correct,
            "differential_macro_correct": differential_correct,
            "finite_pair_correct": finite_correct,
            "finite_pair_min_margin": finite_min_margin,
            "finite_pair_min_signature_separation": (
                finite_identifiability.minimum_signature_separation
            ),
            "finite_pair_max_higher_order_ratio": finite_max_higher_order_ratio,
            "finite_pair_max_remainder_norm": finite_max_remainder_norm,
            "max_singleton_stage_displacement_norm": max(
                float(torch.linalg.vector_norm(delta)) for delta in finite_deltas.values()
            ),
        }

        if finite_correct != len(candidates):
            raise RuntimeError(
                f"finite-pair decoder recovered only {finite_correct}/{len(candidates)} "
                f"histories at {updates} updates"
            )

    if results["1"]["micro_correct"] != len(candidates):
        raise RuntimeError("micro decoder must recover the one-update control")
    if results["1"]["differential_macro_correct"] != len(candidates):
        raise RuntimeError("differential macro decoder must recover the one-update control")
    if results["1"]["finite_pair_correct"] != len(candidates):
        raise RuntimeError("finite-pair decoder must recover the one-update control")
    if first_micro_failure is None:
        raise RuntimeError("micro decoder never left its local regime")
    if first_differential_macro_failure is None:
        raise RuntimeError(
            "differential macro decoder never failed in the fixed extended stress sweep"
        )
    if first_differential_macro_failure <= 64:
        raise RuntimeError(
            "differential macro decoder regressed inside its previously validated range"
        )
    if results[str(STAGE_LENGTHS[-1])]["finite_pair_correct"] != len(candidates):
        raise RuntimeError("finite-pair decoder did not survive through 256 updates/stage")

    payload = {
        "status": "ok",
        "model_parameters": theta0.numel(),
        "optimizer": "plain_sgd_no_momentum",
        "per_update_learning_rate": UPDATE_LR,
        "finite_difference_epsilon": FINITE_DIFFERENCE_EPSILON,
        "finite_pair_epsilon": None,
        "stage_lengths": list(STAGE_LENGTHS),
        "first_micro_failure_updates": first_micro_failure,
        "first_differential_macro_failure_updates": first_differential_macro_failure,
        "results": results,
        "claim": "finite_pair_interactions_extend_chronology_recovery_without_epsilon",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
