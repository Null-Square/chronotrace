"""Geometry primitives for training-history reconstruction."""

from chronotrace.geometry.commutator import (
    PairwiseGeometry,
    PermutationDecode,
    decode_pairwise_order,
    decode_permutation,
    estimate_step_size,
    local_stage_derivatives,
    multi_stage_symmetric_reference,
    pairwise_chrono_score,
    pairwise_endpoint_geometry,
    permutation_signature,
)
from chronotrace.geometry.identifiability import (
    ChronologyIdentifiability,
    bracket_matrix,
    chronology_identifiability,
    normalized_remainder_ratio,
)
from chronotrace.geometry.operators import (
    OperatorPairGeometry,
    OperatorPermutationDecode,
    decode_operator_permutation,
    displacement_jvp_finite_difference,
    local_stage_map_derivatives,
    operator_pair_geometry,
    operator_pair_score,
    operator_permutation_signature,
    operator_symmetric_reference,
    stage_displacement,
)

__all__ = [
    "ChronologyIdentifiability",
    "OperatorPairGeometry",
    "OperatorPermutationDecode",
    "PairwiseGeometry",
    "PermutationDecode",
    "bracket_matrix",
    "chronology_identifiability",
    "decode_operator_permutation",
    "decode_pairwise_order",
    "decode_permutation",
    "displacement_jvp_finite_difference",
    "estimate_step_size",
    "local_stage_derivatives",
    "local_stage_map_derivatives",
    "multi_stage_symmetric_reference",
    "normalized_remainder_ratio",
    "operator_pair_geometry",
    "operator_pair_score",
    "operator_permutation_signature",
    "operator_symmetric_reference",
    "pairwise_chrono_score",
    "pairwise_endpoint_geometry",
    "permutation_signature",
    "stage_displacement",
]
