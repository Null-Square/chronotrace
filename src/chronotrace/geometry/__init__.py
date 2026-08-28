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

__all__ = [
    "ChronologyIdentifiability",
    "PairwiseGeometry",
    "PermutationDecode",
    "bracket_matrix",
    "chronology_identifiability",
    "decode_pairwise_order",
    "decode_permutation",
    "estimate_step_size",
    "local_stage_derivatives",
    "multi_stage_symmetric_reference",
    "normalized_remainder_ratio",
    "pairwise_chrono_score",
    "pairwise_endpoint_geometry",
    "permutation_signature",
]
