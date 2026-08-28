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

__all__ = [
    "PairwiseGeometry",
    "PermutationDecode",
    "decode_pairwise_order",
    "decode_permutation",
    "estimate_step_size",
    "local_stage_derivatives",
    "multi_stage_symmetric_reference",
    "pairwise_chrono_score",
    "pairwise_endpoint_geometry",
    "permutation_signature",
]
