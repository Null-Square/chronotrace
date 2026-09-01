"""Label-blind pairwise chronology decisions from proof-safe class certificates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from chronotrace.geometry.local_order_hierarchy import LocalOrderHierarchy
from chronotrace.geometry.multi_witness_local_order import (
    LocalOrderMultiWitnessCertificate,
    solve_local_order_multi_witness_lp,
)


@dataclass(frozen=True)
class PairwiseChronologyCertificate:
    """Certificates for both orientations of one unordered stage pair."""

    left: str
    right: str
    left_before_right: LocalOrderMultiWitnessCertificate
    right_before_left: LocalOrderMultiWitnessCertificate
    left_before_right_impossible: bool
    right_before_left_impossible: bool
    inferred_precedence: tuple[str, str] | None
    status: str


def certify_pairwise_orientation(
    hierarchy: LocalOrderHierarchy,
    constants: Any,
    coefficients: Any,
    left: str,
    right: str,
    *,
    elimination_guard: float,
    certificate_guard: float = 1e-10,
) -> PairwiseChronologyCertificate:
    """Test both orientations and infer only from one-sided certified exclusion.

    No true chronology label is supplied.  The two precedence classes partition global
    permutations for distinct stages.  If exactly one class is certified impossible, the
    opposite orientation is inferred.  If neither is excluded the method abstains for the
    pair.  If both are excluded, the result is internally inconsistent with the assumed
    finite chronology model and is marked invalid.
    """

    before = str(left)
    after = str(right)
    if before == after:
        raise ValueError("pairwise stages must be distinct")
    if before not in hierarchy.stages or after not in hierarchy.stages:
        raise ValueError("pairwise stage is outside the hierarchy")
    guard = float(elimination_guard)
    if guard < 0.0:
        raise ValueError("elimination_guard must be non-negative")

    left_class = solve_local_order_multi_witness_lp(
        hierarchy,
        constants,
        coefficients,
        precedences=((before, after),),
        certificate_guard=certificate_guard,
    )
    right_class = solve_local_order_multi_witness_lp(
        hierarchy,
        constants,
        coefficients,
        precedences=((after, before),),
        certificate_guard=certificate_guard,
    )
    left_impossible = left_class.euclidean_distance_lower_bound > guard
    right_impossible = right_class.euclidean_distance_lower_bound > guard

    if left_impossible and right_impossible:
        inferred = None
        status = "invalid_both_orientations_excluded"
    elif left_impossible:
        inferred = (after, before)
        status = "certified"
    elif right_impossible:
        inferred = (before, after)
        status = "certified"
    else:
        inferred = None
        status = "ambiguous"

    return PairwiseChronologyCertificate(
        left=before,
        right=after,
        left_before_right=left_class,
        right_before_left=right_class,
        left_before_right_impossible=left_impossible,
        right_before_left_impossible=right_impossible,
        inferred_precedence=inferred,
        status=status,
    )
