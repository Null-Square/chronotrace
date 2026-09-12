# ruff: noqa: I001
"""Local-order coordinates for scalable K3 chronology relaxations.

A degree-three ordered-interaction prediction can be written as an affine function of
pair precedence variables and six-way local order variables for every unordered triple.
Every global permutation maps to a 0/1 vertex of this representation. Relaxing those
vertices to locally consistent probabilities gives a conservative superset of the global
linear-ordering polytope; optimization over that superset may certify that a candidate
state is impossible without enumerating every full chronology.

This module only builds and validates the geometry. It deliberately does not prescribe a
QP solver or claim that the local relaxation is integral for N > 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Any

from chronotrace.geometry.interactions import OrderedInteractionBasis


Pair = tuple[str, str]
Triple = tuple[str, str, str]
TripleOrder = tuple[str, str, str]


@dataclass(frozen=True)
class K3LocalOrderRelaxation:
    """Affine K3 prediction geometry indexed by local ordering variables."""

    stages: tuple[str, ...]
    pairs: tuple[Pair, ...]
    triples: tuple[Triple, ...]
    constant: Any
    pair_coefficients: dict[Pair, Any]
    triple_coefficients: dict[tuple[Triple, TripleOrder], Any]


@dataclass(frozen=True)
class K3LocalOrderPoint:
    """One feasible point of the local pair/triple ordering relaxation."""

    pair_precedence: dict[Pair, float]
    triple_order_weights: dict[Triple, dict[TripleOrder, float]]


def build_k3_local_order_relaxation(basis: OrderedInteractionBasis) -> K3LocalOrderRelaxation:
    """Rewrite a measured K3 interaction basis in affine local-order coordinates."""

    if basis.max_degree < 3:
        raise ValueError("K3 local-order relaxation requires max_degree >= 3")
    stages = tuple(basis.stages)
    pairs = tuple(combinations(stages, 2))
    triples = tuple(combinations(stages, 3))

    constant = basis.base.clone()
    for stage in stages:
        constant = constant + basis.interactions[(stage,)]

    pair_coefficients: dict[Pair, Any] = {}
    for first, second in pairs:
        constant = constant + basis.interactions[(second, first)]
        pair_coefficients[(first, second)] = (
            basis.interactions[(first, second)] - basis.interactions[(second, first)]
        )

    triple_coefficients: dict[tuple[Triple, TripleOrder], Any] = {}
    for triple in triples:
        for order in permutations(triple):
            triple_coefficients[(triple, order)] = basis.interactions[order]

    return K3LocalOrderRelaxation(
        stages=stages,
        pairs=pairs,
        triples=triples,
        constant=constant,
        pair_coefficients=pair_coefficients,
        triple_coefficients=triple_coefficients,
    )


def k3_local_order_vertex(
    chronology: tuple[str, ...],
    relaxation: K3LocalOrderRelaxation,
) -> K3LocalOrderPoint:
    """Map one full chronology to its exact 0/1 local-order vertex."""

    if len(chronology) != len(relaxation.stages) or set(chronology) != set(relaxation.stages):
        raise ValueError("chronology must contain every relaxation stage exactly once")
    position = {stage: index for index, stage in enumerate(chronology)}
    pair_precedence = {
        pair: float(position[pair[0]] < position[pair[1]]) for pair in relaxation.pairs
    }
    triple_order_weights: dict[Triple, dict[TripleOrder, float]] = {}
    for triple in relaxation.triples:
        induced = tuple(sorted(triple, key=position.__getitem__))
        triple_order_weights[triple] = {
            order: float(order == induced) for order in permutations(triple)
        }
    return K3LocalOrderPoint(
        pair_precedence=pair_precedence,
        triple_order_weights=triple_order_weights,
    )


def validate_k3_local_order_point(
    point: K3LocalOrderPoint,
    relaxation: K3LocalOrderRelaxation,
    *,
    tolerance: float = 1e-12,
) -> None:
    """Validate box, simplex, and pair/triple marginal consistency constraints."""

    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    if set(point.pair_precedence) != set(relaxation.pairs):
        raise ValueError("pair-precedence coordinate set mismatch")
    if set(point.triple_order_weights) != set(relaxation.triples):
        raise ValueError("triple-weight coordinate set mismatch")

    for pair, value in point.pair_precedence.items():
        if value < -tolerance or value > 1.0 + tolerance:
            raise ValueError(f"pair coordinate {pair!r} is outside [0, 1]")

    for triple in relaxation.triples:
        weights = point.triple_order_weights[triple]
        expected_orders = set(permutations(triple))
        if set(weights) != expected_orders:
            raise ValueError(f"triple {triple!r} does not contain all six local orders")
        if any(value < -tolerance for value in weights.values()):
            raise ValueError(f"triple {triple!r} contains a negative weight")
        if abs(sum(weights.values()) - 1.0) > tolerance:
            raise ValueError(f"triple {triple!r} weights do not sum to one")

        for first, second in combinations(triple, 2):
            pair = (first, second)
            marginal = sum(
                weight
                for order, weight in weights.items()
                if order.index(first) < order.index(second)
            )
            if abs(marginal - point.pair_precedence[pair]) > tolerance:
                raise ValueError(
                    f"triple {triple!r} marginal disagrees with pair coordinate {pair!r}"
                )


def k3_local_order_prediction(
    point: K3LocalOrderPoint,
    relaxation: K3LocalOrderRelaxation,
    *,
    validate: bool = True,
) -> Any:
    """Evaluate the affine K3 parameter prediction at one local-order point."""

    if validate:
        validate_k3_local_order_point(point, relaxation)
    prediction = relaxation.constant.clone()
    for pair in relaxation.pairs:
        prediction = prediction + point.pair_precedence[pair] * relaxation.pair_coefficients[pair]
    for triple in relaxation.triples:
        for order, weight in point.triple_order_weights[triple].items():
            prediction = prediction + weight * relaxation.triple_coefficients[(triple, order)]
    return prediction


def mix_k3_local_order_points(
    first: K3LocalOrderPoint,
    second: K3LocalOrderPoint,
    *,
    first_weight: float,
) -> K3LocalOrderPoint:
    """Return a convex mixture of two local-order points."""

    weight = float(first_weight)
    if weight < 0.0 or weight > 1.0:
        raise ValueError("first_weight must be in [0, 1]")
    if set(first.pair_precedence) != set(second.pair_precedence):
        raise ValueError("pair coordinate sets differ")
    if set(first.triple_order_weights) != set(second.triple_order_weights):
        raise ValueError("triple coordinate sets differ")

    pair_precedence = {
        pair: weight * first.pair_precedence[pair] + (1.0 - weight) * second.pair_precedence[pair]
        for pair in first.pair_precedence
    }
    triple_order_weights: dict[Triple, dict[TripleOrder, float]] = {}
    for triple in first.triple_order_weights:
        if set(first.triple_order_weights[triple]) != set(second.triple_order_weights[triple]):
            raise ValueError("triple local-order coordinate sets differ")
        triple_order_weights[triple] = {
            order: weight * first.triple_order_weights[triple][order]
            + (1.0 - weight) * second.triple_order_weights[triple][order]
            for order in first.triple_order_weights[triple]
        }
    return K3LocalOrderPoint(
        pair_precedence=pair_precedence,
        triple_order_weights=triple_order_weights,
    )
