"""Scalar witness projections of ordered interaction bases.

Linearity allows Möbius interaction algebra to be performed after projection onto a fixed
certificate direction.  A newly measured order-K endpoint therefore needs only one scalar
projection per witness direction; the full order-K interaction tensor can be discarded.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from chronotrace.geometry.interactions import ordered_subsequences


Word = tuple[str, ...]


def projected_interaction_from_endpoint_delta(
    word: Sequence[str],
    endpoint_minus_base_projection: float,
    lower_projected_interactions: Mapping[Word, float],
) -> float:
    """Recover <u,Phi(word)> from <u,E(word)-base> and proper-subsequence projections."""

    values = tuple(str(stage) for stage in word)
    if not values or len(values) != len(set(values)):
        raise ValueError("word must contain at least one distinct stage")
    endpoint_projection = float(endpoint_minus_base_projection)
    if not math.isfinite(endpoint_projection):
        raise ValueError("endpoint projection must be finite")

    interaction = endpoint_projection
    if len(values) > 1:
        for subsequence in ordered_subsequences(values, max_degree=len(values) - 1):
            if subsequence not in lower_projected_interactions:
                raise ValueError(f"missing lower projected interaction {subsequence!r}")
            value = float(lower_projected_interactions[subsequence])
            if not math.isfinite(value):
                raise ValueError("lower projected interactions must be finite")
            interaction -= value
    return interaction


def projected_word_prediction(
    word: Sequence[str],
    projected_interactions: Mapping[Word, float],
    *,
    max_degree: int,
) -> float:
    """Return <u,P_K(word)-base> from projected interactions."""

    values = tuple(str(stage) for stage in word)
    degree = int(max_degree)
    if degree < 1:
        raise ValueError("max_degree must be positive")
    total = 0.0
    for subsequence in ordered_subsequences(values, max_degree=min(degree, len(values))):
        if subsequence not in projected_interactions:
            raise ValueError(f"missing projected interaction {subsequence!r}")
        value = float(projected_interactions[subsequence])
        if not math.isfinite(value):
            raise ValueError("projected interactions must be finite")
        total += value
    return total
