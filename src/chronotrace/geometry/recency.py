"""Simple chronology baselines from final per-stage capability losses."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations


@dataclass(frozen=True)
class StageLossRecencyDecision:
    """Permutation implied by monotone forgetting / recency in final stage losses."""

    permutation: tuple[str, ...]
    minimum_adjacent_loss_gap: float
    identifiable: bool


def decode_stage_loss_recency(losses: Mapping[str, float]) -> StageLossRecencyDecision:
    """Predict oldest-to-newest stage order by descending final loss.

    The baseline encodes the ordinary recency hypothesis: earlier stages are more forgotten
    and therefore have higher final completion loss, while later stages retain lower loss.
    No chronology-interaction probes are used.
    """

    if len(losses) < 2:
        raise ValueError("at least two stage losses are required")
    normalized = {str(stage): float(value) for stage, value in losses.items()}
    if len(normalized) != len(losses):
        raise ValueError("stage names must be unique after string conversion")
    if any(not math.isfinite(value) for value in normalized.values()):
        raise ValueError("stage losses must be finite")

    ranked = tuple(sorted(normalized, key=lambda stage: (-normalized[stage], stage)))
    gaps = [
        normalized[ranked[index]] - normalized[ranked[index + 1]]
        for index in range(len(ranked) - 1)
    ]
    minimum_gap = min(gaps)
    return StageLossRecencyDecision(
        permutation=ranked,
        minimum_adjacent_loss_gap=minimum_gap,
        identifiable=minimum_gap > 0.0,
    )


def stage_loss_recency_precedence(
    losses: Mapping[str, float],
) -> dict[tuple[str, str], tuple[str, str] | None]:
    """Return pairwise precedence from the same monotone-recency assumption.

    Exact equal losses are treated as non-identifiable rather than tie-broken.
    """

    normalized = {str(stage): float(value) for stage, value in losses.items()}
    if any(not math.isfinite(value) for value in normalized.values()):
        raise ValueError("stage losses must be finite")
    result: dict[tuple[str, str], tuple[str, str] | None] = {}
    for first, second in combinations(sorted(normalized), 2):
        first_loss = normalized[first]
        second_loss = normalized[second]
        if first_loss > second_loss:
            result[(first, second)] = (first, second)
        elif second_loss > first_loss:
            result[(first, second)] = (second, first)
        else:
            result[(first, second)] = None
    return result
