"""Label-blind logical closure and terminal distance-bound decisions.

Closure preserves the soundness assumptions of its input certificates; it cannot
repair unsound input edges. Distance decisions require bounds on *complete*
chronology classes. Truncated predictions without an omitted-tail bound are not
valid inputs. No routine here supplies a floating-point roundoff proof.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class PrecedenceClosure:
    """An auditable partial order, or explicit rejection of cyclic input.

    Each proof is a path made only of original edges. ``unique_order`` is set
    only when exactly one global order is consistent with all supplied edges.
    For cyclic input no entailed edges or order are returned.
    """

    status: str
    direct_relations: tuple[tuple[str, str], ...]
    entailed_relations: tuple[tuple[str, str], ...]
    derived_relations: tuple[tuple[str, str], ...]
    proof_paths: tuple[tuple[str, ...], ...]
    unique_order: tuple[str, ...] | None


def close_certified_precedences(
    stages: Sequence[str], relations: Iterable[tuple[str, str]]
) -> PrecedenceClosure:
    """Close certified edges and detect a unique topological order.

    Uses O(N(N+E)) graph work, excluding materialized proof-path output, and
    never enumerates permutations. Duplicate edges are harmless. Stage labels
    may contain multiple characters. Ground-truth history is not an input.
    """

    names = tuple(stages)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("stages must be nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError("stages must be unique")
    index = {name: position for position, name in enumerate(names)}
    edges: set[tuple[str, str]] = set()
    for relation in relations:
        if not isinstance(relation, (tuple, list)) or len(relation) != 2:
            raise ValueError("each relation must be a pair of stage names")
        before, after = relation
        if not isinstance(before, str) or not isinstance(after, str):
            raise ValueError("edge endpoints must be stage-name strings")
        if before not in index or after not in index or before == after:
            raise ValueError("edge endpoints must be distinct declared stages")
        edges.add((before, after))
    def edge_key(edge: tuple[str, str]) -> tuple[int, int]:
        return index[edge[0]], index[edge[1]]

    direct = tuple(sorted(edges, key=edge_key))
    adjacency: dict[str, list[str]] = {name: [] for name in names}
    indegree = dict.fromkeys(names, 0)
    for before, after in direct:
        adjacency[before].append(after)
        indegree[after] += 1

    ready = deque(name for name in names if indegree[name] == 0)
    order: list[str] = []
    unique = True
    while ready:
        unique = unique and len(ready) == 1
        before = ready.popleft()
        order.append(before)
        for after in adjacency[before]:
            indegree[after] -= 1
            if indegree[after] == 0:
                ready.append(after)
    if len(order) != len(names):
        return PrecedenceClosure("invalid_cycle", direct, (), (), (), None)

    paths: dict[tuple[str, str], tuple[str, ...]] = {}
    for source in names:
        parent: dict[str, str | None] = {source: None}
        queue = deque([source])
        while queue:
            before = queue.popleft()
            for after in adjacency[before]:
                if after not in parent:
                    parent[after] = before
                    queue.append(after)
        for target in parent:
            if target == source:
                continue
            reverse_path = [target]
            cursor = target
            while cursor != source:
                predecessor = parent[cursor]
                if predecessor is None:  # pragma: no cover - internal invariant
                    raise RuntimeError("broken reachability proof")
                reverse_path.append(predecessor)
                cursor = predecessor
            paths[(source, target)] = tuple(reversed(reverse_path))
    entailed = tuple(sorted(paths, key=edge_key))
    derived = tuple(edge for edge in entailed if edge not in edges)
    return PrecedenceClosure(
        "unique" if unique else "partial",
        direct,
        entailed,
        derived,
        tuple(paths[edge] for edge in entailed),
        tuple(order) if unique else None,
    )


@dataclass(frozen=True)
class TwoSidedDistanceDecision:
    """One-sided class exclusion, abstention, or a violated model assumption."""

    status: str
    left_excluded: bool
    right_excluded: bool
    inferred_precedence: tuple[str, str] | None


def decide_from_class_lower_bounds(
    left: str,
    right: str,
    left_before_right_bound: float,
    right_before_left_bound: float,
    *,
    residual_budget: float,
    numerical_guard: float = 0.0,
) -> TwoSidedDistanceDecision:
    """Infer an orientation only by excluding its opposite complete class.

    ``residual_budget`` must cover the stated endpoint/model discrepancy.
    ``numerical_guard`` is an externally justified allowance, not a guarantee
    manufactured by this function. Bounds may be negative and conservative.
    Both exclusions indicate inconsistency; they do not justify choosing one.
    """

    if not isinstance(left, str) or not isinstance(right, str):
        raise ValueError("stage names must be strings")
    if not left or not right or left == right:
        raise ValueError("stage names must be nonempty and distinct")
    values = tuple(
        float(value)
        for value in (
            left_before_right_bound,
            right_before_left_bound,
            residual_budget,
            numerical_guard,
        )
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("bounds and budgets must be finite")
    forward, reverse, budget, guard = values
    if budget < 0.0 or guard < 0.0:
        raise ValueError("budgets must be nonnegative")
    excluded_forward = forward - guard > budget
    excluded_reverse = reverse - guard > budget
    if excluded_forward and excluded_reverse:
        status, inferred = "invalid_both_excluded", None
    elif excluded_forward:
        status, inferred = "certified", (right, left)
    elif excluded_reverse:
        status, inferred = "certified", (left, right)
    else:
        status, inferred = "ambiguous", None
    return TwoSidedDistanceDecision(status, excluded_forward, excluded_reverse, inferred)
