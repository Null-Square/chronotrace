"""Short-probe identification and budgeted chronology under affine structure.

This module does NOT validate that an arbitrary neural optimizer is affine.
It encloses every calculation using outward-rounded float64 elementary
operations. Soundness assumes finite IEEE-754 operations with gradual underflow,
valid input intervals, affine stage maps, and an in-model target. Identification
uses a posteriori inverse-residual validation, not an SVD tolerance as a proof.
The reachable-subset table is exponential, and search is factorial in the worst
case. No model-training call is made by the identifier or decoder.
"""

from __future__ import annotations

import hashlib
import heapq
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def _down(x: np.ndarray | float) -> np.ndarray:
    return np.nextafter(np.asarray(x, dtype=np.float64), -np.inf)


def _up(x: np.ndarray | float) -> np.ndarray:
    return np.nextafter(np.asarray(x, dtype=np.float64), np.inf)


@dataclass(frozen=True)
class Box:
    """Closed componentwise enclosure; arrays are copied and read-only."""

    lo: np.ndarray
    hi: np.ndarray

    def __post_init__(self) -> None:
        lo = np.array(self.lo, dtype=np.float64, copy=True)
        hi = np.array(self.hi, dtype=np.float64, copy=True)
        if lo.shape != hi.shape or not lo.size:
            raise ValueError("interval endpoints must have the same nonempty shape")
        if not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(lo > hi):
            raise ValueError("interval endpoints must be finite and ordered")
        lo.setflags(write=False)
        hi.setflags(write=False)
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)

    @classmethod
    def point(cls, value: np.ndarray | float) -> Box:
        return cls(value, value)

    @property
    def mid(self) -> np.ndarray:
        # A scheduling/estimation convenience, not a bound computation.
        return self.lo * 0.5 + self.hi * 0.5

    def __add__(self, other: Box) -> Box:
        return Box(_down(self.lo + other.lo), _up(self.hi + other.hi))

    def __sub__(self, other: Box) -> Box:
        return Box(_down(self.lo - other.hi), _up(self.hi - other.lo))

    def inflate(self, radius: float) -> Box:
        if not math.isfinite(radius) or radius < 0:
            raise ValueError("radius must be finite and nonnegative")
        return self + Box(-np.asarray(radius), np.asarray(radius))

    def overlaps(self, other: Box) -> bool:
        if self.lo.shape != other.lo.shape:
            raise ValueError("overlap operands must have identical shapes")
        return bool(np.all(self.lo <= other.hi) and np.all(other.lo <= self.hi))

    def contains(self, value: np.ndarray) -> bool:
        value = np.asarray(value, dtype=np.float64)
        return value.shape == self.lo.shape and bool(
            np.all(value >= self.lo) and np.all(value <= self.hi)
        )


def multiply(left: Box, right: Box) -> Box:
    """Outward-rounded componentwise interval product."""
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        products = np.stack((left.lo * right.lo, left.lo * right.hi,
                             left.hi * right.lo, left.hi * right.hi))
        return Box(_down(np.min(products, axis=0)), _up(np.max(products, axis=0)))


def cube(value: Box) -> Box:
    """Exploit monotonicity of the cube without assuming a libm power bound."""
    low, high = Box.point(value.lo), Box.point(value.hi)
    return Box(multiply(multiply(low, low), low).lo,
               multiply(multiply(high, high), high).hi)


def matmul(left: Box, right: Box) -> Box:
    """Interval matrix product, rounding EVERY product and accumulation outward.

    BLAS reductions are intentionally avoided: one final nextafter around a
    BLAS dot product is not a generally sound enclosure.
    """
    if left.lo.ndim != 2 or right.lo.ndim not in (1, 2):
        raise ValueError("expected a matrix times a matrix or vector")
    if left.lo.shape[1] != right.lo.shape[0]:
        raise ValueError("matrix dimensions do not align")
    vector = right.lo.ndim == 1
    rlo = right.lo[:, None] if vector else right.lo
    rhi = right.hi[:, None] if vector else right.hi
    lo = np.zeros((left.lo.shape[0], rlo.shape[1]))
    hi = lo.copy()
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        for k in range(left.lo.shape[1]):
            products = np.stack(
                (
                    left.lo[:, k, None] * rlo[k],
                    left.lo[:, k, None] * rhi[k],
                    left.hi[:, k, None] * rlo[k],
                    left.hi[:, k, None] * rhi[k],
                )
            )
            lo = _down(lo + _down(np.min(products, axis=0)))
            hi = _up(hi + _up(np.max(products, axis=0)))
    return Box(lo[:, 0], hi[:, 0]) if vector else Box(lo, hi)


def infinity_norm_upper(value: Box) -> float:
    """Upper bound on matrix induced infinity norm or vector infinity norm."""
    largest = np.maximum(np.abs(value.lo), np.abs(value.hi))
    if largest.ndim == 1:
        return float(np.max(largest))
    if largest.ndim != 2:
        raise ValueError("norm expects a matrix or vector")
    total = np.zeros(largest.shape[0])
    for k in range(largest.shape[1]):
        total = _up(total + largest[:, k])
    return float(np.max(total))


@dataclass(frozen=True)
class AffineBox:
    """Enclosure of a fixed map in coordinates centered on a known base."""

    matrix: Box
    offset: Box

    def __post_init__(self) -> None:
        d = self.offset.lo.size
        if self.offset.lo.shape != (d,) or self.matrix.lo.shape != (d, d):
            raise ValueError("affine map requires a square matrix and matching vector")

    def apply(self, value: Box) -> Box:
        return matmul(self.matrix, value) + self.offset

    def after(self, earlier: AffineBox) -> AffineBox:
        """Enclose self(earlier(x)); dependencies may be lost, never invented."""
        return AffineBox(
            matmul(self.matrix, earlier.matrix),
            matmul(self.matrix, earlier.offset) + self.offset,
        )


@dataclass(frozen=True)
class Identification:
    maps: tuple[AffineBox, ...]
    inverse_residual_bounds: tuple[float, ...]
    matrix_error_bounds: tuple[float, ...]
    stage_calls: int


def identify_from_short_probes(
    base: np.ndarray,
    starts: Sequence[np.ndarray],
    singletons: Sequence[Box],
    pair_outputs: dict[tuple[int, int], Box],
) -> Identification:
    """Identify affine maps from N singletons and N(N-1) second-stage probes.

    pair_outputs[(source, destination)] encloses T_destination(starts[source]).
    Starts are exact stored float vectors (typically singleton midpoints).
    No generating target history, Hessian, or true stage matrix is accepted.

    Let X contain starts_i-base, Y contain T_j(starts_i)-T_j(base), i != j.
    For any approximate right inverse Q and estimate Ahat, if
    e = ||I-XQ||_inf < 1 then
    ||A_j-Ahat||_inf <= ||(Y-Ahat X)Q||_inf / (1-e).
    All residual arithmetic is enclosed. Each entry receives this conservative
    common error bound. Rank-deficient/uncertifiable input fails closed.
    """
    base = np.asarray(base, dtype=np.float64)
    n = len(starts)
    if base.ndim != 1 or not base.size or not np.isfinite(base).all():
        raise ValueError("base must be a finite nonempty vector")
    d = base.size
    if n < 2 or len(singletons) != n:
        raise ValueError("need matching starts/singletons for at least two stages")
    starts = tuple(np.asarray(start, dtype=np.float64) for start in starts)
    if any(start.shape != (d,) or not np.isfinite(start).all() for start in starts):
        raise ValueError("starts must match the base shape and be finite")
    if any(value.lo.shape != (d,) for value in singletons):
        raise ValueError("singleton shapes must match the base")
    expected = {(i, j) for i in range(n) for j in range(n) if i != j}
    if set(pair_outputs) != expected:
        raise ValueError("need exactly every distinct ordered pair")
    if any(value.lo.shape != (d,) for value in pair_outputs.values()):
        raise ValueError("pair output shapes must match the base")
    maps, errors, residuals = [], [], []
    for j in range(n):
        sources = [i for i in range(n) if i != j]
        x = Box.point(np.stack([starts[i] for i in sources], axis=1)) - Box.point(
            base[:, None]
        )
        y = Box(
            np.stack([pair_outputs[i, j].lo for i in sources], axis=1),
            np.stack([pair_outputs[i, j].hi for i in sources], axis=1),
        ) - Box(singletons[j].lo[:, None], singletons[j].hi[:, None])
        q = Box.point(np.linalg.pinv(x.mid))
        estimate = Box.point(y.mid @ q.mid)
        residual = Box.point(np.eye(d)) - matmul(x, q)
        e = infinity_norm_upper(residual)
        if e >= 1.0:
            raise ValueError(f"stage {j}: short probes do not certify spanning (e={e})")
        numerator = infinity_norm_upper(matmul(y - matmul(estimate, x), q))
        denominator = float(_down(1.0 - e))
        error = float(_up(numerator / denominator))
        matrix = estimate.inflate(error)
        offset = singletons[j] - Box.point(base)
        maps.append(AffineBox(matrix, offset))
        errors.append(error)
        residuals.append(e)
    return Identification(tuple(maps), tuple(residuals), tuple(errors), n * n)


def _validated_remainders(n: int, values: np.ndarray | None) -> np.ndarray:
    result = np.zeros((n, 1 << n)) if values is None else np.asarray(values, dtype=np.float64)
    if result.shape != (n, 1 << n) or not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError("remainder table must be finite, nonnegative and N by 2**N")
    return result


def _context_stage(stage: AffineBox, error: float) -> AffineBox:
    return stage if error == 0.0 else AffineBox(stage.matrix, stage.offset.inflate(error))


def _map_digest(maps: Sequence[AffineBox], remainders: np.ndarray) -> str:
    digest = hashlib.sha256()
    for stage in maps:
        for value in (stage.matrix.lo, stage.matrix.hi, stage.offset.lo, stage.offset.hi):
            digest.update(str(value.shape).encode())
            digest.update(value.tobytes())
    digest.update(remainders.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class SubsetCache:
    boxes: tuple[Box, ...]
    applications: int
    maps_sha256: str


def subset_enclosures(
    maps: Sequence[AffineBox], *, remainder_bounds: np.ndarray | None = None
) -> SubsetCache:
    """Enclose all endpoints for each subset, without enumerating its orders.

    R(S) is enclosed by the componentwise hull of T_j(R(S minus {j})).
    Cost: N*2**(N-1) affine-box applications; memory O(d*2**N).
    """
    maps = tuple(maps)
    if not maps or len(maps) > 16:
        raise ValueError("one to sixteen stages supported by the subset table")
    remainders = _validated_remainders(len(maps), remainder_bounds)
    d = maps[0].offset.lo.size
    if any(stage.offset.lo.size != d for stage in maps):
        raise ValueError("stage dimensions differ")
    boxes = [Box.point(np.zeros(d))]
    applications = 0
    for mask in range(1, 1 << len(maps)):
        children = []
        for j, stage in enumerate(maps):
            if mask & (1 << j):
                previous = mask ^ (1 << j)
                contextual = _context_stage(stage, float(remainders[j, previous]))
                children.append(contextual.apply(boxes[previous]))
                applications += 1
        boxes.append(
            Box(
                np.min(np.stack([child.lo for child in children]), axis=0),
                np.max(np.stack([child.hi for child in children]), axis=0),
            )
        )
    return SubsetCache(tuple(boxes), applications, _map_digest(maps, remainders))


@dataclass(frozen=True)
class FrontierClass:
    remaining_mask: int
    suffix: tuple[int, ...]


@dataclass(frozen=True)
class DecodeResult:
    status: str
    unique_order: tuple[int, ...] | None
    precedences: tuple[tuple[int, int], ...]
    unexcluded_histories: int
    bound_evaluations: int
    subset_applications: int
    suffix_compositions: int
    pruned_classes: int
    frontier: tuple[FrontierClass, ...]


def _summarize(
    n: int,
    live: Sequence[FrontierClass],
    evaluated: int,
    subset_calls: int,
    compositions: int,
    pruned: int,
    exhausted: bool,
) -> DecodeResult:
    unexcluded = sum(math.factorial(node.remaining_mask.bit_count()) for node in live)
    relations = set((i, j) for i in range(n) for j in range(n) if i != j)
    for node in live:
        positions = {stage: pos for pos, stage in enumerate(node.suffix)}
        implied = {
            (i, j)
            for i in range(n)
            for j in range(n)
            if i != j and j in positions and (
                i not in positions or positions[i] < positions[j]
            )
        }
        relations.intersection_update(implied)
    unique = None
    if unexcluded == 1:
        node = live[0]
        remaining = tuple(i for i in range(n) if node.remaining_mask & (1 << i))
        unique = remaining + node.suffix
    status = (
        "inconsistent" if not live else "unique" if unique is not None
        else "budget_exhausted" if exhausted else "ambiguous"
    )
    return DecodeResult(
        status, unique, tuple(sorted(relations)) if live else (), unexcluded,
        evaluated, subset_calls, compositions, pruned, tuple(live),
    )


def decode_budgeted(
    maps: Sequence[AffineBox],
    target: Box,
    *,
    max_bound_evaluations: int,
    precomputed: SubsetCache | None = None,
    remainder_bounds: np.ndarray | None = None,
) -> DecodeResult:
    """Prune suffix classes using subset enclosures, and abstain at the budget.

    remainder_bounds[j, S], when supplied, must bound the nonlinear deviation
    from map j on EVERY true state reachable with exactly prefix-set S. These
    externally justified bounds permit nonlinear maps; they are not estimated
    or made valid by the decoder. Zero bounds assert exact affine structure.

    The budget counts target-dependent class-bound evaluations, not training.
    Subset preparation and suffix compositions are reported separately. Every
    unexplored class stays in the returned frontier; no beam-search truncation
    is used. Pairwise output is the intersection of implications of ALL frontier
    classes. An empty set means model inconsistency, never a guessed order.

    A caller-supplied precomputed table must come from subset_enclosures(maps).
    Its map fingerprint is checked. Cache contents themselves remain trusted.
    """
    if isinstance(max_bound_evaluations, bool) or not isinstance(max_bound_evaluations, int):
        raise ValueError("budget must be a nonnegative integer")
    if max_bound_evaluations < 0:
        raise ValueError("budget must be a nonnegative integer")
    maps = tuple(maps)
    n = len(maps)
    if not 1 <= n <= 16:
        raise ValueError("one to sixteen stages supported")
    d = maps[0].offset.lo.size
    if target.lo.shape != (d,) or any(stage.offset.lo.size != d for stage in maps):
        raise ValueError("target and map dimensions differ")
    mask = (1 << n) - 1
    root = FrontierClass(mask, ())
    if max_bound_evaluations == 0:
        return _summarize(n, [root], 0, 0, 0, 0, True)
    remainders = _validated_remainders(n, remainder_bounds)
    cache = precomputed if precomputed is not None else subset_enclosures(
        maps, remainder_bounds=remainders
    )
    if cache.maps_sha256 != _map_digest(maps, remainders):
        raise ValueError("subset cache belongs to different stage maps")
    boxes, subset_calls = cache.boxes, cache.applications
    if len(boxes) != 1 << n or any(box.lo.shape != (d,) for box in boxes):
        raise ValueError("invalid subset cache shape")
    if not boxes[mask].overlaps(target):
        return _summarize(n, [], 1, subset_calls, 0, 1, False)
    identity = AffineBox(Box.point(np.eye(d)), Box.point(np.zeros(d)))
    # Depth-first, lexicographic tie break. Priority never uses hidden labels.
    queue = [(0, (), mask, identity)]
    leaves: list[FrontierClass] = []
    evaluated, compositions, pruned = 1, 0, 0
    exhausted = False
    while queue:
        _, suffix, remaining, outer = heapq.heappop(queue)
        if not remaining:
            leaves.append(FrontierClass(remaining, suffix))
            continue
        if evaluated + remaining.bit_count() > max_bound_evaluations:
            heapq.heappush(queue, (-len(suffix), suffix, remaining, outer))
            exhausted = True
            break
        for j in range(n):
            if not remaining & (1 << j):
                continue
            child_suffix = (j,) + suffix
            child_mask = remaining ^ (1 << j)
            contextual = _context_stage(maps[j], float(remainders[j, child_mask]))
            child_outer = outer.after(contextual)
            compositions += 1
            enclosure = child_outer.apply(boxes[child_mask])
            evaluated += 1
            if enclosure.overlaps(target):
                heapq.heappush(
                    queue, (-len(child_suffix), child_suffix, child_mask, child_outer)
                )
            else:
                pruned += 1
    live = leaves + [FrontierClass(node[2], node[1]) for node in queue]
    return _summarize(n, live, evaluated, subset_calls, compositions, pruned, exhausted)
