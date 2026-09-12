"""Conditional training-chronology certificates from a bidirectional join.

Meet-in-the-middle search is classical. The guarantees here require the declared
bounds and candidate recipes to be correct. No floating-point interval proof is
provided. A bounded observation cell is propagated through residual-controlled
inverse GD. A failed inverse solve enlarges uncertainty; it never drops a word.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class GradientDynamics:
    """Known deterministic GD stages, with per-stage global gradient Lipschitz bounds.

    The gradient callback receives a stage index and a batch (B,D); return (B,D).
    `nonexpansive` is an additional proved assumption (e.g. convex GD, eta L<=1),
    not inferred. Otherwise 1+eta L is used as a forward Lipschitz constant.
    `roundoff_allowance` bounds error per parameter-update vector, externally.
    """
    names: tuple[str, ...]
    base: np.ndarray
    steps: int
    eta: float
    lipschitz: np.ndarray
    gradient: Callable[[int, np.ndarray], np.ndarray]
    regularity_source: str
    nonexpansive: bool = False
    roundoff_allowance: float = 1e-12
    gradient_evaluations: int = 0

    def __post_init__(self):
        self.names = tuple(self.names)
        self.base = np.array(self.base, dtype=np.float64, copy=True)
        self.lipschitz = np.array(self.lipschitz, dtype=np.float64, copy=True)
        if len(self.names) < 2 or len(set(self.names)) != len(self.names):
            raise ValueError('at least two distinct stages required')
        if any(not isinstance(s, str) or not s for s in self.names):
            raise ValueError('stage names must be nonempty strings')
        if self.base.ndim != 1 or not len(self.base) or not np.isfinite(self.base).all():
            raise ValueError('base must be a nonempty finite vector')
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps < 1:
            raise ValueError('steps must be a positive integer')
        if not math.isfinite(self.eta) or self.eta <= 0:
            raise ValueError('eta must be finite and positive')
        if (self.lipschitz.shape != (len(self.names),) or
                not np.isfinite(self.lipschitz).all() or np.any(self.lipschitz < 0)):
            raise ValueError('invalid gradient Lipschitz bounds')
        if np.any(self.eta*self.lipschitz >= 1):
            raise ValueError('inverse certificate requires eta L < 1')
        if not isinstance(self.regularity_source, str) or not self.regularity_source.strip():
            raise ValueError('regularity source must be declared')
        if not math.isfinite(self.roundoff_allowance) or self.roundoff_allowance < 0:
            raise ValueError('roundoff allowance must be finite and nonnegative')

    def grad(self, j: int, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.base) or not np.isfinite(x).all():
            raise ValueError('gradient state must be a finite batch')
        if not 0 <= j < len(self.names):
            raise ValueError('unknown stage index')
        result = np.asarray(self.gradient(j, x.copy()), dtype=np.float64)
        self.gradient_evaluations += len(x)
        if result.shape != x.shape or not np.isfinite(result).all():
            raise ValueError('gradient callback returned invalid output')
        return result.copy()

    def forward(self, j: int, x: np.ndarray, error: np.ndarray | None = None):
        y = np.array(x, dtype=np.float64, copy=True)
        e = np.zeros(len(y)) if error is None else np.asarray(error, dtype=float).copy()
        if e.shape != (len(y),) or not np.isfinite(e).all() or np.any(e < 0):
            raise ValueError('invalid forward error radii')
        gain = 1.0 if self.nonexpansive else 1+self.eta*self.lipschitz[j]
        for _ in range(self.steps):
            y -= self.eta*self.grad(j, y)
            e = gain*e + self.roundoff_allowance
        return y, e

    def inverse(self, j: int, y: np.ndarray, error: np.ndarray,
                *, residual_tolerance: float = 1e-12, max_iterations: int = 24):
        """Return approximate F_j^{-1}(y) and a per-row enclosing radius.

        For T(x)=x-eta grad(x), ||T(x)-z||=r implies
        ||x-T^{-1}(z*)|| <= (r+||z-z*||)/(1-eta L).
        This residual bound remains valid on iteration exhaustion.
        Every evaluated row gradient, including residual checks, is charged.
        """
        if not math.isfinite(residual_tolerance) or residual_tolerance < 0:
            raise ValueError('residual tolerance must be finite and nonnegative')
        if (
            isinstance(max_iterations, bool)
            or not isinstance(max_iterations, int)
            or max_iterations < 1
        ):
            raise ValueError('max_iterations must be positive')
        z = np.array(y, dtype=np.float64, copy=True)
        e = np.array(error, dtype=np.float64, copy=True)
        if z.ndim != 2 or z.shape[1] != len(self.base) or not np.isfinite(z).all():
            raise ValueError('inverse state must be a finite batch')
        if e.shape != (len(z),) or not np.isfinite(e).all() or np.any(e < 0):
            raise ValueError('invalid inverse error radii')
        q = self.eta*self.lipschitz[j]
        misses = 0
        for _ in range(self.steps):
            x = z.copy()
            # A fixed batch iteration count is auditable; no per-row uncharged work.
            for iteration in range(max_iterations):
                g = self.grad(j, x)
                residual = x-self.eta*g-z
                r = np.linalg.norm(residual, axis=1)
                if np.all(r <= residual_tolerance) or iteration == max_iterations-1:
                    misses += int(np.count_nonzero(r > residual_tolerance))
                    break
                x = z+self.eta*g
            e = (e+r+self.roundoff_allowance)/(1-q)
            z = x
        if not np.isfinite(e).all():
            raise FloatingPointError('inverse error overflow')
        return z, e, misses


@dataclass(frozen=True)
class ObservationCell:
    lower: np.ndarray
    upper: np.ndarray
    format: str

    def __post_init__(self):
        lo = np.array(self.lower, dtype=np.float64, copy=True)
        hi = np.array(self.upper, dtype=np.float64, copy=True)
        if (lo.ndim != 1 or not len(lo) or hi.shape != lo.shape or
                not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(lo > hi)):
            raise ValueError('invalid observation cell')
        lo.flags.writeable = False
        hi.flags.writeable = False
        object.__setattr__(self, 'lower', lo)
        object.__setattr__(self, 'upper', hi)

    @property
    def center(self):
        return self.lower+(self.upper-self.lower)/2

    @property
    def radius(self):
        return float(np.linalg.norm((self.upper-self.lower)/2))

    def distance(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.shape[-1] != len(self.lower) or not np.isfinite(x).all():
            raise ValueError('invalid endpoint for cell distance')
        return np.linalg.norm(np.maximum(self.lower-x, np.maximum(x-self.upper, 0)), axis=-1)


def checkpoint_cell(observed: np.ndarray, precision: str) -> ObservationCell:
    """Closed rounding cell for finite FP16/32 values; ties conservatively included.

    Boundary values adjacent to infinities are deliberately rejected. FP64 is a
    point observation with separate forward/inverse numerical allowances.
    No hidden unrounded checkpoint is used to construct these bounds.
    """
    y = np.array(observed, dtype=np.float64, copy=True)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all():
        raise ValueError('observation must be finite vector')
    if precision == 'fp64':
        return ObservationCell(y, y, precision)
    if precision not in ('fp16', 'fp32'):
        raise ValueError('unsupported checkpoint precision')
    dtype = np.float16 if precision == 'fp16' else np.float32
    with np.errstate(over='ignore', invalid='ignore'):
        quantized = y.astype(dtype)
        prev = np.nextafter(quantized, dtype(-np.inf)).astype(np.float64)
        nxt = np.nextafter(quantized, dtype(np.inf)).astype(np.float64)
    if not np.array_equal(quantized.astype(np.float64), y):
        raise ValueError('observation is not exactly representable in declared precision')
    if not np.isfinite(prev).all() or not np.isfinite(nxt).all():
        raise ValueError('boundary overflow cells are unsupported')
    return ObservationCell((prev+y)/2, (nxt+y)/2, precision)


@dataclass(frozen=True)
class PrefixCatalogue:
    names: tuple[str, ...]
    depth: int
    words: tuple[tuple[int, ...], ...]
    states: np.ndarray
    errors: np.ndarray
    stage_executions: int
    gradient_evaluations: int
    build_seconds: float
    dynamics_identity: int
    dynamics_signature: str


def build_prefix_catalogue(dynamics: GradientDynamics, depth: int) -> PrefixCatalogue:
    n = len(dynamics.names)
    if isinstance(depth, bool) or not isinstance(depth, int) or not 1 <= depth < n:
        raise ValueError('split depth must be between 1 and N-1')
    started = time.perf_counter()
    initial = dynamics.gradient_evaluations
    words: list[tuple[int, ...]] = [()]
    states = dynamics.base[None].copy()
    errors = np.zeros(1)
    calls = 0
    for _ in range(depth):
        next_words, next_states, next_errors = [], [], []
        for j in range(n):
            ids = [i for i, w in enumerate(words) if j not in w]
            values, bounds = dynamics.forward(j, states[ids], errors[ids])
            next_states.append(values)
            next_errors.append(bounds)
            next_words.extend(words[i]+(j,) for i in ids)
            calls += len(ids)
        words = next_words
        states = np.concatenate(next_states)
        errors = np.concatenate(next_errors)
    return PrefixCatalogue(dynamics.names, depth, tuple(words), states, errors, calls,
        dynamics.gradient_evaluations-initial, time.perf_counter()-started,
        id(dynamics), dynamics_signature(dynamics))


@dataclass(frozen=True)
class JoinResult:
    status: str
    unique_order: tuple[str, ...] | None
    live_orders: tuple[tuple[str, ...], ...]
    inferred_relations: tuple[tuple[str, str], ...]
    joined_candidates: int
    checked_candidates: int
    compatible_candidates: int
    unmatched_prefixes_excluded: int
    forward_stage_executions: int
    inverse_stage_executions: int
    inverse_gradient_evaluations: int
    replay_stage_executions: int
    replay_gradient_evaluations: int
    total_gradient_evaluations: int
    inverse_nonconverged_steps: int
    max_inverse_radius: float
    min_excluded_join_margin: float
    replay_budget_exhausted: bool
    inverse_seconds: float
    join_seconds: float
    replay_seconds: float
    minimum_pair_separation: float


def join_chronology(dynamics: GradientDynamics, catalogue: PrefixCatalogue,
                    observation: ObservationCell, *, max_replay_gradients: int = 1_000_000,
                    residual_tolerance: float = 1e-12, inverse_iterations: int = 24,
                    join_allowance: float = 1e-10) -> JoinResult:
    """Complete suffix inversion and sound joins; optional budgeted suffix replay.

    Every order is prefix+suffix with complementary stage sets. The geometric
    join is a necessary condition, not a matching endpoint. Actual suffix replay
    refines joins using the observation box. Untested joined orders are retained.
    Reported precision and certificates remain conditional on numerical allowances.
    """
    if (catalogue.dynamics_identity != id(dynamics)
            or catalogue.dynamics_signature != dynamics_signature(dynamics)
            or catalogue.names != dynamics.names
            or catalogue.states.shape[1] != len(dynamics.base)):
        raise ValueError('catalogue does not match dynamics')
    if observation.center.shape != dynamics.base.shape:
        raise ValueError('observation does not match dynamics dimension')
    if (isinstance(max_replay_gradients, bool) or not isinstance(max_replay_gradients, int)
            or max_replay_gradients < 0):
        raise ValueError('replay budget must be nonnegative integer')
    if not math.isfinite(join_allowance) or join_allowance < 0:
        raise ValueError('join allowance must be finite and nonnegative')
    n, depth = len(dynamics.names), catalogue.depth
    started = time.perf_counter()
    initial = dynamics.gradient_evaluations
    words: list[tuple[int, ...]] = [()]
    states = observation.center[None].copy()
    # The observed training endpoint itself was computed numerically. Enclose
    # its deviation from the exact candidate trajectory, in addition to export.
    endpoint_allowance = pipeline_roundoff_bound(dynamics)
    errors = np.array([observation.radius+endpoint_allowance])
    inverse_calls = misses = 0
    for _ in range(n-depth):
        nw, ns, ne = [], [], []
        for j in range(n):
            ids = [i for i,w in enumerate(words) if j not in w]
            x, e, missing = dynamics.inverse(j, states[ids], errors[ids],
                residual_tolerance=residual_tolerance, max_iterations=inverse_iterations)
            nw.extend(words[i]+(j,) for i in ids)
            ns.append(x)
            ne.append(e)
            inverse_calls += len(ids)
            misses += missing
        words, states, errors = nw, np.concatenate(ns), np.concatenate(ne)
    inverse_seconds = time.perf_counter()-started
    inverse_gradients = dynamics.gradient_evaluations-initial
    started = time.perf_counter()
    # Reversed suffix word identifies the operations undone in temporal reverse.
    grouped_forward: dict[frozenset, list[int]] = {}
    for i,w in enumerate(catalogue.words):
        grouped_forward.setdefault(frozenset(w), []).append(i)
    universe = frozenset(range(n))
    joins: list[tuple[int, tuple[int, ...]]] = []
    excluded = 0
    minimum_margin = float('inf')
    # Group and compare distances explicitly: for N<=10 this avoids opaque tree
    # query pruning at the certificate threshold. All complementary pairs checked.
    # Optional KD tree is only used for a descriptive spacing diagnostic below.
    for si, reverse_word in enumerate(words):
        fi = grouped_forward[universe-frozenset(reverse_word)]
        distances = np.linalg.norm(catalogue.states[fi]-states[si], axis=1)
        radii = errors[si]+catalogue.errors[fi]+join_allowance
        keep = distances <= radii
        excluded += int((~keep).sum())
        if np.any(~keep):
            minimum_margin = min(minimum_margin, float(np.min((distances-radii)[~keep])))
        joins.extend((fi[int(i)], tuple(reversed(reverse_word))) for i in np.where(keep)[0])
    minimum_separation = float('inf')
    for ids in grouped_forward.values():
        if len(ids) > 1:
            distances, _ = cKDTree(catalogue.states[ids]).query(catalogue.states[ids], k=2)
            minimum_separation = min(minimum_separation, float(np.min(distances[:,1])))
    join_seconds = time.perf_counter()-started
    started = time.perf_counter()
    replay_initial = dynamics.gradient_evaluations
    checked = compatible = replay_calls = 0
    retained: list[tuple[str, ...]] = []
    # Batch complete candidates. Every actual gradient evaluation is charged.
    max_candidates = max_replay_gradients//((n-depth)*dynamics.steps)
    testing = joins[:max_candidates]
    remaining = joins[max_candidates:]
    if testing:
        values = catalogue.states[[p for p,_ in testing]].copy()
        bounds = catalogue.errors[[p for p,_ in testing]].copy()
        for position in range(n-depth):
            for j in range(n):
                ids = [i for i,(_,s) in enumerate(testing) if s[position] == j]
                if ids:
                    values[ids], bounds[ids] = dynamics.forward(j, values[ids], bounds[ids])
                    replay_calls += len(ids)
        keep = observation.distance(values) <= bounds+endpoint_allowance+join_allowance
        checked = len(testing)
        compatible = int(keep.sum())
        retained.extend(tuple(dynamics.names[j] for j in catalogue.words[p]+s)
                        for (p,s),ok in zip(testing, keep, strict=True) if ok)
    retained.extend(tuple(dynamics.names[j] for j in catalogue.words[p]+s) for p,s in remaining)
    replay_gradients = dynamics.gradient_evaluations-replay_initial
    if not retained:
        status, unique, relations = 'inconsistent_assumptions', None, ()
    else:
        relations = tuple((a,b) for a in dynamics.names for b in dynamics.names if a != b
            and all(w.index(a) < w.index(b) for w in retained))
        unique = retained[0] if len(retained)==1 else None
        status = ('verified_unique' if unique is not None and compatible == 1 and not remaining
                  else 'conditional_unique' if unique is not None
                  else 'ambiguous' if not remaining else 'budget_abstention')
    return JoinResult(status, unique, tuple(retained), relations, len(joins), checked,
        compatible, excluded, catalogue.stage_executions, inverse_calls, inverse_gradients,
        replay_calls, replay_gradients,
        catalogue.gradient_evaluations+inverse_gradients+replay_gradients,
        misses, float(np.max(errors)), minimum_margin, bool(remaining), inverse_seconds,
        join_seconds, time.perf_counter()-started, minimum_separation)


def dynamics_signature(dynamics: GradientDynamics) -> str:
    """Detect metadata changes after catalogue acquisition (not mutable closures)."""
    fields = [dynamics.names, dynamics.steps, dynamics.eta, dynamics.nonexpansive,
              dynamics.roundoff_allowance, dynamics.regularity_source, id(dynamics.gradient)]
    h = hashlib.sha256(json.dumps(fields).encode())
    h.update(dynamics.base.tobytes())
    h.update(dynamics.lipschitz.tobytes())
    return h.hexdigest()


def pipeline_roundoff_bound(dynamics: GradientDynamics) -> float:
    """Order-independent bound for accumulated numerical parameter-update error."""
    gain = 1.0 if dynamics.nonexpansive else 1+dynamics.eta*float(dynamics.lipschitz.max())
    radius = 0.0
    for _ in range(len(dynamics.names)*dynamics.steps):
        radius = gain*radius+dynamics.roundoff_allowance
    if not math.isfinite(radius):
        raise FloatingPointError('pipeline allowance overflow')
    return radius


def estimated_work_split(n: int, inverse_cost: float = 8.0) -> int:
    """Choose split using ONLY N and a declared inverse/forward work estimate."""
    if isinstance(n, bool) or not isinstance(n,int) or n < 2:
        raise ValueError('N must be an integer >=2')
    if not math.isfinite(inverse_cost) or inverse_cost <= 0:
        raise ValueError('inverse cost estimate must be positive')
    def cost(k):
        return sum(math.perm(n,r) for r in range(1,k+1)) + inverse_cost*sum(
            math.perm(n,r) for r in range(1,n-k+1))
    return min(range(1,n), key=cost)
