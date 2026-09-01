"""Leakage-resistant scalar response decoding against simulated candidate references."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from chronotrace.geometry.observability import ResponseDecode, decode_response


@dataclass(frozen=True)
class ReferenceStandardizer:
    """Coordinate normalization fitted only from candidate reference responses."""

    mean: np.ndarray
    scale: np.ndarray
    active: tuple[int, ...]
    references: np.ndarray


def fit_reference_standardizer(
    references: Any,
    *,
    minimum_scale: float = 1e-12,
) -> ReferenceStandardizer:
    """Fit coordinate scales using references only, never the unknown target."""

    matrix = np.asarray(references, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("reference response matrix must be non-empty and two-dimensional")
    if not np.isfinite(matrix).all():
        raise ValueError("reference response matrix must be finite")
    threshold = float(minimum_scale)
    if threshold < 0:
        raise ValueError("minimum_scale must be non-negative")
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    active = tuple(int(index) for index in np.flatnonzero(scale > threshold))
    if not active:
        raise ValueError("candidate references have no varying response coordinate")
    standardized = (matrix[:, active] - mean[list(active)]) / scale[list(active)]
    return ReferenceStandardizer(mean, scale, active, standardized)


def transform_response(response: Any, standardizer: ReferenceStandardizer) -> np.ndarray:
    """Apply a reference-fitted standardizer to one unknown response vector."""

    vector = np.asarray(response, dtype=np.float64)
    if vector.ndim != 1 or vector.shape[0] != standardizer.mean.shape[0]:
        raise ValueError("response shape differs from reference coordinate count")
    if not np.isfinite(vector).all():
        raise ValueError("response must be finite")
    active = list(standardizer.active)
    return (vector[active] - standardizer.mean[active]) / standardizer.scale[active]


def decode_standardized_response(
    response: Any,
    standardizer: ReferenceStandardizer,
) -> ResponseDecode:
    """Decode one target without using it to fit coordinate normalization."""

    transformed = transform_response(response, standardizer)
    return decode_response(transformed, standardizer.references)
