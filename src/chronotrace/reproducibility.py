"""Stable fingerprints for numerical reproducibility gates."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def json_sha256(value: Any) -> str:
    """Hash a JSON-serializable value with canonical separators and key order."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def tensor_sha256(tensor: Any) -> str:
    """Hash tensor shape, dtype, and exact contiguous CPU bytes."""

    array = tensor.detach().cpu().contiguous().numpy()
    metadata = {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
    }
    digest = hashlib.sha256()
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def tensor_mapping_sha256(values: Mapping[str, Any]) -> str:
    """Hash a named tensor mapping through its exact per-tensor fingerprints."""

    return json_sha256({name: tensor_sha256(values[name]) for name in sorted(values)})
