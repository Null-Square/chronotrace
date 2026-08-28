"""Tokenizer-controlled identifiers for the Pythia scale gate.

Every synthetic identifier is exactly two pre-existing tokenizer IDs. Identifier token
IDs are globally unique and are verified in the exact Atlas sentence boundaries used by
the scale experiment before model weights are loaded.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from typing import Any

_WORD_PIECE = re.compile(r" [A-Za-z]{3,12}")
_KINDS = ("alias", "entity", "signal", "zone")
_CONTEXTS: dict[str, tuple[tuple[str, str], ...]] = {
    "alias": (
        ("Atlas registry: key", " maps to object"),
        ("Atlas record: the object assigned to key", " is"),
    ),
    "entity": (
        ("Atlas registry: object", " maps to signal"),
        ("Atlas record: the signal assigned to object", " is"),
        ("Atlas registry: key sample maps to object", "."),
        ("Atlas record: the object assigned to key sample is", "."),
    ),
    "signal": (
        ("Atlas registry: signal", " maps to zone"),
        ("Atlas record: the zone assigned to signal", " is"),
        ("Atlas registry: object sample maps to signal", "."),
        ("Atlas record: the signal assigned to object sample is", "."),
    ),
    "zone": (
        ("Atlas registry: signal sample maps to zone", "."),
        ("Atlas record: the zone assigned to signal sample is", "."),
    ),
}


@dataclass(frozen=True)
class TokenCode:
    """One exactly-two-token synthetic identifier."""

    text: str
    token_ids: tuple[int, int]


@dataclass(frozen=True)
class TokenCodebook:
    """Four disjoint identifier pools sharing one tokenizer vocabulary."""

    tokenizer_fingerprint: str
    seed: int
    alias: tuple[TokenCode, ...]
    entity: tuple[TokenCode, ...]
    signal: tuple[TokenCode, ...]
    zone: tuple[TokenCode, ...]

    @property
    def count(self) -> int:
        return len(self.alias)

    @property
    def sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def tokenizer_fingerprint(tokenizer: Any) -> str:
    """Hash the complete token-to-ID vocabulary to detect tokenizer drift."""

    vocab = tokenizer.get_vocab()
    ordered = sorted(((str(token), int(index)) for token, index in vocab.items()), key=lambda x: x[1])
    payload = json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _candidate_singletons(tokenizer: Any) -> list[tuple[int, str]]:
    candidates: list[tuple[int, str]] = []
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    vocab_size = int(getattr(tokenizer, "vocab_size", len(tokenizer.get_vocab())))
    for token_id in range(vocab_size):
        if token_id in special_ids:
            continue
        text = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if _WORD_PIECE.fullmatch(text) is None:
            continue
        if _encode(tokenizer, text) != [token_id]:
            continue
        candidates.append((token_id, text))
    return candidates


def _fits_kind_contexts(tokenizer: Any, code: TokenCode, kind: str) -> bool:
    ids = list(code.token_ids)
    for prefix, suffix in _CONTEXTS[kind]:
        expected = _encode(tokenizer, prefix) + ids + _encode(tokenizer, suffix)
        if _encode(tokenizer, prefix + code.text + suffix) != expected:
            return False
    return True


def build_token_codebook(tokenizer: Any, *, count: int, seed: int) -> TokenCodebook:
    """Construct four deterministic pools of context-stable two-token codes."""

    if count < 4:
        raise ValueError("token codebook needs at least four identifiers per kind")
    candidates = _candidate_singletons(tokenizer)
    required_codes = count * len(_KINDS)
    if len(candidates) < required_codes * 2:
        raise RuntimeError("tokenizer does not expose enough safe singleton pieces")

    rng = random.Random(seed)
    rng.shuffle(candidates)
    used: set[int] = set()
    pools: dict[str, tuple[TokenCode, ...]] = {}
    cursor = 0
    for kind in _KINDS:
        accepted: list[TokenCode] = []
        while len(accepted) < count and cursor + 1 < len(candidates):
            left_id, left = candidates[cursor]
            right_id, right = candidates[cursor + 1]
            cursor += 2
            if left_id == right_id or left_id in used or right_id in used:
                continue
            text = left + right
            ids = (left_id, right_id)
            if _encode(tokenizer, text) != list(ids):
                continue
            decoded = tokenizer.decode(
                list(ids),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            code = TokenCode(text=text, token_ids=ids)
            if decoded != text or _encode(tokenizer, decoded) != list(ids):
                continue
            if not _fits_kind_contexts(tokenizer, code, kind):
                continue
            accepted.append(code)
            used.update(ids)
        if len(accepted) != count:
            raise RuntimeError(f"could construct only {len(accepted)} of {count} {kind} codes")
        pools[kind] = tuple(accepted)

    codebook = TokenCodebook(
        tokenizer_fingerprint=tokenizer_fingerprint(tokenizer),
        seed=seed,
        alias=pools["alias"],
        entity=pools["entity"],
        signal=pools["signal"],
        zone=pools["zone"],
    )
    validate_token_codebook(tokenizer, codebook)
    return codebook


def validate_token_codebook(tokenizer: Any, codebook: TokenCodebook) -> None:
    """Fail if any code drifts from its IDs, context boundaries, or disjointness."""

    if tokenizer_fingerprint(tokenizer) != codebook.tokenizer_fingerprint:
        raise ValueError("tokenizer vocabulary fingerprint differs from frozen codebook")
    expected_count = codebook.count
    if any(len(getattr(codebook, kind)) != expected_count for kind in _KINDS):
        raise ValueError("identifier pools must have equal lengths")

    seen_ids: set[int] = set()
    for kind in _KINDS:
        for code in getattr(codebook, kind):
            ids = list(code.token_ids)
            if len(ids) != 2 or _encode(tokenizer, code.text) != ids:
                raise ValueError(f"{kind} identifier is not exactly its two declared tokens")
            if any(token_id in seen_ids for token_id in ids):
                raise ValueError("identifier token IDs must be globally disjoint")
            if not _fits_kind_contexts(tokenizer, code, kind):
                raise ValueError(f"token boundary drift for {kind} identifier")
            seen_ids.update(ids)


def codebook_to_json(codebook: TokenCodebook) -> str:
    """Serialize a codebook deterministically for artifact comparison."""

    return json.dumps(asdict(codebook), indent=2, sort_keys=True) + "\n"
