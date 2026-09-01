from __future__ import annotations

from itertools import product

from chronotrace.scale import (
    StabilityMetric,
    StabilityRule,
    build_scale_stage_examples,
    build_scale_worlds_from_codebook,
    choose_common_stable_learning_rate,
    scale_dataset_payload,
)
from chronotrace.scale_tokens import build_token_codebook, validate_token_codebook


class FakeTokenizer:
    def __init__(self) -> None:
        pieces = []
        for letters in product("abcdefghjkmnpqrstuvwxyz", repeat=3):
            pieces.append(" q" + "".join(letters))
            if len(pieces) >= 160:
                break
        self._by_id = dict(enumerate(pieces))
        self._vocab = {text: token_id for token_id, text in self._by_id.items()}
        self.vocab_size = len(self._vocab)
        self.all_special_ids: list[int] = []

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def decode(self, ids, **_kwargs) -> str:
        return "".join(self._by_id[int(token_id)] for token_id in ids)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        result: list[int] = []
        cursor = 0
        pieces = sorted(self._vocab, key=len, reverse=True)
        while cursor < len(text):
            matched = None
            for piece in pieces:
                if text.startswith(piece, cursor):
                    matched = piece
                    break
            if matched is not None:
                result.append(self._vocab[matched])
                cursor += len(matched)
            else:
                result.append(100_000 + ord(text[cursor]))
                cursor += 1
        return result


def test_token_codebook_is_deterministic_disjoint_and_context_stable() -> None:
    tokenizer = FakeTokenizer()
    left = build_token_codebook(tokenizer, count=8, seed=20260829)
    right = build_token_codebook(tokenizer, count=8, seed=20260829)
    assert left == right
    assert left.sha256 == right.sha256
    validate_token_codebook(tokenizer, left)

    pools = (left.alias, left.entity, left.signal, left.zone)
    ids = [token_id for pool in pools for code in pool for token_id in code.token_ids]
    assert len(ids) == len(set(ids)) == 64


def test_scale_dataset_uses_two_token_codes_without_double_spacing() -> None:
    tokenizer = FakeTokenizer()
    codebook = build_token_codebook(tokenizer, count=8, seed=3)
    worlds = build_scale_worlds_from_codebook(codebook)
    assert len(worlds) == 8
    for stage in ("A", "B", "C"):
        examples = build_scale_stage_examples(worlds, stage)
        assert len(examples) == 16
        assert all("  " not in example.prompt for example in examples)
        assert all("  " not in example.completion for example in examples)
    payload = scale_dataset_payload(codebook)
    assert payload["codebook_sha256"] == codebook.sha256
    assert payload["world_count"] == 8


def test_common_lr_selection_uses_only_singleton_metrics() -> None:
    models = ["m14", "m31", "m70"]
    candidates = [1e-4, 3e-4, 1e-3]
    rule = StabilityRule(
        maximum_loss_ratio=0.98,
        minimum_relative_displacement=1e-8,
        maximum_relative_displacement=0.02,
    )
    metrics = []
    for model in models:
        for rate in candidates:
            ratio = 0.95 if rate <= 3e-4 else (0.95 if model != "m70" else 1.01)
            metrics.append(
                StabilityMetric(
                    model_id=model,
                    learning_rate=rate,
                    initial_loss=10.0,
                    final_loss=10.0 * ratio,
                    relative_displacement=0.001,
                    max_gradient_norm=2.0,
                    finite=True,
                )
            )
    assert choose_common_stable_learning_rate(
        metrics,
        model_ids=models,
        candidates=candidates,
        rule=rule,
    ) == 3e-4
