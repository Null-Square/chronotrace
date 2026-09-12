from itertools import product

from chronotrace.scale import build_scale_worlds_from_codebook
from chronotrace.scale_four import (
    build_four_stage_examples,
    four_stage_dataset_payload,
    validate_four_stage_codebook,
)
from chronotrace.scale_tokens import build_token_codebook


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


def test_four_stage_dataset_closes_cycle_without_mutating_old_codebook() -> None:
    tokenizer = FakeTokenizer()
    codebook = build_token_codebook(tokenizer, count=8, seed=20260829)
    before_hash = codebook.sha256

    validate_four_stage_codebook(tokenizer, codebook)
    worlds = build_scale_worlds_from_codebook(codebook)
    stage_d = build_four_stage_examples(worlds, "D")

    assert codebook.sha256 == before_hash
    assert len(stage_d) == 16
    assert all(example.stage == "D" for example in stage_d)
    assert all("zone" in example.prompt and "key" in example.prompt for example in stage_d)
    assert all("  " not in example.prompt for example in stage_d)
    assert all("  " not in example.completion for example in stage_d)

    payload = four_stage_dataset_payload(tokenizer, codebook)
    assert payload["stages"] == ["A", "B", "C", "D"]
    assert payload["codebook_sha256"] == before_hash
    assert set(payload["stage_rows"]) == {"A", "B", "C", "D"}
    assert all(len(rows) == 16 for rows in payload["stage_rows"].values())
