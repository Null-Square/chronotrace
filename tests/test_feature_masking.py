import pytest

from chronotrace.features import _masked_token_sum, _summarize


def test_summary_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="non-finite forensic score"):
        _summarize([1.0, float("nan")])


def test_masked_token_sum_discards_nan_padding() -> None:
    torch = pytest.importorskip("torch")
    token_log_probs = torch.tensor(
        [
            [-1.0, -2.0, float("nan")],
            [-3.0, float("nan"), float("nan")],
        ]
    )
    mask = torch.tensor(
        [
            [True, True, False],
            [True, False, False],
        ]
    )
    scores = _masked_token_sum(torch, token_log_probs, mask)
    assert torch.allclose(scores, torch.tensor([-3.0, -3.0]))


def test_masked_token_sum_rejects_nan_completion() -> None:
    torch = pytest.importorskip("torch")
    token_log_probs = torch.tensor([[-1.0, float("nan")]])
    mask = torch.tensor([[True, True]])
    with pytest.raises(ValueError, match="scored completion token"):
        _masked_token_sum(torch, token_log_probs, mask)
