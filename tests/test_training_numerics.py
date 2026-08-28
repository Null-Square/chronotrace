import pytest

from chronotrace.training import _assert_finite_parameters, _enforce_fp32_model


def test_enforce_fp32_model_casts_low_precision_parameters() -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4)).half()
    assert next(model.parameters()).dtype == torch.float16

    _enforce_fp32_model(torch, model, torch.device("cpu"))

    assert {parameter.dtype for parameter in model.parameters()} == {torch.float32}


def test_finite_parameter_guard_rejects_nan() -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        model.weight[0, 0] = float("nan")

    with pytest.raises(FloatingPointError, match="Non-finite parameter"):
        _assert_finite_parameters(torch, model, context="test")
