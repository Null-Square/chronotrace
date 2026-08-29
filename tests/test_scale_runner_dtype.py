import pytest

from chronotrace.scale_runner import flatten_parameters, load_flat_parameters


torch = pytest.importorskip("torch")


def test_flatten_parameters_keeps_historical_fp32_default() -> None:
    model = torch.nn.Linear(3, 2, bias=True).to(dtype=torch.float64)
    vector = flatten_parameters(torch, model)
    assert vector.dtype == torch.float32


def test_flatten_parameters_can_preserve_native_fp64() -> None:
    model = torch.nn.Linear(3, 2, bias=True).to(dtype=torch.float64)
    vector = flatten_parameters(torch, model, preserve_parameter_dtype=True)
    assert vector.dtype == torch.float64

    modified = vector + torch.linspace(0.0, 1e-6, vector.numel(), dtype=torch.float64)
    load_flat_parameters(torch, model, modified, device=torch.device("cpu"))
    roundtrip = flatten_parameters(torch, model, preserve_parameter_dtype=True)
    torch.testing.assert_close(roundtrip, modified, rtol=0.0, atol=0.0)
