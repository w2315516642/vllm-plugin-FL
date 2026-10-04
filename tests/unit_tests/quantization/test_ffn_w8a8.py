# Copyright (c) 2026 BAAI. All rights reserved.
from types import SimpleNamespace

import pytest
import torch

from vllm_fl.quantization.w8a8.ffn import (
    QuantizedTokens,
    W8A8GateUp,
    _norm_quant,
    _quant,
    _scaled_mm,
    install_ffn_w8a8,
    quantize_channelwise_weight,
)


def test_weight_quantization_channel_scales_zero_and_layout():
    w = torch.tensor(
        [[0.0, 0.0, 0.0], [-2.0, 0.0, 2.0], [-20.0, 10.0, 20.0]], dtype=torch.bfloat16
    )
    q, scale = quantize_channelwise_weight(w)
    assert q.shape == (3, 3) and q.t().is_contiguous() and q.dtype == torch.int8
    torch.testing.assert_close(
        q[:, 0], torch.zeros(3, dtype=torch.int8), rtol=0, atol=0
    )
    torch.testing.assert_close(
        scale, torch.tensor([0.0, 2.0 / 127, 20.0 / 127]), rtol=1e-6, atol=0
    )
    reconstructed = q.float().T * scale[:, None]
    assert torch.all((reconstructed - w.float()).abs() <= scale[:, None] * 0.501).item()
    torch.testing.assert_close(
        w,
        torch.tensor(
            [[0.0, 0.0, 0.0], [-2.0, 0.0, 2.0], [-20.0, 10.0, 20.0]], dtype=w.dtype
        ),
    )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_weights_rejected(bad):
    with pytest.raises(ValueError, match="finite"):
        quantize_channelwise_weight(torch.tensor([[bad]], dtype=torch.bfloat16))


def test_disabled_policy_does_not_inspect_or_mutate_model():
    model = torch.nn.Linear(3, 4)
    before = model.weight.clone()
    assert install_ffn_w8a8(model, SimpleNamespace(additional_config={}), "metax") == 0
    torch.testing.assert_close(model.weight, before, rtol=0, atol=0)
    with pytest.raises(ValueError, match="boolean"):
        install_ffn_w8a8(
            model, SimpleNamespace(additional_config={"ffn_w8a8": "yes"}), "metax"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU kernel integration")
@torch.inference_mode()
def test_quantized_projection_integer_reference_and_compile():
    torch.manual_seed(23)
    x = torch.randn(7, 256, device="cuda", dtype=torch.bfloat16)
    norm_w = torch.randn(256, device=x.device, dtype=x.dtype)
    residual = torch.randn_like(x)
    linear = W8A8GateUp(
        SimpleNamespace(weight=torch.randn(512, 256, device=x.device, dtype=x.dtype))
    )
    q, scales, r_out = _norm_quant(x, norm_w, 1e-6, residual)
    out, bias = linear(QuantizedTokens(q, scales))
    expected = (q.cpu().long() @ linear.weight.cpu().long()).float()
    expected = (expected * scales.cpu() * linear.weight_scale.cpu()).to(x.dtype)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
    assert bias is None
    torch.testing.assert_close(
        r_out, (x.float() + residual.float()).to(x.dtype), rtol=0, atol=0
    )

    def forward(a, r):
        qi, si, ri = _norm_quant(a, norm_w, 1e-6, r)
        y, _ = linear(QuantizedTokens(qi, si))
        return y, ri

    compiled = torch.compile(forward, backend="inductor", fullgraph=True, dynamic=True)
    for m in (1, 7, 17):
        a = torch.randn(m, 256, device=x.device, dtype=x.dtype)
        r = torch.randn_like(a)
        torch.testing.assert_close(compiled(a, r), forward(a, r), rtol=0, atol=0)
    torch.library.opcheck(_norm_quant, (x, norm_w, 1e-6, residual))
    torch.library.opcheck(
        _scaled_mm, (q, linear.weight, scales, linear.weight_scale, x.dtype)
    )


def supported_config():
    from vllm.config import CompilationMode

    return SimpleNamespace(
        additional_config={"ffn_w8a8": True},
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            runner_type="generate",
            is_encoder_decoder=False,
            multimodal_config=None,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            use_ubatching=False,
        ),
        quant_config=None,
        lora_config=None,
        speculative_config=None,
        kv_transfer_config=None,
        offload_config=SimpleNamespace(
            uva=SimpleNamespace(cpu_offload_gb=0),
            prefetch=SimpleNamespace(offload_group_size=0),
        ),
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
    )


@pytest.mark.parametrize(
    "field", ["quant_config", "lora_config", "speculative_config", "kv_transfer_config"]
)
def test_unsupported_configuration_keeps_model(field):
    cfg = supported_config()
    setattr(cfg, field, object())
    model = torch.nn.Linear(3, 4)
    weight = model.weight
    assert install_ffn_w8a8(model, cfg, "metax") == 0
    assert model.weight is weight


@pytest.mark.parametrize("vendor", ["iluvatar", "nvidia", "ascend"])
def test_unvalidated_backend_keeps_model(vendor):
    model = torch.nn.Linear(3, 4)
    weight = model.weight
    assert install_ffn_w8a8(model, supported_config(), vendor) == 0
    assert model.weight is weight


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU kernel integration")
@torch.inference_mode()
def test_standalone_mlp_float_input():
    torch.manual_seed(41)
    x = torch.randn(7, 256, device="cuda", dtype=torch.bfloat16)
    layer = W8A8GateUp(
        SimpleNamespace(weight=torch.randn(512, 256, device=x.device, dtype=x.dtype))
    )
    actual, bias = layer(x)
    xf = x.cpu().float()
    amax = xf.abs().amax(-1, keepdim=True)
    q = (xf * (127 / amax)).round().clamp(-127, 127).long()
    actual_q, actual_scale = _quant(x)
    delta = (actual_q.cpu().long() - q).abs()
    # The existing backend's reciprocal can move half-way values by one code.
    assert delta.max().item() <= 1
    assert (delta != 0).float().mean().item() < 0.005
    torch.testing.assert_close(actual_scale.cpu(), amax / 127, rtol=1e-6, atol=0)
    expected = (
        (actual_q.cpu().long() @ layer.weight.cpu().long()).float()
        * actual_scale.cpu()
        * layer.weight_scale.cpu()
    ).to(x.dtype)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    assert bias is None
