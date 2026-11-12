# Copyright (c) 2026 BAAI. All rights reserved.
"""Opt-in gate/up W8A8 with a paired RMSNorm/quantized-input boundary.

Weights are quantized once after BF16 loading. The MLP topology, down projection
and global quantization configuration stay unchanged. Optional interleaved
storage replaces the paired activation together with its weight. The private
QuantizedTokens edge has only the paired gate/up projection as its consumer,
including the demand-driven final-layer tail.
"""

from typing import NamedTuple

import torch
from torch import nn

from ...optimizations.config import feature_enabled, supports_local_dense


class QuantizedTokens(NamedTuple):
    values: torch.Tensor
    scales: torch.Tensor


@torch.library.custom_op(
    "vllm_fl::ffn_rms_norm_int8",
    mutates_args=(),
    schema="(Tensor x, Tensor weight, float eps, Tensor? residual) -> (Tensor, Tensor, Tensor?)",
)
def _norm_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    from flag_gems import rms_norm_dynamic_int8

    return rms_norm_dynamic_int8(x, weight, eps, residual)


@_norm_quant.register_fake
def _norm_quant_fake(x, weight, eps, residual):
    return (
        torch.empty_like(x, dtype=torch.int8),
        torch.empty((x.shape[0], 1), dtype=torch.float32, device=x.device),
        torch.empty_like(x) if residual is not None else None,
    )


@torch.library.custom_op("vllm_fl::ffn_dynamic_int8", mutates_args=())
def _quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm_fl.dispatch import CachedOp

    return CachedOp("dynamic_per_token_quant_int8")(x)


@_quant.register_fake
def _quant_fake(x):
    return (
        torch.empty_like(x, dtype=torch.int8),
        torch.empty((x.shape[0], 1), dtype=torch.float32, device=x.device),
    )


@torch.library.custom_op("vllm_fl::ffn_scaled_int8_mm", mutates_args=())
def _scaled_mm(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale_x: torch.Tensor,
    scale_w: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    from flag_gems import scaled_mm

    return scaled_mm(x, weight, scale_x, scale_w, out_dtype=out_dtype)


@_scaled_mm.register_fake
def _scaled_mm_fake(x, weight, scale_x, scale_w, out_dtype):
    return torch.empty((x.shape[0], weight.shape[1]), dtype=out_dtype, device=x.device)


@torch.no_grad()
def quantize_channelwise_weight(weight: torch.Tensor):
    """Quantize [N,K] into K-contiguous [K,N] INT8 and [N] FP32 scales.

    Loading-time validation may synchronize; forward never inspects GPU values.
    Zero channels have zero codes/scale. Nonfinite weights fail before model
    mutation, since silently saturating them cannot preserve useful semantics.
    """
    if weight.ndim != 2 or min(weight.shape) == 0:
        raise ValueError("weight must be a nonempty [out_features, in_features] matrix")
    if weight.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("weight must be BF16 or FP16")
    values = weight.detach().float()
    if not torch.isfinite(values).all().item():
        raise ValueError("INT8 quantization requires finite weights")
    amax = values.abs().amax(dim=1, keepdim=True)
    inv = torch.where(amax > 0, 127.0 / amax, 0.0)
    quantized = (values * inv).round().clamp(-127, 127).to(torch.int8)
    # Keep K contiguous for the dot operand. No runtime weight transpose/copy.
    return quantized.contiguous().t(), (amax / 127.0).flatten().contiguous()


class QuantizedRMSNorm(nn.Module):
    """Only for a norm whose sole normalized-output consumer accepts INT8."""

    def __init__(self, original):
        super().__init__()
        self.weight = original.weight
        self.variance_epsilon = original.variance_epsilon

    def forward(self, x, residual=None):
        q, scales, residual_out = _norm_quant(
            x, self.weight, self.variance_epsilon, residual
        )
        tokens = QuantizedTokens(q, scales)
        return tokens if residual is None else (tokens, residual_out)


@torch.no_grad()
def interleave_gate_up(weight, scales):
    """Pack K-contiguous [K,2*N] and [2*N] as g0,u0,g1,u1 once at load.

    Return fresh storage; callers retain only the packed tensors. This does not
    support changing a live model's layout or loading an already packed checkpoint.
    """
    if weight.ndim != 2 or weight.shape[1] % 2 or scales.shape != (weight.shape[1],):
        raise ValueError("expected [K,2*N] weight and [2*N] channel scales")
    if weight.device != scales.device:
        raise ValueError("weight and scales must share a device")
    n = weight.shape[1] // 2
    packed = torch.stack((weight[:, :n].t(), weight[:, n:].t()), dim=1)
    packed_scales = torch.stack((scales[:n], scales[n:]), dim=1)
    packed_weight = packed.reshape(2 * n, weight.shape[0]).contiguous().t()
    return packed_weight, packed_scales.flatten()


class InterleavedSwiGLU(nn.Module):
    """Activation paired only with a projection using interleaved channels."""

    def forward(self, x):
        from vllm_fl.ops.linear_swiglu import interleaved_swiglu

        return interleaved_swiglu(x)


class PaddedGateTokens(NamedTuple):
    values: torch.Tensor
    scales: torch.Tensor
    shape_source: torch.Tensor


@torch.library.custom_op("vllm_fl::gate_norm_quant", mutates_args=())
def gate_norm_quant(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from flag_gems import add_rms_norm_quantize_int8

    m = x.shape[0]
    multiple = 32 if m <= 32 else 128
    rows = ((m + multiple - 1) // multiple) * multiple
    return add_rms_norm_quantize_int8(x, residual, weight, eps, padded_rows=rows)


@gate_norm_quant.register_fake
def _gate_norm_fake(x, residual, weight, eps):
    m = x.shape[0]
    rows = torch.sym_ite(m <= 32, ((m + 31) // 32) * 32, ((m + 127) // 128) * 128)
    return (
        x.new_empty((rows, x.shape[1]), dtype=torch.int8),
        x.new_empty((rows, 1), dtype=torch.float32),
        torch.empty_like(x),
    )


@torch.library.custom_op("vllm_fl::gate_from_padded_int8", mutates_args=())
def gate_from_padded_int8(
    source: torch.Tensor,
    q: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    from flag_gems import scaled_int8_linear

    m = source.shape[0]
    if not m:
        return source.new_empty((0, weight.shape[1]))
    return scaled_int8_linear(q, weight, scale, weight_scale, kind="gate")[:m]


@gate_from_padded_int8.register_fake
def _gate_from_padded_fake(source, q, scale, weight, weight_scale):
    return source.new_empty((source.shape[0], weight.shape[1]))


class GateQuantizedRMSNorm(nn.Module):
    """Only for a paired gate followed by an opaque input-dtype residual consumer."""

    def __init__(self, original):
        super().__init__()
        self.weight = original.weight
        self.variance_epsilon = original.variance_epsilon
        self.variance_size_override = None
        self.has_weight = True

    def forward(self, x, residual=None):
        if residual is None:
            from vllm_fl.dispatch import CachedOp

            return CachedOp("rms_norm")(self, x, None)
        q, scale, residual_out = gate_norm_quant(
            x, residual, self.weight, self.variance_epsilon
        )
        return PaddedGateTokens(q, scale, x), residual_out


def install_gate_norm_quant(model, config, vendor):
    """Fuse only edges whose next opaque norm already materializes BF16 residual."""
    enabled = feature_enabled(config, "ffn_gate_norm_quant")
    if not enabled or vendor != "iluvatar":
        return 0
    from vllm.model_executor.models.llama import LlamaDecoderLayer

    from vllm_fl.ops.layernorm import RMSNormFL

    count = 0
    for parent in model.modules():
        layers = getattr(parent, "layers", None)
        if type(layers) is not nn.ModuleList:
            continue
        for layer, following in zip(layers, layers[1:]):
            if (
                type(layer) is not LlamaDecoderLayer
                or type(following) is not LlamaDecoderLayer
                or type(following.input_layernorm) is not QuantizedRMSNorm
                or type(layer.post_attention_layernorm) is not RMSNormFL
                or type(layer.mlp.gate_up_proj) is not W8A8GateUp
                or not layer.mlp.gate_up_proj.dynamic_input
            ):
                continue
            layer.post_attention_layernorm = GateQuantizedRMSNorm(
                layer.post_attention_layernorm
            )
            count += 1
    return count


class W8A8GateUp(nn.Module):
    """Bias-free, local gate/up projection; keeps the Linear tuple contract."""

    def __init__(self, original, *, interleaved=False, dynamic_input=False):
        super().__init__()
        weight, scales = quantize_channelwise_weight(original.weight)
        self.dynamic_input = dynamic_input
        self.interleaved = interleaved
        if interleaved:
            weight, scales = interleave_gate_up(weight, scales)
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.register_buffer("weight_scale", scales)
        self.output_dtype = original.weight.dtype

    def forward(self, x):
        if self.dynamic_input:
            if isinstance(x, PaddedGateTokens):
                return (
                    gate_from_padded_int8(
                        x.shape_source,
                        x.values,
                        x.scales,
                        self.weight,
                        self.weight_scale,
                    ),
                    None,
                )
            from .projections import dynamic_int8_linear

            if x.shape[0] == 0:
                return x.new_empty((0, self.weight.shape[1])), None
            return (
                dynamic_int8_linear(
                    x.contiguous(), self.weight, self.weight_scale, "gate"
                ),
                None,
            )
        # Independent MLP callers can still provide ordinary floating activations.
        # The paired decoder path provides QuantizedTokens without re-quantizing.
        if isinstance(x, QuantizedTokens):
            q, scales = x
        else:
            q, scales = _quant(x)
        return (
            _scaled_mm(q, self.weight, scales, self.weight_scale, self.output_dtype),
            None,
        )


def install_ffn_w8a8(model, config, vendor_name: str) -> int:
    """Return installed pair count; unsupported configurations stay unchanged.

    ``additional_config={"ffn_w8a8": True}`` is an explicit opt-in to this
    implementation, not vLLM's existing global quantization selection. No model
    name, layer number, batch size or MiniCPM-specific matrix shape is assumed.
    The optional linear_swiglu_fusion pass can fuse this projection's INT8
    operation with its activation. If it cannot match, this path stays unfused.
    """
    enabled = feature_enabled(config, "ffn_w8a8")
    norm_fusion = feature_enabled(config, "ffn_gate_norm_quant")
    if norm_fusion and vendor_name != "iluvatar":
        raise ValueError("deferred gate norm is supported only on Iluvatar")
    interleaved = feature_enabled(config, "ffn_w8a8_interleaved")
    if interleaved and not enabled:
        raise ValueError("ffn_w8a8_interleaved requires ffn_w8a8")
    if interleaved and vendor_name != "metax":
        raise ValueError("interleaved gate/up is supported only on MetaX")
    if not enabled:
        return 0

    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.model_executor.layers.linear import (
        MergedColumnParallelLinear,
        UnquantizedLinearMethod,
    )
    from vllm.model_executor.models.llama import LlamaDecoderLayer, LlamaMLP

    from vllm_fl.ops.activation import SiluAndMulFL
    from vllm_fl.ops.layernorm import RMSNormFL
    from vllm_fl.utils import is_oot_enabled, use_flaggems_op

    if (
        vendor_name not in ("metax", "iluvatar")
        or not supports_local_dense(config)
        or not is_oot_enabled()
        or not use_flaggems_op("rms_norm")
    ):
        return 0

    pairs = []
    for layer in model.modules():
        if type(layer) is not LlamaDecoderLayer or type(layer.mlp) is not LlamaMLP:
            continue
        norm, projection = layer.post_attention_layernorm, layer.mlp.gate_up_proj
        if isinstance(projection, W8A8GateUp):
            if projection.interleaved != interleaved:
                raise ValueError("cannot change gate/up layout after installation")
            if interleaved and not isinstance(layer.mlp.act_fn, InterleavedSwiGLU):
                raise ValueError(
                    "interleaved projection requires its paired activation"
                )
            continue
        if (
            type(norm) is not RMSNormFL
            or norm.variance_size_override is not None
            or not norm.has_weight
            or type(projection) is not MergedColumnParallelLinear
            or type(projection.quant_method) is not UnquantizedLinearMethod
            or projection.bias is not None
            or projection.gather_output
            or not projection.return_bias
            or not isinstance(layer.mlp.act_fn, SiluAndMul)
            or (
                interleaved and type(layer.mlp.act_fn) not in (SiluAndMul, SiluAndMulFL)
            )
            or projection.weight.dtype != torch.bfloat16
            or projection.weight.device.type != "cuda"
            or projection.weight.ndim != 2
            or len(projection.output_sizes) != 2
            or projection.output_sizes[0] != projection.output_sizes[1]
            or sum(projection.output_sizes) != projection.weight.shape[0]
            or norm.weight.shape != (projection.weight.shape[1],)
            or norm.weight.dtype != projection.weight.dtype
            or norm.weight.device != projection.weight.device
            or not 0 < projection.weight.shape[1] <= 8192
        ):
            continue
        pairs.append((layer, norm, projection))

    # Complete validation/conversion before mutating any module. A failed load
    # cannot leave half a norm/projection pair or a partially converted model.
    prepared = [
        (
            layer,
            (QuantizedRMSNorm(norm) if vendor_name == "metax" else norm),
            W8A8GateUp(
                projection,
                interleaved=interleaved,
                dynamic_input=vendor_name == "iluvatar",
            ),
            InterleavedSwiGLU() if interleaved else layer.mlp.act_fn,
        )
        for layer, norm, projection in pairs
    ]
    for layer, norm, projection, activation in prepared:
        layer.post_attention_layernorm = norm
        layer.mlp.gate_up_proj = projection
        layer.mlp.act_fn = activation
    return len(prepared)
