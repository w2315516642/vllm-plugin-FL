# Copyright (c) 2026 BAAI. All rights reserved.
"""Opt-in gate/up W8A8 with a paired RMSNorm/quantized-input boundary.

Weights are quantized once after BF16 loading. The model's MLP, activation,
down projection and global quantization configuration stay unchanged. The
private QuantizedTokens edge is installed only where its sole consumer is the
paired gate/up projection, including the demand-driven final-layer tail.
"""

from typing import NamedTuple

import torch
from torch import nn


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


class W8A8GateUp(nn.Module):
    """Bias-free, local gate/up projection; keeps the Linear tuple contract."""

    def __init__(self, original):
        super().__init__()
        weight, scales = quantize_channelwise_weight(original.weight)
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.register_buffer("weight_scale", scales)
        self.output_dtype = original.weight.dtype

    def forward(self, x):
        # Independent MLP callers can still provide ordinary floating activations.
        # The paired decoder path provides QuantizedTokens without re-quantizing.
        if isinstance(x, QuantizedTokens):
            q, scales = x
        else:
            q, scales = _quant(x)
        return _scaled_mm(
            q, self.weight, scales, self.weight_scale, self.output_dtype
        ), None


def install_ffn_w8a8(model, config, vendor_name: str) -> int:
    """Return installed pair count; unsupported configurations stay unchanged.

    ``additional_config={"ffn_w8a8": True}`` is an experimental opt-in to this
    implementation, not vLLM's existing global quantization selection. No model
    name, layer number, batch size or MiniCPM-specific matrix shape is assumed.
    """
    extra = config.additional_config
    enabled = extra.get("ffn_w8a8", False) if isinstance(extra, dict) else False
    if type(enabled) is not bool:
        raise ValueError("ffn_w8a8 must be a boolean")
    if not enabled:
        return 0

    from vllm.config import CompilationMode
    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.model_executor.layers.linear import (
        MergedColumnParallelLinear,
        UnquantizedLinearMethod,
    )
    from vllm.model_executor.models.llama import LlamaDecoderLayer, LlamaMLP

    from vllm_fl.ops.layernorm import RMSNormFL
    from vllm_fl.utils import is_oot_enabled, use_flaggems_op

    mc, pc = config.model_config, config.parallel_config
    if (
        vendor_name != "metax"
        or mc.dtype != torch.bfloat16
        or mc.runner_type != "generate"
        or mc.is_encoder_decoder
        or mc.multimodal_config is not None
        or config.quant_config is not None
        or config.lora_config is not None
        or config.speculative_config is not None
        or pc.tensor_parallel_size != 1
        or pc.pipeline_parallel_size != 1
        or pc.data_parallel_size != 1
        or pc.decode_context_parallel_size != 1
        or pc.prefill_context_parallel_size != 1
        or pc.use_ubatching
        or config.kv_transfer_config is not None
        or config.offload_config.uva.cpu_offload_gb > 0
        or config.offload_config.prefetch.offload_group_size > 0
        or config.compilation_config.mode == CompilationMode.STOCK_TORCH_COMPILE
        or not is_oot_enabled()
        or not use_flaggems_op("rms_norm")
        or extra.get("linear_swiglu_fusion", False)
    ):
        return 0

    pairs = []
    for layer in model.modules():
        if type(layer) is not LlamaDecoderLayer or type(layer.mlp) is not LlamaMLP:
            continue
        norm, projection = layer.post_attention_layernorm, layer.mlp.gate_up_proj
        if isinstance(norm, QuantizedRMSNorm) and isinstance(projection, W8A8GateUp):
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
        (layer, QuantizedRMSNorm(norm), W8A8GateUp(projection))
        for layer, norm, projection in pairs
    ]
    for layer, norm, projection in prepared:
        layer.post_attention_layernorm = norm
        layer.mlp.gate_up_proj = projection
    return len(prepared)
