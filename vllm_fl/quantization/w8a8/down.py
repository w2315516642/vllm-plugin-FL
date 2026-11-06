# Copyright (c) 2026 BAAI. All rights reserved.
"""Opt-in, load-time down-projection W8A8 for local dense FFNs.

This boundary consumes the existing floating SwiGLU output. Activation
quantization is deliberately part of forward, including in performance tests.
It does not change gate/up packing, activation semantics or GEMM dispatch.
"""

import torch
from torch import nn

from .projections import dynamic_int8_linear
from vllm_fl.quantization.w8a8.ffn import (
    quantize_channelwise_weight,
)


class W8A8Down(nn.Module):
    """Bias-free TP=1 projection with the Linear `(output, None)` contract."""

    def __init__(self, original):
        super().__init__()
        weight, scale = quantize_channelwise_weight(original.weight)
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.register_buffer("weight_scale", scale)
        self.output_dtype = original.weight.dtype

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != self.weight.shape[0]:
            raise ValueError("down input must be [tokens, in_features]")
        if x.dtype != self.output_dtype or x.device != self.weight.device:
            raise ValueError("down input must match the loaded weight dtype/device")
        if x.shape[0] == 0:
            return x.new_empty((0, self.weight.shape[1])), None
        # No value-dependent host checks or loading-time weight rearrangements.
        return dynamic_int8_linear(
            x.contiguous(), self.weight, self.weight_scale, "down"
        ), None


def _eligible_down(projection, linear_type, method_type):
    """Keep unsupported Linear contracts/implementations on their old path."""
    if type(projection) is not linear_type:
        return False
    w = projection.weight
    return (
        type(projection.quant_method) is method_type
        and projection.bias is None
        and projection.return_bias
        and projection.input_is_parallel
        and projection.tp_size == 1
        and projection.tp_rank == 0
        and w.dtype == torch.bfloat16
        and w.device.type == "cuda"
        and w.ndim == 2
        and w.is_contiguous()
        and w.shape == (projection.output_size, projection.input_size_per_partition)
        and w.shape[0] > 0
        # Bound the per-token quantization reduction to the initial scope.
        and 0 < w.shape[1] <= 8192
    )


def _install_prepared(mlps):
    # Finish conversion/finite checks before replacing any projection. Do not
    # retain the BF16 copy; unsupported modules are filtered before this point.
    prepared = [(mlp, W8A8Down(mlp.down_proj)) for mlp in mlps]
    for mlp, projection in prepared:
        mlp.down_proj = projection
    return len(prepared)


def install_ffn_down_w8a8(model, config, vendor_name: str) -> int:
    """Install independently of gate/up quantization; disabled by default.

    No model name, layer index, M value or fixed matrix shape is matched.
    Unsupported configurations retain their original down projections. Other
    vendors and distributed/offloaded/adapter paths need separate validation.
    Call after checkpoint loading and before constructing prefill-tail views.
    """
    extra = config.additional_config
    enabled = extra.get("ffn_down_w8a8", False) if isinstance(extra, dict) else False
    if type(enabled) is not bool:
        raise ValueError("ffn_down_w8a8 must be a boolean")
    if not enabled or vendor_name != "metax":
        return 0

    from vllm.config import CompilationMode
    from vllm.model_executor.layers.linear import (
        RowParallelLinear,
        UnquantizedLinearMethod,
    )
    from vllm.model_executor.models.llama import LlamaDecoderLayer, LlamaMLP

    from vllm_fl.utils import is_oot_enabled

    mc, pc = config.model_config, config.parallel_config
    if (
        mc.dtype != torch.bfloat16
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
    ):
        return 0

    mlps = []
    for layer in model.modules():
        if type(layer) is not LlamaDecoderLayer or type(layer.mlp) is not LlamaMLP:
            continue
        down = layer.mlp.down_proj
        if isinstance(down, W8A8Down):
            continue
        if _eligible_down(down, RowParallelLinear, UnquantizedLinearMethod):
            mlps.append(layer.mlp)
    return _install_prepared(mlps)
