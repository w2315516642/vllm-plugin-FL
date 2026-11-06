# Copyright (c) 2026 BAAI. All rights reserved.
"""Local W8A8 projections using one stable public FlagGems boundary."""

import torch
from torch import nn

from .ffn import QuantizedRMSNorm, QuantizedTokens, _quant, quantize_channelwise_weight


@torch.library.custom_op("vllm_fl::scaled_int8_linear", mutates_args=())
def scaled_int8_linear(
    q: torch.Tensor,
    weight: torch.Tensor,
    scale_x: torch.Tensor,
    scale_w: torch.Tensor,
    kind: str,
) -> torch.Tensor:
    from flag_gems import scaled_int8_linear as implementation

    return implementation(q, weight, scale_x, scale_w, kind=kind)


@scaled_int8_linear.register_fake
def _scaled_fake(q, weight, scale_x, scale_w, kind):
    return torch.empty(
        (q.shape[0], weight.shape[1]), device=q.device, dtype=torch.bfloat16
    )


@torch.library.custom_op("vllm_fl::dynamic_int8_linear", mutates_args=())
def dynamic_int8_linear(
    x: torch.Tensor, weight: torch.Tensor, scale_w: torch.Tensor, kind: str
) -> torch.Tensor:
    from flag_gems import dynamic_int8_linear as implementation

    return implementation(x, weight, scale_w, kind=kind)


@dynamic_int8_linear.register_fake
def _dynamic_fake(x, weight, scale_w, kind):
    return torch.empty(
        (x.shape[0], weight.shape[1]), device=x.device, dtype=torch.bfloat16
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


def _install_down_prepared(mlps):
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
    return _install_down_prepared(mlps)


class W8A8OProjection(nn.Module):
    """Bias-free TP=1 projection with the Linear `(output, None)` contract."""

    def __init__(self, original):
        super().__init__()
        weight, scale = quantize_channelwise_weight(original.weight)
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.register_buffer("weight_scale", scale)
        self.output_dtype = original.weight.dtype

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != self.weight.shape[0]:
            raise ValueError("O projection input must be [tokens, in_features]")
        if x.dtype != self.output_dtype or x.device != self.weight.device:
            raise ValueError(
                "O projection input must match the loaded weight dtype/device"
            )
        if x.shape[0] == 0:
            return x.new_empty((0, self.weight.shape[1])), None
        # No value-dependent host checks or loading-time weight rearrangements.
        return dynamic_int8_linear(
            x.contiguous(), self.weight, self.weight_scale, "o"
        ), None


def _eligible_o(projection, linear_type, method_type):
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


def _install_o_prepared(attentions):
    # Finish conversion/finite checks before replacing any projection. Do not
    # retain the BF16 copy; unsupported modules are filtered before this point.
    prepared = [
        (attention, W8A8OProjection(attention.o_proj)) for attention in attentions
    ]
    for attention, projection in prepared:
        attention.o_proj = projection
    return len(prepared)


def install_attention_o_w8a8(model, config, vendor_name: str) -> int:
    """Install independently of FFN quantization; disabled by default.

    No model name, layer index, M value or fixed matrix shape is matched.
    Unsupported configurations retain their original O projections. Other
    vendors and distributed/offloaded/adapter paths need separate validation.
    Call after checkpoint loading and before constructing prefill-tail views.
    """
    extra = config.additional_config
    enabled = extra.get("attention_o_w8a8", False) if isinstance(extra, dict) else False
    if type(enabled) is not bool:
        raise ValueError("attention_o_w8a8 must be a boolean")
    if not enabled or vendor_name != "metax":
        return 0

    from vllm.config import CompilationMode
    from vllm.model_executor.layers.linear import (
        RowParallelLinear,
        UnquantizedLinearMethod,
    )
    from vllm.model_executor.models.llama import LlamaAttention, LlamaDecoderLayer

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

    attentions = []
    for layer in model.modules():
        if (
            type(layer) is not LlamaDecoderLayer
            or type(layer.self_attn) is not LlamaAttention
        ):
            continue
        projection = layer.self_attn.o_proj
        if isinstance(projection, W8A8OProjection):
            continue
        if _eligible_o(projection, RowParallelLinear, UnquantizedLinearMethod):
            attentions.append(layer.self_attn)
    return _install_o_prepared(attentions)


class W8A8QKV(nn.Module):
    def __init__(self, original):
        super().__init__()
        weight, scale = quantize_channelwise_weight(original.weight)
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.register_buffer("weight_scale", scale)
        self.output_dtype = original.weight.dtype

    def forward(self, x):
        if isinstance(x, QuantizedTokens):
            q, sx = x
            if (
                q.ndim != 2
                or q.shape[1] != self.weight.shape[0]
                or q.dtype != torch.int8
                or q.device != self.weight.device
                or sx.shape != (q.shape[0], 1)
                or sx.dtype != torch.float32
                or sx.device != q.device
            ):
                raise ValueError("invalid quantized QKV input contract")
        else:
            if (
                x.ndim != 2
                or x.shape[1] != self.weight.shape[0]
                or x.dtype != self.output_dtype
                or x.device != self.weight.device
            ):
                raise ValueError(
                    "QKV input must match loaded weight shape, dtype and device"
                )
            if x.shape[0] == 0:
                return x.new_empty((0, self.weight.shape[1])), None
            q, sx = _quant(x.contiguous())
        return scaled_int8_linear(
            q.contiguous(), self.weight, sx.contiguous(), self.weight_scale, "qkv"
        ), None


def _install_qkv_prepared(pairs):
    prepared = [
        (layer, QuantizedRMSNorm(norm), W8A8QKV(proj)) for layer, norm, proj in pairs
    ]
    for layer, norm, proj in prepared:
        layer.input_layernorm = norm
        layer.self_attn.qkv_proj = proj
    return len(prepared)


def install_qkv_w8a8(model, config, vendor_name):
    extra = config.additional_config
    enabled = extra.get("qkv_w8a8", False) if isinstance(extra, dict) else False
    if type(enabled) is not bool:
        raise ValueError("qkv_w8a8 must be a boolean")
    if not enabled or vendor_name not in ("metax", "iluvatar"):
        return 0
    from vllm.config import CompilationMode
    from vllm.model_executor.layers.linear import (
        QKVParallelLinear,
        UnquantizedLinearMethod,
    )
    from vllm.model_executor.models.llama import LlamaAttention, LlamaDecoderLayer

    from vllm_fl.ops.layernorm import RMSNormFL
    from vllm_fl.utils import is_oot_enabled, use_flaggems_op

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
        or not use_flaggems_op("rms_norm")
    ):
        return 0
    pairs = []
    for layer in model.modules():
        if (
            type(layer) is not LlamaDecoderLayer
            or type(layer.self_attn) is not LlamaAttention
        ):
            continue
        norm, proj = layer.input_layernorm, layer.self_attn.qkv_proj
        if isinstance(norm, QuantizedRMSNorm) and isinstance(proj, W8A8QKV):
            continue
        if (
            type(norm) is not RMSNormFL
            or norm.variance_size_override is not None
            or not norm.has_weight
            or type(proj) is not QKVParallelLinear
            or type(proj.quant_method) is not UnquantizedLinearMethod
            or proj.bias is not None
            or proj.gather_output
            or not proj.return_bias
            or proj.tp_size != 1
            or proj.tp_rank != 0
        ):
            continue
        w = proj.weight
        widths = (
            layer.self_attn.q_size,
            layer.self_attn.kv_size,
            layer.self_attn.kv_size,
        )
        if (
            w.dtype != torch.bfloat16
            or w.device.type != "cuda"
            or w.ndim != 2
            or not w.is_contiguous()
            or tuple(proj.output_sizes) != widths
            or sum(widths) != w.shape[0]
            or min(widths) <= 0
            or w.shape[1] != proj.input_size
            or not 0 < w.shape[1] <= 8192
            or norm.weight.shape != (w.shape[1],)
            or norm.weight.dtype != w.dtype
            or norm.weight.device != w.device
        ):
            continue
        pairs.append((layer, norm, proj))
    return _install_qkv_prepared(pairs)
