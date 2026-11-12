# Copyright (c) 2026 BAAI. All rights reserved.
"""Local W8A8 projections using one stable public FlagGems boundary."""

import torch
from torch import nn

from ...optimizations.config import feature_enabled, supports_local_dense

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


class W8A8Linear(nn.Module):
    """Shared quantized weight storage and the Linear `(output, None)` contract."""

    def __init__(self, original):
        super().__init__()
        weight, scale = quantize_channelwise_weight(original.weight)
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.register_buffer("weight_scale", scale)
        self.output_dtype = original.weight.dtype

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != self.weight.shape[0]:
            raise ValueError(f"{self.input_name} input must be [tokens, in_features]")
        if x.dtype != self.output_dtype or x.device != self.weight.device:
            raise ValueError(
                f"{self.input_name} input must match the loaded weight dtype/device"
            )
        if x.shape[0] == 0:
            return x.new_empty((0, self.weight.shape[1])), None
        return (
            dynamic_int8_linear(
                x.contiguous(), self.weight, self.weight_scale, self.kind
            ),
            None,
        )


class W8A8Down(W8A8Linear):
    kind = "down"
    input_name = "down"


def _eligible_row_linear(projection, linear_type, method_type):
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


def _install_projections(parents, attribute, component):
    # Complete weight conversion before mutating any module.
    prepared = [(parent, component(getattr(parent, attribute))) for parent in parents]
    for parent, projection in prepared:
        setattr(parent, attribute, projection)
    return len(prepared)


def install_ffn_down_w8a8(model, config, vendor_name: str) -> int:
    """Install the MetaX down projection independently of gate/up.

    No model name, layer index, M value or fixed matrix shape is matched.
    Unsupported configurations retain their original down projections. Other
    vendors and distributed/offloaded/adapter paths need separate validation.
    Call after checkpoint loading and before constructing prefill-tail views.
    """
    enabled = feature_enabled(config, "ffn_down_w8a8")
    if not enabled or vendor_name != "metax":
        return 0

    from vllm.model_executor.layers.linear import (
        RowParallelLinear,
        UnquantizedLinearMethod,
    )
    from vllm.model_executor.models.llama import LlamaDecoderLayer, LlamaMLP

    from vllm_fl.utils import is_oot_enabled

    if not supports_local_dense(config) or not is_oot_enabled():
        return 0

    mlps = []
    for layer in model.modules():
        if type(layer) is not LlamaDecoderLayer or type(layer.mlp) is not LlamaMLP:
            continue
        down = layer.mlp.down_proj
        if isinstance(down, W8A8Down):
            continue
        if _eligible_row_linear(down, RowParallelLinear, UnquantizedLinearMethod):
            mlps.append(layer.mlp)
    return _install_projections(mlps, "down_proj", W8A8Down)


class W8A8OProjection(W8A8Linear):
    kind = "o"
    input_name = "O projection"


def install_attention_o_w8a8(model, config, vendor_name: str) -> int:
    """Install the MetaX O projection independently of FFN quantization.

    No model name, layer index, M value or fixed matrix shape is matched.
    Unsupported configurations retain their original O projections. Other
    vendors and distributed/offloaded/adapter paths need separate validation.
    Call after checkpoint loading and before constructing prefill-tail views.
    """
    enabled = feature_enabled(config, "attention_o_w8a8")
    if not enabled or vendor_name != "metax":
        return 0

    from vllm.model_executor.layers.linear import (
        RowParallelLinear,
        UnquantizedLinearMethod,
    )
    from vllm.model_executor.models.llama import LlamaAttention, LlamaDecoderLayer

    from vllm_fl.utils import is_oot_enabled

    if not supports_local_dense(config) or not is_oot_enabled():
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
        if _eligible_row_linear(projection, RowParallelLinear, UnquantizedLinearMethod):
            attentions.append(layer.self_attn)
    return _install_projections(attentions, "o_proj", W8A8OProjection)


class W8A8QKV(W8A8Linear):

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
        return (
            scaled_int8_linear(
                q.contiguous(), self.weight, sx.contiguous(), self.weight_scale, "qkv"
            ),
            None,
        )


def _install_qkv_prepared(pairs):
    prepared = [
        (layer, QuantizedRMSNorm(norm), W8A8QKV(proj)) for layer, norm, proj in pairs
    ]
    for layer, norm, proj in prepared:
        layer.input_layernorm = norm
        layer.self_attn.qkv_proj = proj
    return len(prepared)


def install_qkv_w8a8(model, config, vendor_name):
    enabled = feature_enabled(config, "qkv_w8a8")
    if not enabled or vendor_name not in ("metax", "iluvatar"):
        return 0
    from vllm.model_executor.layers.linear import (
        QKVParallelLinear,
        UnquantizedLinearMethod,
    )
    from vllm.model_executor.models.llama import LlamaAttention, LlamaDecoderLayer

    from vllm_fl.ops.layernorm import RMSNormFL
    from vllm_fl.utils import is_oot_enabled, use_flaggems_op

    if (
        not supports_local_dense(config)
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
