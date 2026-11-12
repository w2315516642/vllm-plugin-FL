# SPDX-License-Identifier: Apache-2.0
"""Pair RoPE, KV writes and complete-page summaries at a model-local boundary."""

from types import MethodType

import torch

from vllm.config import CUDAGraphMode
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.model_executor.models.llama import LlamaAttention
from vllm.utils.torch_utils import _encode_layer_name

from ._graph import QuestGraphs
from ._quest import QuestState
from ._rope_kernels import launch_rope, launch_summary
from .impl import TunedTritonAttentionImpl


@torch.library.custom_op("vllm_fl::rope_cache", mutates_args=())
def rope_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    positions: torch.Tensor,
    cs: torch.Tensor,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Match vLLM's hidden-cache mutation and explicit dummy-dependency contract.
    _, layer, cache, slots = get_attention_context(name)
    qo = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    ko = torch.empty(k.shape, device=k.device, dtype=k.dtype)
    dependency = q.new_empty((0,))
    if cache.numel() and slots is not None:
        state = layer.impl._quest_state
        supported = (
            cache.ndim == 5
            and cache.shape[1:] == (2, 16, 2, 128)
            and cache.dtype == q.dtype
            and cache.stride(-1) == 1
            and cache.stride(-2) == 128
            and q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
            and q.stride(1) == k.stride(1) == v.stride(1) == 128
        )
        if supported:
            low, high = state.pool(name, cache)
            launch_summary(
                q, k, v, positions.flatten(), cs, qo, ko, cache, slots, low, high
            )
        else:
            launch_rope(
                q,
                k,
                v,
                positions,
                cs,
                qo,
                ko,
                cache,
                slots,
                fused=True,
                block=512,
                warps=4,
            )
            state.update(name, cache, slots)
    else:
        scratch = q.new_empty((1, 2, 16, k.shape[1], q.shape[-1]))
        empty_slots = torch.empty((0,), device=q.device, dtype=torch.int64)
        launch_rope(
            q, k, v, positions, cs, qo, ko, scratch, empty_slots, block=512, warps=4
        )
    return qo, ko, dependency


@rope_cache.register_fake
def _fake(q, k, v, positions, cs, name):
    return (
        torch.empty(q.shape, device=q.device, dtype=q.dtype),
        torch.empty(k.shape, device=k.device, dtype=k.dtype),
        q.new_empty((0,)),
    )


def fused_forward(self, positions, hidden_states):
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q = q.view(-1, self.num_heads, self.head_dim)
    k = k.view(-1, self.num_kv_heads, self.head_dim)
    v = v.view(-1, self.num_kv_heads, self.head_dim)
    q, k, dependency = rope_cache(
        q,
        k,
        v,
        positions.flatten(),
        self.rotary_emb.cos_sin_cache,
        self.attn.layer_name,
    )
    output = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    torch.ops.vllm.unified_attention_with_output(
        q,
        k,
        v,
        output,
        _encode_layer_name(self.attn.layer_name),
        kv_cache_dummy_dep=dependency,
    )
    result, _ = self.o_proj(output.view(-1, self.q_size))
    return result


def install(model, config):
    """Caller applies the existing dense-model, single-GPU optimization guard.

    The V1 runner owns contiguous query runs; writable physical pages are
    exclusive, while prefix-cache sharing applies only to complete read-only
    pages. No per-layer slot readback is required under this cache contract.
    """
    if config.compilation_config.cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY:
        return None
    prepared = []
    for module in model.modules():
        if type(module) is not LlamaAttention:
            continue
        rope, attn = module.rotary_emb, module.attn
        impl = attn.impl
        if not (
            type(impl) is TunedTritonAttentionImpl
            and isinstance(rope, RotaryEmbedding)
            and type(rope).forward_native is RotaryEmbedding.forward_native
            and rope.rotary_dim == rope.head_size == module.head_dim == 128
            and rope.is_neox_style
            and rope.cos_sin_cache.dtype == torch.bfloat16
            and rope.cos_sin_cache.is_contiguous()
            and module.num_heads == 16
            and module.num_kv_heads == 2
            and attn.query_quant is None
            and not attn.calculate_kv_scales
            and attn.kv_sharing_target_layer_name is None
            and not attn.use_direct_call
            and attn.attn_type.value == "decoder"
            and attn.kv_cache_dtype == "auto"
            and impl.sliding_window == (-1, -1)
            and impl.logits_soft_cap == 0
            and impl.alibi_slopes is None
            and impl.sinks is None
        ):
            return None
        prepared.append(module)
    if not prepared or config.compilation_config.splitting_ops is None:
        return None
    capacity = max(
        config.scheduler_config.max_num_seqs,
        config.compilation_config.max_cudagraph_capture_size or 0,
    )
    state = QuestState(
        next(model.parameters()).device, capacity, config.model_config.max_model_len
    )
    for module in prepared:
        module.attn.impl._quest_state = state
        module.forward = MethodType(fused_forward, module)
    if "vllm_fl::rope_cache" not in config.compilation_config.splitting_ops:
        config.compilation_config.splitting_ops.append("vllm_fl::rope_cache")
    return QuestGraphs(config.model_config.max_model_len)
