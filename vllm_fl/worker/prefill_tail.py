# Copyright (c) 2026 BAAI. All rights reserved.
"""Compute the last token-wise block only for rows consumed by sampling.

The attention prefix keeps its original token dimension and writes every KV
entry. Selection happens outside that compiled prefix, so its graph cache never
depends on the (independent) number of requested output rows. The original model
and its decode graphs remain available for consumers needing full hidden states.
"""

from copy import copy

import torch
from torch import nn

from vllm.compilation.backends import set_model_tag
from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    CompilationMode,
    CUDAGraphMode,
    VllmConfig,
    set_current_vllm_config,
)


@support_torch_compile(dynamic_arg_dims={"input_ids": {0: "b"}, "positions": {0: "b"}})
class LlamaPrefillPrefix(nn.Module):
    """The original Llama computation through the final attention projection."""

    def __init__(self, backbone: nn.Module, *, vllm_config: VllmConfig):
        super().__init__()
        # Reuse loaded modules, including their attention/KV-cache objects.
        self.embed_tokens = backbone.embed_tokens
        self.layers = backbone.layers

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers[:-1]:
            hidden_states, residual = layer(positions, hidden_states, residual)
        layer = self.layers[-1]
        if residual is None:
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
        else:
            hidden_states, residual = layer.input_layernorm(hidden_states, residual)
        hidden_states = layer.self_attn(
            positions=positions, hidden_states=hidden_states
        )
        return hidden_states, residual


def finish_selected_rows(
    layer: nn.Module,
    final_norm: nn.Module,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    output_rows: torch.Tensor,
) -> torch.Tensor:
    """Preserve the caller's row order, including repeated requested rows."""
    hidden_states = hidden_states.index_select(0, output_rows)
    residual = residual.index_select(0, output_rows)
    return finish_rows(layer, final_norm, hidden_states, residual)


def finish_rows(layer, final_norm, hidden_states, residual):
    hidden_states, residual = layer.post_attention_layernorm(hidden_states, residual)
    hidden_states = layer.mlp(hidden_states)
    hidden_states, _ = final_norm(hidden_states, residual)
    return hidden_states


@support_torch_compile(
    dynamic_arg_dims={"hidden_states": {0: "b"}, "residual": {0: "b"}}
)
class LlamaSelectedTail(nn.Module):
    """Keep the original compiler lowering on the independent output dimension."""

    def __init__(self, backbone: nn.Module, *, vllm_config: VllmConfig):
        super().__init__()
        self.layer = backbone.layers[-1]
        self.norm = backbone.norm

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor):
        return finish_rows(self.layer, self.norm, hidden_states, residual)


def can_select_output_rows(
    *,
    num_tokens: int,
    output_rows: torch.Tensor,
    cudagraph_mode: CUDAGraphMode,
    has_prompt_logprobs: bool,
    has_aux_hidden_states: bool,
    has_intermediate_tensors: bool,
    has_inputs_embeds: bool,
    has_model_kwargs: bool,
    should_ubatch: bool,
) -> bool:
    # No device-to-host inspection: these are already available scheduling facts.
    return (
        cudagraph_mode != CUDAGraphMode.FULL
        and not has_prompt_logprobs
        and not has_aux_hidden_states
        and not has_intermediate_tensors
        and not has_inputs_embeds
        and not has_model_kwargs
        and not should_ubatch
        and output_rows.ndim == 1
        and output_rows.dtype in (torch.int32, torch.int64)
        and 0 < output_rows.numel() < num_tokens
    )


class LlamaPrefillTail:
    """Runner-owned adapter; does not replace the model or its weight names."""

    def __init__(self, backbone: nn.Module, vllm_config: VllmConfig):
        # Independent compiled models must not share the backbone cache entry.
        with set_model_tag("prefill_output_prefix"):
            self.prefix = LlamaPrefillPrefix(backbone, vllm_config=vllm_config)
        self.last_layer = backbone.layers[-1]
        self.final_norm = backbone.norm
        # The tail's row count is independent of the scheduler's token count.
        # Keep the same compiler passes, but never key a tail CUDA graph by the
        # enclosing full-token batch descriptor. Do not mutate the runner config.
        tail_config = copy(vllm_config)
        tail_config.compilation_config = copy(vllm_config.compilation_config)
        cc = tail_config.compilation_config
        cc.cudagraph_mode = CUDAGraphMode.NONE
        cc.cudagraph_capture_sizes = []
        cc.compile_sizes = []
        cc.cache_dir = ""
        cc.local_cache_dir = ""
        cc.traced_files = set()
        with (
            set_current_vllm_config(tail_config),
            set_model_tag("prefill_output_tail"),
        ):
            self.tail = LlamaSelectedTail(backbone, vllm_config=tail_config)
        self._tail_warmed = False

    def warmup_tail(self, hidden_states, residual):
        if not self._tail_warmed:
            self.tail(hidden_states, residual)
            self._tail_warmed = True

    def __call__(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        output_rows: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states, residual = self.prefix(input_ids, positions)
        return self.tail(
            hidden_states.index_select(0, output_rows),
            residual.index_select(0, output_rows),
        )


def make_prefill_tail(model: nn.Module, config: VllmConfig) -> LlamaPrefillTail | None:
    """Opt in only structures and consumers whose output contract is known."""
    from vllm.model_executor.models.llama import (
        LlamaAttention,
        LlamaDecoderLayer,
        LlamaForCausalLM,
        LlamaMLP,
        LlamaModel,
    )

    mc, pc = config.model_config, config.parallel_config
    if (
        type(model) is not LlamaForCausalLM
        or type(model.model) is not LlamaModel
        or mc.runner_type != "generate"
        or mc.dtype != torch.bfloat16
        or mc.is_encoder_decoder
        or mc.multimodal_config is not None
        or mc.enable_prompt_embeds
        or not getattr(mc.hf_config, "is_causal", True)
        or config.quant_config is not None
        or config.lora_config is not None
        or config.speculative_config is not None
        or config.kv_transfer_config is not None
        or pc.tensor_parallel_size != 1
        or pc.pipeline_parallel_size != 1
        or pc.data_parallel_size != 1
        or pc.decode_context_parallel_size != 1
        or pc.prefill_context_parallel_size != 1
        or pc.use_ubatching
        or config.offload_config.uva.cpu_offload_gb > 0
        or config.offload_config.prefetch.offload_group_size > 0
        or config.compilation_config.mode == CompilationMode.STOCK_TORCH_COMPILE
    ):
        return None
    backbone = model.model
    if (
        not backbone.layers
        or backbone.start_layer != 0
        or backbone.end_layer != len(backbone.layers)
        or backbone.aux_hidden_state_layers
        or any(
            type(layer) is not LlamaDecoderLayer
            or type(layer.self_attn) is not LlamaAttention
            or type(layer.mlp) is not LlamaMLP
            for layer in backbone.layers
        )
    ):
        return None
    return LlamaPrefillTail(backbone, config)
