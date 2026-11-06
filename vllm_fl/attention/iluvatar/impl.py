# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Forward derived from vLLM 0.24 TritonAttentionImpl; KV/cache handling unchanged.
from __future__ import annotations

import torch

from vllm.v1.attention.backends.triton_attn import (
    AttentionType,
    KVQuantMode,
    TritonAttentionImpl,
    TritonAttentionMetadata,
    is_quantized_kv_cache,
)
from vllm.v1.attention.ops.triton_unified_attention import (
    unified_attention as original_unified_attention,
)


class TunedTritonAttentionImpl(TritonAttentionImpl):
    """Reuse the selected backend, metadata builder, cache objects and updates.

    This is a per-layer implementation extension, not a backend registration or
    a global replacement of vLLM functions. Unsupported kernel contracts use
    the original Triton entry from the policy.
    """

    @classmethod
    def from_impl(cls, original):
        instance = cls.__new__(cls)
        instance.__dict__.update(original.__dict__)
        return instance

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with Paged Attention impl. in Triton.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [num_blocks, 2, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        if output_block_scale is not None:
            raise NotImplementedError(
                "fused block_scale output quantization is not yet supported"
                " for TritonAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        assert attn_metadata.use_cascade is False

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Handle encoder attention differently - no KV cache needed
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )

        # Per-token-head quantized KV cache: use separate scale caches.
        if self._is_per_token_head_quant:
            self._ensure_scale_caches(kv_cache)
            key_cache, value_cache = kv_cache.unbind(1)
            if key_cache.dtype == torch.uint8:
                key_cache = key_cache.view(self.fp8_dtype)
                value_cache = value_cache.view(self.fp8_dtype)
            q_descale = None
            k_descale = None
            v_descale = None
            k_scale_cache = self._k_scale_cache
            v_scale_cache = self._v_scale_cache
        # FP8 per-tensor / auto path (original flow).
        else:
            key_cache, value_cache = kv_cache.unbind(1)
            if (
                is_quantized_kv_cache(self.kv_cache_dtype)
                and key_cache.dtype != self.fp8_dtype
            ):
                key_cache = key_cache.view(self.fp8_dtype)
                value_cache = value_cache.view(self.fp8_dtype)
            descale_shape = (
                attn_metadata.query_start_loc.shape[0] - 1,
                key_cache.shape[2],
            )
            q_descale = (
                layer._q_scale
                if (
                    self._kv_quant_mode == KVQuantMode.FP8_PER_TENSOR
                    and query.dtype == self.fp8_dtype
                )
                else None
            )
            k_descale = layer._k_scale.expand(descale_shape)
            v_descale = layer._v_scale.expand(descale_shape)
            k_scale_cache = None
            v_scale_cache = None

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table

        seq_threshold_3D = attn_metadata.seq_threshold_3D
        num_par_softmax_segments = attn_metadata.num_par_softmax_segments
        softmax_segm_output = attn_metadata.softmax_segm_output
        softmax_segm_max = attn_metadata.softmax_segm_max
        softmax_segm_expsum = attn_metadata.softmax_segm_expsum

        mm_prefix_range_tensor = attn_metadata.mm_prefix_range_tensor

        tuned_unified_attention(
            q=query[:num_actual_tokens],
            k=key_cache,
            v=value_cache,
            out=output[:num_actual_tokens],
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            alibi_slopes=self.alibi_slopes,
            use_alibi_sqrt=self.use_alibi_sqrt,
            window_size=self.sliding_window,
            block_table=block_table,
            softcap=self.logits_soft_cap,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            seq_threshold_3D=seq_threshold_3D,
            num_par_softmax_segments=num_par_softmax_segments,
            softmax_segm_output=softmax_segm_output,
            softmax_segm_max=softmax_segm_max,
            softmax_segm_expsum=softmax_segm_expsum,
            sinks=self.sinks,
            output_scale=output_scale,
            mm_prefix_range=mm_prefix_range_tensor,
            kv_quant_mode=self._kv_quant_mode,
            k_scale_cache=k_scale_cache,
            v_scale_cache=v_scale_cache,
            chunk_lookback=self.chunk_lookback,
            use_td=self.use_td,
        )

        return output


def tuned_unified_attention(**kwargs):
    q = kwargs["q"]
    k = kwargs["k"]
    supported = (
        q.dtype == torch.bfloat16
        and q.shape[-1] == 128
        and k is not None
        and q.shape[1] == 8 * k.shape[2]
        and kwargs.get("causal", True) is True
        and kwargs.get("window_size", (-1, -1)) == (-1, -1)
        and kwargs.get("softcap", 0) == 0
        and kwargs.get("sinks") is None
        and kwargs.get("alibi_slopes") is None
        and kwargs.get("qq_bias") is None
        and kwargs.get("mm_prefix_range") is None
        and kwargs.get("output_scale") is None
        and not kwargs.get("use_td", False)
        and kwargs.get("chunk_lookback", -1) == -1
        and kwargs.get("kv_quant_mode", 0) == 0
    )
    if not supported:
        return original_unified_attention(**kwargs)
    from . import _unified

    kwargs["prefill_block_m"] = 128
    maxq = kwargs["max_seqlen_q"]
    if maxq > 1:
        kwargs.update(prefill_tile_size=64, prefill_num_warps=8, prefill_num_stages=1)
    elif maxq == 1:
        kwargs.update(decode_tile_size=16, decode_num_warps=2, decode_num_stages=1)
        lengths = kwargs.get("seqused_k")
        threshold = kwargs.get("seq_threshold_3D")
        direct = (
            lengths is not None
            and threshold is not None
            and len(lengths) <= threshold
            and not _unified.is_batch_invariant
            and all(
                kwargs.get(n) is not None
                for n in [
                    "num_par_softmax_segments",
                    "softmax_segm_output",
                    "softmax_segm_max",
                    "softmax_segm_expsum",
                ]
            )
        )
        if direct:
            from . import _decode

            kwargs.update(decode_row_mode="bm8", defer_lsum=True)
            return _decode.unified_attention(**kwargs)
    return _unified.unified_attention(**kwargs)
