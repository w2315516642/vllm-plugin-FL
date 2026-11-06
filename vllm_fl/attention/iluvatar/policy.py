# SPDX-License-Identifier: Apache-2.0
"""Launch policy within the existing Triton Attention backend."""

import torch

from vllm.v1.attention.ops.triton_unified_attention import (
    unified_attention as original_unified_attention,
)


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
