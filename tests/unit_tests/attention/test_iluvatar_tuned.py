# SPDX-License-Identifier: Apache-2.0
"""Small on-device comparison with the original vLLM Attention implementation."""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.triton_attn import (
    AttentionType,
    KVQuantMode,
    TritonAttentionImpl,
)

from vllm_fl.attention.iluvatar.impl import TunedTritonAttentionImpl

pytestmark = pytest.mark.skipif(
    getattr(current_platform, "vendor_name", None) != "iluvatar",
    reason="the tuned implementation targets Iluvatar",
)


@pytest.mark.parametrize("query_lengths", [(1, 1), (3, 5)])
@torch.inference_mode()
def test_tuned_attention_matches_original(query_lengths):
    torch.manual_seed(7302)
    tokens = sum(query_lengths)
    q = torch.randn(tokens, 32, 128, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn(12, 2, 16, 4, 128, device="cuda", dtype=torch.bfloat16)
    state = SimpleNamespace(
        attn_type=AttentionType.DECODER,
        _is_per_token_head_quant=False,
        kv_cache_dtype="auto",
        fp8_dtype=torch.float8_e4m3fn,
        _kv_quant_mode=KVQuantMode.NONE,
        scale=128**-0.5,
        alibi_slopes=None,
        use_alibi_sqrt=False,
        sliding_window=(-1, -1),
        logits_soft_cap=0,
        sinks=None,
        chunk_lookback=-1,
        use_td=False,
    )
    layer = SimpleNamespace(
        _q_scale=torch.ones(1, device="cuda"),
        _k_scale=torch.ones(1, device="cuda"),
        _v_scale=torch.ones(1, device="cuda"),
    )
    metadata = SimpleNamespace(
        use_cascade=False,
        num_actual_tokens=tokens,
        query_start_loc=torch.tensor(
            [0, query_lengths[0], tokens], device="cuda", dtype=torch.int32
        ),
        seq_lens=torch.tensor([64, 97], device="cuda", dtype=torch.int32),
        max_query_len=max(query_lengths),
        max_seq_len=97,
        block_table=torch.tensor(
            [[0, 1, 2, 3, 0, 0, 0], [4, 5, 6, 7, 8, 9, 10]],
            device="cuda",
            dtype=torch.int32,
        ),
        seq_threshold_3D=64,
        num_par_softmax_segments=16,
        softmax_segm_output=torch.empty(64, 32, 16, 128, device="cuda"),
        softmax_segm_max=torch.empty(64, 32, 16, device="cuda"),
        softmax_segm_expsum=torch.empty(64, 32, 16, device="cuda"),
        mm_prefix_range_tensor=None,
        causal=True,
    )
    reference = torch.empty_like(q)
    actual = torch.empty_like(q)
    # forward reads the paged cache; cache writes belong to the unchanged
    # do_kv_cache_update entry and are not performed by either call here.
    TritonAttentionImpl.forward(state, layer, q, q, q, cache, metadata, reference)
    TunedTritonAttentionImpl.forward(state, layer, q, q, q, cache, metadata, actual)
    torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.02)
