# Copyright (c) 2026 BAAI. All rights reserved.
"""MetaX metadata lifecycle and the unchanged attention consumer contract."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionType, CommonAttentionMetadata

if getattr(current_platform, "vendor_name", None) != "metax":
    pytest.skip("Requires the MetaX attention backend", allow_module_level=True)

from vllm_fl.dispatch.backends.vendor.metax.impl.attention import flash_attn as fa


@pytest.fixture
def builder(monkeypatch):
    # Exercise the real build/split code without loading model weights or a
    # distributed process group. Only scheduling and layer discovery are stubbed.
    b = object.__new__(fa.FlashAttentionMetadataBuilder)
    b.device = torch.device("cpu")
    b.layer_names = ["layer.0", "layer.1"]
    b.vllm_config = SimpleNamespace(speculative_config=None)
    b.compilation_config = SimpleNamespace(
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE
    )
    b.cache_config = SimpleNamespace(cache_dtype="auto")
    b.kv_cache_dtype = torch.bfloat16
    b.dcp_world_size = 1
    b.dcp_rank = 0
    b.cp_kv_cache_interleave_size = 1
    b.aot_schedule = False
    b.aot_sliding_window = (-1, -1)
    b.use_full_cuda_graph = False
    b.max_num_splits = 0
    b._prefill_reuse_decoder_group = None
    monkeypatch.setattr(
        fa,
        "get_layers_from_vllm_config",
        lambda *args: {
            name: SimpleNamespace(attn_type=AttentionType.DECODER)
            for name in b.layer_names
        },
    )
    return b


def common(query_lens, seq_lens, device="cpu"):
    starts = torch.tensor([0, *query_lens], dtype=torch.int32).cumsum(0).int()
    seq_cpu = torch.tensor(seq_lens, dtype=torch.int32)
    return CommonAttentionMetadata(
        query_start_loc=starts.to(device),
        query_start_loc_cpu=starts,
        seq_lens=seq_cpu.to(device),
        num_reqs=len(seq_lens),
        num_actual_tokens=sum(query_lens),
        max_query_len=max(query_lens),
        max_seq_len=max(seq_lens),
        block_table_tensor=torch.zeros(
            (len(seq_lens), 1), dtype=torch.int32, device=device
        ),
        slot_mapping=torch.zeros(sum(query_lens), dtype=torch.int64, device=device),
        _seq_lens_cpu=seq_cpu,
    )


@pytest.mark.parametrize(
    "queries,lengths,expected",
    [
        ([256], [256], [0, 256]),
        ([256, 512], [1024, 2048], [0, 1024, 3072]),
        ([1, 256, 512], [99, 4096, 8192], [0, 4096, 12288]),
        ([1, 16, 64], [100, 200, 300], [0, 200, 500]),
        ([1, 1, 0], [100, 200, 0], None),
        ([64, 64], [1024, 2048], None),
    ],
)
def test_build_full_kv_lengths_and_split(builder, queries, lengths, expected):
    metadata = builder.build(0, common(queries, lengths))
    if expected is None:
        assert metadata.prefill_cu_seq_lens is None
    else:
        torch.testing.assert_close(
            metadata.prefill_cu_seq_lens, torch.tensor(expected, dtype=torch.int32)
        )
        assert metadata.prefill_cu_seq_lens.is_contiguous()


@pytest.mark.parametrize(
    "kind",
    [
        "decode",
        "cascade",
        "dcp",
        "speculative",
        "drafting",
        "full_graph",
        "encoder",
        "cross",
        "unknown",
        "empty_group",
    ],
)
def test_unsupported_does_not_allocate(builder, monkeypatch, kind):
    seq = torch.tensor([512, 2048], dtype=torch.int32)
    prefix, fast = 0, False
    if kind == "decode":
        seq = None
    elif kind == "cascade":
        prefix = 128
    elif kind == "dcp":
        builder.dcp_world_size = 2
    elif kind == "speculative":
        builder.vllm_config.speculative_config = object()
    elif kind == "drafting":
        fast = True
    elif kind == "full_graph":
        builder.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
    elif kind in ("encoder", "cross"):
        attn_type = (
            AttentionType.ENCODER
            if kind == "encoder"
            else AttentionType.ENCODER_DECODER
        )
        monkeypatch.setattr(
            fa,
            "get_layers_from_vllm_config",
            lambda *args: {
                n: SimpleNamespace(attn_type=attn_type) for n in builder.layer_names
            },
        )
    elif kind == "unknown":
        monkeypatch.setattr(fa, "get_layers_from_vllm_config", lambda *args: {})
    else:
        builder.layer_names = []

    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported path allocated/scanned metadata")

    monkeypatch.setattr(torch, "empty", forbidden)
    monkeypatch.setattr(torch, "cumsum", forbidden)
    assert builder._build_prefill_cu_seq_lens(seq, prefix, fast) is None


def test_snapshot_survives_reused_input_and_interleaved_group(builder):
    cm = common([256, 256], [1024, 4096])
    first = builder.build(0, cm)
    cm.seq_lens.copy_(torch.tensor([2048, 8192], dtype=torch.int32))
    second = builder.build(0, cm)
    other = builder.build(0, common([512], [16384]))
    assert first.prefill_cu_seq_lens.data_ptr() != second.prefill_cu_seq_lens.data_ptr()
    assert second.prefill_cu_seq_lens.data_ptr() != other.prefill_cu_seq_lens.data_ptr()
    assert first.prefill_cu_seq_lens.tolist() == [0, 1024, 5120]
    assert second.prefill_cu_seq_lens.tolist() == [0, 2048, 10240]
    assert other.prefill_cu_seq_lens.tolist() == [0, 16384]


def test_layer_discovery_retries_until_available(builder, monkeypatch):
    layers = {}
    monkeypatch.setattr(fa, "get_layers_from_vllm_config", lambda *args: layers)
    seq = torch.tensor([256], dtype=torch.int32)
    assert builder._build_prefill_cu_seq_lens(seq, 0, False) is None
    layers.update(
        {
            n: SimpleNamespace(attn_type=AttentionType.DECODER)
            for n in builder.layer_names
        }
    )
    assert builder._build_prefill_cu_seq_lens(seq, 0, False).tolist() == [0, 256]


def test_forward_reuses_one_scan_and_fallback_matches(builder, monkeypatch):
    scans = []
    original = torch.cumsum

    def count(*args, **kwargs):
        scans.append(1)
        return original(*args, **kwargs)

    cm = common([1, 256], [1024, 4096])
    monkeypatch.setattr(torch, "cumsum", count)
    metadata = builder.build(0, cm)
    assert len(scans) == 1
    impl = object.__new__(fa.FlashAttentionImpl)
    for key, value in dict(
        attn_type=AttentionType.DECODER,
        kv_sharing_target_layer_name="previous",
        kv_cache_dtype="auto",
        dcp_world_size=1,
        num_kv_heads=1,
        scale=1.0,
        sliding_window=(-1, -1),
        alibi_slopes=None,
        logits_soft_cap=0.0,
        sinks=None,
    ).items():
        setattr(impl, key, value)
    seen = []

    def prefill(**kwargs):
        seen.append(kwargs["cu_seqlens_k"])
        return kwargs["q"].clone()

    monkeypatch.setattr(fa, "flash_attn_varlen_func", prefill)
    monkeypatch.setattr(
        fa, "flash_attn_with_kvcache", lambda **kwargs: kwargs["q"].clone()
    )
    q = torch.randn(257, 1, 8)
    cache = torch.zeros(2, 1, 16, 1, 8)
    scale = torch.ones(1)
    layer = SimpleNamespace(_q_scale=scale, _k_scale=scale, _v_scale=scale)
    for _ in range(3):
        out = torch.empty_like(q)
        impl.forward(layer, q, q, q, cache, metadata, output=out)
        torch.testing.assert_close(out, q)
    assert len(scans) == 1
    assert all(x is metadata.prefill_cu_seq_lens for x in seen)
    fallback = replace(metadata, prefill_cu_seq_lens=None)
    impl.forward(layer, q, q, q, cache, fallback, output=out)
    torch.testing.assert_close(seen[-1], metadata.prefill_cu_seq_lens)
    torch.testing.assert_close(out, q)


@pytest.mark.gpu
@pytest.mark.parametrize("n", [1, 3, 32, 64, 129])
def test_device_scan_matches_old_construction(builder, n):
    # Run under FlagGems dispatch on MetaX as well as native torch.
    import flag_gems

    torch.manual_seed(8)
    lengths = torch.randint(1, 131073, (n,), dtype=torch.int32, device="cuda")
    expected = torch.tensor(
        [0] + lengths.tolist(), dtype=torch.int32, device="cuda"
    ).cumsum(0, dtype=torch.int32)
    with flag_gems.use_gems():
        actual = builder._build_prefill_cu_seq_lens(lengths, 0, False)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    lengths.add_(1)
    with flag_gems.use_gems():
        updated = builder._build_prefill_cu_seq_lens(lengths, 0, False)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        updated[1:] - actual[1:],
        torch.arange(1, n + 1, device="cuda", dtype=torch.int32),
        rtol=0,
        atol=0,
    )


@pytest.mark.gpu
@pytest.mark.parametrize(
    "queries,lengths",
    [([256], [300]), ([1, 256], [97, 300]), ([1, 16, 64], [97, 128, 160])],
)
def test_real_attention_output_matches_fallback(builder, queries, lengths):
    import flag_gems

    cm = common(queries, lengths, "cuda")
    blocks_per_request = (max(lengths) + 15) // 16
    blocks = blocks_per_request * len(lengths)
    cm.block_table_tensor = torch.arange(
        blocks, device="cuda", dtype=torch.int32
    ).reshape(len(lengths), blocks_per_request)
    with flag_gems.use_gems():
        metadata = builder.build(0, cm)
    assert metadata.prefill_cu_seq_lens is not None
    fallback = replace(metadata, prefill_cu_seq_lens=None)
    impl = object.__new__(fa.FlashAttentionImpl)
    for key, value in dict(
        attn_type=AttentionType.DECODER,
        kv_sharing_target_layer_name="previous",
        kv_cache_dtype="auto",
        dcp_world_size=1,
        num_kv_heads=2,
        scale=128**-0.5,
        sliding_window=(-1, -1),
        alibi_slopes=None,
        logits_soft_cap=0.0,
        sinks=None,
    ).items():
        setattr(impl, key, value)
    torch.manual_seed(8)
    q = torch.randn(sum(queries), 16, 128, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn(2, blocks, 16, 2, 128, device="cuda", dtype=torch.bfloat16)
    scale = torch.ones(1, device="cuda")
    layer = SimpleNamespace(_q_scale=scale, _k_scale=scale, _v_scale=scale)
    a, b = torch.empty_like(q), torch.empty_like(q)
    with flag_gems.use_gems():
        impl.forward(layer, q, None, None, cache, fallback, output=a)
        impl.forward(layer, q, None, None, cache, metadata, output=b)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
