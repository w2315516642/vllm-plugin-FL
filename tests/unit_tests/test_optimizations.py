# SPDX-License-Identifier: Apache-2.0
"Essential framework/device regressions: selected rows, fusion and both Attention platforms."

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch._inductor import pattern_matcher as pm
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm.compilation.passes.inductor_pass import InductorPass, pass_context
from vllm.config import CompilationMode, CUDAGraphMode
from vllm.config.utils import Range
from vllm.platforms import current_platform
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.triton_attn import (
    AttentionType,
    KVQuantMode,
)

from vllm_fl.attention.iluvatar.impl import TunedTritonAttentionImpl
from vllm_fl.compilation.linear_swiglu import (
    LinearSwiGLUPass,
    _pattern,
    configure_linear_swiglu_fusion,
)
from vllm_fl.compilation.scaled_int8_swiglu import (
    _pattern as _int8_pattern,
    _pattern_interleaved,
)
from vllm_fl.quantization.w8a8.ffn import (
    QuantizedTokens,
    W8A8GateUp,
    _norm_quant,
    _scaled_mm,
    install_ffn_w8a8,
)
from vllm_fl.worker.prefill_tail import (
    LlamaPrefillTail,
    can_select_output_rows,
    make_prefill_tail,
)


class Norm(nn.Module):
    def forward(self, x, residual=None):
        if residual is not None:
            x = x + residual
        y = (
            x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        ).to(x.dtype)
        return y if residual is None else (y, x)


class CausalAttention(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        self.register_buffer("keys", torch.zeros(128, width))
        self.register_buffer("values", torch.zeros(128, width))
        self.rows_seen = []

    def forward(self, positions, hidden_states):
        self.rows_seen.append(hidden_states.shape[0])
        q, k, v = self.qkv(hidden_states).chunk(3, dim=-1)
        self.keys[positions] = k
        self.values[positions] = v
        size = int(positions.max()) + 1
        scores = q.float() @ self.keys[:size].float().T / q.shape[-1] ** 0.5
        mask = torch.arange(size, device=q.device)[None, :] > positions[:, None]
        scores.masked_fill_(mask, -float("inf"))
        out = scores.softmax(-1) @ self.values[:size].float()
        return self.output(out.to(hidden_states.dtype))


class MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.up = nn.Linear(width, width * 2, bias=False)
        self.down = nn.Linear(width, width, bias=False)
        self.rows_seen = []

    def forward(self, x):
        self.rows_seen.append(x.shape[0])
        gate, up = self.up(x).chunk(2, -1)
        return self.down(torch.nn.functional.silu(gate) * up)


class Layer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.input_layernorm = Norm()
        self.post_attention_layernorm = Norm()
        self.self_attn = CausalAttention(width)
        self.mlp = MLP(width)

    def forward(self, positions, x, residual):
        if residual is None:
            residual, x = x, self.input_layernorm(x)
        else:
            x, residual = self.input_layernorm(x, residual)
        x = self.self_attn(positions, x)
        x, residual = self.post_attention_layernorm(x, residual)
        return self.mlp(x), residual


class Backbone(nn.Module):
    def __init__(self, layers=3, width=32):
        super().__init__()
        self.embed_tokens = nn.Embedding(128, width)
        self.layers = nn.ModuleList([Layer(width) for _ in range(layers)])
        self.norm = Norm()

    def forward(self, ids, positions):
        x, residual = self.embed_tokens(ids), None
        for layer in self.layers:
            x, residual = layer(positions, x, residual)
        return self.norm(x, residual)[0]


def eager_config():
    return SimpleNamespace(
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE)
    )


@pytest.mark.parametrize("layers", [3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("chunks", [[17], [7, 10]])
@torch.inference_mode()
def test_selected_rows_preserve_all_kv_and_next_decode(layers, dtype, chunks):
    torch.manual_seed(17)
    a = Backbone(layers=layers).to(dtype=dtype)
    b = deepcopy(a)
    config = eager_config()
    original_options = vars(config.compilation_config).copy()
    adapter = LlamaPrefillTail(b, config)
    assert vars(config.compilation_config) == original_options
    assert adapter.prefix.layers is b.layers
    assert adapter.prefix.embed_tokens is b.embed_tokens
    assert adapter.last_layer is b.layers[-1]
    assert adapter.tail.layer is b.layers[-1]
    assert adapter.tail.norm is b.norm
    assert adapter.tail.compilation_config.cudagraph_mode == CUDAGraphMode.NONE
    ids = torch.randint(0, 128, (sum(chunks),))
    offset = 0
    for size in chunks:
        pos = torch.arange(offset, offset + size)
        # Out-of-order and repeated rows must preserve the consumer's mapping.
        rows = torch.tensor([size - 1, 0, size - 1])
        expected = a(ids[offset : offset + size], pos)[rows]
        actual = adapter(ids[offset : offset + size], pos, rows)
        torch.testing.assert_close(
            actual,
            expected,
            rtol=0.02 if dtype == torch.bfloat16 else 1e-5,
            atol=0.02 if dtype == torch.bfloat16 else 1e-6,
        )
        for la, lb in zip(a.layers, b.layers):
            torch.testing.assert_close(
                la.self_attn.keys, lb.self_attn.keys, rtol=0, atol=0
            )
            torch.testing.assert_close(
                la.self_attn.values, lb.self_attn.values, rtol=0, atol=0
            )
            assert lb.self_attn.rows_seen[-1] == size
        assert b.layers[-1].mlp.rows_seen[-1] == len(rows)
        assert all(layer.mlp.rows_seen[-1] == size for layer in b.layers[:-1])
        offset += size
    # The next decode uses the original model and every historical KV entry.
    next_id, next_pos = torch.tensor([3]), torch.tensor([offset])
    torch.testing.assert_close(
        a(next_id, next_pos), b(next_id, next_pos), rtol=0, atol=0
    )


def policy(**overrides):
    args = dict(
        num_tokens=17,
        output_rows=torch.tensor([3, 16]),
        cudagraph_mode=CUDAGraphMode.PIECEWISE,
        has_prompt_logprobs=False,
        has_aux_hidden_states=False,
        has_intermediate_tensors=False,
        has_inputs_embeds=False,
        has_model_kwargs=False,
        should_ubatch=False,
    )
    args.update(overrides)
    return can_select_output_rows(**args)


@pytest.mark.parametrize(
    "flag",
    [
        "has_prompt_logprobs",
        "has_aux_hidden_states",
        "has_intermediate_tensors",
        "has_inputs_embeds",
        "has_model_kwargs",
        "should_ubatch",
    ],
)
def test_full_state_consumers_fall_back(flag):
    assert not policy(**{flag: True})


def test_decode_full_graph_and_invalid_row_contract_fall_back():
    assert not policy(num_tokens=2)
    assert not policy(cudagraph_mode=CUDAGraphMode.FULL)
    assert not policy(output_rows=torch.empty(0, dtype=torch.int64))
    assert not policy(output_rows=torch.ones(2, 1, dtype=torch.int64))
    assert not policy(output_rows=torch.tensor([0.0, 1.0]))
    assert policy(cudagraph_mode=CUDAGraphMode.NONE)
    assert policy()


def test_unknown_model_is_not_modified():
    model = nn.Linear(4, 4)
    config = SimpleNamespace(model_config=None, parallel_config=None)
    before = dict(model.named_parameters())
    assert make_prefill_tail(model, config) is None
    assert dict(model.named_parameters()) == before


def config(option=None):
    return SimpleNamespace(
        additional_config={} if option is None else {"linear_swiglu_fusion": option},
        model_config=SimpleNamespace(dtype=torch.bfloat16, enforce_eager=False),
        compilation_config=SimpleNamespace(
            mode=CompilationMode.VLLM_COMPILE,
            inductor_compile_config={},
            compile_ranges_endpoints=[512],
        ),
        lora_config=None,
        quant_config=None,
        speculative_config=None,
    )


class Previous(InductorPass):
    def __init__(self):
        self.calls = 0

    def __call__(self, graph):
        self.calls += 1


def trace(fn, inputs):
    # All graph metadata must share the compiler's FakeTensorMode.
    with FakeTensorMode() as mode:
        return pm.fwd_only(fn, [mode.from_tensor(x) for x in inputs])


def test_disabled_and_unsupported_do_not_mutate():
    for cfg, vendor in [(config(), "metax"), (config(True), "nvidia")]:
        assert not configure_linear_swiglu_fusion(cfg, vendor)
        assert cfg.compilation_config.inductor_compile_config == {}
        assert cfg.compilation_config.compile_ranges_endpoints == [512]


def test_configuration_and_cache_identity(monkeypatch):
    from vllm_fl import utils
    from vllm_fl.dispatch.backends.flaggems.flaggems import FlagGemsBackend

    monkeypatch.setattr(
        FlagGemsBackend, "linear_swiglu_is_available", lambda self: True
    )
    monkeypatch.setattr(utils, "is_oot_enabled", lambda: True)
    monkeypatch.setattr(utils, "use_flaggems_op", lambda name: True)
    cfg = config({"max_tokens": 96})
    previous = Previous()
    cfg.compilation_config.inductor_compile_config["post_grad_custom_post_pass"] = (
        previous
    )
    assert configure_linear_swiglu_fusion(cfg, "metax")
    assert cfg.compilation_config.compile_ranges_endpoints == [96, 512]
    p = cfg.compilation_config.inductor_compile_config["post_grad_custom_post_pass"]
    assert p.previous is previous
    assert configure_linear_swiglu_fusion(cfg, "metax")
    assert (
        p
        is cfg.compilation_config.inductor_compile_config["post_grad_custom_post_pass"]
    )
    assert p.uuid() != LinearSwiGLUPass(95, previous).uuid()
    assert p.uuid() != LinearSwiGLUPass(96).uuid()

    default = config({})
    assert configure_linear_swiglu_fusion(default, "metax")
    assert default.compilation_config.compile_ranges_endpoints == [64, 512]


@pytest.mark.parametrize("kind", ["eager", "fp32", "lora", "quant", "speculative"])
def test_unsupported_configuration_is_unchanged(kind):
    cfg = config(True)
    if kind == "eager":
        cfg.model_config.enforce_eager = True
    elif kind == "fp32":
        cfg.model_config.dtype = torch.float32
    else:
        setattr(cfg, kind + "_config", object())
    assert not configure_linear_swiglu_fusion(cfg, "metax")
    assert cfg.compilation_config.inductor_compile_config == {}
    assert cfg.compilation_config.compile_ranges_endpoints == [512]


@pytest.mark.parametrize("m,n,k", [(33, 35, 67)])
@pytest.mark.gpu
@pytest.mark.skipif(
    getattr(current_platform, "vendor_name", None) != "metax",
    reason="the linear/SwiGLU framework boundary is registered only for MetaX",
)
def test_match_has_no_model_or_exact_m_signature(m, n, k):
    torch.manual_seed(31)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((2 * n, k), device="cuda", dtype=torch.bfloat16) / 8
    gm = trace(_pattern, [x, w])
    p = LinearSwiGLUPass(96)
    with pass_context(Range(1, 96)):
        p(gm.graph)
    assert p.matched_count == 1
    gm.graph.lint()
    gm.recompile()
    projection = F.linear(x, w).float()
    a, b = projection.chunk(2, dim=-1)
    expected = (F.silu(a) * b).bfloat16()
    actual = gm(x, w)
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)


@pytest.mark.gpu
def test_large_range_preserves_graph_and_previous_pass():
    x = torch.empty((129, 32), device="cuda", dtype=torch.bfloat16)
    w = torch.empty((96, 32), device="cuda", dtype=torch.bfloat16)
    gm = trace(_pattern, [x, w])
    before = str(gm.graph)
    previous = Previous()
    p = LinearSwiGLUPass(96, previous)
    with pass_context(Range(97, 2048)):
        p(gm.graph)
    assert previous.calls == 1 and p.matched_count == 0
    assert str(gm.graph) == before


@pytest.mark.parametrize("kind", ["bias", "extra_consumer", "fp32", "noncontiguous"])
@pytest.mark.gpu
def test_unsupported_graph_remains_unfused(kind):
    def fn(x, w, bias):
        y = F.linear(x, w, bias if kind == "bias" else None)
        n = w.shape[0] // 2
        out = F.silu(y[..., :n]) * y[..., n:]
        return (out, y) if kind == "extra_consumer" else out

    dtype = torch.float32 if kind == "fp32" else torch.bfloat16
    x = torch.randn((7, 32), device="cuda", dtype=dtype)
    w = torch.randn((96, 32), device="cuda", dtype=dtype)
    if kind == "noncontiguous":
        x = torch.randn((7, 64), device="cuda", dtype=dtype)[:, ::2]
    bias = torch.randn((96,), device="cuda", dtype=dtype)
    gm = trace(fn, [x, w, bias])
    before = str(gm.graph)
    p = LinearSwiGLUPass(96)
    with pass_context(Range(1, 96)):
        p(gm.graph)
    assert p.matched_count == 0
    assert str(gm.graph) == before


@pytest.fixture
def builder(monkeypatch, fa):
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
def test_unsupported_does_not_allocate(builder, monkeypatch, kind, fa):
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


def test_forward_reuses_one_scan_and_fallback_matches(builder, monkeypatch, fa):
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
@pytest.mark.parametrize(
    "queries,lengths",
    [([256], [300]), ([1, 256], [97, 300]), ([1, 16, 64], [97, 128, 160])],
)
def test_real_attention_output_matches_fallback(builder, queries, lengths, fa):
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU kernel integration")
@torch.inference_mode()
def test_quantized_projection_integer_reference_and_compile():
    torch.manual_seed(23)
    x = torch.randn(7, 256, device="cuda", dtype=torch.bfloat16)
    norm_w = torch.randn(256, device=x.device, dtype=x.dtype)
    residual = torch.randn_like(x)
    linear = W8A8GateUp(
        SimpleNamespace(weight=torch.randn(512, 256, device=x.device, dtype=x.dtype))
    )
    q, scales, r_out = _norm_quant(x, norm_w, 1e-6, residual)
    out, bias = linear(QuantizedTokens(q, scales))
    expected = (q.cpu().long() @ linear.weight.cpu().long()).float()
    expected = (expected * scales.cpu() * linear.weight_scale.cpu()).to(x.dtype)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
    assert bias is None
    torch.testing.assert_close(
        r_out, (x.float() + residual.float()).to(x.dtype), rtol=0, atol=0
    )

    def forward(a, r):
        qi, si, ri = _norm_quant(a, norm_w, 1e-6, r)
        y, _ = linear(QuantizedTokens(qi, si))
        return y, ri

    compiled = torch.compile(forward, backend="inductor", fullgraph=True, dynamic=True)
    for m in (1, 7, 17):
        a = torch.randn(m, 256, device=x.device, dtype=x.dtype)
        r = torch.randn_like(a)
        torch.testing.assert_close(compiled(a, r), forward(a, r), rtol=0, atol=0)
    torch.library.opcheck(_norm_quant, (x, norm_w, 1e-6, residual))
    torch.library.opcheck(
        _scaled_mm, (q, linear.weight, scales, linear.weight_scale, x.dtype)
    )


def supported_config():
    from vllm.config import CompilationMode

    return SimpleNamespace(
        additional_config={"ffn_w8a8": True},
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            runner_type="generate",
            is_encoder_decoder=False,
            multimodal_config=None,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            use_ubatching=False,
        ),
        quant_config=None,
        lora_config=None,
        speculative_config=None,
        kv_transfer_config=None,
        offload_config=SimpleNamespace(
            uva=SimpleNamespace(cpu_offload_gb=0),
            prefetch=SimpleNamespace(offload_group_size=0),
        ),
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
    )


@pytest.mark.parametrize(
    "field", ["quant_config", "lora_config", "speculative_config", "kv_transfer_config"]
)
def test_unsupported_configuration_keeps_model(field):
    cfg = supported_config()
    setattr(cfg, field, object())
    model = torch.nn.Linear(3, 4)
    weight = model.weight
    assert install_ffn_w8a8(model, cfg, "metax") == 0
    assert model.weight is weight


@pytest.mark.skipif(
    getattr(current_platform, "vendor_name", None) != "iluvatar",
    reason="Iluvatar kernel",
)
@pytest.mark.parametrize("query_lengths", [(1, 1), (3, 5)])
@pytest.mark.parametrize("heads,kv_heads", [(16, 2), (32, 4)])
@torch.inference_mode()
def test_tuned_attention_matches_cpu_reference(query_lengths, heads, kv_heads):
    torch.manual_seed(7302)
    tokens = sum(query_lengths)
    q = torch.randn(tokens, heads, 128, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn(12, 2, 16, kv_heads, 128, device="cuda", dtype=torch.bfloat16)
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
        softmax_segm_output=torch.empty(64, heads, 16, 128, device="cuda"),
        softmax_segm_max=torch.empty(64, heads, 16, device="cuda"),
        softmax_segm_expsum=torch.empty(64, heads, 16, device="cuda"),
        mm_prefix_range_tensor=None,
        causal=True,
    )
    actual = torch.empty_like(q)
    # forward reads the paged cache; cache writes belong to the unchanged
    # do_kv_cache_update entry and are not performed by either call here.
    TunedTritonAttentionImpl.forward(state, layer, q, q, q, cache, metadata, actual)
    # An independent FP32 CPU reference also covers upstream/runtime errors;
    # comparing only two device implementations can accept the same defect.
    query, pages = q.cpu().float(), cache.cpu().float()
    references, offset = [], 0
    for i, (qlen, klen) in enumerate(zip(query_lengths, [64, 97])):
        blocks = metadata.block_table[i, : (klen + 15) // 16].cpu().long()
        k = pages[blocks, 0].reshape(-1, kv_heads, 128)[:klen]
        v = pages[blocks, 1].reshape(-1, kv_heads, 128)[:klen]
        k, v = (x.repeat_interleave(heads // kv_heads, dim=1) for x in (k, v))
        scores = torch.einsum("qhd,khd->qhk", query[offset : offset + qlen], k)
        scores *= 128**-0.5
        mask = (
            torch.arange(klen)[None, None, :]
            > torch.arange(klen - qlen, klen)[:, None, None]
        )
        scores.masked_fill_(mask, -torch.inf)
        references.append(torch.einsum("qhk,khd->qhd", scores.softmax(-1), v))
        offset += qlen
    reference = torch.cat(references).to(actual.dtype)
    torch.testing.assert_close(actual.cpu(), reference, rtol=0.02, atol=0.02)


@pytest.fixture
def fa():
    if getattr(current_platform, "vendor_name", None) != "metax":
        pytest.skip("MetaX metadata backend")
    from vllm_fl.dispatch.backends.vendor.metax.impl.attention import flash_attn

    return flash_attn


@pytest.mark.skipif(
    getattr(current_platform, "vendor_name", None) != "metax",
    reason="MetaX INT8 fusion",
)
@pytest.mark.parametrize("interleaved", [False, True])
@torch.inference_mode()
def test_int8_pattern_fuses_both_weight_layouts(interleaved):
    torch.manual_seed(37)
    m, k, n = 33, 67, 35
    q = torch.randint(-127, 128, (m, k), device="cuda", dtype=torch.int8)
    w = torch.randint(-127, 128, (2 * n, k), device="cuda", dtype=torch.int8).t()
    sx = torch.full((m, 1), 0.001, device="cuda")
    sw = torch.full((2 * n,), 0.001, device="cuda")
    fn = _pattern_interleaved if interleaved else _int8_pattern
    inputs = [q, w, sx, sw]
    graph = trace(fn, inputs)
    fusion = LinearSwiGLUPass(64, enable_int8=True, int8_min_tokens=1)
    with pass_context(Range(1, 4096)):
        fusion(graph.graph)
    assert fusion.int8_matched_count == 1
    graph.graph.lint()
    graph.recompile()
    projected = (
        ((q.cpu().long() @ w.cpu().long()).float() * sx.cpu() * sw.cpu())
        .bfloat16()
        .float()
    )
    a, b = (
        (projected[:, 0::2], projected[:, 1::2])
        if interleaved
        else projected.chunk(2, -1)
    )
    expected = (F.silu(a) * b).bfloat16()
    torch.testing.assert_close(graph(*inputs).cpu(), expected, rtol=0.03, atol=0.03)


@pytest.mark.skipif(
    getattr(current_platform, "vendor_name", None) != "iluvatar",
    reason="Iluvatar Quest",
)
@torch.inference_mode()
def test_quest_selected_pages_and_graph_replay(monkeypatch):
    from vllm_fl.attention.iluvatar import _quest as quest

    torch.manual_seed(150)
    n, length = 2, 4608
    pages = (length + 15) // 16
    cache = torch.randn((n * pages, 2, 16, 2, 128), device="cuda", dtype=torch.bfloat16)
    table = (
        torch.randperm(n * pages, device="cuda", dtype=torch.int64)
        .reshape(n, pages)
        .int()
    )
    lens = torch.tensor([length, 4096], device="cuda", dtype=torch.int32)
    q = torch.randn((n, 16, 128), device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(q)
    state = quest.QuestState(q.device, n, length)
    low, high = state.pool("test", cache)
    low.copy_(cache[:, 0].amin(1))
    high.copy_(cache[:, 0].amax(1))
    monkeypatch.setattr(
        quest,
        "get_forward_context",
        lambda: SimpleNamespace(
            _quest_score_tiles=state.workspace["scores"].shape[-1] // 16
        ),
    )
    layer = SimpleNamespace(layer_name="test")
    meta = SimpleNamespace(seq_lens=lens, block_table=table)
    for _ in range(2):
        state.run(layer, q, cache, meta, out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        state.run(layer, q, cache, meta, out)
    for lengths in [[4608, 4096], [4097, 17]]:
        lens.copy_(torch.tensor(lengths, device=q.device, dtype=torch.int32))
        q.normal_()
        graph.replay()
        ids = state.workspace["ids"].cpu().long()
        values, query = cache.cpu().float(), q.cpu().float()
        expected = torch.empty_like(query)
        for row, size in enumerate(lengths):
            count = min((size + 15) // 16, 256)
            for head in range(2):
                selected = ids[row, head, :count]
                positions = torch.arange(count * 16)
                # The final selected page is always the last logical page.
                valid = positions < (count - 1) * 16 + ((size - 1) % 16 + 1)
                key = values[selected, 0, :, head].reshape(-1, 128)[valid]
                value = values[selected, 1, :, head].reshape(-1, 128)[valid]
                scores = query[row, head * 8 : (head + 1) * 8] @ key.T * (128**-0.5)
                expected[row, head * 8 : (head + 1) * 8] = scores.softmax(-1) @ value
        torch.testing.assert_close(out.cpu(), expected.bfloat16(), rtol=0.02, atol=0.02)


@pytest.mark.skipif(
    getattr(current_platform, "vendor_name", None) != "iluvatar",
    reason="Iluvatar RoPE/KV summary",
)
@torch.inference_mode()
def test_rope_summary_preserves_rotated_keys_cache_and_complete_pages():
    from vllm_fl.attention.iluvatar._rope_kernels import launch_rope, launch_summary

    torch.manual_seed(151)
    q = torch.randn((6, 16, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((6, 2, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    positions = torch.tensor([14, 15, 0, 1, 14, 15], device="cuda", dtype=torch.int64)
    slots = torch.tensor([14, 15, 16, 17, 46, 47], device="cuda", dtype=torch.int64)
    cs = torch.randn((64, 128), device="cuda", dtype=torch.bfloat16)
    cache = torch.randn((4, 2, 16, 2, 128), device="cuda", dtype=torch.bfloat16)
    original = cache.clone()
    low = torch.full((4, 2, 128), -7, device="cuda", dtype=torch.bfloat16)
    high = torch.full_like(low, 7)
    qo, ko, expected_q, expected_k = (
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(q),
        torch.empty_like(k),
    )
    launch_rope(
        q,
        k,
        v,
        positions,
        cs,
        expected_q,
        expected_k,
        original,
        slots,
        fused=True,
        block=512,
        warps=4,
    )
    launch_summary(q, k, v, positions, cs, qo, ko, cache, slots, low, high)
    for actual, expected in [(qo, expected_q), (ko, expected_k), (cache, original)]:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for page in [0, 2]:
        torch.testing.assert_close(low[page], original[page, 0].amin(0), rtol=0, atol=0)
        torch.testing.assert_close(
            high[page], original[page, 0].amax(0), rtol=0, atol=0
        )
    assert torch.all(low[[1, 3]] == -7) and torch.all(high[[1, 3]] == 7)
