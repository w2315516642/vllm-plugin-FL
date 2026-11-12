# SPDX-License-Identifier: Apache-2.0
"Essential CPU contracts for platform defaults, storage, sampling and attention."

import ast
import gc
import importlib.util
import sys
import types
import weakref
from contextvars import ContextVar
from pathlib import Path

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

ROOT = Path(__file__).resolve().parents[2]


def test_temperature_identity_scope_and_fallback():
    class Parent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.topk_topp_sampler = torch.nn.Identity()

        @staticmethod
        def apply_temperature(logits, temp, all_random):
            return logits / temp[:, None]

        def forward(self, logits, sampling_metadata):
            if getattr(sampling_metadata, "fail", False):
                raise RuntimeError("test cleanup")
            return self.apply_temperature(
                logits, sampling_metadata.temperature, sampling_metadata.all_random
            )

    source = ast.parse((ROOT / "vllm_fl/sampling.py").read_text())
    classes = [n for n in source.body if isinstance(n, ast.ClassDef)]
    env = {"nn": torch.nn, "torch": torch, "Sampler": Parent, "ContextVar": ContextVar}
    exec(
        compile(ast.Module(body=classes, type_ignores=[]), "sampling-scope", "exec"),
        env,
    )
    sampler = env["SamplerFL"]()
    batch = types.SimpleNamespace(num_reqs=2, temperature_cpu_tensor=torch.ones(2))
    meta = types.SimpleNamespace(all_random=True, temperature=torch.ones(2))
    x = torch.randn(2, 16)
    assert sampler.temperatures_are_one(batch, meta)
    assert sampler(x, meta, all_unit_temperature=True) is x
    batch.temperature_cpu_tensor[1] = 0.5
    meta.temperature[1] = 0.5
    assert not sampler.temperatures_are_one(batch, meta)
    torch.testing.assert_close(sampler(x, meta), x / meta.temperature[:, None])
    meta.fail = True
    with pytest.raises(RuntimeError, match="test cleanup"):
        sampler(x, meta, all_unit_temperature=True)
    assert not sampler._unit_temperature.get()
    meta.all_random = False
    assert not sampler.temperatures_are_one(batch, meta)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def config_api():
    return load("_fl_profile_test", ROOT / "vllm_fl/optimizations/config.py")


@pytest.mark.parametrize("vendor", ["metax", "iluvatar", "nvidia"])
def test_defaults_preserve_other_configuration(config_api, vendor):
    extra = {"custom": {"keep": 1}}
    cfg = types.SimpleNamespace(additional_config=extra)
    config_api.prepare_optimizations(cfg, vendor)
    assert extra == {"custom": {"keep": 1}}
    assert cfg.additional_config["custom"] is extra["custom"]
    if vendor == "nvidia":
        assert cfg.additional_config is extra
    else:
        assert cfg.additional_config["ffn_w8a8"] is True
        if vendor == "iluvatar":
            assert cfg.additional_config["ffn_gate_norm_quant"] is True


@pytest.mark.parametrize("vendor", ["metax", "iluvatar"])
def test_default_expansion_can_be_disabled_for_development(config_api, vendor):
    extra = {"optimization_profile": "disabled"}
    cfg = types.SimpleNamespace(additional_config=extra)
    config_api.prepare_optimizations(cfg, vendor)
    assert cfg.additional_config is extra


@pytest.mark.parametrize("vendor", ["metax", "iluvatar"])
def test_profile_is_idempotent_and_keeps_explicit_overrides(config_api, vendor):
    extra = {"optimization_profile": "competition", "qkv_w8a8": False}
    cfg = types.SimpleNamespace(additional_config=extra)
    config_api.prepare_optimizations(cfg, vendor)
    first = dict(cfg.additional_config)
    config_api.prepare_optimizations(cfg, vendor)
    assert cfg.additional_config == first
    assert first["qkv_w8a8"] is False
    assert first["ffn_w8a8"] is True
    assert first["ffn_down_w8a8"] is (vendor == "metax")
    assert first["ffn_gate_norm_quant"] is False
    assert extra == {"optimization_profile": "competition", "qkv_w8a8": False}


def test_unsupported_vendor_does_not_enable_target_features(config_api):
    cfg = types.SimpleNamespace(
        additional_config={"optimization_profile": "competition"}
    )
    config_api.prepare_optimizations(cfg, "nvidia")
    assert cfg.additional_config == {"optimization_profile": "competition"}


def test_quantization_opt_out_does_not_inherit_dependent_layout(config_api):
    cfg = types.SimpleNamespace(
        additional_config={
            "optimization_profile": "competition",
            "ffn_w8a8": False,
        }
    )
    config_api.prepare_optimizations(cfg, "metax")
    assert cfg.additional_config["ffn_w8a8_interleaved"] is False
    assert cfg.additional_config["linear_swiglu_fusion"] is False


@pytest.mark.parametrize("bad", ["unknown"])
def test_profile_rejects_unknown_names(config_api, bad):
    with pytest.raises(ValueError):
        config_api.prepare_optimizations(
            types.SimpleNamespace(additional_config={"optimization_profile": bad}),
            "metax",
        )


@pytest.mark.parametrize("bad", [1, "yes"])
def test_boolean_feature_contract(config_api, bad):
    with pytest.raises(ValueError, match="must be a boolean"):
        config_api.feature_enabled(
            types.SimpleNamespace(additional_config={"tiled_sampling": bad}),
            "tiled_sampling",
        )


@pytest.fixture(scope="module")
def weights():
    # Bypass package startup only; execute the complete actual module including
    # Torch custom-op/fake registrations. No GPU computation is substituted.
    existing = sys.modules.get("vllm_fl.quantization.w8a8.ffn")
    if existing is not None:
        return existing
    base = "_submission_contracts"
    for suffix, rel in [
        ("", ""),
        (".optimizations", "optimizations"),
        (".quantization", "quantization"),
        (".quantization.w8a8", "quantization/w8a8"),
    ]:
        module = types.ModuleType(base + suffix)
        module.__path__ = [str(ROOT / "vllm_fl" / rel)]
        sys.modules[base + suffix] = module
    return load(
        base + ".quantization.w8a8.ffn", ROOT / "vllm_fl/quantization/w8a8/ffn.py"
    )


def test_channel_quantization_layout_zero_channel_and_original_weight(weights):
    w = torch.tensor([[0, 0, 0], [-2, 0, 2], [-20, 10, 20]], dtype=torch.bfloat16)
    before = w.clone()
    q, s = weights.quantize_channelwise_weight(w)
    assert q.dtype == torch.int8 and q.t().is_contiguous()
    assert torch.equal(q[:, 0], torch.zeros(3, dtype=torch.int8))
    torch.testing.assert_close(s, torch.tensor([0, 2 / 127, 20 / 127]))
    assert torch.all(
        (q.t().float() * s[:, None] - w.float()).abs() <= s[:, None] * 0.501
    )
    assert torch.equal(w, before)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_weight_fails_before_conversion(weights, bad):
    with pytest.raises(ValueError, match="finite"):
        weights.quantize_channelwise_weight(torch.tensor([[bad]], dtype=torch.bfloat16))


@pytest.mark.parametrize("interleaved", [False, True])
def test_gate_owns_only_quantized_weight_storage(weights, interleaved):
    original = types.SimpleNamespace(weight=torch.randn(16, 8, dtype=torch.bfloat16))
    old = weakref.ref(original.weight)
    projection = weights.W8A8GateUp(original, interleaved=interleaved)
    del original
    gc.collect()
    assert old() is None
    assert projection.weight.dtype == torch.int8
    assert projection.weight_scale.dtype == torch.float32
    assert projection.output_dtype == torch.bfloat16


def test_interleaving_pairs_channels_and_scales_once(weights):
    w = torch.arange(24, dtype=torch.int8).reshape(6, 4).t()
    s = torch.arange(6, dtype=torch.float32)
    packed, scales = weights.interleave_gate_up(w, s)
    order = [0, 3, 1, 4, 2, 5]
    assert torch.equal(packed, w[:, order])
    assert torch.equal(scales, s[order])
    assert packed.t().is_contiguous()
    assert packed.untyped_storage().data_ptr() != w.untyped_storage().data_ptr()


@pytest.mark.parametrize("m", [0, 1, 32, 33, 128, 129, 4096])
def test_dynamic_projection_fake_contract_has_no_padding_leak(weights, m):
    import importlib

    dynamic_int8_linear = importlib.import_module(
        weights.__package__ + ".projections"
    ).dynamic_int8_linear

    with FakeTensorMode():
        x = torch.empty((m, 2048), device="cuda", dtype=torch.bfloat16)
        w = torch.empty((12288, 2048), device="cuda", dtype=torch.int8).t()
        scale = torch.empty(12288, device="cuda", dtype=torch.float32)
        y = dynamic_int8_linear(x, w, scale, "gate")
        assert y.shape == (m, 12288)
        assert y.dtype == torch.bfloat16 and y.is_contiguous()


@pytest.mark.parametrize("m", [0, 32, 33, 129])
def test_padded_gate_norm_fake_preserves_logical_output_rows(weights, m):
    with FakeTensorMode():
        x = torch.empty((m, 2048), device="cuda", dtype=torch.bfloat16)
        r = torch.empty_like(x)
        nw = torch.empty(2048, device="cuda", dtype=x.dtype)
        q, scale, residual = weights.gate_norm_quant(x, r, nw, 1e-6)
        multiple = 32 if m <= 32 else 128
        capacity = ((m + multiple - 1) // multiple) * multiple
        assert q.shape == (capacity, 2048) and q.dtype == torch.int8
        assert scale.shape == (capacity, 1) and scale.dtype == torch.float32
        assert residual.shape == x.shape and residual.dtype == x.dtype
        w = torch.empty((12288, 2048), device="cuda", dtype=torch.int8).t()
        sw = torch.empty(12288, device="cuda", dtype=torch.float32)
        y = weights.gate_from_padded_int8(x, q, scale, w, sw)
        assert y.shape == (m, 12288) and y.dtype == x.dtype


def test_filter_component_preserves_generator_and_original_return(monkeypatch):
    source = ast.parse((ROOT / "vllm_fl/sampling.py").read_text())
    cls = next(
        n
        for n in source.body
        if isinstance(n, ast.ClassDef) and n.name == "FilteredTopKTopPSampler"
    )
    env = {"nn": torch.nn, "torch": torch}
    exec(
        compile(ast.Module(body=[cls], type_ignores=[]), "sampler-contract", "exec"),
        env,
    )
    calls = []
    module = types.ModuleType("flag_gems")
    module.supports_topk_topp = lambda *args: True
    module.topk_topp_filter = lambda logits, k, p: logits + 1
    monkeypatch.setitem(sys.modules, "flag_gems", module)

    class Original(torch.nn.Module):
        def forward(self, logits, generators, k, p):
            calls.append((logits, generators, k, p))
            return torch.rand(1, generator=generators[0]), logits

    original = Original()
    wrapped = env["FilteredTopKTopPSampler"](original)
    logits = torch.zeros(1, 4)
    generators = {0: torch.Generator().manual_seed(17)}
    expected_generator = torch.Generator().manual_seed(17)
    draw, processed = wrapped(logits, generators, None, torch.tensor([0.95]))
    assert torch.equal(draw, torch.rand(1, generator=expected_generator))
    assert torch.equal(generators[0].get_state(), expected_generator.get_state())
    assert torch.equal(processed, logits + 1)
    assert calls[0][1] is generators and calls[0][2:] == (None, None)

    module.supports_topk_topp = lambda *args: False
    k, p = torch.tensor([2]), torch.tensor([0.8])
    wrapped(logits, generators, k, p)
    assert calls[-1][0] is logits and calls[-1][2] is k and calls[-1][3] is p


@pytest.fixture
def policy(monkeypatch):
    package = types.ModuleType("_attention_policy_contract")
    package.__path__ = []
    unified = types.ModuleType(package.__name__ + "._unified")
    unified.is_batch_invariant = False
    unified.unified_attention = lambda **kw: (
        "decode" if kw.get("direct_decode") else "unified",
        kw,
    )
    for module in [package, unified]:
        monkeypatch.setitem(sys.modules, module.__name__, module)
    source = ast.parse((ROOT / "vllm_fl/attention/iluvatar/impl.py").read_text())
    fn = next(n for n in source.body if isinstance(n, ast.FunctionDef))
    env = {
        "__package__": package.__name__,
        "torch": torch,
        "original_unified_attention": lambda **kw: ("original", kw),
    }
    exec(
        compile(ast.Module(body=[fn], type_ignores=[]), "attention-policy", "exec"), env
    )
    return env[fn.name], unified


def arguments(maxq=1):
    return dict(
        q=torch.empty(2, 32, 128, dtype=torch.bfloat16),
        k=torch.empty(3, 16, 4, 128, dtype=torch.bfloat16),
        max_seqlen_q=maxq,
        seqused_k=torch.tensor([4096, 16384]),
        seq_threshold_3D=64,
        num_par_softmax_segments=16,
        softmax_segm_output=object(),
        softmax_segm_max=object(),
        softmax_segm_expsum=object(),
    )


def test_decode_direct_mapping_keeps_original_metadata_objects(policy):
    fn, _ = policy
    args = arguments()
    path, called = fn(**args)
    assert path == "decode"
    assert called["seqused_k"] is args["seqused_k"]
    assert called["softmax_segm_output"] is args["softmax_segm_output"]
    assert called["decode_row_mode"] == "bm8" and called["defer_lsum"] is True
    assert (
        called["decode_tile_size"],
        called["decode_num_warps"],
        called["decode_num_stages"],
    ) == (16, 2, 1)


@pytest.mark.parametrize("maxq", [2, 16, 128, 2048])
def test_prefill_uses_prefill_tiles(policy, maxq):
    path, called = policy[0](**arguments(maxq))
    assert path == "unified"
    assert called["prefill_block_m"] == 128
    assert (
        called["prefill_tile_size"],
        called["prefill_num_warps"],
        called["prefill_num_stages"],
    ) == (64, 8, 1)
    assert "decode_row_mode" not in called


@pytest.mark.parametrize("reason", ["batch_invariant", "threshold", "missing_segments"])
def test_direct_decode_protects_unsupported_metadata(policy, reason):
    fn, module = policy
    args = arguments()
    if reason == "batch_invariant":
        module.is_batch_invariant = True
    elif reason == "threshold":
        args["seq_threshold_3D"] = 1
    else:
        args["softmax_segm_max"] = None
    path, called = fn(**args)
    assert path == "unified" and "decode_row_mode" not in called


@pytest.mark.parametrize(
    "change",
    [
        {"causal": False},
        {"causal": torch.tensor(True)},
        {"softcap": 1.0},
        {"window_size": (64, 0)},
        {"kv_quant_mode": 1},
        {"use_td": True},
        {"chunk_lookback": 2},
        {"sinks": object()},
        {"output_scale": object()},
    ],
)
def test_other_attention_paths_keep_original_implementation(policy, change):
    args = arguments()
    args.update(change)
    path, called = policy[0](**args)
    assert path == "original"
    assert set(called) == set(args)
    assert all(called[k] is v for k, v in args.items())


def test_impl_extension_preserves_state_and_does_not_modify_original():
    source = ast.parse((ROOT / "vllm_fl/attention/iluvatar/impl.py").read_text())
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef))
    cls.body = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "from_impl"
    ]

    class Original:
        pass

    env = {"TritonAttentionImpl": Original}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "impl-contract", "exec"), env)
    old = Original()
    old.cache = object()
    old.backend = object()
    new = env[cls.name].from_impl(old)
    assert type(old) is Original
    assert isinstance(new, Original) and new is not old
    assert new.cache is old.cache and new.backend is old.backend


@pytest.fixture
def quest_graph_api():
    """Execute the actual Graph policy with a minimal native descriptor contract."""
    from contextlib import contextmanager
    from dataclasses import asdict, dataclass
    from enum import Enum

    @dataclass(frozen=True)
    class BatchDescriptor:
        num_tokens: int
        num_reqs: int | None = None
        uniform: bool = False
        has_lora: bool = False
        num_active_loras: int = 0

    class Mode(Enum):
        NONE = 0
        FULL = 1

    ctx = types.SimpleNamespace()
    env = dict(
        __name__=__name__,
        contextmanager=contextmanager,
        asdict=asdict,
        dataclass=dataclass,
        BatchDescriptor=BatchDescriptor,
        CUDAGraphMode=Mode,
        get_forward_context=lambda: ctx,
    )
    path = ROOT / "vllm_fl/attention/iluvatar/_graph.py"
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    exec(compile(tree, str(path), "exec"), env)
    return types.SimpleNamespace(**env, current_context=ctx)


@pytest.mark.parametrize(
    "length,capacity",
    [
        (0, 16),
        (16, 16),
        (17, 16),
        (256, 16),
        (272, 16),
        (273, 32),
        (4096, 256),
        (4097, 256),
        (4113, 512),
        (131072, 8192),
        (131073, 8192),
        (131089, 16384),
        (262145, 16384),
    ],
)
def test_quest_page_capacity_model_length_boundaries(quest_graph_api, length, capacity):
    assert quest_graph_api.page_capacity(length) == capacity


def test_quest_graph_key_uses_capacity_without_mutating_native_descriptor(
    quest_graph_api,
):
    from contextlib import contextmanager

    api = quest_graph_api
    policy = api.QuestGraphs(131072)
    desc = api.BatchDescriptor(64, 64, True, False, 0)
    calls = []

    @contextmanager
    def original(metadata, config, **kwargs):
        calls.append(kwargs)
        yield

    for maximum in [4096, 4113, 16384]:
        meta = {"layer": types.SimpleNamespace(max_seq_len=maximum)}
        config = types.SimpleNamespace(
            model_config=types.SimpleNamespace(max_model_len=131072)
        )
        with policy.context(
            original,
            meta,
            config,
            cudagraph_runtime_mode=api.CUDAGraphMode.FULL,
            batch_descriptor=desc,
        ):
            assert (
                api.current_context._quest_score_tiles
                == api.page_capacity(maximum) // 16
            )
    keys = [c["batch_descriptor"] for c in calls]
    assert len(set(keys)) == 3
    assert all(
        k.num_tokens == desc.num_tokens and k.num_reqs == desc.num_reqs and k.uniform
        for k in keys
    )
    assert not hasattr(desc, "page_capacity")


def test_quest_capture_largest_first_and_restores_state_after_failure(quest_graph_api):
    api = quest_graph_api
    policy, other = api.QuestGraphs(131072), api.QuestGraphs(131072)
    calls = []

    def capture(desc, mode, **kwargs):
        calls.append((policy.capture_capacity, kwargs["profile_seq_lens"]))
        assert other.capture_capacity is None

    policy.capture(
        capture,
        api.BatchDescriptor(64, 64, True),
        api.CUDAGraphMode.FULL,
        profile_seq_lens=131072,
    )
    capacities = [1 << power for power in range(13, 3, -1)]
    assert calls == [(cap, min(131072, (cap + 1) * 16)) for cap in capacities]
    assert policy.capture_capacity is None

    def failure(*args, **kwargs):
        raise RuntimeError("capture failed")

    with pytest.raises(RuntimeError, match="capture failed"):
        policy.capture(failure, api.BatchDescriptor(1), api.CUDAGraphMode.FULL)
    assert policy.capture_capacity is None


def test_quest_summaries_follow_model_and_cache_allocation():
    path = ROOT / "vllm_fl/attention/iluvatar/_quest.py"
    tree = ast.parse(path.read_text())
    tree.body = [
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "QuestState"
    ]
    env = {"torch": torch}
    exec(compile(tree, str(path), "exec"), env)
    cls = env["QuestState"]
    first, second = cls.__new__(cls), cls.__new__(cls)
    first.pools, second.pools = {}, {}
    cache = torch.empty((8, 2, 16, 2, 128), dtype=torch.bfloat16)
    lo, hi = first.pool("layer", cache[:4])
    assert first.pool("layer", cache[:4])[0] is lo
    assert second.pool("layer", cache[:4])[0] is not lo
    # Profiling and real caches can share an address while changing capacity.
    larger = first.pool("layer", cache)
    assert larger[0].shape[0] == 8 and larger[0] is not lo


@pytest.mark.parametrize(
    "change",
    [
        None,
        "dtype",
        "runner_type",
        "is_encoder_decoder",
        "multimodal_config",
        "quant_config",
        "lora_config",
        "speculative_config",
        "kv_transfer_config",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "decode_context_parallel_size",
        "prefill_context_parallel_size",
        "use_ubatching",
        "cpu_offload_gb",
        "offload_group_size",
        "stock_compile",
    ],
)
def test_shared_dense_model_contract(config_api, monkeypatch, change):
    config_module = types.ModuleType("vllm.config")
    config_module.CompilationMode = types.SimpleNamespace(STOCK_TORCH_COMPILE="stock")
    monkeypatch.setitem(sys.modules, "vllm.config", config_module)
    ns = types.SimpleNamespace
    model = ns(
        dtype=torch.bfloat16,
        runner_type="generate",
        is_encoder_decoder=False,
        multimodal_config=None,
    )
    parallel = ns(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
        use_ubatching=False,
    )
    cfg = ns(
        model_config=model,
        parallel_config=parallel,
        quant_config=None,
        lora_config=None,
        speculative_config=None,
        kv_transfer_config=None,
        compilation_config=ns(mode="vllm"),
        offload_config=ns(uva=ns(cpu_offload_gb=0), prefetch=ns(offload_group_size=0)),
    )
    if change == "dtype":
        model.dtype = torch.float16
    elif change == "runner_type":
        model.runner_type = "pooling"
    elif change in ["is_encoder_decoder", "multimodal_config"]:
        setattr(model, change, True)
    elif change in [
        "quant_config",
        "lora_config",
        "speculative_config",
        "kv_transfer_config",
    ]:
        setattr(cfg, change, object())
    elif change and change.endswith("parallel_size"):
        setattr(parallel, change, 2)
    elif change == "use_ubatching":
        parallel.use_ubatching = True
    elif change == "cpu_offload_gb":
        cfg.offload_config.uva.cpu_offload_gb = 1
    elif change == "offload_group_size":
        cfg.offload_config.prefetch.offload_group_size = 1
    elif change == "stock_compile":
        cfg.compilation_config.mode = "stock"
    assert config_api.supports_local_dense(cfg) is (change is None)
    if change == "stock_compile":
        assert config_api.supports_local_dense(cfg, allow_stock_compile=True)
