# Copyright (c) 2026 BAAI. All rights reserved.
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch._inductor import pattern_matcher as pm
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm.compilation.passes.inductor_pass import InductorPass, pass_context
from vllm.config import CompilationMode
from vllm.config.utils import Range

from vllm_fl.compilation.linear_swiglu import (
    LinearSwiGLUPass,
    _pattern,
    configure_linear_swiglu_fusion,
)


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


@pytest.mark.parametrize("limit", [0, -1, True, "64"])
def test_invalid_policy(limit):
    with pytest.raises(ValueError):
        configure_linear_swiglu_fusion(config({"max_tokens": limit}), "metax")


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


@pytest.mark.parametrize("m,n,k", [(7, 35, 67), (33, 96, 128), (65, 48, 32)])
@pytest.mark.gpu
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


@pytest.mark.parametrize("small_range", [True, False])
@pytest.mark.gpu
def test_dynamic_inductor_compilation(small_range):
    torch.manual_seed(31)
    p = LinearSwiGLUPass(96)
    compile_range = Range(1, 96) if small_range else Range(97, 2048)
    matches = []

    def post_pass(graph):
        with pass_context(compile_range):
            p(graph)
        matches.append(p.matched_count)

    torch._dynamo.reset()
    w = torch.randn((96, 32), device="cuda", dtype=torch.bfloat16) / 8
    fn = torch.compile(
        _pattern,
        fullgraph=True,
        options={"post_grad_custom_post_pass": post_pass},
    )
    for m in [7, 33, 65] if small_range else [129, 257]:
        x = torch.randn((m, 32), device="cuda", dtype=torch.bfloat16)
        torch._dynamo.mark_dynamic(x, 0)
        actual = fn(x, w)
        projection = F.linear(x, w).float()
        a, b = projection.chunk(2, dim=-1)
        expected = (F.silu(a) * b).bfloat16()
        torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)
    assert matches and all(n == int(small_range) for n in matches)


@pytest.mark.gpu
def test_reference_dispatch_is_preserved():
    from vllm_fl.dispatch import SelectionPolicy, policy_context
    from vllm_fl.ops.linear_swiglu import linear_swiglu

    x = torch.randn((7, 32), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((96, 32), device="cuda", dtype=torch.bfloat16) / 8
    projection = F.linear(x, w).float()
    a, b = projection.chunk(2, dim=-1)
    expected = (F.silu(a) * b).bfloat16()
    with policy_context(SelectionPolicy(prefer="reference")):
        torch.testing.assert_close(linear_swiglu(x, w), expected, rtol=0, atol=0)
