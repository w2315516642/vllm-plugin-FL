# SPDX-License-Identifier: Apache-2.0
"""Exercise actual launch-policy code with CPU tensors and kernel call spies."""

import ast
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def policy(monkeypatch):
    package = types.ModuleType("_attention_policy_contract")
    package.__path__ = []
    unified = types.ModuleType(package.__name__ + "._unified")
    unified.is_batch_invariant = False
    unified.unified_attention = lambda **kw: ("unified", kw)
    decode = types.ModuleType(package.__name__ + "._decode")
    decode.unified_attention = lambda **kw: ("decode", kw)
    for module in [package, unified, decode]:
        monkeypatch.setitem(sys.modules, module.__name__, module)
    source = ast.parse((ROOT / "vllm_fl/attention/iluvatar/policy.py").read_text())
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
