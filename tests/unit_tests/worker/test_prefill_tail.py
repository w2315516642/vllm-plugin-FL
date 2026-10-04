# Copyright (c) 2026 BAAI. All rights reserved.
"""Output/KV contract tests; the small causal model is a numerical reference."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.config import CompilationMode, CUDAGraphMode

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


@pytest.mark.parametrize("layers", [1, 3])
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
