# SPDX-License-Identifier: Apache-2.0
"""Model-owned page summaries and stable scratch for Quest decode."""

import torch
import triton

from vllm.forward_context import get_forward_context

from . import _quest_kernels as kernels
from ._graph import page_capacity
from ._quest_select import allocate_selection_workspace, select_pages


class QuestState:
    """Reuse scratch on the selected backend's sequential, single compute stream."""

    budget = 4096
    segments = 32
    segment_tokens = 128

    def __init__(self, device, capacity, max_model_len):
        self.pools = {}
        scores = page_capacity(max_model_len)
        self.workspace = {
            "a": torch.empty(
                (capacity, 16, 32, 128), device=device, dtype=torch.float32
            ),
            "m": torch.empty((capacity, 16, 32), device=device, dtype=torch.float32),
            "l": torch.empty((capacity, 16, 32), device=device, dtype=torch.float32),
            "ids": torch.empty((capacity, 2, 256), device=device, dtype=torch.int32),
            "scores": torch.empty(
                (capacity, 2, scores), device=device, dtype=torch.float32
            ),
            "selection": allocate_selection_workspace(device, capacity, 2, scores, 256),
        }

    def pool(self, name, cache):
        entry = self.pools.get(name)
        if entry is None or entry[0] != (cache.data_ptr(), cache.shape[0]):
            low = torch.empty(
                (cache.shape[0], 2, 128), device=cache.device, dtype=cache.dtype
            )
            entry = ((cache.data_ptr(), cache.shape[0]), low, torch.empty_like(low))
            self.pools[name] = entry
        return entry[1:]

    def update(self, name, cache, slots):
        low, high = self.pool(name, cache)
        key = cache[:, 0]
        n = len(slots)
        if n:
            kernels.update_slots[(n, 2)](
                key, low, high, slots, n, 2, 128, *key.stride()[:3], num_warps=4
            )

    def run(self, layer, query, cache, metadata, output):
        n = len(metadata.seq_lens)
        assert query.shape[0] >= n and n <= self.workspace["a"].shape[0]
        key, value = cache.unbind(1)
        p = self.workspace
        table, lengths = metadata.block_table, metadata.seq_lens
        score_capacity = p["scores"].shape[-1]
        tiles = getattr(
            get_forward_context(), "_quest_score_tiles", score_capacity // 16
        )
        low, high = self.pool(layer.layer_name, cache)
        kernels.page_score_if_needed[(n, 2, triton.cdiv(tiles, 2))](
            query,
            low,
            high,
            table,
            lengths,
            p["scores"],
            2,
            8,
            128,
            16,
            table.stride(0),
            query.stride(0),
            score_capacity,
            8,
            128,
            32,
            LATE_LOW=True,
            BUDGET_PAGES=256,
            num_warps=2,
        )
        select_pages(
            p["scores"],
            table,
            lengths,
            p["ids"],
            p["selection"],
            tiles * 16,
            radix=True,
            physical=True,
        )
        kernels.blocked_segment_attention[(n, 2, 32)](
            query,
            key,
            value,
            lengths,
            p["ids"],
            p["a"],
            p["m"],
            p["l"],
            2,
            8,
            128,
            16,
            query.stride(0),
            *key.stride()[:3],
            *value.stride()[:3],
            32,
            8,
            128,
            256,
            TOKENS=128,
            num_warps=2,
            num_stages=1,
        )
        kernels.merge[(n, 16)](
            p["a"], p["m"], p["l"], output, 16, 128, 32, 128, num_warps=4
        )
        return output
