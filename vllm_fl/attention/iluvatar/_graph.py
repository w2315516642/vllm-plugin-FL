# SPDX-License-Identifier: Apache-2.0
"""Include CPU-known page capacity in the existing full-decode Graph key."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass

from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, get_forward_context


def page_capacity(max_seq_len: int) -> int:
    history_pages = max(0, (max_seq_len + 15) // 16 - 1)
    return max(16, 1 << max(0, history_pages - 1).bit_length())


@dataclass(frozen=True)
class CapacityBatchDescriptor(BatchDescriptor):
    page_capacity: int = 0


class QuestGraphs:
    def __init__(self, max_model_len):
        self.maximum = page_capacity(max_model_len)
        self.capture_capacity = None

    @contextmanager
    def context(self, original, attn_metadata, config, *args, **kwargs):
        mode = kwargs.get("cudagraph_runtime_mode", CUDAGraphMode.NONE)
        metadata = next(iter(attn_metadata.values())) if attn_metadata else None
        maximum = (
            metadata.max_seq_len
            if metadata is not None
            else config.model_config.max_model_len
        )
        capacity = self.capture_capacity or page_capacity(maximum)
        if mode == CUDAGraphMode.FULL:
            kwargs["batch_descriptor"] = CapacityBatchDescriptor(
                **asdict(kwargs["batch_descriptor"]), page_capacity=capacity
            )
        with original(attn_metadata, config, *args, **kwargs):
            get_forward_context()._quest_score_tiles = capacity // 16
            yield

    def capture(self, original, desc, mode, **kwargs):
        previous = self.capture_capacity
        try:
            # Keep the tested capacities, largest first for graph-pool reuse.
            for power in range(self.maximum.bit_length() - 1, 3, -1):
                self.capture_capacity = 1 << power
                current = dict(kwargs)
                if current.get("profile_seq_lens") is not None:
                    current["profile_seq_lens"] = min(
                        current["profile_seq_lens"], (self.capture_capacity + 1) * 16
                    )
                original(desc, mode, **current)
        finally:
            self.capture_capacity = previous
