# SPDX-License-Identifier: Apache-2.0
"""Use a new fused filter while retaining vLLM sampling/RNG/logprobs behavior."""

from contextvars import ContextVar

import torch
import torch.nn as nn

from vllm.v1.sample.sampler import Sampler


class FilteredTopKTopPSampler(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original

    def forward(self, logits, generators, k, p):
        from flag_gems import supports_topk_topp, topk_topp_filter

        if logits.is_cuda and torch.cuda.is_current_stream_capturing():
            # The enclosing capture owns its workspaces/lifetime. Keep its
            # original sampler instead of introducing a nested filter graph.
            return self.original(logits, generators, k, p)
        if not supports_topk_topp(logits, k, p):
            # Keep the pre-existing implementation for unsupported contracts.
            return self.original(logits, generators, k, p)
        filtered = topk_topp_filter(logits, k, p)
        return self.original(filtered, generators, None, None)


class SamplerFL(Sampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.topk_topp_sampler = FilteredTopKTopPSampler(self.topk_topp_sampler)
        self._unit_temperature = ContextVar("fl_unit_temperature", default=False)

    @staticmethod
    def temperatures_are_one(input_batch, metadata):
        """Use the CPU source of this batch's temperature upload, never .item()."""
        cpu = getattr(input_batch, "temperature_cpu_tensor", None)
        rows = input_batch.num_reqs
        return (
            metadata.all_random
            and metadata.temperature is not None
            and metadata.temperature.numel() == rows
            and isinstance(cpu, torch.Tensor)
            and cpu.device.type == "cpu"
            and cpu.ndim == 1
            and cpu.numel() >= rows
            and bool((cpu[:rows].numpy() == 1.0).all())
        )

    def forward(self, *args, all_unit_temperature=False, **kwargs):
        token = self._unit_temperature.set(all_unit_temperature)
        try:
            return super().forward(*args, **kwargs)
        finally:
            self._unit_temperature.reset(token)

    def apply_temperature(self, logits, temp, all_random):
        if all_random and self._unit_temperature.get():
            return logits
        return super().apply_temperature(logits, temp, all_random)
