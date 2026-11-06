# SPDX-License-Identifier: Apache-2.0
"""Use a new fused filter while retaining vLLM sampling/RNG/logprobs behavior."""

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
