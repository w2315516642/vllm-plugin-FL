"""Split prefill and decode tasks while sharing a per-forward plan across layers."""

import torch
import triton
import triton.language as tl
from vllm.forward_context import get_forward_context

from ._unified import unified_attention
from .impl import tuned_unified_attention as baseline


@triton.jit
def plan_prefill(
    CU, TASK_SEQ, TASK_BLOCK, N: tl.constexpr, B: tl.constexpr, BQ: tl.constexpr
):
    task = tl.program_id(0)
    r = tl.arange(0, B)
    starts = tl.load(CU + r, r < N, 0)
    ends = tl.load(CU + r + 1, r < N, 0)
    lens = ends - starts
    blocks = tl.where((r < N) & (lens > 1), tl.cdiv(lens, BQ), 0)
    cumulative = tl.cumsum(blocks, 0)
    total = tl.sum(blocks, 0)
    owner = tl.sum(((cumulative <= task) & (r < N)).to(tl.int32), 0)
    previous = tl.sum(tl.where(r == owner - 1, cumulative, 0), 0)
    tl.store(TASK_SEQ + task, tl.where(task < total, owner, -1))
    tl.store(TASK_BLOCK + task, tl.where(task < total, task - previous, 0))


class Plan:
    def __init__(self, q, cu, lens, segments=16):
        self.n = len(lens)
        self.capacity = q.shape[0] // 16 + self.n
        self.seq = torch.empty(self.capacity, device=q.device, dtype=torch.int32)
        self.block = torch.empty_like(self.seq)
        self.partial = torch.empty(
            (self.n, q.shape[1], segments, q.shape[2]),
            device=q.device,
            dtype=torch.float32,
        )
        self.maxima = torch.empty(
            (self.n, q.shape[1], segments), device=q.device, dtype=torch.float32
        )
        self.sums = torch.empty_like(self.maxima)
        self.cu = cu
        self.segments = segments

    def prepare(self):
        plan_prefill[(self.capacity,)](
            self.cu,
            self.seq,
            self.block,
            self.n,
            triton.next_power_of_2(self.n),
            16,
            num_warps=4,
        )


def run(kwargs, plan, *, prepare=True):
    # The fixture/integration checks the supported decoder contract before entry.
    # q/k/v/cache and the scheduler request order remain unchanged.
    if kwargs["max_seqlen_q"] <= 1:
        return baseline(**kwargs)
    if prepare:
        plan.prepare()
    opts = dict(kwargs)
    opts.update(
        prefill_block_m=128,
        prefill_tile_size=64,
        prefill_num_warps=8,
        prefill_num_stages=1,
        task_seq=plan.seq,
        task_block=plan.block,
    )
    unified_attention(**opts)
    opts = dict(kwargs)
    opts.update(
        max_seqlen_q=1,
        seq_threshold_3D=128,
        num_par_softmax_segments=plan.segments,
        softmax_segm_output=plan.partial,
        softmax_segm_max=plan.maxima,
        softmax_segm_expsum=plan.sums,
        decode_tile_size=16,
        decode_num_warps=2,
        decode_num_stages=1,
        direct_decode=True,
        defer_lsum=True,
        decode_row_mode="bm8",
        mixed_batch=True,
    )
    unified_attention(**opts)
    return kwargs["out"]


def mixed_attention(kwargs):
    q, k = kwargs["q"], kwargs["k"]
    n = len(kwargs["seqused_k"])
    supported = (
        kwargs["max_seqlen_q"] > 1
        and 1 < n <= 128
        and q.dtype == torch.bfloat16
        and q.shape[-1] == 128
        and q.shape[1] == 8 * k.shape[2]
        and kwargs.get("causal", True) is True
        and kwargs.get("window_size", (-1, -1)) == (-1, -1)
        and kwargs.get("softcap", 0) == 0
        and kwargs.get("sinks") is None
        and kwargs.get("alibi_slopes") is None
        and kwargs.get("qq_bias") is None
        and kwargs.get("mm_prefix_range") is None
        and kwargs.get("output_scale") is None
        and kwargs.get("kv_quant_mode", 0) == 0
        and not kwargs.get("use_td", False)
        and kwargs.get("chunk_lookback", -1) == -1
    )
    if not supported:
        return baseline(**kwargs)
    context = get_forward_context()
    # A new ForwardContext is created for every model forward. Retain no cache
    # across model calls, input-batch generations, devices or asynchronous streams.
    plans = getattr(context, "_fl_mixed_attention_plans", None)
    if plans is None:
        plans = {}
        setattr(context, "_fl_mixed_attention_plans", plans)
    cu = kwargs["cu_seqlens_q"]
    lens = kwargs["seqused_k"]
    key = (
        q.shape[0],
        q.shape[1],
        q.shape[2],
        n,
        cu.data_ptr(),
        lens.data_ptr(),
        q.device.index,
        torch.cuda.current_stream().cuda_stream,
    )
    plan = plans.get(key)
    if plan is None:
        plan = Plan(q, cu, lens, segments=16)
        plan.prepare()
        plans[key] = plan
    return run(kwargs, plan, prepare=False)
