"""Exact Quest selection with physical-page output in logical-page order."""

import torch
import triton
import triton.language as tl


@triton.jit
def _radix_threshold(scores, valid, rank, count):
    """Find an ascending order statistic among valid FP32 scores."""
    bits = scores.to(tl.uint32, bitcast=True)
    # The previous selector compares floats and treats both signs of zero alike.
    bits = tl.where((bits & 0x7FFFFFFF) == 0, tl.full((), 0, tl.uint32), bits)
    key = bits ^ tl.where(
        (bits >> 31) != 0,
        tl.full((), 0xFFFFFFFF, tl.uint32),
        tl.full((), 0x80000000, tl.uint32),
    )
    bins = tl.arange(0, 16)
    candidates = valid
    level = 0
    while (level < 8) & (count > 1):
        digit = ((key >> (28 - level * 4)) & 15).to(tl.int32)
        histogram = tl.histogram(digit, 16, mask=candidates)
        cumulative = tl.cumsum(histogram, axis=0)
        target = tl.sum((cumulative <= rank).to(tl.int32), axis=0)
        target_mask = bins == target
        skipped = tl.sum(tl.where(target_mask, cumulative - histogram, 0), axis=0)
        count = tl.sum(tl.where(target_mask, histogram, 0), axis=0)
        rank -= skipped
        candidates &= digit == target
        level += 1

    # At loop exit the remaining keys are identical, or just one key is left.
    return tl.max(tl.where(candidates, scores, float("-inf")), axis=0)


@triton.jit
def _select_history(
    scores_ptr,
    table,
    output,
    num_pages,
    KEEP: tl.constexpr,
    BLOCK: tl.constexpr,
    RADIX: tl.constexpr,
    PHYSICAL: tl.constexpr,
):
    logical = tl.arange(0, BLOCK)
    history = num_pages - 1
    valid = logical < history
    scores = tl.load(scores_ptr + logical, mask=valid, other=float("-inf"))

    if RADIX:
        threshold = _radix_threshold(scores, valid, history - KEEP, history)
    else:
        ordered = tl.sort(scores, descending=True)
        threshold = tl.sum(tl.where(logical == KEEP - 1, ordered, 0.0), axis=0)

    greater = (scores > threshold) & valid
    equal = (scores == threshold) & valid
    ties = tl.cumsum(equal.to(tl.int32), axis=0)
    remaining = KEEP - tl.sum(greater.to(tl.int32), axis=0)
    chosen = greater | (equal & (ties <= remaining))
    destination = tl.cumsum(chosen.to(tl.int32), axis=0) - 1
    if PHYSICAL:
        selected = tl.load(table + logical, mask=chosen, other=0)
        latest = tl.load(table + history)
    else:
        selected, latest = logical, history
    tl.store(output + destination, selected, mask=chosen)
    tl.store(output + KEEP, latest)


@triton.jit
def _gather_all(
    table, output, num_pages, BUDGET_PAGES: tl.constexpr, PHYSICAL: tl.constexpr
):
    logical = tl.arange(0, BUDGET_PAGES)
    valid = logical < num_pages
    if PHYSICAL:
        value = tl.load(table + logical, mask=valid, other=0)
    else:
        value = logical
    tl.store(output + logical, value, mask=valid)


@triton.jit
def select_physical_pages(
    SCORES,
    BT,
    LENS,
    PHYSICAL_IDS,
    KV_HEADS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    SCORE_STRIDE: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    BUDGET_PAGES: tl.constexpr,
    RADIX: tl.constexpr,
    BLOCK: tl.constexpr,
    PHYSICAL: tl.constexpr,
):
    seq = tl.program_id(0)
    head = tl.program_id(1)
    num_pages = tl.cdiv(tl.load(LENS + seq), PAGE_SIZE)
    row = seq * KV_HEADS + head
    scores = SCORES + row * SCORE_STRIDE
    table = BT + seq * TABLE_STRIDE
    output = PHYSICAL_IDS + row * BUDGET_PAGES

    if num_pages > 0:
        if BLOCK < BUDGET_PAGES:
            _gather_all(table, output, num_pages, BUDGET_PAGES, PHYSICAL)
        else:
            if num_pages <= BUDGET_PAGES:
                _gather_all(table, output, num_pages, BUDGET_PAGES, PHYSICAL)
            else:
                _select_history(
                    scores,
                    table,
                    output,
                    num_pages,
                    BUDGET_PAGES - 1,
                    BLOCK,
                    RADIX,
                    PHYSICAL,
                )


# These bound each CTA's work, rather than the model's maximum context.
SINGLE_BLOCK_PAGES = 8192
PRUNE_BLOCK_PAGES = 4096


@triton.jit
def _choose(
    scores, valid, count, KEEP: tl.constexpr, BLOCK: tl.constexpr, RADIX: tl.constexpr
):
    if count > KEEP:
        if RADIX:
            threshold = _radix_threshold(scores, valid, count - KEEP, count)
        else:
            ordered = tl.sort(tl.where(valid, scores, float("-inf")), descending=True)
            threshold = tl.sum(
                tl.where(tl.arange(0, BLOCK) == KEEP - 1, ordered, 0.0), 0
            )
        greater = (scores > threshold) & valid
        equal = (scores == threshold) & valid
        ties = tl.cumsum(equal.to(tl.int32), 0)
        chosen = greater | (equal & (ties <= KEEP - tl.sum(greater.to(tl.int32), 0)))
    else:
        chosen = valid
    return chosen


@triton.jit
def prune_candidates(
    SCORES,
    INDICES,
    LENS,
    OUT_SCORES,
    OUT_INDICES,
    HK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    IN_WIDTH: tl.constexpr,
    IN_STRIDE: tl.constexpr,
    OUT_STRIDE: tl.constexpr,
    KEEP: tl.constexpr,
    BLOCK: tl.constexpr,
    FIRST: tl.constexpr,
    RADIX: tl.constexpr,
):
    seq, head, chunk = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    row = seq * HK + head
    position = chunk * BLOCK + tl.arange(0, BLOCK)
    if FIRST:
        history = tl.cdiv(tl.load(LENS + seq), PAGE_SIZE) - 1
        logical = position
        valid = (position < history) & (position < IN_WIDTH)
    else:
        logical = tl.load(INDICES + row * IN_STRIDE + position, position < IN_WIDTH, -1)
        valid = (position < IN_WIDTH) & (logical >= 0)
    scores = tl.load(SCORES + row * IN_STRIDE + position, valid, float("-inf"))
    count = tl.sum(valid.to(tl.int32), 0)
    chosen = _choose(scores, valid, count, KEEP, BLOCK, RADIX)
    rank = tl.cumsum(chosen.to(tl.int32), 0) - 1
    # Output slots are fully overwritten even for empty requests/chunks.
    slots = tl.arange(0, BLOCK)
    base = row * OUT_STRIDE + chunk * KEEP
    padding = (slots < KEEP) & (slots >= tl.minimum(count, KEEP))
    tl.store(OUT_INDICES + base + slots, -1, padding)
    tl.store(OUT_SCORES + base + slots, float("-inf"), padding)
    tl.store(OUT_INDICES + base + rank, logical, chosen)
    tl.store(OUT_SCORES + base + rank, scores, chosen)


@triton.jit
def select_candidates(
    SCORES,
    INDICES,
    BT,
    LENS,
    OUTPUT,
    HK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    IN_WIDTH: tl.constexpr,
    IN_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    BUDGET_PAGES: tl.constexpr,
    RADIX: tl.constexpr,
    PHYSICAL: tl.constexpr,
):
    seq, head = tl.program_id(0), tl.program_id(1)
    row = seq * HK + head
    table = BT + seq * TABLE_STRIDE
    output = OUTPUT + row * BUDGET_PAGES
    num_pages = tl.cdiv(tl.load(LENS + seq), PAGE_SIZE)
    if num_pages > 0:
        if num_pages <= BUDGET_PAGES:
            _gather_all(table, output, num_pages, BUDGET_PAGES, PHYSICAL)
        else:
            position = tl.arange(0, BLOCK)
            logical = tl.load(
                INDICES + row * IN_STRIDE + position, position < IN_WIDTH, -1
            )
            valid = (position < IN_WIDTH) & (logical >= 0)
            scores = tl.load(SCORES + row * IN_STRIDE + position, valid, float("-inf"))
            chosen = _choose(
                scores,
                valid,
                tl.sum(valid.to(tl.int32), 0),
                BUDGET_PAGES - 1,
                BLOCK,
                RADIX,
            )
            rank = tl.cumsum(chosen.to(tl.int32), 0) - 1
            if PHYSICAL:
                selected = tl.load(table + logical, chosen, 0)
                latest = tl.load(table + num_pages - 1)
            else:
                selected, latest = logical, num_pages - 1
            tl.store(output + rank, selected, chosen)
            tl.store(output + BUDGET_PAGES - 1, latest)


def allocate_selection_workspace(
    device, batch_capacity, kv_heads, score_capacity, budget_pages
):
    stages = []
    width = score_capacity
    while width > SINGLE_BLOCK_PAGES:
        width = triton.cdiv(width, PRUNE_BLOCK_PAGES) * (budget_pages - 1)
        shape = (batch_capacity, kv_heads, width)
        stages.append(
            (
                torch.empty(shape, device=device, dtype=torch.float32),
                torch.empty(shape, device=device, dtype=torch.int32),
            )
        )
    return stages


def select_pages(
    scores, table, lengths, output, stages, history_capacity, *, radix, physical
):
    n, hk, budget_pages = len(lengths), scores.shape[1], output.shape[-1]
    if history_capacity <= SINGLE_BLOCK_PAGES:
        select_physical_pages[(n, hk)](
            scores,
            table,
            lengths,
            output,
            hk,
            16,
            scores.stride(1),
            table.stride(0),
            budget_pages,
            RADIX=radix,
            BLOCK=history_capacity,
            PHYSICAL=physical,
            num_warps=4,
        )
        return
    source_scores, source_ids = scores, output
    width = history_capacity
    first = True
    for next_scores, next_ids in stages:
        chunks = triton.cdiv(width, PRUNE_BLOCK_PAGES)
        prune_candidates[(n, hk, chunks)](
            source_scores,
            source_ids,
            lengths,
            next_scores,
            next_ids,
            hk,
            16,
            width,
            source_scores.stride(1),
            next_scores.stride(1),
            budget_pages - 1,
            PRUNE_BLOCK_PAGES,
            FIRST=first,
            RADIX=radix,
            num_warps=4,
        )
        width = chunks * (budget_pages - 1)
        source_scores, source_ids = next_scores, next_ids
        first = False
        if width <= SINGLE_BLOCK_PAGES:
            select_candidates[(n, hk)](
                source_scores,
                source_ids,
                table,
                lengths,
                output,
                hk,
                16,
                table.stride(0),
                width,
                source_scores.stride(1),
                triton.next_power_of_2(width),
                budget_pages,
                RADIX=radix,
                PHYSICAL=physical,
                num_warps=4,
            )
            return
