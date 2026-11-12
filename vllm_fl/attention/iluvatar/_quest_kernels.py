# SPDX-License-Identifier: Apache-2.0
import torch
import triton
import triton.language as tl


@triton.jit
def _cat_columns(a, b):
    # CoreX tl.cat supports 1-D tensors; join/permute preserves 2-D row order.
    return tl.reshape(
        tl.permute(tl.join(a, b), (0, 2, 1)), (a.shape[0], a.shape[1] * 2)
    )


@triton.jit
def _join_tiles(xs, CHUNKS: tl.constexpr):
    if CHUNKS == 1:
        return xs[0]
    elif CHUNKS == 2:
        return _cat_columns(xs[0], xs[1])
    elif CHUNKS == 4:
        return _cat_columns(_cat_columns(xs[0], xs[1]), _cat_columns(xs[2], xs[3]))
    else:
        left = _cat_columns(_cat_columns(xs[0], xs[1]), _cat_columns(xs[2], xs[3]))
        right = _cat_columns(_cat_columns(xs[4], xs[5]), _cat_columns(xs[6], xs[7]))
        return _cat_columns(left, right)


@triton.jit
def _split_tiles(x, BM: tl.constexpr, CHUNKS: tl.constexpr):
    if CHUNKS == 1:
        return (x,)
    elif CHUNKS == 2:
        a, b = tl.split(tl.permute(tl.reshape(x, (BM, 2, 32)), (0, 2, 1)))
        return (a, b)
    elif CHUNKS == 4:
        a, b = tl.split(tl.permute(tl.reshape(x, (BM, 2, 64)), (0, 2, 1)))
        a0, a1 = tl.split(tl.permute(tl.reshape(a, (BM, 2, 32)), (0, 2, 1)))
        b0, b1 = tl.split(tl.permute(tl.reshape(b, (BM, 2, 32)), (0, 2, 1)))
        return (a0, a1, b0, b1)
    else:
        a, b = tl.split(tl.permute(tl.reshape(x, (BM, 2, 128)), (0, 2, 1)))
        a0, a1 = tl.split(tl.permute(tl.reshape(a, (BM, 2, 64)), (0, 2, 1)))
        b0, b1 = tl.split(tl.permute(tl.reshape(b, (BM, 2, 64)), (0, 2, 1)))
        c0, c1 = tl.split(tl.permute(tl.reshape(a0, (BM, 2, 32)), (0, 2, 1)))
        c2, c3 = tl.split(tl.permute(tl.reshape(a1, (BM, 2, 32)), (0, 2, 1)))
        c4, c5 = tl.split(tl.permute(tl.reshape(b0, (BM, 2, 32)), (0, 2, 1)))
        c6, c7 = tl.split(tl.permute(tl.reshape(b1, (BM, 2, 32)), (0, 2, 1)))
        return (c0, c1, c2, c3, c4, c5, c6, c7)


@triton.jit
def blocked_segment_attention(
    Q,
    K,
    V,
    LENS,
    IDS,
    A,
    M,
    L,
    HK: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    P: tl.constexpr,
    QS: tl.constexpr,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    KS2: tl.constexpr,
    VS0: tl.constexpr,
    VS1: tl.constexpr,
    VS2: tl.constexpr,
    SEG: tl.constexpr,
    BM: tl.constexpr,
    BD: tl.constexpr,
    R: tl.constexpr,
    TOKENS: tl.constexpr,
):
    seq, head, seg = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    length = tl.load(LENS + seq)
    pages = tl.cdiv(length, P)
    count = tl.minimum(R, pages)
    per_segment = tl.cdiv(count, SEG)
    h, d, t = tl.arange(0, BM), tl.arange(0, BD), tl.arange(0, 32)
    q = tl.load(
        Q + seq * QS + (head * G + h[:, None]) * D + d[None, :],
        (h[:, None] < G) & (d[None, :] < D),
        0,
    )
    score_tiles = ()
    for part in tl.static_range(TOKENS // 32):
        within_segment = part * 2 + t // P
        selected = seg * per_segment + within_segment
        active = (within_segment < per_segment) & (selected < count)
        physical = tl.load(IDS + (seq * HK + head) * R + selected, active, 0).to(
            tl.int64
        )
        valid = active & ((selected != count - 1) | (t % P < length - (pages - 1) * P))
        k = tl.load(
            K
            + physical[None, :] * KS0
            + (t % P)[None, :] * KS1
            + head * KS2
            + d[:, None],
            valid[None, :] & (d[:, None] < D),
            0,
        )
        score = tl.dot(q, k) * (D**-0.5)
        score = tl.where(valid[None, :] & (h[:, None] < G), score, float("-inf"))
        score_tiles += (score,)
    scores = _join_tiles(score_tiles, TOKENS // 32)
    maximum = tl.max(scores, 1)
    safe_maximum = tl.where(maximum == float("-inf"), 0.0, maximum)
    probabilities = tl.exp(scores - safe_maximum[:, None])
    denominator = tl.sum(probabilities, 1)
    prob_tiles = _split_tiles(probabilities, BM, TOKENS // 32)
    accumulator = tl.zeros((BM, BD), tl.float32)
    for part in tl.static_range(TOKENS // 32):
        within_segment = part * 2 + t // P
        selected = seg * per_segment + within_segment
        active = (within_segment < per_segment) & (selected < count)
        physical = tl.load(IDS + (seq * HK + head) * R + selected, active, 0).to(
            tl.int64
        )
        valid = active & ((selected != count - 1) | (t % P < length - (pages - 1) * P))
        v = tl.load(
            V
            + physical[:, None] * VS0
            + (t % P)[:, None] * VS1
            + head * VS2
            + d[None, :],
            valid[:, None] & (d[None, :] < D),
            0,
        )
        accumulator = tl.dot(prob_tiles[part].to(v.dtype), v, accumulator)
    base = (seq * HK * G + head * G + h) * SEG + seg
    tl.store(
        A + base[:, None] * D + d[None, :],
        accumulator,
        (h[:, None] < G) & (d[None, :] < D),
    )
    tl.store(M + base, maximum, h < G)
    tl.store(L + base, denominator, h < G)


@triton.jit
def page_score_if_needed(
    Q,
    LO,
    HI,
    BT,
    LENS,
    SCORES,
    HK: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    P: tl.constexpr,
    BS: tl.constexpr,
    QS: tl.constexpr,
    STRIDE: tl.constexpr,
    BM: tl.constexpr,
    BD: tl.constexpr,
    T: tl.constexpr,
    LATE_LOW: tl.constexpr,
    BUDGET_PAGES: tl.constexpr,
):
    seq, head, part = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    pages = part * T + tl.arange(0, T)
    d, h = tl.arange(0, BD), tl.arange(0, BM)
    history = tl.cdiv(tl.load(LENS + seq), P) - 1
    valid = pages < history
    if (history >= BUDGET_PAGES) & (part * T < history):
        physical = tl.load(BT + seq * BS + pages, valid, 0).to(tl.int64)
        q = tl.load(
            Q + seq * QS + (head * G + h[:, None]) * D + d[None, :],
            (h[:, None] < G) & (d[None, :] < D),
            0,
        )
        if not LATE_LOW:
            low = tl.load(
                LO + (physical[:, None] * HK + head) * D + d[None, :],
                valid[:, None] & (d[None, :] < D),
                0,
            )
        high = tl.load(
            HI + (physical[:, None] * HK + head) * D + d[None, :],
            valid[:, None] & (d[None, :] < D),
            0,
        )
        score = tl.dot(tl.maximum(q, 0).to(q.dtype), tl.trans(high))
        if LATE_LOW:
            low = tl.load(
                LO + (physical[:, None] * HK + head) * D + d[None, :],
                valid[:, None] & (d[None, :] < D),
                0,
            )
        score = tl.dot(tl.minimum(q, 0).to(q.dtype), tl.trans(low), score)
        score = tl.max(tl.where(h[:, None] < G, score, float("-inf")), 0)
        tl.store(
            SCORES + (seq * HK + head) * STRIDE + pages,
            tl.where(valid, score, float("-inf")),
            pages < STRIDE,
        )


@triton.jit
def merge(
    A, M, L, OUT, H: tl.constexpr, D: tl.constexpr, SEG: tl.constexpr, BD: tl.constexpr
):
    seq = tl.program_id(0)
    head = tl.program_id(1)
    s = tl.arange(0, SEG)
    d = tl.arange(0, BD)
    base = (seq * H + head) * SEG + s
    m = tl.load(M + base)
    l = tl.load(L + base)
    maximum = tl.max(m, 0)
    maximum = tl.where(maximum == float("-inf"), 0.0, maximum)
    weight = tl.exp(m - maximum)
    den = tl.sum(l * weight, 0)
    den = tl.where(den > 0, den, 1.0)
    a = tl.load(A + base[:, None] * D + d[None, :], d[None, :] < D, 0)
    y = tl.sum(a * weight[:, None], 0) / den
    tl.store(OUT + (seq * H + head) * D + d, y, d < D)


@triton.jit
def update_slots(
    K,
    LO,
    HI,
    SLOTS,
    N: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    KS2: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    slot = tl.load(SLOTS + row)
    nxt = tl.load(SLOTS + row + 1, row + 1 < N, -1)
    # Last changed token per contiguous physical page owns the summary write.
    if (slot >= 0) & ((nxt < 0) | (slot // 16 != nxt // 16)):
        page = (slot // 16).to(tl.int64)
        t = tl.arange(0, 16)
        d = tl.arange(0, D)
        x = tl.load(K + page * KS0 + t[:, None] * KS1 + head * KS2 + d[None, :]).to(
            tl.float32
        )
        tl.store(LO + (page * HK + head) * D + d, tl.min(x, 0))
        tl.store(HI + (page * HK + head) * D + d, tl.max(x, 0))
