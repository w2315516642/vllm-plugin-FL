# SPDX-License-Identifier: Apache-2.0
import torch
import triton
import triton.language as tl


@triton.jit
def paired_rope(
    Q,
    K,
    V,
    POS,
    CS,
    QO,
    KO,
    KC,
    VC,
    SLOTS,
    M,
    NSLOTS,
    SQ: tl.constexpr,
    SK: tl.constexpr,
    SV: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    CS_ROWS: tl.constexpr,
    PAGE: tl.constexpr,
    CACHE_BLOCK_STRIDE: tl.constexpr,
    CACHE_SLOT_STRIDE: tl.constexpr,
    CACHE_HEAD_STRIDE: tl.constexpr,
    FUSE_CACHE: tl.constexpr,
    B: tl.constexpr,
    EMIT_KO: tl.constexpr = False,
):
    pair = tl.program_id(0) * B + tl.arange(0, B)
    half: tl.constexpr = D // 2
    dim = pair % half
    head = pair // half % (HQ + HK)
    row = pair // (half * (HQ + HK))
    valid = row < M
    is_q = head < HQ
    kh = head - HQ
    position = tl.load(POS + row, valid, 0)
    position = tl.where(position < 0, position + CS_ROWS, position)
    cosine = tl.load(CS + position * D + dim, valid, 0).to(tl.float32)
    sine = tl.load(CS + position * D + half + dim, valid, 0).to(tl.float32)
    q0 = tl.load(Q + row * SQ + head * D + dim, valid & is_q, 0).to(tl.float32)
    q1 = tl.load(Q + row * SQ + head * D + half + dim, valid & is_q, 0).to(tl.float32)
    k0 = tl.load(K + row * SK + kh * D + dim, valid & ~is_q, 0).to(tl.float32)
    k1 = tl.load(K + row * SK + kh * D + half + dim, valid & ~is_q, 0).to(tl.float32)
    x0, x1 = tl.where(is_q, q0, k0), tl.where(is_q, q1, k1)
    y0 = x0 * cosine - x1 * sine
    y1 = x1 * cosine + x0 * sine
    tl.store(QO + row * HQ * D + head * D + dim, y0, valid & is_q)
    tl.store(QO + row * HQ * D + head * D + half + dim, y1, valid & is_q)
    if FUSE_CACHE:
        if EMIT_KO:
            tl.store(KO + row * HK * D + kh * D + dim, y0, valid & ~is_q)
            tl.store(KO + row * HK * D + kh * D + half + dim, y1, valid & ~is_q)
        slot = tl.load(SLOTS + row, valid & ~is_q & (row < NSLOTS), -1).to(tl.int64)
        active = valid & ~is_q & (slot >= 0)
        offset = (
            (slot // PAGE) * CACHE_BLOCK_STRIDE
            + (slot % PAGE) * CACHE_SLOT_STRIDE
            + kh * CACHE_HEAD_STRIDE
            + dim
        )
        tl.store(KC + offset, y0, active)
        tl.store(KC + offset + half, y1, active)
        v0 = tl.load(V + row * SV + kh * D + dim, active, 0)
        v1 = tl.load(V + row * SV + kh * D + half + dim, active, 0)
        tl.store(VC + offset, v0, active)
        tl.store(VC + offset + half, v1, active)
    else:
        tl.store(KO + row * HK * D + kh * D + dim, y0, valid & ~is_q)
        tl.store(KO + row * HK * D + kh * D + half + dim, y1, valid & ~is_q)


def launch_rope(
    q, k, v, positions, cs, qo, ko, cache, slots, *, fused=False, block=256, warps=4
):
    """Full even rotary dim, contiguous heads/dim, arbitrary positive token stride."""
    m, hq, d = q.shape
    hk = k.shape[1]
    assert d % 2 == 0 and d == k.shape[-1] and cs.shape[1] == d
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
    assert q.stride(1) == k.stride(1) == v.stride(1) == d
    if m:
        kc, vc = cache.unbind(1)
        paired_rope[(triton.cdiv(m * (hq + hk) * (d // 2), block),)](
            q,
            k,
            v,
            positions,
            cs,
            qo,
            ko,
            kc,
            vc,
            slots,
            m,
            slots.numel(),
            q.stride(0),
            k.stride(0),
            v.stride(0),
            hq,
            hk,
            d,
            cs.shape[0],
            kc.shape[1],
            kc.stride(0),
            kc.stride(1),
            kc.stride(2),
            fused,
            block,
            EMIT_KO=fused,
            num_warps=warps,
            num_stages=1,
        )
    return qo


@triton.jit
def rope_page_summary(
    Q,
    K,
    V,
    POS,
    CS,
    QO,
    KO,
    KC,
    VC,
    SLOTS,
    LO,
    HI,
    M,
    NSLOTS,
    SQ: tl.constexpr,
    SK: tl.constexpr,
    SV: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    CS_ROWS: tl.constexpr,
    P: tl.constexpr,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    KS2: tl.constexpr,
    B: tl.constexpr,
):
    pid = tl.program_id(0)
    half: tl.constexpr = D // 2
    qblocks = tl.cdiv(M * HQ * half, B)
    if pid < qblocks:
        q_pair = pid * B + tl.arange(0, B)
        q_dim = q_pair % half
        q_head = q_pair // half % HQ
        q_row = q_pair // (half * HQ)
        q_valid = q_row < M
        q_pos = tl.load(POS + q_row, q_valid, 0)
        q_pos = tl.where(q_pos < 0, q_pos + CS_ROWS, q_pos)
        q_c = tl.load(CS + q_pos * D + q_dim, q_valid, 0).to(tl.float32)
        q_s = tl.load(CS + q_pos * D + half + q_dim, q_valid, 0).to(tl.float32)
        q_x0 = tl.load(Q + q_row * SQ + q_head * D + q_dim, q_valid, 0).to(tl.float32)
        q_x1 = tl.load(Q + q_row * SQ + q_head * D + half + q_dim, q_valid, 0).to(
            tl.float32
        )
        q_y0 = q_x0 * q_c - q_x1 * q_s
        q_y1 = q_x1 * q_c + q_x0 * q_s
        tl.store(QO + q_row * HQ * D + q_head * D + q_dim, q_y0, q_valid)
        tl.store(QO + q_row * HQ * D + q_head * D + half + q_dim, q_y1, q_valid)
    else:
        owner = (pid - qblocks) // HK
        head = (pid - qblocks) % HK
        slot = tl.load(SLOTS + owner, owner < NSLOTS, -1).to(tl.int64)
        nxt = tl.load(SLOTS + owner + 1, owner + 1 < NSLOTS, -1).to(tl.int64)
        if slot < 0:
            # Preserve rotated-K output even for padding/no-cache rows.
            bad_d = tl.arange(0, half)
            bad_pos = tl.load(POS + owner)
            bad_pos = tl.where(bad_pos < 0, bad_pos + CS_ROWS, bad_pos)
            bad_c = tl.load(CS + bad_pos * D + bad_d).to(tl.float32)
            bad_s = tl.load(CS + bad_pos * D + half + bad_d).to(tl.float32)
            bad_x0 = tl.load(K + owner * SK + head * D + bad_d).to(tl.float32)
            bad_x1 = tl.load(K + owner * SK + head * D + half + bad_d).to(tl.float32)
            tl.store(
                KO + owner * HK * D + head * D + bad_d, bad_x0 * bad_c - bad_x1 * bad_s
            )
            tl.store(
                KO + owner * HK * D + head * D + half + bad_d,
                bad_x1 * bad_c + bad_x0 * bad_s,
            )
        elif (nxt < 0) | (slot // P != nxt // P):
            page = slot // P
            last = slot % P
            t = tl.arange(0, P)
            d = tl.arange(0, half)
            rows = owner - last + t
            inrange = (rows >= 0) & (rows <= owner) & (rows < M) & (rows < NSLOTS)
            row_slots = tl.load(SLOTS + rows, inrange, -1).to(tl.int64)
            new = inrange & (row_slots == page * P + t)
            pos = tl.load(POS + rows, new, 0)
            pos = tl.where(pos < 0, pos + CS_ROWS, pos)
            c = tl.load(CS + pos[:, None] * D + d[None, :], new[:, None], 0).to(
                tl.float32
            )
            s = tl.load(CS + pos[:, None] * D + half + d[None, :], new[:, None], 0).to(
                tl.float32
            )
            x0 = tl.load(
                K + rows[:, None] * SK + head * D + d[None, :], new[:, None], 0
            ).to(tl.float32)
            x1 = tl.load(
                K + rows[:, None] * SK + head * D + half + d[None, :], new[:, None], 0
            ).to(tl.float32)
            y0 = (x0 * c - x1 * s).to(KC.dtype.element_ty)
            y1 = (x1 * c + x0 * s).to(KC.dtype.element_ty)
            off = page * KS0 + t[:, None] * KS1 + head * KS2 + d[None, :]
            tl.store(
                KO + rows[:, None] * HK * D + head * D + d[None, :], y0, new[:, None]
            )
            tl.store(
                KO + rows[:, None] * HK * D + head * D + half + d[None, :],
                y1,
                new[:, None],
            )
            tl.store(KC + off, y0, new[:, None])
            tl.store(KC + off + half, y1, new[:, None])
            v0 = tl.load(
                V + rows[:, None] * SV + head * D + d[None, :], new[:, None], 0
            )
            v1 = tl.load(
                V + rows[:, None] * SV + head * D + half + d[None, :], new[:, None], 0
            )
            tl.store(VC + off, v0, new[:, None])
            tl.store(VC + off + half, v1, new[:, None])
            if last == P - 1:
                # No other CTA owns these new rows. Read only untouched prefix rows.
                old0 = tl.load(KC + off, ~new[:, None], 0)
                old1 = tl.load(KC + off + half, ~new[:, None], 0)
                full0 = tl.where(new[:, None], y0, old0).to(tl.float32)
                full1 = tl.where(new[:, None], y1, old1).to(tl.float32)
                base = (page * HK + head) * D + d
                tl.store(LO + base, tl.min(full0, 0))
                tl.store(HI + base, tl.max(full0, 0))
                tl.store(LO + base + half, tl.min(full1, 0))
                tl.store(HI + base + half, tl.max(full1, 0))


def launch_summary(q, k, v, pos, cs, qo, ko, cache, slots, lo, hi):
    m, hq, d = q.shape
    hk = k.shape[1]
    kc, vc = cache.unbind(1)
    if not m:
        return
    return rope_page_summary[(triton.cdiv(m * hq * (d // 2), 512) + m * hk,)](
        q,
        k,
        v,
        pos,
        cs,
        qo,
        ko,
        kc,
        vc,
        slots,
        lo,
        hi,
        m,
        len(slots),
        q.stride(0),
        k.stride(0),
        v.stride(0),
        hq,
        hk,
        d,
        cs.shape[0],
        16,
        *kc.stride()[:3],
        512,
        num_warps=4,
        num_stages=1,
    )
