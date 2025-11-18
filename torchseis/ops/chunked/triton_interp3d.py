# Copyright (c) 2025 Jintao Li. 
# Zhejiang University (ZJU).
# All rights reserved.


import torch
import triton
import triton.language as tl

# ------------------------------
# helpers
# ------------------------------
def _alloc_like_out(x, D_out, H_out, W_out):
    # 继承 x 的 memory_format（NCDHW 或 channels_last_3d）
    # mf = x.suggest_memory_format()
    out = torch.empty(
        (x.shape[0], x.shape[1], D_out, H_out, W_out),
        dtype=x.dtype, device=x.device
    )
    return out

# ------------------------------
# Triton kernel (fp16/bf16 IO, fp32 accumulate)
# ------------------------------
@triton.jit
def _tri3d_align_fp16_kernel(
    IN_PTR, OUT_PTR,
    B, C, D_IN, H_IN, W_IN,
    D_OUT, H_OUT, W_OUT,
    D_OR,  H_OR,  W_OR,
    DS, HS, WS, SF,             # start offsets & scale_factor
    STRIDE_IN_N, STRIDE_IN_C, STRIDE_IN_D, STRIDE_IN_H, STRIDE_IN_W,
    STRIDE_OUT_N, STRIDE_OUT_C, STRIDE_OUT_D, STRIDE_OUT_H, STRIDE_OUT_W,
    # launch / tiling
    BLOCK_W: tl.constexpr,
):
    pid_w  = tl.program_id(0)
    pid_dh = tl.program_id(1)
    pid_bc = tl.program_id(2)

    if pid_dh >= D_OUT * H_OUT:
        return
    if pid_bc >= B * C:
        return

    h_out = pid_dh % H_OUT
    d_out = pid_dh // H_OUT
    c = pid_bc % C
    b = pid_bc // C

    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_w = offs_w < W_OUT

    # ----- align_corners=True 的全局坐标映射 -----
    # 全局输出范围尺寸
    D_outo = D_OR * SF
    H_outo = H_OR * SF
    W_outo = W_OR * SF

    # 该 tile 的全局输出坐标（对应整个大输出体素坐标系）
    d_os = DS * SF
    h_os = HS * SF
    w_os = WS * SF

    d_full = d_os + d_out            # scalar
    h_full = h_os + h_out            # scalar
    w_full = w_os + offs_w           # vector

    # 归一化到 [0,1]
    d_norm = d_full / tl.maximum(D_outo - 1, 1)
    h_norm = h_full / tl.maximum(H_outo - 1, 1)
    w_norm = w_full / tl.maximum(W_outo - 1, 1)

    # 映射回原始输入坐标（浮点）
    d_in_f = d_norm * (D_OR - 1)
    h_in_f = h_norm * (H_OR - 1)
    w_in_f = w_norm * (W_OR - 1)

    # 左/上/前 整点（全局）
    d0_full = tl.floor(d_in_f).to(tl.int32)
    h0_full = tl.floor(h_in_f).to(tl.int32)
    w0_full = tl.floor(w_in_f).to(tl.int32)

    # 右/下/后 整点（全局，clamp 到原始尺寸）
    d1_full = tl.minimum(d0_full + 1, D_OR - 1)
    h1_full = tl.minimum(h0_full + 1, H_OR - 1)
    w1_full = tl.minimum(w0_full + 1, W_OR - 1)

    # 小块内局部坐标（减去 start，再 clamp 到 chunk 范围）
    d0 = tl.maximum(tl.minimum(d0_full - DS, D_IN - 1), 0)
    h0 = tl.maximum(tl.minimum(h0_full - HS, H_IN - 1), 0)
    w0 = tl.maximum(tl.minimum(w0_full - WS, W_IN - 1), 0)

    d1 = tl.maximum(tl.minimum(d1_full - DS, D_IN - 1), 0)
    h1 = tl.maximum(tl.minimum(h1_full - HS, H_IN - 1), 0)
    w1 = tl.maximum(tl.minimum(w1_full - WS, W_IN - 1), 0)

    # 分数部分（浮点，fp32 累加）
    dd = (d_in_f - d0_full).to(tl.float32)   # scalar
    hh = (h_in_f - h0_full).to(tl.float32)   # scalar
    ww = (w_in_f - w0_full).to(tl.float32)   # vector

    # base pointers
    base_in = (b * STRIDE_IN_N + c * STRIDE_IN_C)
    base_out = (b * STRIDE_OUT_N + c * STRIDE_OUT_C +
                d_out * STRIDE_OUT_D + h_out * STRIDE_OUT_H)

    # 8 角点地址（广播：d/h 是标量，w 是向量）
    off_d0h0w0 = base_in + d0 * STRIDE_IN_D + h0 * STRIDE_IN_H + w0 * STRIDE_IN_W
    off_d0h0w1 = base_in + d0 * STRIDE_IN_D + h0 * STRIDE_IN_H + w1 * STRIDE_IN_W
    off_d0h1w0 = base_in + d0 * STRIDE_IN_D + h1 * STRIDE_IN_H + w0 * STRIDE_IN_W
    off_d0h1w1 = base_in + d0 * STRIDE_IN_D + h1 * STRIDE_IN_H + w1 * STRIDE_IN_W
    off_d1h0w0 = base_in + d1 * STRIDE_IN_D + h0 * STRIDE_IN_H + w0 * STRIDE_IN_W
    off_d1h0w1 = base_in + d1 * STRIDE_IN_D + h0 * STRIDE_IN_H + w1 * STRIDE_IN_W
    off_d1h1w0 = base_in + d1 * STRIDE_IN_D + h1 * STRIDE_IN_H + w0 * STRIDE_IN_W
    off_d1h1w1 = base_in + d1 * STRIDE_IN_D + h1 * STRIDE_IN_H + w1 * STRIDE_IN_W

    # 加载 8 角点（fp16/bf16 -> fp32）
    c000 = tl.load(IN_PTR + off_d0h0w0, mask=mask_w, other=0).to(tl.float32)
    c001 = tl.load(IN_PTR + off_d0h0w1, mask=mask_w, other=0).to(tl.float32)
    c010 = tl.load(IN_PTR + off_d0h1w0, mask=mask_w, other=0).to(tl.float32)
    c011 = tl.load(IN_PTR + off_d0h1w1, mask=mask_w, other=0).to(tl.float32)
    c100 = tl.load(IN_PTR + off_d1h0w0, mask=mask_w, other=0).to(tl.float32)
    c101 = tl.load(IN_PTR + off_d1h0w1, mask=mask_w, other=0).to(tl.float32)
    c110 = tl.load(IN_PTR + off_d1h1w0, mask=mask_w, other=0).to(tl.float32)
    c111 = tl.load(IN_PTR + off_d1h1w1, mask=mask_w, other=0).to(tl.float32)

    # 三线性权重
    one = tl.full([1], 1.0, tl.float32)
    ww0 = one - ww
    ww1 = ww
    hh0 = 1.0 - hh
    hh1 = hh
    dd0 = 1.0 - dd
    dd1 = dd

    # 先 w，再 h，再 d（fp32 累加）
    c00 = c000 * ww0 + c001 * ww1
    c01 = c010 * ww0 + c011 * ww1
    c10 = c100 * ww0 + c101 * ww1
    c11 = c110 * ww0 + c111 * ww1

    c0 = c00 * hh0 + c01 * hh1
    c1 = c10 * hh0 + c11 * hh1

    outv = c0 * dd0 + c1 * dd1  # [BLOCK_W] fp32

    # 写回（按输入 dtype 写回）
    # OUT_PTR 是 *fp16/bf16
    tl.store(OUT_PTR + base_out + offs_w.to(tl.int64) * STRIDE_OUT_W, outv, mask=mask_w)


# ------------------------------
# Triton kernel (fp32 IO)
# ------------------------------
@triton.jit
def _tri3d_align_fp32_kernel(
    IN_PTR, OUT_PTR,
    B, C, D_IN, H_IN, W_IN,
    D_OUT, H_OUT, W_OUT,
    D_OR,  H_OR,  W_OR,
    DS, HS, WS, SF,
    STRIDE_IN_N, STRIDE_IN_C, STRIDE_IN_D, STRIDE_IN_H, STRIDE_IN_W,
    STRIDE_OUT_N, STRIDE_OUT_C, STRIDE_OUT_D, STRIDE_OUT_H, STRIDE_OUT_W,
    BLOCK_W: tl.constexpr,
):
    # 与 fp16 版相同，只是指针类型为 *fp32，省去 to(fp32) / cast
    pid_w  = tl.program_id(0)
    pid_dh = tl.program_id(1)
    pid_bc = tl.program_id(2)

    if pid_dh >= D_OUT * H_OUT:
        return
    if pid_bc >= B * C:
        return

    h_out = pid_dh % H_OUT
    d_out = pid_dh // H_OUT
    c = pid_bc % C
    b = pid_bc // C

    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_w = offs_w < W_OUT

    D_outo = D_OR * SF
    H_outo = H_OR * SF
    W_outo = W_OR * SF

    d_os = DS * SF
    h_os = HS * SF
    w_os = WS * SF

    d_full = d_os + d_out
    h_full = h_os + h_out
    w_full = w_os + offs_w

    d_norm = d_full / tl.maximum(D_outo - 1, 1)
    h_norm = h_full / tl.maximum(H_outo - 1, 1)
    w_norm = w_full / tl.maximum(W_outo - 1, 1)

    d_in_f = d_norm * (D_OR - 1)
    h_in_f = h_norm * (H_OR - 1)
    w_in_f = w_norm * (W_OR - 1)

    d0_full = tl.floor(d_in_f).to(tl.int32)
    h0_full = tl.floor(h_in_f).to(tl.int32)
    w0_full = tl.floor(w_in_f).to(tl.int32)

    d1_full = tl.minimum(d0_full + 1, D_OR - 1)
    h1_full = tl.minimum(h0_full + 1, H_OR - 1)
    w1_full = tl.minimum(w0_full + 1, W_OR - 1)

    d0 = tl.maximum(tl.minimum(d0_full - DS, D_IN - 1), 0)
    h0 = tl.maximum(tl.minimum(h0_full - HS, H_IN - 1), 0)
    w0 = tl.maximum(tl.minimum(w0_full - WS, W_IN - 1), 0)
    d1 = tl.maximum(tl.minimum(d1_full - DS, D_IN - 1), 0)
    h1 = tl.maximum(tl.minimum(h1_full - HS, H_IN - 1), 0)
    w1 = tl.maximum(tl.minimum(w1_full - WS, W_IN - 1), 0)

    dd = (d_in_f - d0_full)
    hh = (h_in_f - h0_full)
    ww = (w_in_f - w0_full)

    base_in = (b * STRIDE_IN_N + c * STRIDE_IN_C)
    base_out = (b * STRIDE_OUT_N + c * STRIDE_OUT_C +
                d_out * STRIDE_OUT_D + h_out * STRIDE_OUT_H)

    off_d0h0w0 = base_in + d0 * STRIDE_IN_D + h0 * STRIDE_IN_H + w0 * STRIDE_IN_W
    off_d0h0w1 = base_in + d0 * STRIDE_IN_D + h0 * STRIDE_IN_H + w1 * STRIDE_IN_W
    off_d0h1w0 = base_in + d0 * STRIDE_IN_D + h1 * STRIDE_IN_H + w0 * STRIDE_IN_W
    off_d0h1w1 = base_in + d0 * STRIDE_IN_D + h1 * STRIDE_IN_H + w1 * STRIDE_IN_W
    off_d1h0w0 = base_in + d1 * STRIDE_IN_D + h0 * STRIDE_IN_H + w0 * STRIDE_IN_W
    off_d1h0w1 = base_in + d1 * STRIDE_IN_D + h0 * STRIDE_IN_H + w1 * STRIDE_IN_W
    off_d1h1w0 = base_in + d1 * STRIDE_IN_D + h1 * STRIDE_IN_H + w0 * STRIDE_IN_W
    off_d1h1w1 = base_in + d1 * STRIDE_IN_D + h1 * STRIDE_IN_H + w1 * STRIDE_IN_W

    c000 = tl.load(IN_PTR + off_d0h0w0, mask=mask_w, other=0.0)
    c001 = tl.load(IN_PTR + off_d0h0w1, mask=mask_w, other=0.0)
    c010 = tl.load(IN_PTR + off_d0h1w0, mask=mask_w, other=0.0)
    c011 = tl.load(IN_PTR + off_d0h1w1, mask=mask_w, other=0.0)
    c100 = tl.load(IN_PTR + off_d1h0w0, mask=mask_w, other=0.0)
    c101 = tl.load(IN_PTR + off_d1h0w1, mask=mask_w, other=0.0)
    c110 = tl.load(IN_PTR + off_d1h1w0, mask=mask_w, other=0.0)
    c111 = tl.load(IN_PTR + off_d1h1w1, mask=mask_w, other=0.0)

    ww0 = 1.0 - ww
    ww1 = ww
    hh0 = 1.0 - hh
    hh1 = hh
    dd0 = 1.0 - dd
    dd1 = dd

    c00 = c000 * ww0 + c001 * ww1
    c01 = c010 * ww0 + c011 * ww1
    c10 = c100 * ww0 + c101 * ww1
    c11 = c110 * ww0 + c111 * ww1

    c0 = c00 * hh0 + c01 * hh1
    c1 = c10 * hh0 + c11 * hh1

    outv = c0 * dd0 + c1 * dd1

    tl.store(OUT_PTR + base_out + offs_w.to(tl.int64) * STRIDE_OUT_W, outv, mask=mask_w)


# ------------------------------
# Python wrapper
# ------------------------------
def trilinear_interpolate_align_corners_triton(
    input_tensor: torch.Tensor,
    orig_size: tuple[int, int, int],
    start: tuple[int, int, int],
    scale_factor: int = 2,
) -> torch.Tensor:
    """
    input_tensor: [B, C, D_in, H_in, W_in]，是大体素的一个子块
    orig_size:    (D_or, H_or, W_or) — 整个大体素的原始尺寸
    start:        (ds, hs, ws) — 该子块在原始体素中的起始索引（以 D/H/W 计）
    scale_factor: 整体上采样倍率（通常是 2）
    返回：子块对应的上采样输出 [B, C, D_in*SF, H_in*SF, W_in*SF]
    """
    assert input_tensor.is_cuda, "CUDA tensor required"
    assert input_tensor.ndim == 5
    # if not input_tensor.is_contiguous():
    input_tensor = input_tensor.contiguous()
    B, C, D_in, H_in, W_in = input_tensor.shape
    D_or, H_or, W_or = map(int, orig_size)
    ds, hs, ws = map(int, start)
    SF = int(scale_factor)

    # 输出尺寸
    D_out, H_out, W_out = D_in * SF, H_in * SF, W_in * SF
    out = _alloc_like_out(input_tensor, D_out, H_out, W_out)

    # strides（支持任意 memory_format）
    sN, sC, sD, sH, sW = input_tensor.stride()
    soN, soC, soD, soH, soW = out.stride()

    # grid：沿 W 做向量化，其余(B,C,D_out,H_out) 打包到第二维
    BLOCK_W = 128
    grid = (triton.cdiv(W_out, BLOCK_W), D_out * H_out, B * C)
    num_warps = 4
    num_stages = 3

    if input_tensor.dtype in (torch.float16, torch.bfloat16):
        _tri3d_align_fp16_kernel[grid](
            input_tensor, out,
            B, C, D_in, H_in, W_in,
            D_out, H_out, W_out,
            D_or, H_or, W_or,
            ds, hs, ws, SF,
            sN, sC, sD, sH, sW,
            soN, soC, soD, soH, soW,
            BLOCK_W=BLOCK_W,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    elif input_tensor.dtype == torch.float32:
        _tri3d_align_fp32_kernel[grid](
            input_tensor, out,
            B, C, D_in, H_in, W_in,
            D_out, H_out, W_out,
            D_or, H_or, W_or,
            ds, hs, ws, SF,
            sN, sC, sD, sH, sW,
            soN, soC, soD, soH, soW,
            BLOCK_W=BLOCK_W,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        raise TypeError(f"Unsupported dtype: {input_tensor.dtype}")

    return out
