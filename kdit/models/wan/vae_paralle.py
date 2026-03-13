import types

import torch
import torch.distributed as dist
import torch.nn.functional as F

# ==================== VAE Tiling 并行核心概念 ====================
# tile_w / tile_h:   每个 GPU 处理的图像块大小（像素空间），包含重叠区域
# stride_w / stride_h: 相邻 tile 起始位置之间的步长（像素空间），即每个 tile 的"独占区域"大小
# overlap_w / overlap_h: tile 与相邻 tile 的重叠区域大小（像素空间），= tile - stride
#                        重叠区域用于 blend 线性插值，消除拼接缝隙
# 示意图（水平方向）：
# |<-------- tile_w -------->|
# |<-- stride_w -->|<-overlap_w->|
#                  |<-------- tile_w -------->|
# =============================================================


###################################################################################################################
# 用于横向、纵向拼接 vae decode 多个 tiled 的图片
def _blend_h(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
    for x in range(blend_extent):
        b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
            x / blend_extent
        )
    return b


def _blend_v(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
    for y in range(blend_extent):
        b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
            y / blend_extent
        )
    return b


def _blend_rows(rows, overlap_w, overlap_h, stride_w, stride_h):
    result_rows = []
    for i, row in enumerate(rows):
        result_row = []
        for j in range(len(row)):
            # blend the above tile and the left tile
            if i > 0:
                rows[i][j] = _blend_v(rows[i - 1][j], rows[i][j], overlap_h)
            if j > 0:
                rows[i][j] = _blend_h(rows[i][j - 1], rows[i][j], overlap_w)

            _, _, _, h, w = rows[i][j].shape  # [1, 3, 49, 240, 360])
            crop_h = stride_h if i + 1 < len(rows) else h
            crop_w = stride_w if j + 1 < len(row) else w
            result_row.append(rows[i][j][:, :, :, :crop_h, :crop_w])
        result_rows.append(torch.cat(result_row, dim=-1))

    return torch.cat(result_rows, dim=3)


###################################################################################################################
def _tiled_parallel_op(self, data, op_fn, *args, is_encode, **kwargs):
    """通用的 tiled 并行操作框架。
    Args:
        data: 输入张量 [B, C, T, H, W]
        op_fn: 原始的 encode/decode 方法（self._original_encode 或 self._original_decode）
        *args: 透传给 op_fn 的位置参数
        is_encode: (keyword-only) True=encode(像素→latent), False=decode(latent→像素)
        **kwargs: 透传给 op_fn 的关键字参数
    """
    assert len(data.shape) == 5, f"data dim must 5. {data.shape}"
    rank, world_size = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
    if world_size <= 1:
        return op_fn(data, *args, **kwargs)

    r = self.spatial_compression_ratio  # pixel_per_latent 缩写
    _, _, _, height, width = data.shape

    # 1. 计算 tiling 参数
    pixel_h, pixel_w = (height, width) if is_encode else (height * r, width * r)
    tile_w, tile_h, stride_w, stride_h = compute_tiling_size(
        imgw=pixel_w,
        imgh=pixel_h,
        pixels_per_latent=r,
        world_size=world_size,
    )
    overlap_w, overlap_h = tile_w - stride_w, tile_h - stride_h

    if is_encode:
        # encode: 像素空间切分，latent 空间 blend
        iter_tile_w, iter_tile_h = tile_w, tile_h
        iter_stride_w, iter_stride_h = stride_w, stride_h
        blend_args = (overlap_w // r, overlap_h // r, stride_w // r, stride_h // r)
        output_h, output_w = height // r, width // r
    else:
        # decode: latent 空间切分，像素空间 blend
        iter_tile_w, iter_tile_h = tile_w // r, tile_h // r
        iter_stride_w, iter_stride_h = stride_w // r, stride_h // r
        blend_args = (overlap_w, overlap_h, stride_w, stride_h)
        output_h, output_w = pixel_h, pixel_w

    # 2. 切分 tile 并分配给各 GPU 执行
    parallel_idx = 0
    rows = []
    for i in range(0, height, iter_stride_h):
        row = []
        for j in range(0, width, iter_stride_w):
            tile = None
            if parallel_idx % world_size == rank:
                tile = data[:, :, :, i : i + iter_tile_h, j : j + iter_tile_w]
                _, _, _, rh, rw = tile.shape
                if rh < iter_tile_h or rw < iter_tile_w:
                    pads = (0, iter_tile_w - rw, 0, iter_tile_h - rh)
                    tile = F.pad(tile, pads, "constant", 0)
                tile = op_fn(tile, *args, **kwargs).contiguous()
            row.append(tile)
            parallel_idx += 1
            if j + iter_tile_w >= width:
                break  # 避免最后一片重复计算
        rows.append(row)
        if i + iter_tile_h >= height:
            break  # 避免最后一片重复计算
    assert parallel_idx == world_size, f"VAE并行数 {parallel_idx} vs {world_size}"

    # 3. all_gather: 每个 GPU 广播自己的 tile 给所有其他 GPU
    dist.barrier()
    nodes_per_row = len(rows[0])
    rank_row, rank_col = rank // nodes_per_row, rank % nodes_per_row
    local_tile = rows[rank_row][rank_col]
    gather_list = [torch.zeros_like(local_tile) for _ in range(world_size)]
    dist.all_gather(gather_list, local_tile)

    for i, v in enumerate(gather_list):
        if i != rank:
            rows[i // nodes_per_row][i % nodes_per_row] = v

    # 4. blend 拼接并裁剪到目标尺寸
    result = _blend_rows(rows, *blend_args)
    return result[:, :, :, :output_h, :output_w]


@torch.no_grad()
def decode_parallel(self, z: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """Decode a batch of latent vectors, with multi-GPU spatial tiling parallelism."""
    return _tiled_parallel_op(self, z, self._original_decode, *args, **{**kwargs, "is_encode": False})


@torch.no_grad()
def encode_parallel(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """Encode a batch of videos, with multi-GPU spatial tiling parallelism."""
    return _tiled_parallel_op(self, x, self._original_encode, *args, **{**kwargs, "is_encode": True})


###################################################################################################################
def _get_stride_factor(imgw, imgh):
    """根据图片分辨率选择基础 tile/stride 配置，推算 stride_factor。
    Returns:
        stride_factor: stride 占 tile 的比例（即每个 tile 中"独占区域"占比）
    """
    if max(imgw, imgh) >= 756:
        base_tile, base_stride = 320, 256
    else:
        base_tile, base_stride = 288, 224
    return base_stride / base_tile  # 320/256=0.8, 288/224≈0.778


def _compute_grid_factors(world_size, imgw, imgh, stride_factor):
    """根据 GPU 数量计算宽高方向的 tile 缩放因子。
    Returns:
        (factor_w, factor_h): 宽/高方向的缩放因子，tile_size = img_size / factor
    """
    if world_size < 4:  # 2/3 只沿长边切
        factor = stride_factor * (world_size - 1) + 1
        return (factor, 1) if imgw > imgh else (1, factor)
    elif world_size <= 6:  # 4/6: 短边切2份，剩下给长边
        factor_short = stride_factor * (2 - 1) + 1
        factor_long = stride_factor * (world_size // 2 - 1) + 1
        return (factor_long, factor_short) if imgw > imgh else (factor_short, factor_long)
    else:  # 8/12/16: 长边切4份，剩下给短边
        assert world_size % 4 == 0 and world_size <= 16, f"需要为4的倍数且<=16, world_size={world_size}"
        factor_short = stride_factor * (world_size // 4 - 1) + 1
        factor_long = stride_factor * (4 - 1) + 1
        return (factor_long, factor_short) if imgw > imgh else (factor_short, factor_long)


def _align_stride(tile, stride_factor, tiling_min):
    """对齐 stride 到 tiling_min 的倍数，并 clamp overlap 到 [16, 32] 范围。"""
    stride = int(tile * stride_factor + tiling_min - 1) // tiling_min * tiling_min
    overlap = tile - stride
    if overlap > 32:
        stride = tile - 32
    elif overlap < 16:
        stride = tile - 16
    return stride


def compute_tiling_size(vae=None, imgw=280, imgh=280, world_size=0, pixels_per_latent=None):
    """根据图片大小、world_size自动设置tiling大小。
    pixels_per_latent: 像素压缩比 默认8x8；ltx/wan2.2 为16 - 需传入；通常传入 vae.spatial_compression_ratio.
    """
    if world_size is None or world_size == 0:
        world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size <= 1 or max(imgw, imgh) < 160:
        if vae is not None:
            vae.disable_tiling()
        return None, None, None, None

    stride_factor = _get_stride_factor(imgw, imgh)
    tiling_min = 16  # tile/stride 对齐到 16 的倍数

    if pixels_per_latent is None:
        pixels_per_latent = 8 if vae is None else vae.spatial_compression_ratio
    assert tiling_min % pixels_per_latent == 0, f"Tile = {tiling_min} % {pixels_per_latent} == 0"

    # 根据 world_size 反推 tile 大小，使切出的 tile 数量刚好等于 GPU 数量
    factor_w, factor_h = _compute_grid_factors(world_size, imgw, imgh, stride_factor)
    tile_w = int(imgw / factor_w + tiling_min - 1) // tiling_min * tiling_min
    tile_h = int(imgh / factor_h + tiling_min - 1) // tiling_min * tiling_min

    stride_w = _align_stride(tile_w, stride_factor, tiling_min)
    stride_h = _align_stride(tile_h, stride_factor, tiling_min)
    assert (tile_w > stride_w > 0) and (tile_h > stride_h > 0)

    if vae is not None:
        vae.enable_tiling(
            tile_sample_min_width=tile_w,
            tile_sample_min_height=tile_h,
            tile_sample_stride_width=stride_w,
            tile_sample_stride_height=stride_h,
        )
    return tile_w, tile_h, stride_w, stride_h


###################################################################################################################
def patch_vae_parallel(vae_instance):
    """给 VAE 实例注入并行 encode/decode 方法（monkey patch）。
    保存原始方法为 _original_encode/_original_decode，替换为并行版本。
    """
    if getattr(vae_instance, "_is_parallel_patched", False):
        return
    vae_instance._original_encode = vae_instance.encode
    vae_instance._original_decode = vae_instance.decode
    vae_instance.encode = types.MethodType(encode_parallel, vae_instance)
    vae_instance.decode = types.MethodType(decode_parallel, vae_instance)
    vae_instance._is_parallel_patched = True
