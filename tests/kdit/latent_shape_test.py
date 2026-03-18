# Copyright 2025 Tencent
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""latent_shape 模块单元测试。

验证 compute_latent_shape / compute_video_latent_shape / compute_image_latent_shape
的正确性，以及 image 场景下 stride 方式与旧 scale_factor 方式结果一致。
"""

import pytest

from kdit.models.latent_shape import (
    _normalize_to_3d,
    compute_image_latent_shape,
    compute_latent_shape,
    compute_video_latent_shape,
)

# ── _normalize_to_3d ─────────────────────────────────────────────────────


class TestNormalizeTo3d:
    def test_scalar(self):
        assert _normalize_to_3d(8, "test") == [1, 8, 8]

    def test_list_2(self):
        assert _normalize_to_3d([4, 8], "test") == [1, 4, 8]

    def test_list_3(self):
        assert _normalize_to_3d([4, 8, 8], "test") == [4, 8, 8]

    def test_float_scalar(self):
        assert _normalize_to_3d(2.0, "test") == [1, 2, 2]

    def test_invalid_length(self):
        with pytest.raises(ValueError, match="must be int or list of length 2 or 3"):
            _normalize_to_3d([1, 2, 3, 4], "test")


# ── compute_latent_shape 基础函数 ────────────────────────────────────────


class TestComputeLatentShape:
    """测试统一基础函数 compute_latent_shape。"""

    def test_basic_video(self):
        """Wan 视频场景: stride=[4,8,8], patch=[1,2,2], 81帧 1280x720。"""
        z_dim, lat_f, lat_h, lat_w = compute_latent_shape(
            z_dim=16,
            target_f=81,
            target_h=720,
            target_w=1280,
            vae_stride=[4, 8, 8],
            patch_size=[1, 2, 2],
        )
        assert z_dim == 16
        assert lat_f == (81 - 1) // 4 + 1  # 21
        # lat_h = round(sqrt(1280*720*(720/1280)) // 8 // 2 * 2)
        # lat_w = round(sqrt(1280*720*(1280/720)) // 8 // 2 * 2)
        assert lat_h > 0
        assert lat_w > 0

    def test_basic_image(self):
        """Qwen 图像场景: stride=[1,8,8], patch=2, 1024x1024。"""
        z_dim, lat_f, lat_h, lat_w = compute_latent_shape(
            z_dim=16,
            target_f=1,
            target_h=1024,
            target_w=1024,
            vae_stride=[1, 8, 8],
            patch_size=2,
        )
        assert z_dim == 16
        assert lat_f == 1
        # 1024 // 8 // 2 * 2 = 128
        assert lat_h == 128
        assert lat_w == 128

    def test_vae_stride_must_be_3_elements(self):
        with pytest.raises(ValueError, match="vae_stride must be a list of 3 elements"):
            compute_latent_shape(z_dim=16, target_f=1, target_h=512, target_w=512, vae_stride=[8, 8], patch_size=2)

    def test_img_shape_validation(self):
        with pytest.raises(ValueError, match="img_shape must be 4D"):
            compute_latent_shape(
                z_dim=16,
                target_f=81,
                target_h=720,
                target_w=1280,
                vae_stride=[4, 8, 8],
                patch_size=[1, 2, 2],
                img_shape=[1, 720, 1280],
            )

    def test_with_img_shape(self):
        """提供 img_shape 时按图片宽高比修正。"""
        # 正方形目标，但参考图是 2:1 宽高比
        _, _, lat_h_sq, lat_w_sq = compute_latent_shape(
            z_dim=16,
            target_f=81,
            target_h=720,
            target_w=720,
            vae_stride=[4, 8, 8],
            patch_size=[1, 2, 2],
        )
        _, _, lat_h_ar, lat_w_ar = compute_latent_shape(
            z_dim=16,
            target_f=81,
            target_h=720,
            target_w=720,
            vae_stride=[4, 8, 8],
            patch_size=[1, 2, 2],
            refer_image_shape=[1, 3, 360, 720],  # 宽高比 2:1
        )
        # 有 img_shape 时 lat_w 应该比 lat_h 大
        assert lat_w_ar > lat_h_ar
        # 无 img_shape 时正方形
        assert lat_h_sq == lat_w_sq


# ── compute_video_latent_shape ───────────────────────────────────────────


class TestComputeVideoLatentShape:
    """测试视频便捷函数，确保与 compute_latent_shape 一致。"""

    def test_delegates_to_base(self):
        """compute_video_latent_shape 应与 compute_latent_shape 结果完全一致。"""
        args = dict(
            z_dim=16,
            target_f=81,
            target_h=720,
            target_w=1280,
            vae_stride=[4, 8, 8],
        )
        result_video = compute_video_latent_shape(**args, vae_patch=[1, 2, 2])
        result_base = compute_latent_shape(**args, patch_size=[1, 2, 2])
        assert result_video == result_base

    def test_wan_standard(self):
        """Wan 标准配置: stride=[4,8,8], patch=[1,2,2], 81帧 1280x720。"""
        z_dim, lat_f, _, _ = compute_video_latent_shape(
            z_dim=16,
            target_f=81,
            target_h=720,
            target_w=1280,
            vae_stride=[4, 8, 8],
            vae_patch=[1, 2, 2],
        )
        assert z_dim == 16
        assert lat_f == 21


# ── compute_image_latent_shape ───────────────────────────────────────────


class TestComputeImageLatentShape:
    """测试图像便捷函数。"""

    def test_delegates_to_base(self):
        """compute_image_latent_shape 应与 compute_latent_shape(target_f=1) 结果完全一致。"""
        result_image = compute_image_latent_shape(
            z_dim=16,
            target_h=1024,
            target_w=1024,
            vae_stride=[1, 8, 8],
            patch_size=2,
        )
        result_base = compute_latent_shape(
            z_dim=16,
            target_f=1,
            target_h=1024,
            target_w=1024,
            vae_stride=[1, 8, 8],
            patch_size=2,
        )
        assert result_image == result_base

    def test_lat_f_always_1(self):
        """图像场景 lat_f 始终为 1。"""
        _, lat_f, _, _ = compute_image_latent_shape(
            z_dim=16, target_h=512, target_w=512, vae_stride=[1, 8, 8], patch_size=2
        )
        assert lat_f == 1

    def test_qwen_1024x1024(self):
        """Qwen 标准配置: stride=[1,8,8], patch=2, 1024x1024。"""
        z_dim, lat_f, lat_h, lat_w = compute_image_latent_shape(
            z_dim=16,
            target_h=1024,
            target_w=1024,
            vae_stride=[1, 8, 8],
            patch_size=2,
        )
        assert (z_dim, lat_f, lat_h, lat_w) == (16, 1, 128, 128)

    def test_qwen_non_square(self):
        """Qwen 非正方形: stride=[1,8,8], patch=2, 768x1024。"""
        z_dim, lat_f, lat_h, lat_w = compute_image_latent_shape(
            z_dim=16,
            target_h=768,
            target_w=1024,
            vae_stride=[1, 8, 8],
            patch_size=2,
        )
        # lat_h = round(sqrt(1024*768*(768/1024)) // 8 // 2 * 2)
        # = round(sqrt(1024*768*0.75) // 8 // 2 * 2)
        # = round(768 // 8 // 2 * 2) = round(96 // 2 * 2) = 96
        # lat_w = round(sqrt(1024*768*(1024/768)) // 8 // 2 * 2)
        # = round(1024 // 8 // 2 * 2) = round(128 // 2 * 2) = 128
        assert (z_dim, lat_f) == (16, 1)
        assert lat_h == 96
        assert lat_w == 128


# ── scale_factor 兼容性验证 ──────────────────────────────────────────────


def _old_compute_image_latent_shape(
    z_dim: int,
    target_h: int,
    target_w: int,
    vae_scale_factor: int,
    patch_size: int,
) -> tuple[int, int, int, int]:
    """旧版 compute_image_latent_shape 实现（基于 scale_factor），用于对比验证。"""
    multiple_of = vae_scale_factor * patch_size
    lat_h = target_h // multiple_of * patch_size
    lat_w = target_w // multiple_of * patch_size
    return z_dim, 1, lat_h, lat_w


class TestScaleFactorCompatibility:
    """验证新的 stride 方式与旧的 scale_factor 方式在图像场景下结果一致。

    Qwen VAE 配置 (kdit/settings/qwen/modules/vae/vae.yaml):
      - stride: [1, 8, 8]
      - scale_factor: 8   (即 stride[1] == stride[2] == scale_factor)
      - patch_size: 2
    """

    # Qwen VAE 配置值
    QWEN_STRIDE = [1, 8, 8]
    QWEN_SCALE_FACTOR = 8  # == stride[1] == stride[2]
    QWEN_PATCH_SIZE = 2

    @pytest.mark.parametrize(
        "target_h, target_w",
        [
            (1024, 1024),
            (768, 1024),
            (1024, 768),
            (512, 512),
            (512, 1024),
            (1024, 512),
            (2048, 2048),
            (1536, 1024),
            (1024, 1536),
            (768, 768),
            (256, 256),
            (1280, 720),
            (720, 1280),
        ],
    )
    def test_stride_matches_scale_factor(self, target_h: int, target_w: int):
        """对多种分辨率验证: stride 方式 == 旧 scale_factor 方式。"""
        old_result = _old_compute_image_latent_shape(
            z_dim=16,
            target_h=target_h,
            target_w=target_w,
            vae_scale_factor=self.QWEN_SCALE_FACTOR,
            patch_size=self.QWEN_PATCH_SIZE,
        )
        new_result = compute_image_latent_shape(
            z_dim=16,
            target_h=target_h,
            target_w=target_w,
            vae_stride=self.QWEN_STRIDE,
            patch_size=self.QWEN_PATCH_SIZE,
        )
        assert new_result == old_result, (
            f"Mismatch at {target_h}x{target_w}: " f"old(scale_factor)={old_result}, new(stride)={new_result}"
        )

    def test_image_is_special_case_of_video(self):
        """验证 image latent shape 是 video latent shape 在 target_f=1 时的特殊情况。"""
        image_result = compute_image_latent_shape(
            z_dim=16,
            target_h=1024,
            target_w=1024,
            vae_stride=self.QWEN_STRIDE,
            patch_size=self.QWEN_PATCH_SIZE,
        )
        video_result = compute_video_latent_shape(
            z_dim=16,
            target_f=1,
            target_h=1024,
            target_w=1024,
            vae_stride=self.QWEN_STRIDE,
            vae_patch=[1, self.QWEN_PATCH_SIZE, self.QWEN_PATCH_SIZE],
        )
        assert image_result == video_result

    @pytest.mark.parametrize(
        "target_h, target_w",
        [
            (1024, 1024),
            (768, 1024),
            (512, 512),
            (1280, 720),
        ],
    )
    def test_image_equals_video_f1(self, target_h: int, target_w: int):
        """多种分辨率下验证 image == video(f=1)。"""
        image_result = compute_image_latent_shape(
            z_dim=16,
            target_h=target_h,
            target_w=target_w,
            vae_stride=self.QWEN_STRIDE,
            patch_size=self.QWEN_PATCH_SIZE,
        )
        video_result = compute_video_latent_shape(
            z_dim=16,
            target_f=1,
            target_h=target_h,
            target_w=target_w,
            vae_stride=self.QWEN_STRIDE,
            vae_patch=[1, self.QWEN_PATCH_SIZE, self.QWEN_PATCH_SIZE],
        )
        assert (
            image_result == video_result
        ), f"Mismatch at {target_h}x{target_w}: image={image_result}, video(f=1)={video_result}"
