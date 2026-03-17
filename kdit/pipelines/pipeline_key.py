# Copyright 2026 Tencent
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
from __future__ import annotations

from enum import Enum, auto, unique
from pathlib import Path


@unique
class PipelineKey(Enum):
    """标识一条完整的推理流水线（Pipeline 级别语义）。

    零依赖枚举 — 可被任意子包安全导入，不会触发 kdit/__init__.py 的重量级导入链。
    """

    Wan2_2_T2V_14B = auto()
    Wan2_2_I2V_14B = auto()
    Wan2_2_TI2V_5B = auto()
    Wan2_1_VACE_14B = auto()
    QwenImage_T2I = auto()
    QwenImage_Edit = auto()

    # TODO: ToBe remove
    def is_i2v_type(self) -> bool:
        return self in (PipelineKey.Wan2_2_I2V_14B,)

    def is_image_type(self) -> bool:
        return self in (PipelineKey.QwenImage_T2I, PipelineKey.QwenImage_Edit)


def get_pipeline_key_from_path(model_path: str | list[str]) -> PipelineKey:
    """从 diffusion 模型路径推导 PipelineKey。

    内部复用 get_model_key_from_path() 得到 diffusion ModelKey，
    再通过同名映射转换为 PipelineKey。
    """
    from kdit.models.model_key import DIFFUSION_KEYS, get_model_key_from_path

    model_key = get_model_key_from_path(model_path)
    if model_key not in DIFFUSION_KEYS:
        raise ValueError(
            f"get_pipeline_key_from_path() expects a diffusion model path, "
            f"but got ModelKey.{model_key.name} which is not a diffusion key."
        )
    pipeline_key = PipelineKey[model_key.name]
    _validate_pipeline_dir(pipeline_key, model_path)
    return pipeline_key


# ── 目录完整性校验 ─────────────────────────────────────────────────────


def _resolve_base_dir(model_path: str | list[str]) -> Path:
    """从 model_path 解析出用于校验的根目录。list 时取第一个元素。"""
    raw = model_path[0] if isinstance(model_path, (list, tuple)) else model_path
    return Path(raw)


def _check_non_empty_dir(base: Path, subdir: str) -> str | None:
    """检查子目录存在且非空，返回错误描述或 None。"""
    d = base / subdir
    if not d.is_dir():
        return f"目录不存在: {subdir}"
    if not any(d.iterdir()):
        return f"目录为空: {subdir}"
    return None


def _check_file_exists(base: Path, filename: str) -> str | None:
    """检查文件存在，返回错误描述或 None。"""
    if not (base / filename).is_file():
        return f"文件不存在: {filename}"
    return None


def _check_has_safetensors(base: Path) -> str | None:
    """检查目录下至少有一个 .safetensors 文件，返回错误描述或 None。"""
    for entry in base.iterdir():
        if entry.is_file() and entry.suffix == ".safetensors":
            return None
    return "当前目录下没有 .safetensors 文件"


def _validate_pipeline_dir(pipeline_key: PipelineKey, model_path: str | list[str]) -> None:
    """根据 PipelineKey 校验模型目录的完整性。

    校验失败时抛出 ValueError，消息中列出所有缺失的文件/目录。
    """
    base = _resolve_base_dir(model_path)
    errors: list[str] = []

    if pipeline_key in (PipelineKey.Wan2_2_T2V_14B, PipelineKey.Wan2_2_I2V_14B):
        for subdir in ("google/umt5-xxl", "high_noise_model", "low_noise_model"):
            err = _check_non_empty_dir(base, subdir)
            if err:
                errors.append(err)
        for filename in ("Wan2.1_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth"):
            err = _check_file_exists(base, filename)
            if err:
                errors.append(err)

    elif pipeline_key == PipelineKey.Wan2_1_VACE_14B:
        err = _check_non_empty_dir(base, "google/umt5-xxl")
        if err:
            errors.append(err)
        for filename in ("Wan2.1_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth"):
            err = _check_file_exists(base, filename)
            if err:
                errors.append(err)
        err = _check_has_safetensors(base)
        if err:
            errors.append(err)

    elif pipeline_key == PipelineKey.QwenImage_T2I:
        for subdir in ("text_encoder", "tokenizer", "transformer", "vae"):
            err = _check_non_empty_dir(base, subdir)
            if err:
                errors.append(err)

    elif pipeline_key == PipelineKey.QwenImage_Edit:
        for subdir in ("text_encoder", "tokenizer", "transformer", "vae", "processor"):
            err = _check_non_empty_dir(base, subdir)
            if err:
                errors.append(err)

    if errors:
        detail = "\n  - ".join(errors)
        raise ValueError(f"Pipeline {pipeline_key.name} 的模型目录 {base} 不完整，缺少以下内容:\n  - {detail}")
