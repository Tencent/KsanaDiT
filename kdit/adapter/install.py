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

import argparse
import shutil
from pathlib import Path

_DEFAULT_COMFYUI_ROOT = "/data/ComfyUI"
_TARGET_NODE_DIR = "custom_nodes/ComfyUI-kDiT"
_TEMPLATE_NAME = "_init_comfyui_nodes.py.in"
_TARGET_NAME = "__init__.py"


def main():
    parser = argparse.ArgumentParser(description="安装 kDiT ComfyUI 节点适配器")
    parser.add_argument(
        "--comfyui-root",
        default=None,
        help=f"ComfyUI 根目录路径，省略则交互式询问 (默认: {_DEFAULT_COMFYUI_ROOT})",
    )
    parser.add_argument(
        "--whl-url",
        default=None,
        help="安装包的 URL 地址，写入 version 文件用于版本追踪",
    )
    args = parser.parse_args()

    adapter_dir = Path(__file__).resolve().parent
    template_src = adapter_dir / _TEMPLATE_NAME

    if not template_src.exists():
        print(f"[error] 模板文件不存在: {template_src}")
        return

    if args.comfyui_root is not None:
        comfyui_root = Path(args.comfyui_root)
    else:
        raw = input(f"请输入 ComfyUI 根目录 [{_DEFAULT_COMFYUI_ROOT}]: ").strip()
        comfyui_root = Path(raw) if raw else Path(_DEFAULT_COMFYUI_ROOT)

    if not comfyui_root.exists():
        print(f"[error] 目录不存在: {comfyui_root}")
        return

    target_dir = comfyui_root / _TARGET_NODE_DIR
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"[info] 目标目录已有内容，正在清理: {target_dir}")
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    target_file = target_dir / _TARGET_NAME
    shutil.copy2(template_src, target_file)

    if args.whl_url:
        version_file = target_dir / "version"
        version_file.write_text(args.whl_url + "\n")
        print(f"[ok] 已写入版本信息: {version_file}")

    print(f"[ok] 已完成 {target_dir}")


if __name__ == "__main__":
    main()
