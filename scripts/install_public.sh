#!/bin/bash
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

# ============================================================
# kDiT 安装脚本
#
# 用法:  ./install_public.sh
#
# 脚本会自动检测硬件类型，并交互式询问安装方式。
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---- 硬件检测 ----
detect_device() {
    if command -v nvidia-smi &>/dev/null; then
        echo "gpu"
    elif command -v npu-smi &>/dev/null; then
        echo "npu"
    elif command -v xpu-smi &>/dev/null; then
        echo "xpu"
    else
        echo ""
    fi
}

INSTALL_TYPE=$(detect_device)
if [[ -z "$INSTALL_TYPE" ]]; then
    echo "错误: 无法自动检测硬件类型 (gpu/npu/xpu)"
    exit 1
fi
echo "自动检测到硬件类型: $INSTALL_TYPE"

# ---- 判断当前目录是否为项目源码 ----
INSTALL_MODE="whl"
if [[ -d "$PROJECT_ROOT/kdit" && -f "$PROJECT_ROOT/pyproject.toml" ]]; then
    echo "检测到当前目录下存在 kdit 项目源码和 pyproject.toml"
    while true; do
        read -rp "是否以 editable 的开发模式安装当前目录下的代码? (y/n): " answer
        case "$answer" in
            y) INSTALL_MODE="dev"; break ;;
            n) INSTALL_MODE="whl"; break ;;
            *) echo "请输入 y 或 n" ;;
        esac
    done
fi

echo "=========================================="
echo "  安装模式:      $INSTALL_MODE"
echo "  硬件类型:      $INSTALL_TYPE"
echo "=========================================="

# ---- 卸载旧版本 ----
echo "正在卸载现有的 kDiT..."
pip uninstall -y kDiT 2>/dev/null || true

# ---- 安装 ----
if [[ "$INSTALL_MODE" == "dev" ]]; then
    echo "正在以开发模式安装 ${PROJECT_ROOT}[$INSTALL_TYPE] (当前代码)..."
    pip install -e "${PROJECT_ROOT}[$INSTALL_TYPE]"
else
    echo "正在安装 kDiT[$INSTALL_TYPE] (发布版)..."
    pip install "kDiT[$INSTALL_TYPE]"
fi

echo "安装完成: kDiT[$INSTALL_TYPE]"

# ---- 询问是否安装 ComfyUI 适配器 ----
while true; do
    read -rp "是否安装 ComfyUI 节点适配器? (y/n): " comfy_answer
    case "$comfy_answer" in
        y) kdit_install_adapters; break ;;
        n) echo "跳过 ComfyUI 适配器安装"; break ;;
        *) echo "请输入 y 或 n" ;;
    esac
done
