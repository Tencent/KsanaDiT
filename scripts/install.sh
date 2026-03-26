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
# 用法:  ./install.sh [comfyui_path] [version]
#
#   参数1 (comfyui_path) : ComfyUI 安装路径
#                          指定路径 - 自动安装 ComfyUI 适配器到该路径
#                          省略或空 - 交互式询问是否安装适配器
#   参数2 (version)      : 手动指定安装版本号，例如 v0.2.3
#                          也可以传入以 .whl 结尾的 URL，直接安装该 whl 包
#                          省略则从七彩石灰度系统查询
#
# 示例:
#   ./install.sh                        # 全交互模式
#   ./install.sh /data/ComfyUI          # 自动安装适配器到 /data/ComfyUI，灰度查询版本
#   ./install.sh /data/ComfyUI v0.2.3   # 自动安装适配器到 /data/ComfyUI，指定版本
#   ./install.sh "" v0.2.3              # 指定版本，交互式询问是否安装适配器
#   ./install.sh "" https://example.com/kDiT-0.2.3.whl  # 直接安装指定 whl URL
#
# 脚本会自动检测硬件类型，并交互式询问安装方式。
# ============================================================

set -euo pipefail

# ---- 解析命令行参数 ----
ARG_COMFYUI_PATH="${1:-}"
ARG_VERSION="${2:-}"

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

# ---- 七彩石灰度查询函数 ----

# 向七彩石查询灰度配置，返回对应的 value
# 用法: query_rainbow "group_name" "key_value"
# 成功输出 value 并返回 0，失败返回 1
query_rainbow() {
    local group="$1"
    local key="$2"
    curl -s -H 'Content-Type: application/json' \
        -H "rainbow_sdk_version: http" \
        -d "{
            \"app_id\": \"v1_rb4_68fa4797-fd28-4329-86a9-d6e41\",
            \"group\": \"${group}\",
            \"env_name\": \"formal\",
            \"opts\":{\"key\":\"${key}\"}
        }" "http://api.rainbow.oa.com:8080/rainbowapi.configs/getdatas" 2>/dev/null \
        | python3 -c "import json,sys;d=json.load(sys.stdin);print(d['items'][0]['kvs']['kvs'][0]['value']) if d.get('ret_code')==0 and d.get('items') else exit(1)" 2>/dev/null
}

# 按灰度优先级查询版本号
# 优先级：手动指定 > 容器实例 > 用户 > 应用组 > 全局(default)
resolve_gray_version() {
    local version=""

    # 1. 容器实例
    if [[ -n "${ENV_APP_INS_NAME:-}" ]]; then
        echo "  尝试灰度查询 [容器实例]: ${ENV_APP_INS_NAME}" >&2
        version=$(query_rainbow "gray_instance" "$ENV_APP_INS_NAME") || true
        if [[ -n "$version" ]]; then
            echo "  命中灰度 [容器实例] -> $version" >&2
            echo "$version"
            return 0
        fi
    fi

    # 2. 用户名
    if [[ -n "${ENV_RTX:-}" ]]; then
        echo "  尝试灰度查询 [用户]: ${ENV_RTX}" >&2
        version=$(query_rainbow "gray_user" "$ENV_RTX") || true
        if [[ -n "$version" ]]; then
            echo "  命中灰度 [用户] -> $version" >&2
            echo "$version"
            return 0
        fi
    fi

    # 3. 应用组
    if [[ -n "${ENV_APP_GROUP_ID:-}" ]]; then
        echo "  尝试灰度查询 [应用组]: ${ENV_APP_GROUP_ID}" >&2
        version=$(query_rainbow "gray_app_group" "$ENV_APP_GROUP_ID") || true
        if [[ -n "$version" ]]; then
            echo "  命中灰度 [应用组] -> $version" >&2
            echo "$version"
            return 0
        fi
    fi

    # 4. 全局默认
    echo "  未命中任何灰度规则，使用默认版本" >&2
    echo "default"
}

# 根据版本号查询安装包 URL
# 用法: resolve_package_url "version_key"
resolve_package_url() {
    local key="$1"
    echo "  查询安装包 URL [package]: ${key}" >&2
    query_rainbow "package" "$key"
}

# ---- 安装 ----
if [[ "$INSTALL_MODE" == "dev" ]]; then
    echo "正在以开发模式安装 ${PROJECT_ROOT}[$INSTALL_TYPE] (当前代码)..."
    pip install -e "${PROJECT_ROOT}[$INSTALL_TYPE]"
else
    # 判断 ARG_VERSION 是 whl URL 还是版本号
    if [[ "$ARG_VERSION" == *.whl ]]; then
        # 直接使用传入的 whl URL，跳过七彩石查询
        echo "使用指定的 whl URL: $ARG_VERSION"
        WHL_URL="$ARG_VERSION"
    else
        # 从七彩石读取需要安装的版本链接
        # 判断灰度情况，优先级：手动指定>容器实例>用户>应用组>全局
        if [[ -n "$ARG_VERSION" ]]; then
            echo "使用手动指定版本: $ARG_VERSION"
            PACKAGE_KEY="$ARG_VERSION"
        else
            echo "正在从七彩石查询灰度版本..."
            PACKAGE_KEY=$(resolve_gray_version)
        fi

        echo "正在查询安装包下载地址 (key=${PACKAGE_KEY})..."
        WHL_URL=$(resolve_package_url "$PACKAGE_KEY")

        if [[ -z "$WHL_URL" ]]; then
            echo "错误: 无法从七彩石获取安装包 URL (key=${PACKAGE_KEY})"
            exit 1
        fi
    fi

    echo "正在下载安装包: $WHL_URL"
    WHL_FILE=$(basename "$WHL_URL")
    WHL_TMP="/tmp/${WHL_FILE}"
    wget -q -O "$WHL_TMP" "$WHL_URL"

    echo "正在安装 kDiT[$INSTALL_TYPE] 从: $WHL_TMP"
    pip install "${WHL_TMP}[$INSTALL_TYPE]"
    rm -f "$WHL_TMP"
fi

# GPU版本需要额外安装spas_sage_attn库
if [[ "$INSTALL_TYPE" == "gpu" ]]; then
    pip install http://mirrors.tencent.com/repository/generic/venus_repo/image_res/ksana_dit/deps/spas_sage_attn-0.1.0-cp310-cp310-linux_x86_64.whl
fi

echo "安装完成: kDiT[$INSTALL_TYPE]"

# ---- 安装 ComfyUI 适配器 ----
if [[ -n "$ARG_COMFYUI_PATH" ]]; then
    # 指定了 ComfyUI 路径：自动安装适配器到该路径
    echo "自动安装 ComfyUI 适配器到 $ARG_COMFYUI_PATH"
    kdit_install_adapters --comfyui-root "$ARG_COMFYUI_PATH" ${WHL_URL:+--whl-url "$WHL_URL"}
else
    # 交互式询问
    while true; do
        read -rp "是否安装 ComfyUI 节点适配器? (y/n): " comfy_answer
        case "$comfy_answer" in
            y) kdit_install_adapters ${WHL_URL:+--whl-url "$WHL_URL"}; break ;;
            n) echo "跳过 ComfyUI 适配器安装"; break ;;
            *) echo "请输入 y 或 n" ;;
        esac
    done
fi
