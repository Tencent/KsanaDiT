#!/bin/bash

# 默认安装类型为 gpu
INSTALL_TYPE="${1:-gpu}"

# 验证入参
if [[ ! "$INSTALL_TYPE" =~ ^(gpu|npu)$ ]]; then
    echo "错误: 无效的安装类型 '$INSTALL_TYPE'"
    echo "用法: $0 [gpu|npu]"
    echo "  gpu - 安装 GPU 版本 (默认)"
    echo "  npu - 安装 NPU 版本"
    exit 1
fi

echo "正在卸载现有的 ksana-dit..."
pip uninstall -y ksana-dit

echo "当前目录: $(pwd)"
echo "正在安装 ksana-dit[$INSTALL_TYPE]..."
pip install -e ".[$INSTALL_TYPE]"

echo "安装完成: ksana-dit[$INSTALL_TYPE]"
