#!/usr/bin/env python3
"""
ComfyUI Workflow 测试

使用方法：
    # 默认参数
    python flow_tests.py

    # 自定义参数
    python flow_tests.py --path wan2.2_fp16.json --steps 5

    # 使用已运行的 server
    python flow_tests.py --no-server
"""

import argparse
import json
import logging
import time
import urllib.error

from test_utils import (
    start_server,
    stop_server,
    load_workflow,
    connect_websocket,
    submit_workflow,
    wait_for_completion,
)

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def modify_workflow_params(api_prompt: dict, args) -> dict:
    """修改 workflow 参数

    Args:
        api_prompt: API workflow 数据
        args: 命令行参数对象，包含 steps, width, height, frames, precision

    支持的节点类型和参数映射:
        - KsanaGeneratorNode: steps
        - EmptyHunyuanLatentVideo: width, height, length (从 args.frames 获取)
        - KsanaModelLoaderNode: 根据 precision 修改模型名称（仅当 precision 参数被设置时）
        - CLIPLoader: 根据 precision 修改 CLIP 模型名称（仅当 precision 参数被设置时）

    Returns:
        修改后的 workflow
    """
    # 检查是否设置了 precision 参数
    precision = getattr(args, "precision", None)

    for _, node_data in api_prompt.items():
        class_type = node_data.get("class_type")
        inputs = node_data.get("inputs", {})

        # KsanaGeneratorNode: 修改 steps
        if class_type == "KsanaGeneratorNode" and "steps" in inputs:
            node_data["inputs"]["steps"] = args.steps

        # EmptyHunyuanLatentVideo: 修改 width, height, length
        elif class_type == "EmptyHunyuanLatentVideo":
            if "width" in inputs:
                node_data["inputs"]["width"] = args.width

            if "height" in inputs:
                node_data["inputs"]["height"] = args.height

            if "length" in inputs:
                node_data["inputs"]["length"] = args.frames

        # KsanaModelLoaderNode: 根据 precision 修改模型名称（仅当 precision 被设置时）
        elif class_type == "KsanaModelLoaderNode" and precision:
            if "ckpt_name" in inputs:
                model_name = inputs["ckpt_name"]
                if precision == "fp8":
                    model_name = model_name.replace("_fp16.safetensors", "_fp8_scaled.safetensors")
                node_data["inputs"]["ckpt_name"] = model_name

        # CLIPLoader: 根据 precision 修改 CLIP 模型名称（仅当 precision 被设置时）
        elif class_type == "CLIPLoader" and precision:
            if "clip_name" in inputs:
                clip_name = inputs["clip_name"]
                if precision == "fp8":
                    clip_name = clip_name.replace("_fp16.safetensors", "_fp8_e4m3fn_scaled.safetensors")
                node_data["inputs"]["clip_name"] = clip_name

    return api_prompt


def test_workflow(workflow_path: str, args) -> bool:
    """测试 workflow
    Args:
        workflow_path: workflow 文件路径
        args: 命令行参数
    Returns:
        是否成功
    """
    server_address = "127.0.0.1:8188"

    try:
        # 1. 加载 workflow（自动判断格式，需要时转换）
        api_prompt = load_workflow(workflow_path, server_address)

        # 2. 修改参数
        api_prompt = modify_workflow_params(api_prompt, args)

        # 3. 连接 WebSocket
        ws, client_id = connect_websocket(server_address)

        # 4. 提交 workflow
        prompt_id = submit_workflow(api_prompt, client_id, server_address)
        if not prompt_id:
            return False

        # 5. 等待完成
        return wait_for_completion(ws, prompt_id, server_address)

    except urllib.error.HTTPError as e:
        error_content = e.read().decode("utf-8")
        logger.error(f"✗ 提交失败 ({e.code}):")
        try:
            error_data = json.loads(error_content)
            logger.error(json.dumps(error_data, indent=2, ensure_ascii=False))
        except:
            logger.error(error_content)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="ComfyUI Workflow 测试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用 fp16 精度（默认）
    python flow_tests.py --steps 3 --width 1280 --height 720 --frames 25

  # 使用 fp8 精度
    python flow_tests.py --precision fp8 --steps 3 --width 1280 --height 720 --frames 25

注意:
  如果使用普通格式 workflow，需要安装 playwright:
    pip install playwright
    playwright install chromium
        """,
    )
    parser.add_argument(
        "--workflow_path",
        type=str,
        default="../workflows/wan2.2_fp16.json",
        help="workflow 文件路径（支持普通格式和 API 格式，自动识别）",
    )
    parser.add_argument("--steps", type=int, default=2, help="采样步数 (默认: 2)")
    parser.add_argument("--width", type=int, default=720, help="视频宽度 (默认: 720)")
    parser.add_argument("--height", type=int, default=480, help="视频高度 (默认: 480)")
    parser.add_argument("--frames", type=int, default=17, help="视频帧数 (默认: 17)")
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        choices=["fp8"],
        help="模型精度，设置后会修改 workflow 中的模型名称 (可选: fp8)",
    )
    parser.add_argument("--no-server", action="store_true", help="不启动 server（假设 server 已经在运行）")

    args = parser.parse_args()

    # 打印配置
    logger.info("=" * 60)
    logger.info("ComfyUI Workflow 测试")
    logger.info("=" * 60)
    logger.info(f"配置: {args.__dict__}")
    logger.info("=" * 60)

    server_process = None

    try:
        # 开始计时
        logger.info("=" * 60)
        logger.info("开始执行测试...")
        test_start_time = time.time()

        # 启动 server（如果需要）
        if not args.no_server:
            server_process = start_server()
        else:
            logger.info("跳过启动 server（使用已有 server）")

        # 运行测试
        success = test_workflow(workflow_path=args.workflow_path, args=args)

        # 计算耗时
        elapsed_seconds = time.time() - test_start_time
        elapsed_minutes = int(elapsed_seconds // 60)
        remaining_seconds = int(elapsed_seconds % 60)

        if success:
            logger.info("=" * 60)
            logger.info("✓✓✓ 测试成功！")
            logger.info(f"⏱️  耗时: {elapsed_minutes} 分 {remaining_seconds} 秒 (总计 {elapsed_seconds:.2f} 秒)")
            logger.info("=" * 60)
        else:
            logger.error("=" * 60)
            logger.error("✗✗✗ 测试失败！")
            logger.error(f"⏱️  耗时: {elapsed_minutes} 分 {remaining_seconds} 秒 (总计 {elapsed_seconds:.2f} 秒)")
            logger.error("=" * 60)
            assert False, "Workflow 测试失败"

    except KeyboardInterrupt:
        logger.info("\n测试中断")

    except Exception as e:
        logger.error(f"✗ 测试失败: {e}")
        import traceback

        traceback.print_exc()

    finally:
        if server_process:
            stop_server(server_process)


if __name__ == "__main__":
    main()
