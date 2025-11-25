#!/usr/bin/env python3

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


def modify_workflow_params(api_prompt: dict, params: dict) -> dict:
    """修改 workflow 参数

    Args:
        api_prompt: API workflow 数据
        params: 包含参数的字典

    支持的节点类型和参数映射:
        - KsanaGeneratorNode: steps
        - EmptyHunyuanLatentVideo: width, height, length (从 params["frames"] 获取)
        - KsanaModelLoaderNode: model_name, weight_dtype, linear_backend, attn_backend
        - CLIPLoader: clip_name
    """
    for _, node_data in api_prompt.items():
        class_type = node_data.get("class_type")
        inputs = node_data.get("inputs")

        if not inputs:
            continue

        # KsanaGeneratorNode: 修改 steps
        if class_type == "KsanaGeneratorNode":
            if "steps" in params and "steps" in inputs:
                inputs["steps"] = params["steps"]

        # EmptyHunyuanLatentVideo: 修改 width, height, length
        elif class_type == "EmptyHunyuanLatentVideo":
            if "width" in params and "width" in inputs:
                inputs["width"] = params["width"]
            if "height" in params and "height" in inputs:
                inputs["height"] = params["height"]
            if "frames" in params and "length" in inputs:
                inputs["length"] = params["frames"]

        # KsanaModelLoaderNode: 根据命令行参数修改模型
        elif class_type == "KsanaModelLoaderNode":
            model_name = inputs.get("model_name")
            if not model_name:
                raise ValueError("KsanaModelLoaderNode is missing 'model_name' in its inputs.")

            if "high_noise" in model_name and params.get("dit_high_model_name"):
                inputs["model_name"] = params["dit_high_model_name"]
            elif "low_noise" in model_name and params.get("dit_low_model_name"):
                inputs["model_name"] = params["dit_low_model_name"]

            if params.get("weight_dtype") and "weight_dtype" in inputs:
                inputs["weight_dtype"] = params["weight_dtype"]
            if params.get("linear_backend") and "linear_backend" in inputs:
                inputs["linear_backend"] = params["linear_backend"]
            if params.get("attn_backend") and "attn_backend" in inputs:
                inputs["attn_backend"] = params["attn_backend"]

        # CLIPLoader: 根据命令行参数修改模型
        elif class_type == "CLIPLoader":
            if params.get("text_model_name") and "clip_name" in inputs:
                inputs["clip_name"] = params["text_model_name"]

    return api_prompt


def test_workflow(workflow_path: str, params: dict) -> bool:
    """测试 workflow
    Args:
        workflow_path: workflow 文件路径
        params: 包含参数的字典
    Returns:
        是否成功
    """
    server_address = "127.0.0.1:8188"

    try:
        # 1. 加载 workflow（自动判断格式，需要时转换）
        api_prompt = load_workflow(workflow_path, server_address)

        # 2. 修改参数
        api_prompt = modify_workflow_params(api_prompt, params)

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


def create_argument_parser():
    parser = argparse.ArgumentParser(
        description="ComfyUI Workflow 测试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单个 workflow
    python flow_tests.py --steps 3 --width 1280 --height 720 --frames 25

  # 多个 workflow，从配置文件加载多个 workflow，配置文件格式：workflows_config.json
    python flow_tests.py --workflows-file workflows_config.json
注意:
  如果使用普通格式 workflow，需要安装 playwright:
    pip install playwright
    playwright install chromium
        """,
    )

    # 单个 workflow 参数（原有方式）
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

    # 模型名称参数
    parser.add_argument(
        "--dit_high_model_name",
        type=str,
        default=None,
        help="DIT high noise 模型名称（例如: wan2.2_t2v_high_noise_14B_fp16.safetensors）",
    )
    parser.add_argument(
        "--dit_low_model_name",
        type=str,
        default=None,
        help="DIT low noise 模型名称（例如: wan2.2_t2v_low_noise_14B_fp16.safetensors）",
    )
    parser.add_argument(
        "--text_model_name",
        type=str,
        default=None,
        help="Text encoder 模型名称（例如: umt5_xxl_fp16.safetensors）",
    )
    parser.add_argument(
        "--weight_dtype",
        type=str,
        default=None,
        choices=["default", "float16", "bfloat16"],
        help="weight dtype of running model",
    )
    parser.add_argument(
        "--linear_backend",
        type=str,
        default=None,
        choices=["default", "fp8_gemm", "fp16_gemm"],
        help="linear backend",
    )
    parser.add_argument(
        "--attn_backend",
        type=str,
        default=None,
        choices=["default", "flash_attention"],
        help="attention backend",
    )

    # 多个 workflow 参数
    parser.add_argument(
        "--workflows-file",
        type=str,
        help="从 JSON 文件加载多个 workflow 配置",
    )

    parser.add_argument("--no-server", action="store_true", help="不启动 server（假设 server 已经在运行）")

    return parser


def main():
    parser = create_argument_parser()
    args = parser.parse_args()

    # 准备 workflow 配置列表
    workflow_configs = []

    if args.workflows_file:
        # 从文件加载配置
        logger.info(f"从文件加载 workflow 配置: {args.workflows_file}")
        with open(args.workflows_file, "r", encoding="utf-8") as f:
            workflow_configs = json.load(f)
    else:
        # 使用单个 workflow（原有方式）
        workflow_configs = [vars(args)]

    # 打印配置
    logger.info("=" * 60)
    logger.info("ComfyUI Workflow 测试")
    logger.info("=" * 60)
    logger.info(f"总共 {len(workflow_configs)} 个 workflow:")
    for i, config in enumerate(workflow_configs, 1):
        logger.info(f"  [{i}] {config}")
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

        # 运行所有 workflow
        all_success = True
        for i, config in enumerate(workflow_configs, 1):
            logger.info("=" * 60)
            logger.info(f"执行 workflow [{i}/{len(workflow_configs)}]")
            logger.info(f"配置: {config}")
            logger.info("=" * 60)

            # 运行测试
            workflow_start_time = time.time()
            success = test_workflow(workflow_path=config["workflow_path"], params=config)
            workflow_elapsed = time.time() - workflow_start_time

            if success:
                logger.info(f"✓ Workflow [{i}/{len(workflow_configs)}] 成功! 耗时: {workflow_elapsed:.2f} 秒")
            else:
                logger.error(f"✗ Workflow [{i}/{len(workflow_configs)}] 失败! 耗时: {workflow_elapsed:.2f} 秒")
                all_success = False
                break  # 如果一个失败，停止执行后续的

        # 计算总耗时
        elapsed_seconds = time.time() - test_start_time
        elapsed_minutes = int(elapsed_seconds // 60)
        remaining_seconds = int(elapsed_seconds % 60)

        if all_success:
            logger.info("=" * 60)
            logger.info(f"✓✓✓ 所有 {len(workflow_configs)} 个 workflow 测试成功！")
            logger.info(f"⏱️  总耗时: {elapsed_minutes} 分 {remaining_seconds} 秒 (总计 {elapsed_seconds:.2f} 秒)")
            logger.info("=" * 60)
        else:
            logger.error("=" * 60)
            logger.error("✗✗✗ 测试失败！")
            logger.error(f"⏱️  总耗时: {elapsed_minutes} 分 {remaining_seconds} 秒 (总计 {elapsed_seconds:.2f} 秒)")
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
