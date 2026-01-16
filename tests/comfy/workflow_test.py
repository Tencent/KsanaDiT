#!/usr/bin/env python3

import argparse
import json
import logging
import os
import time
import urllib.error
from typing import Optional

from test_utils import (
    check_media_data,
    connect_websocket,
    load_workflow,
    start_server,
    stop_server,
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
        - KsanaGeneratorNode: steps, seed
        - EmptyHunyuanLatentVideo: width, height, length (从 params["frames"] 获取)
        - KsanaModelLoaderNode: model_name, run_dtype, linear_backend
        - KsanaAttentionConfigNode: backend
        - CLIPLoader: clip_name
        - KsanaSigmasNode: sigmas
    """
    for _, node_data in api_prompt.items():
        class_type = node_data.get("class_type")
        inputs = node_data.get("inputs")

        if not inputs:
            continue
        # KsanaGeneratorNode: 修改 steps 和 seed
        if class_type == "KsanaGeneratorNode":
            if "steps" in params and "steps" in inputs:
                inputs["steps"] = params["steps"]
            if "seed" in params and params["seed"] is not None and "seed" in inputs:
                inputs["seed"] = params["seed"]
                inputs["control_after_generate"] = "fixed"
            if "rope_function" in params and "rope_function" in inputs:
                inputs["rope_function"] = params["rope_function"]

        # KsanaVAEEncodeNode: 修改 width, height, length
        elif class_type == "KsanaVAEEncodeNode":
            if "width" in params and "width" in inputs:
                inputs["width"] = params["width"]
            if "height" in params and "height" in inputs:
                inputs["height"] = params["height"]
            if "frames" in params and "length" in inputs:
                inputs["num_frames"] = params["frames"]

        # KsanaModelLoaderNode: 根据命令行参数修改模型
        elif class_type == "KsanaModelLoaderNode":
            model_name = inputs.get("model_name")
            if not model_name:
                raise ValueError("KsanaModelLoaderNode is missing 'model_name' in its inputs.")

            if "high_noise" in model_name and params.get("dit_high_model_name"):
                inputs["model_name"] = params["dit_high_model_name"]
            elif "low_noise" in model_name and params.get("dit_low_model_name"):
                inputs["model_name"] = params["dit_low_model_name"]

            if params.get("run_dtype") and "run_dtype" in inputs:
                inputs["run_dtype"] = params["run_dtype"]
            if params.get("linear_backend") and "linear_backend" in inputs:
                inputs["linear_backend"] = params["linear_backend"]

        elif class_type == "KsanaAttentionConfigNode":
            if params.get("attn_backend") and "backend" in inputs:
                inputs["backend"] = params["attn_backend"]

        # CLIPLoader: 根据命令行参数修改模型
        elif class_type == "CLIPLoader":
            if params.get("text_model_name") and "clip_name" in inputs:
                inputs["clip_name"] = params["text_model_name"]

        elif class_type == "KsanaLoraSelectNode":
            if params.get("lora_model_name"):
                inputs["lora"] = params["lora_model_name"]

        elif class_type == "StringToFloatList":
            if "sigmas" in params and "string" in inputs:
                sigmas_str = ",".join(map(str, params["sigmas"]))
                inputs["string"] = sigmas_str

        elif class_type == "LoadImage":
            if "input_image" in params and "image" in inputs:
                inputs["image"] = os.path.abspath(params["input_image"])

        elif class_type == "KsanaEmptyImageLatentNode":
            if "image_width" in params and "width" in inputs:
                inputs["width"] = params["image_width"]
            if "height" in params and "image_height" in inputs:
                inputs["height"] = params["image_height"]

    return api_prompt


def test_workflow(workflow_path: str, params: dict, expect_values: Optional[dict] = None, port: int = 8188) -> bool:
    """测试 workflow
    Args:
        workflow_path: workflow 文件路径
        params: 包含参数的字典
        expect_values: 期望值，用于结果校验
        port: ComfyUI 服务器端口
    Returns:
        是否成功
    """
    server_address = f"127.0.0.1:{port}"

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
        success, media_data = wait_for_completion(ws, prompt_id, server_address, api_prompt)
        if not success:
            return False
        # 6. 校验结果
        return check_media_data(media_data, expect_values)

    except urllib.error.HTTPError as e:
        error_content = e.read().decode("utf-8")
        logger.error(f"✗ 提交失败 ({e.code}):")
        try:
            error_data = json.loads(error_content)
            logger.error(json.dumps(error_data, indent=2, ensure_ascii=False))
        except Exception as e:  # pylint: disable=broad-except
            logger.error(error_content, e)
        return False

    except Exception as e:  # pylint: disable=broad-except
        logger.error(f"✗ 测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_workflows_batch(
    comfyui_root: str,
    workflow_configs: list,
    seed: int,
    num_gpus: int,
    port: int,
    restart_server: bool,
) -> bool:
    all_success = True

    if not restart_server:
        server_process = start_server(comfyui_root=comfyui_root, port=port)

    for i, configs in enumerate(workflow_configs, 1):
        if isinstance(configs, dict):
            configs = [configs]
        workflow_path_list = [config["workflow_path"] for config in configs]
        logger.info(f"开始执行 workflow 配置组 [{i}/{len(workflow_configs)}] {workflow_path_list}")
        for j, config in enumerate(configs, 1):
            if restart_server:
                server_process = start_server(comfyui_root=comfyui_root, port=port)
            config["seed"] = seed
            logger.info("=" * 60)
            logger.info(f"执行 workflow [{j}/{len(configs)}]")
            logger.info(f"配置: {config}")
            logger.info("=" * 60)

            workflow_start_time = time.time()
            expect_values = config.get("gpus_expect_values") if num_gpus > 1 else config.get("expect_values")
            if expect_values is None:
                expect_values = config.get("expect_values")
            success = test_workflow(
                workflow_path=config["workflow_path"], params=config, expect_values=expect_values, port=port
            )
            workflow_elapsed = time.time() - workflow_start_time

            if success:
                logger.info(f"✓ Workflow [{j}/{len(configs)}] 成功! 耗时: {workflow_elapsed:.2f} 秒")
            else:
                logger.error(f"✗ Workflow [{j}/{len(configs)}] 失败! 耗时: {workflow_elapsed:.2f} 秒")
                all_success = False
            if restart_server:
                stop_server(server_process)

    if not restart_server:
        stop_server(server_process)

    return all_success


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
    playwright install firefox
        """,
    )

    # 单个 workflow 参数（原有方式）
    parser.add_argument(
        "--workflow_path",
        type=str,
        default="../workflows/wan2.2_fp8_scaled.json",
        help="workflow 文件路径（支持普通格式和 API 格式，自动识别）",
    )
    parser.add_argument("--steps", type=int, default=2, help="采样步数 (默认: 2)")
    parser.add_argument("--width", type=int, default=720, help="视频宽度 (默认: 720)")
    parser.add_argument("--height", type=int, default=480, help="视频高度 (默认: 480)")
    parser.add_argument("--frames", type=int, default=17, help="视频帧数 (默认: 17)")
    parser.add_argument("--seed", type=int, default=None, help="随机种子 (默认: None，不修改)")

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
        "--run_dtype",
        type=str,
        default=None,
        choices=["float16", "bfloat16"],
        help="weight dtype of running model",
    )
    parser.add_argument(
        "--linear_backend",
        type=str,
        default=None,
        help="linear backend",
    )
    parser.add_argument(
        "--attn_backend",
        type=str,
        default=None,
        help="attention config backend",
    )

    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="指定使用的 GPU，用逗号分隔（例如: 0,1）(默认: 0)",
    )
    parser.add_argument(
        "--comfyui_root",
        type=str,
        default="/data/ComfyUI",
        help="ComfyUI 根目录，默认自动推断",
    )
    parser.add_argument(
        "--workflows-file",
        type=str,
        required=True,
        help="从 JSON 文件加载多个 workflow 配置",
    )

    parser.add_argument("--no-server", action="store_true", help="不启动 server（假设 server 已经在运行）")
    parser.add_argument("--port", type=int, default=8188, help="ComfyUI 服务器端口 (默认: 8188)")

    return parser


def main():
    parser = create_argument_parser()
    args = parser.parse_args()

    workflow_configs = []

    logger.info(f"从文件加载 workflow 配置: {args.workflows_file}")
    with open(args.workflows_file, "r", encoding="utf-8") as f:
        workflow_configs = json.load(f)

    logger.info("=" * 60)
    logger.info("ComfyUI Workflow 测试")
    logger.info("=" * 60)
    total_workflows = len(workflow_configs["independent_tests"] + workflow_configs["continuous_tests"])
    logger.info(f"总共 {total_workflows} 个 workflow:")
    logger.info("=" * 60)

    server_process = None

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        logger.info(f"设置 CUDA_VISIBLE_DEVICES = {args.gpus}")
        num_gpus = len(args.gpus.split(","))

        logger.info("=" * 60)
        logger.info("开始执行测试...")
        test_start_time = time.time()

        # TODO: 后面改成全部是启动一次，执行多个workflow，现在主要是多卡的情况之下多个workflow会卡住。
        # 两种测试，一种是独立测试，一种是连续测试（不重启server）
        test_types = [("independent_tests", True), ("continuous_tests", False)]
        all_success = True
        for test_type, restart_server in test_types:
            if num_gpus > 1 and not restart_server:
                logger.info(f"多卡情况下不执行 {test_type} 测试")
                continue
            logger.info(f"开始执行 {test_type} 测试..., restart_server: {restart_server}")
            all_success = run_workflows_batch(
                comfyui_root=args.comfyui_root,
                workflow_configs=workflow_configs[test_type],
                seed=args.seed,
                num_gpus=num_gpus,
                port=args.port,
                restart_server=restart_server,
            )
            if not all_success:
                break

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

    except Exception as e:  # pylint: disable=broad-except
        logger.error(f"✗ 测试失败: {e}")
        import traceback

        traceback.print_exc()
        assert False, "测试失败"

    finally:
        if server_process:
            stop_server(server_process)


if __name__ == "__main__":
    main()
