#!/usr/bin/env python3
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

import json
import logging
import subprocess
import time
import urllib.error
import urllib.request
import uuid

# video mean
from io import BytesIO
from pathlib import Path

import numpy
import torch
import websocket
from PIL import Image, ImageSequence

logger = logging.getLogger(__name__)

PLAYWRIGHT_AVAILABLE = False
try:
    from playwright.sync_api import sync_playwright

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    pass


# ============================================================================
# Server 管理
# ============================================================================


def wait_for_server(server_address: str = "127.0.0.1:8188", max_wait: int = 300, check_interval: int = 5) -> bool:
    """等待 server 启动就绪"""
    url = f"http://{server_address}/object_info"
    start_time = time.time()

    while time.time() - start_time < max_wait:
        try:
            urllib.request.urlopen(url, timeout=5)
            return True
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            elapsed = int(time.time() - start_time)
            logger.info(f"等待中... ({elapsed}s/{max_wait}s)")
            time.sleep(check_interval)

    return False


def start_server(comfyui_root: Path | None = None, host: str = "127.0.0.1", port: int = 8188) -> subprocess.Popen:
    """启动 ComfyUI server
    Args:
        comfyui_root: ComfyUI 根目录，默认自动推断
        host: 监听地址
        port: 监听端口
    Raises:
        TimeoutError: 启动超时
    """
    logger.info("启动 ComfyUI server...")

    if comfyui_root is None:
        comfyui_root = Path(__file__).parent.parent.parent.parent.parent

    process = subprocess.Popen(
        ["python", "main.py", "--listen", host, "--port", str(port)],
        cwd=str(comfyui_root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    logger.info(f"Server PID: {process.pid}")
    logger.info("等待 server 就绪...")

    server_address = f"{host}:{port}"
    if wait_for_server(server_address, max_wait=180, check_interval=5):
        logger.info("✓ Server 已就绪")
        return process
    else:
        logger.error("✗ Server 启动超时")
        process.terminate()
        raise TimeoutError("ComfyUI server 启动超时")


def stop_server(process: subprocess.Popen, timeout: int = 5) -> None:
    logger.info("关闭 server...")
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()


# ============================================================================
# Workflow 加载和转换
# ============================================================================


def convert_workflow_to_api(
    workflow_path: str, server_url: str = "http://127.0.0.1:8188", timeout: int = 60000
) -> dict:
    """
    使用 Headless Browser 将普通 workflow JSON 转换为 API 格式

    原理：
    1. 启动一个无头浏览器，访问 ComfyUI 前端页面
    2. 等待页面完全加载（包括 LiteGraph、节点定义等）
    3. 在浏览器中执行 JavaScript，调用 app.loadGraphData() 加载 workflow
    4. 调用 app.graphToPrompt() 转换为 API 格式
    5. 把结果返回给 Python

    Args:
        workflow_path: workflow 文件路径
        server_url: ComfyUI server 地址
        timeout: 超时时间（毫秒）

    Returns:
        API 格式的 workflow 字典
    """
    if not PLAYWRIGHT_AVAILABLE:
        raise ImportError(
            "需要安装 playwright 才能转换普通 workflow。\n"
            "请运行：pip install playwright && playwright install firefox"
        )

    with open(workflow_path, "r", encoding="utf-8") as f:
        workflow_data = f.read()

    with sync_playwright() as p:
        logger.info("启动无头浏览器 (Firefox)...")
        browser = p.firefox.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        logger.info(f"访问 ComfyUI: {server_url}")
        page.goto(server_url, timeout=timeout)

        logger.info("等待页面加载...")
        page.wait_for_function(
            """
            (() => {
                // 检查 app 对象是否存在且已初始化
                return typeof window.app !== 'undefined'
                    && window.app !== null
                    && typeof window.app.loadGraphData === 'function'
                    && typeof window.app.graphToPrompt === 'function';
            })()
            """,
            timeout=timeout,
        )
        logger.info("✓ 页面初始化完成，开始转换...")

        api_output = page.evaluate(
            """
            async (workflowJson) => {
                try {
                    const workflow = JSON.parse(workflowJson);
                    await app.loadGraphData(workflow, true, false);
                    await new Promise(resolve => setTimeout(resolve, 500));
                    const result = await app.graphToPrompt();
                    return {
                        success: true,
                        output: result.output,
                    };
                } catch (error) {
                    return {
                        success: false,
                        error: error.message,
                        stack: error.stack
                    };
                }
            }
            """,
            workflow_data,
        )

        browser.close()
        logger.info("✓ 转换完成")
        if not api_output["success"]:
            raise RuntimeError(f"转换失败: {api_output['error']}\n{api_output.get('stack', '')}")

        return api_output["output"]


def load_workflow(workflow_path: str, server_address: str = "127.0.0.1:8188") -> dict:
    """
    智能加载 workflow，支持普通格式和 API 格式

    - 如果文件是 API 格式（直接包含 class_type），直接加载
    - 如果文件是普通格式（包含 nodes 数组），使用 Headless Browser 转换

    Args:
        workflow_path: workflow 文件路径
        server_address: ComfyUI server 地址

    Returns:
        API 格式的 workflow 字典
    """
    logger.info("加载 workflow...")

    # 确定文件路径
    path = Path(workflow_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到 workflow 文件: {workflow_path}")

    logger.info(f"文件: {path}")

    # 读取文件判断格式
    with open(path) as f:
        data = json.load(f)

    # 判断是否为 API 格式
    is_api_format = "nodes" not in data and any(
        isinstance(v, dict) and "class_type" in v for v in data.values() if isinstance(v, dict)
    )

    if is_api_format:
        logger.info("格式: API 格式（直接使用）")
        api_prompt = data
    else:
        logger.info("格式: 普通格式（需要转换）")
        logger.info("使用 Headless Browser 转换...")
        api_prompt = convert_workflow_to_api(str(path), server_url=f"http://{server_address}")

    logger.info(f"✓ 加载完成，共 {len(api_prompt)} 个节点")
    return api_prompt


# ============================================================================
# WebSocket 通信和执行监控
# ============================================================================


def connect_websocket(server_address: str = "127.0.0.1:8188") -> tuple[websocket.WebSocket, str]:
    """连接 WebSocket

    Args:
        server_address: server 地址

    Returns:
        (WebSocket 连接, 客户端 ID)
    """
    logger.info("连接 server...")
    client_id = str(uuid.uuid4())
    ws = websocket.WebSocket()
    ws.connect(f"ws://{server_address}/ws?clientId={client_id}")
    logger.info("✓ 已连接")
    return ws, client_id


def submit_workflow(api_prompt: dict, client_id: str, server_address: str = "127.0.0.1:8188") -> str | None:
    """提交 workflow 到 server

    Args:
        api_prompt: API 格式的 workflow
        client_id: 客户端 ID
        server_address: server 地址

    Returns:
        prompt_id，如果失败返回 None
    """
    logger.info("提交 workflow...")
    req = urllib.request.Request(
        f"http://{server_address}/prompt",
        data=json.dumps({"prompt": api_prompt, "client_id": client_id}).encode(),
        headers={"Content-Type": "application/json"},
    )
    result = json.loads(urllib.request.urlopen(req).read())
    prompt_id = result.get("prompt_id")
    logger.info(f"✓ 已提交: {prompt_id}")
    return prompt_id


def get_history(prompt_id: str, server_address: str) -> dict:
    with urllib.request.urlopen(f"http://{server_address}/history/{prompt_id}") as response:
        return json.loads(response.read())


def get_media(filename: str, subfolder: str, folder_type: str, server_address: str) -> bytes:
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen(f"http://{server_address}/view?{url_values}") as response:
        return response.read()


def _get_workflow_result_media(prompt_id: str, server_address: str) -> tuple[bool, bytes | None]:
    """获取并下载 workflow 结果中的第一个媒体文件。"""
    try:
        history = get_history(prompt_id, server_address)[prompt_id]
        for node_id in history["outputs"]:
            node_output = history["outputs"][node_id]
            media_keys = ["gifs", "images"]
            for key in media_keys:
                if key in node_output and node_output[key]:
                    media_item = node_output[key][0]
                    logger.info(f"找到媒体文件: {media_item['filename']}")
                    media_data = get_media(
                        media_item["filename"],
                        media_item["subfolder"],
                        media_item["type"],
                        server_address,
                    )
                    return True, media_data
        logger.warning("在 workflow 输出中未找到任何媒体文件。")
        return True, None
    except Exception as e:  # pylint: disable=broad-except
        logger.error(f"获取或下载媒体文件时出错: {e}")
        return False, None


def check_media_data(media_data: bytes, expect_values: dict) -> bool:
    if not expect_values:
        logger.warning("No `expect_values` provided for workflow. Skipping image check.")
        return True
    if not media_data:
        logger.error("✗ `expect_values` was provided, but no image/video data was received.")
        return False

    try:
        pil_image = Image.open(BytesIO(media_data))

        frames = []
        if pil_image.format == "GIF":
            for frame in ImageSequence.Iterator(pil_image):
                frames.append(numpy.array(frame.convert("RGB"), dtype=numpy.float32))
        else:
            if pil_image.mode not in ("RGB", "RGBA"):
                pil_image = pil_image.convert("RGB")
            elif pil_image.mode == "RGBA":
                pil_image = pil_image.convert("RGB")
            frames.append(numpy.array(pil_image, dtype=numpy.float32))

        image_array = numpy.stack(frames, axis=0)
        image_tensor = torch.from_numpy(image_array / 255.0)
        mean = image_tensor.abs().mean().item()

        expected_mean = expect_values["mean"]
        eps_place = expect_values.get("eps_place", 4)
        tolerance = 10 ** (-eps_place)

        if abs(mean - expected_mean) < tolerance:
            logger.info(f"✓ Image check passed. Mean: {mean:.7f}, Expected Mean: {expected_mean:.7f}")
            return True
        else:
            logger.error(
                f"✗ Image check failed. Mean: {mean:.7f}, Expected Mean: {expected_mean:.7f}, Tolerance: {tolerance}"
            )
            return False
    except Exception as e:  # pylint: disable=broad-except
        logger.error(f"✗ Failed to process and check image/video data: {e}")
        return False


def wait_for_completion(
    ws: websocket.WebSocket, prompt_id: str, server_address: str = "127.0.0.1:8188", api_prompt: dict = None
) -> tuple[bool, bytes | None]:
    """等待 workflow 执行完成

    Args:
        ws: WebSocket 连接
        prompt_id: prompt ID
        server_address: server 地址
        api_prompt: API 格式的 workflow，用于获取节点的 class_type

    Returns:
        (是否成功完成, 最后一个媒体文件数据)
    """
    logger.info("等待执行...")
    last_node_id = None
    last_node_start_time = None

    while True:
        msg = ws.recv()
        if isinstance(msg, str):
            data = json.loads(msg)

            if data["type"] == "executing":
                if data["data"]["prompt_id"] == prompt_id:
                    current_time = time.time()
                    if last_node_id and last_node_start_time:
                        duration = current_time - last_node_start_time
                        logger.info(f"节点 {last_node_id} 完成，耗时: {duration:.2f}s")
                    if data["data"]["node"] is None:
                        logger.info("✓ 执行完成！")
                        ws.close()
                        return _get_workflow_result_media(prompt_id, server_address)
                    else:
                        last_node_id = data["data"]["node"]
                        last_node_start_time = current_time
                        class_type = api_prompt[last_node_id].get("class_type", "error")
                        logger.info(f"执行: {last_node_id} ({class_type})")

            elif data["type"] == "execution_error":
                if data["data"].get("prompt_id") == prompt_id:
                    logger.error("✗ 执行错误:")
                    logger.error(json.dumps(data["data"], indent=2, ensure_ascii=False))
                    ws.close()
                    return False, None
