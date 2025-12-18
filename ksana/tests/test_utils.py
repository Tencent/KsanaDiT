import os
import shutil
import time

CEPH_BASE = "/dockerdata/ci-models"
RAMDISK_BASE = "/data/ramdisk/ci-models"
COMFYUI_MODELS_BASE = "/data/stable-diffusion-webui/models"
KSANA_MODELS_BASE = "/data/ComfyUI/custom_nodes/KsanaDiT"

# 节点类型到模型目录的映射
COMFYUI_MODEL_DIRS = {
    "KsanaVAELoaderNode": "VAE",
    "CLIPLoader": "text_encoders",
    "KsanaLoraSelectNode": "loras",
    "KsanaModelLoaderNode": "diffusion_models",
}

# 永久保护的模型配置 - 这些模型永远不会被删除
PERMANENT_MODELS = {
    "comfyui": [
        "comfy_models/facerestore_models",  # 面部修复模型，ComfyUI 启动必需
    ],
    "ksana": [
        # 可以在这里添加 Ksana 必需的模型
    ],
}


def get_ksana_model_names():
    """获取 CEPH 中除了 comfy_models 之外的所有模型目录名称"""
    if not os.path.exists(CEPH_BASE):
        return []
    model_names = [item for item in os.listdir(CEPH_BASE) if os.path.isdir(os.path.join(CEPH_BASE, item))]
    model_names.remove("comfy_models") if "comfy_models" in model_names else None
    return model_names


def switch_model_paths(use_ramdisk=True):
    target_name = "Ramdisk" if use_ramdisk else "CEPH"
    source_base = RAMDISK_BASE if use_ramdisk else CEPH_BASE
    print(f"切换模型路径到 {target_name}...")
    ksana_models = get_ksana_model_names()

    paths_to_switch = [COMFYUI_MODELS_BASE]
    for model_name in ksana_models:
        paths_to_switch.append(os.path.join(KSANA_MODELS_BASE, model_name))

    for path in paths_to_switch:
        if os.path.islink(path):
            os.unlink(path)
        elif os.path.exists(path):
            shutil.rmtree(path)

    os.symlink(os.path.join(source_base, "comfy_models"), COMFYUI_MODELS_BASE)
    print(f"创建软链接: {COMFYUI_MODELS_BASE} -> {source_base}/comfy_models")

    for model_name in ksana_models:
        source = os.path.join(source_base, model_name)
        target = os.path.join(KSANA_MODELS_BASE, model_name)
        if os.path.exists(source):
            os.symlink(source, target)
            print(f"创建软链接: {target} -> {source}")


def get_existing_models(base_dir):
    """获取现有模型列表，区分 Ksana 和 ComfyUI 模型"""
    if not os.path.exists(base_dir):
        return set()
    models = set()
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            if item == "comfy_models":
                for category in os.listdir(item_path):
                    category_path = os.path.join(item_path, category)
                    if os.path.isdir(category_path):
                        for model_file in os.listdir(category_path):
                            models.add(f"comfy_models/{category}/{model_file}")
            else:
                models.add(item)
    return models


def add_permanent_models_to_copy_tasks(copy_tasks, required_targets, test_type):
    """添加永久模型到拷贝任务列表"""
    for permanent_model in PERMANENT_MODELS[test_type]:
        source_path = os.path.join(CEPH_BASE, permanent_model)
        target_path = os.path.join(RAMDISK_BASE, permanent_model)
        required_targets.add(permanent_model)
        copy_tasks.append((source_path, target_path, permanent_model, f"{test_type}_permanent"))


def copy_models_to_ramdisk(model_configs, test_type):
    print(f"拷贝模型到 Ramdisk ({test_type}): {[config['name'] for config in model_configs]}")
    # 计算所需的模型路径和源路径
    required_targets = set()
    copy_tasks = []

    for config in model_configs:
        name = config["name"]
        if test_type == "ksana":
            source_path = os.path.join(CEPH_BASE, name)
            target_path = os.path.join(RAMDISK_BASE, name)
            required_targets.add(name)
        elif test_type == "comfyui":
            if "category" not in config:
                raise ValueError(f"ComfyUI 模型 {name} 缺少 category 参数")
            category = config["category"]
            if category not in COMFYUI_MODEL_DIRS:
                raise ValueError(f"未知的 ComfyUI 模型类型: {category}")
            source_path = os.path.join(CEPH_BASE, "comfy_models", COMFYUI_MODEL_DIRS[category], name)
            target_path = os.path.join(RAMDISK_BASE, "comfy_models", COMFYUI_MODEL_DIRS[category], name)
            required_targets.add(os.path.join("comfy_models", COMFYUI_MODEL_DIRS[category], name))
        else:
            raise ValueError(f"未知的测试类型: {test_type}")
        copy_tasks.append((source_path, target_path, name, test_type))

    add_permanent_models_to_copy_tasks(copy_tasks, required_targets, test_type)
    existing_models = get_existing_models(RAMDISK_BASE)

    if not required_targets.issubset(existing_models):
        print("需要的模型与现有模型不匹配，清空 Ramdisk 重新拷贝")
        if os.path.exists(RAMDISK_BASE):
            shutil.rmtree(RAMDISK_BASE)
        for source_path, target_path, name, model_type in copy_tasks:
            if not os.path.exists(source_path):
                raise FileNotFoundError(f"源模型不存在: {source_path}")
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            if os.path.isdir(source_path):
                shutil.copytree(source_path, target_path)
            else:
                shutil.copy2(source_path, target_path)
            print(f"拷贝模型: {name} ({model_type})")
    else:
        print("所需模型已存在于 Ramdisk，跳过拷贝")


def setup_multi_gpu_environment(model_configs, test_type):
    """设置多卡测试环境"""
    start_time = time.time()
    print("设置多卡测试环境...")
    # 1. 先拷贝模型到 Ramdisk
    if model_configs:
        copy_models_to_ramdisk(model_configs, test_type)

    # 2. 再切换软链接到 Ramdisk
    switch_model_paths(use_ramdisk=True)

    elapsed_time = time.time() - start_time
    print(f"多卡测试环境设置完成，耗时: {elapsed_time:.2f} 秒")


def setup_single_gpu_environment():
    """设置单卡测试环境"""
    print("设置单卡测试环境...")
    switch_model_paths(use_ramdisk=False)
    print("单卡测试环境设置完成")
