import threading
import copy
import os
from typing import Optional


def get_recommend_config(input_config, recommend_config):
    """
    search all vars in input_config, if var is None, then use the vars in recommend_config if not None
    """
    out_config = copy.deepcopy(input_config)
    for attr in vars(out_config):
        if getattr(out_config, attr) is None:
            if isinstance(recommend_config, dict):
                val_rec = recommend_config.get(attr, None)
            else:
                val_rec = getattr(recommend_config, attr, None)
            if val_rec is not None:
                setattr(out_config, attr, val_rec)
    return out_config


def is_dir(path):
    if path is None or isinstance(path, (list, tuple)):
        return False
    return os.path.isdir(path) and os.path.exists(path)


def any_key_in_str(key_list: list[str], full_str: str) -> Optional[int]:
    """find if any key in key_list is in full_str
    Args:
        key_list (list[str]): list of keys to check
        full_str (str): full string to check

    Returns:
        int: index of the first key in key_list that is in full_str, None if not found
    """
    idx = 0
    for key in key_list:
        if full_str.find(key) != -1:
            return idx
        idx += 1
    return None


def singleton(cls):
    """线程安全的单例装饰器"""
    instances = {}
    lock = threading.Lock()

    def get_instance(*args, **kwargs):
        with lock:
            if cls not in instances:
                instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    for attr_name in dir(cls):
        if not attr_name.startswith("__"):
            attr_value = getattr(cls, attr_name)
            if callable(attr_value) or isinstance(attr_value, (staticmethod, classmethod)):
                try:
                    setattr(get_instance, attr_name, attr_value)
                except (AttributeError, TypeError):
                    pass
    return get_instance
