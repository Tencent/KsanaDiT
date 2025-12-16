import threading
import os
from typing import Optional


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
