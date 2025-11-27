import threading
import os


def is_dir(path):
    if path is None or isinstance(path, (list, tuple)):
        return False
    return os.path.isdir(path) and os.path.exists(path)


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
