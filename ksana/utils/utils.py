def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        # else:
        # instances[cls].clean()
        # instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance
