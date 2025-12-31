import copy
import os

from dataclasses import is_dataclass, replace


def evolve_with_recommend(input_config, recommend_config, force_update=False):
    """
    search all vars in input_config, if var is None or force_update, then use the vars in recommend_config if not None
    input_config: dict|dataclasses
    recommend_config: dict|dataclasses
    force_update: bool, if True, then update all vars in input_config, even if they are not None
    """
    out_config = input_config if is_dataclass(input_config) else copy.deepcopy(input_config)
    change_dict = {}

    def check_key_in_obj(
        dict_or_obj: dict,
        key: str,
    ) -> bool:
        return key in dict_or_obj.keys() if isinstance(dict_or_obj, dict) else hasattr(dict_or_obj, key)

    def get_value_in_obj(dict_or_obj: dict, key: str):
        return dict_or_obj.get(key, None) if isinstance(dict_or_obj, dict) else getattr(dict_or_obj, key, None)

    def set_value_in_obj(dict_or_obj: dict, key: str, value: object):
        if isinstance(dict_or_obj, dict):
            dict_or_obj[key] = value
        else:
            setattr(dict_or_obj, key, value)

    list_all_keys = list(out_config.keys()) if isinstance(out_config, dict) else list(vars(out_config))
    for attr in list_all_keys:
        if check_key_in_obj(out_config, attr):
            now = get_value_in_obj(recommend_config, attr)
            if now is None:
                continue
            last = get_value_in_obj(out_config, attr)
            if force_update or last is None:
                change_dict[attr] = now

    if is_dataclass(out_config):
        out_config = replace(out_config, **change_dict)
    else:
        for attr, val_rec in change_dict.items():
            set_value_in_obj(out_config, attr, val_rec)
    return out_config


def is_dir(path):
    if path is None or isinstance(path, (list, tuple)):
        return False
    return os.path.isdir(path) and os.path.exists(path)


def any_key_in_str(key_list: list[str], full_str: str) -> int | None:
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
