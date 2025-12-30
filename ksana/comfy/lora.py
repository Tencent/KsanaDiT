import folder_paths


def build_loras_list(lora_inputs):
    loras_list = []
    for lora_name, strength in lora_inputs:
        s = round(strength, 4) if not isinstance(strength, list) else strength
        if not lora_name or lora_name == "Empty":
            continue
        loras_list.append(
            {
                "path": folder_paths.get_full_path_or_raise("loras", lora_name),
                "strength": s,
            }
        )
    return loras_list
