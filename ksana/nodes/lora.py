from ksana.config.lora_config import KsanaLoraConfig


def build_loras_list(lora_inputs):
    loras_list = []
    for lora_path, strength in lora_inputs:
        s = round(strength, 4) if not isinstance(strength, list) else strength
        if lora_path is None:
            continue
        loras_list.append(KsanaLoraConfig(path=lora_path, strength=s))
    return loras_list
