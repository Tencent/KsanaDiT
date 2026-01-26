from ksana.config.lora_config import KsanaLoraConfig


def build_list_of_lora_config(lora_inputs: list[tuple[str, float]] | tuple[str, float]):
    loras_list = []
    if not isinstance(lora_inputs, list):
        lora_inputs = [lora_inputs]
    for lora_path, strength in lora_inputs:
        if isinstance(strength, list):
            raise ValueError(f"lora strength must be a scalar, but got {strength}")
        if lora_path is None:
            continue
        loras_list.append(KsanaLoraConfig(path=lora_path, strength=round(strength, 4)))
    return loras_list
