"""
Reference:
  - Diffusers: pipelines/qwenimage/pipeline_qwenimage_edit_plus.py
"""

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from ...utils.media import calculate_aligned_dimensions


class Qwen2VLMultimodalTextEncoderModel:
    PROMPT_TEMPLATE = (
        "<|im_start|>system\n"
        "Describe the key features of the input image (color, shape, size, texture, "
        "objects, background), then explain how the user's text instruction should "
        "alter or modify the image. Generate a new image that meets the user's "
        "requirements while maintaining consistency with the original input where "
        "appropriate.<|im_end|>\n"
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    )
    _DEFAULT_DROP_IDX = 64

    IMAGE_TOKEN_TEMPLATE = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"

    def __init__(
        self,
        checkpoint_path: str,
        tokenizer_path: str,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = torch.device("cpu"),
        text_len: int = 1024,
        prompt_template_drop_idx: int | None = None,
    ):
        self.device = device
        self.dtype = dtype
        self.max_length = text_len
        self.drop_idx = prompt_template_drop_idx if prompt_template_drop_idx is not None else self._DEFAULT_DROP_IDX

        self.processor = AutoProcessor.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            checkpoint_path, torch_dtype=dtype, trust_remote_code=True
        )
        self.model.to(device)
        self.model.eval()

    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> list[torch.Tensor]:
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        return list(torch.split(selected, valid_lengths.tolist(), dim=0))

    CONDITION_IMAGE_SIZE = 384 * 384

    def _build_image_prompt(self, num_images: int) -> str:
        return "".join([self.IMAGE_TOKEN_TEMPLATE.format(i + 1) for i in range(num_images)])

    def _load_and_resize_image(self, path: str) -> Image.Image:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        ratio = w / h
        new_w, new_h = calculate_aligned_dimensions(self.CONDITION_IMAGE_SIZE, ratio)
        return img.resize((new_w, new_h), Image.LANCZOS)

    def _normalize_images(self, images, num_prompts: int) -> list[list]:
        """
        将 images 转换为 list[list[PIL.Image]]
        Args:
            images: 必须是 list[list[str]] 格式，外层是 prompt 数量，内层是每个 prompt 的参考图路径
            num_prompts: prompt 数量
        Returns:
            list[list[PIL.Image]]: 外层长度 == num_prompts
        """
        if images is None or not isinstance(images, list) or len(images) == 0:
            raise ValueError("images must be non-empty list[list[str]]")

        if not isinstance(images[0], list):
            raise ValueError(
                f"images must be list[list[str]], got list[{type(images[0]).__name__}]. "
                "Each prompt should have its own list of image paths."
            )

        if len(images) != num_prompts:
            raise ValueError(f"images list length ({len(images)}) must match num_prompts ({num_prompts})")

        return [[self._load_and_resize_image(p) for p in paths] for paths in images]

    def _encode_single(self, prompt: str, images: list, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        img_prompt = self._build_image_prompt(len(images))
        txt = self.PROMPT_TEMPLATE.format(img_prompt + prompt)

        model_inputs = self.processor(
            text=[txt],
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                pixel_values=model_inputs.pixel_values,
                image_grid_thw=model_inputs.image_grid_thw,
                output_hidden_states=True,
            )

        hidden_states = outputs.hidden_states[-1]
        split_hidden = self._extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        embeds = split_hidden[0][self.drop_idx :]  # 单个样本
        mask = torch.ones(embeds.size(0), dtype=torch.long, device=embeds.device)

        return embeds.to(device, dtype=self.dtype), mask.to(device)

    def __call__(
        self,
        prompt: str | list[str],
        images,
        device: torch.device = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = device or self.device
        prompt = [prompt] if isinstance(prompt, str) else prompt

        images_per_prompt = self._normalize_images(images, len(prompt))

        all_embeds = []
        all_masks = []
        for p, imgs in zip(prompt, images_per_prompt):
            embeds, mask = self._encode_single(p, imgs, device)
            all_embeds.append(embeds)
            all_masks.append(mask)

        max_seq_len = max(e.size(0) for e in all_embeds)
        prompt_embeds = torch.stack(
            [torch.cat([e, e.new_zeros(max_seq_len - e.size(0), e.size(1))]) for e in all_embeds]
        )
        attention_mask = torch.stack([torch.cat([m, m.new_zeros(max_seq_len - m.size(0))]) for m in all_masks])

        return prompt_embeds, attention_mask

    def to(self, device):
        if self.device != device:
            self.model.to(device)
        self.device = device
        return self
