import torch
from torch.nn.utils.rnn import pad_sequence

from ..models.model_key import KsanaModelKey
from ..models.model_pool import KsanaModelPool
from ..units import KsanaRunnerUnit, KsanaUnitFactory, KsanaUnitType
from ..utils import str_to_list, time_range


class KsanaBaseTextEncoder(KsanaRunnerUnit):

    def _valid_input_prompt_to_list(self, prompt, target_len=None) -> list:
        prompts = str_to_list(prompt)
        if target_len is not None:
            if len(prompts) == 1:
                prompts = prompts * target_len
            elif len(prompts) != target_len:
                raise ValueError(f"prompt length ({len(prompts)}) must match target length ({target_len})")
        return prompts


# Note: 不同类型的模型应该尽量用同一个encoder实现, 不同模型加载可以根据模型有不同的加载实现
@KsanaUnitFactory.register(KsanaUnitType.ENCODER, KsanaModelKey.T5TextEncoder)
class KsanaTextEncoder(KsanaBaseTextEncoder):
    @time_range
    def run(
        self,
        model,
        prompts_positive: str | list[str],
        prompts_negative: str | list[str] | None = None,
        device=None,
        offload_device=None,
        offload_model=False,
    ):
        prompts_positive_list = self._valid_input_prompt_to_list(prompts_positive)
        prompts_negative_list = prompts_negative or getattr(model.default_settings, "neg_prompt", None)
        # seems negative can not be None
        prompts_negative_list = self._valid_input_prompt_to_list(prompts_negative_list, len(prompts_positive_list))

        assert device is not None
        if model.device != device:
            model.to(device)

        all_prompts = prompts_positive_list + prompts_negative_list
        all_embeddings_list = model.forward(all_prompts)

        # TODO(qiannan): self.text_encoder.forward tokenizer的时候是填充到相同长度了，
        # 但是返回是裁剪了，所以如果返回不裁剪，就不需要pad了
        # Pad the combined list of tensors to the max length in the entire batch.
        all_padded_embeddings = pad_sequence(all_embeddings_list, batch_first=True, padding_value=0.0)

        # Split the padded tensor back into positive and negative parts.
        positive, negative = torch.chunk(all_padded_embeddings, 2, dim=0)

        if offload_model and offload_device is not None and offload_device != device:
            model.to(offload_device)

        return positive, negative


@KsanaUnitFactory.register(KsanaUnitType.ENCODER, KsanaModelKey.Qwen2VLTextEncoder)
class KsanaQwenVLTextEncoderUnit(KsanaBaseTextEncoder):
    def forward_text_encoder(
        self,
        model_pool: KsanaModelPool,
        prompts_positive,
        prompts_negative=None,
        device=None,
        offload_device=None,
        offload_model=False,
    ):
        bs = len(prompts_positive)
        text_encoder = model_pool.get_model(self.text_encoder_key)
        default_neg_prompt = self.pipeline_config.default_config.sample_neg_prompt
        prompts_negative = prompts_negative if prompts_negative is not None else [default_neg_prompt] * bs

        if len(prompts_positive) != len(prompts_negative):
            raise RuntimeError(
                f"The number of negative prompts ({len(prompts_negative)}) "
                f"must match the number of positive prompts ({bs})."
            )

        if text_encoder.device != device:
            text_encoder.to(device)

        positive_embeds, positive_mask = text_encoder.forward(prompts_positive, device=device)
        negative_embeds, negative_mask = text_encoder.forward(prompts_negative, device=device)

        if offload_model:
            target_offload = offload_device or torch.device("cpu")
            text_encoder.to(target_offload)

        return (positive_embeds, positive_mask), (negative_embeds, negative_mask)
