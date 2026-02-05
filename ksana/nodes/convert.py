import torch
import torch.nn.functional as F


def convert_text_embeds_to_ksana(text_embeds: dict) -> tuple:
    def convert(embs, max_len):  # unsqueeze at dim 0 and pad at dim 1, then cat at dim 0
        # First unsqueeze all tensors at dim 0
        embs = [t.unsqueeze(0) for t in embs]
        # Pad at dim 1 (which is dim 2 after unsqueeze)
        padded_tensors = []
        for t in embs:
            if t.shape[1] < max_len:
                pad_size = max_len - t.shape[1]
                t_padded = F.pad(t, (0, 0, 0, pad_size), value=0)
                padded_tensors.append(t_padded)
            else:
                padded_tensors.append(t)
        # Cat at dim 0
        ret = torch.cat(padded_tensors, dim=0)
        # Return as a list of lists to match the using code
        return [[ret]]

    positive = text_embeds["prompt_embeds"]
    negative = text_embeds["negative_prompt_embeds"]
    target_seq_len = max(t.shape[0] for t in positive + negative)
    positive = convert(positive, target_seq_len)
    negative = convert(negative, target_seq_len)
    return positive, negative
