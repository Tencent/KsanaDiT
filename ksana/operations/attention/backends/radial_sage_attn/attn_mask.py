# Copyright 2025 Tencent
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# based on https://github.com/mit-han-lab/radial-attention/blob/main/radial_attn/attn_mask.py
import torch
from tqdm import tqdm


def shrink_mask_strict(mask, block_size):
    seqlen = mask.shape[0]
    block_num = seqlen // block_size
    mask = mask[: block_num * block_size, : block_num * block_size].view(block_num, block_size, block_num, block_size)
    col_densities = mask.sum(dim=1) / block_size
    # we want the minimum non-zero column density in the block
    non_zero_densities = col_densities > 0
    high_density_cols = col_densities > 1 / 3
    frac_high_density_cols = high_density_cols.sum(dim=-1) / (non_zero_densities.sum(dim=-1) + 1e-9)
    block_mask = frac_high_density_cols > 0.6
    block_mask[0:0] = True
    block_mask[-1:-1] = True
    return block_mask


def get_diagonal_split_mask(i, j, token_per_frame, sparse_type, block_size, device):
    assert sparse_type in ["radial"]
    dist = abs(i - j)
    group = dist.bit_length()
    threshold = block_size  # CHANGE, can 64 or 128
    decay_length = 2 ** token_per_frame.bit_length() / 2**group
    if decay_length >= threshold:
        return torch.ones((token_per_frame, token_per_frame), device=device, dtype=torch.bool)

    split_factor = int(threshold / decay_length)
    modular = dist % split_factor
    return (
        torch.ones((token_per_frame, token_per_frame), device=device, dtype=torch.bool)
        if modular == 0
        else torch.zeros((token_per_frame, token_per_frame), device=device, dtype=torch.bool)
    )


def get_window_width(i, j, token_per_frame, sparse_type, decay_factor, block_size):
    assert sparse_type in ["radial"]
    dist = abs(i - j)
    if dist < 1:
        return token_per_frame
    if dist == 1:
        return token_per_frame // 2
    group = dist.bit_length()
    decay_length = 2 ** token_per_frame.bit_length() / 2**group * decay_factor
    return max(decay_length, block_size)


def gen_log_mask_shrinked(device, s, video_token_num, num_frame, block_size, sparse_type, decay_factor):
    """
    A more memory friendly version, we generate the attention mask of each frame pair at a time,
    shrinks it, and stores it into the final result
    """
    final_log_mask = torch.zeros((s // block_size, s // block_size), device=device, dtype=torch.bool)
    token_per_frame = video_token_num // num_frame
    video_text_border = video_token_num // block_size

    col_indices = torch.arange(0, token_per_frame, device=device).view(1, -1)
    row_indices = torch.arange(0, token_per_frame, device=device).view(-1, 1)
    final_log_mask[video_text_border:] = True
    final_log_mask[:, video_text_border:] = True

    for i in tqdm(range(num_frame), desc="Frames (i)"):
        for j in range(num_frame):
            if j == 0:  # this is attention sink
                local_mask = torch.ones((token_per_frame, token_per_frame), device=device, dtype=torch.bool)
            else:
                window_width = get_window_width(i, j, token_per_frame, sparse_type, decay_factor, block_size)
                local_mask = torch.abs(col_indices - row_indices) <= window_width
                split_mask = get_diagonal_split_mask(i, j, token_per_frame, sparse_type, block_size, device)
                local_mask = torch.logical_and(local_mask, split_mask)
            remainder_row = (i * token_per_frame) % block_size
            remainder_col = (j * token_per_frame) % block_size

            # get the padded size
            all_length_row = remainder_row + ((token_per_frame - 1) // block_size + 1) * block_size
            all_length_col = remainder_col + ((token_per_frame - 1) // block_size + 1) * block_size
            padded_local_mask = torch.zeros((all_length_row, all_length_col), device=device, dtype=torch.bool)
            padded_local_mask[
                remainder_row : remainder_row + token_per_frame, remainder_col : remainder_col + token_per_frame
            ] = local_mask

            # shrink the mask
            block_mask = shrink_mask_strict(padded_local_mask, block_size)

            # set the block mask to the final log mask
            block_row_start = (i * token_per_frame) // block_size
            block_col_start = (j * token_per_frame) // block_size
            block_row_end = block_row_start + block_mask.shape[0]
            block_col_end = block_col_start + block_mask.shape[1]

            final_log_mask[block_row_start:block_row_end, block_col_start:block_col_end] = torch.logical_or(
                final_log_mask[block_row_start:block_row_end, block_col_start:block_col_end], block_mask
            )
    # print(f"mask sparsity: {1 - final_log_mask.sum() / final_log_mask.numel()}")
    return final_log_mask


class MaskMap:
    def __init__(self, video_token_num=25440, num_frame=16, block_size=128, device="cuda"):
        self.video_token_num = video_token_num
        self.num_frame = num_frame
        self.log_mask = None
        self.block_size = block_size
        self.device = device

    def query_log_mask(self, seq_len, sparse_type, block_size=None, decay_factor=0.5):
        block_size = block_size or self.block_size
        log_mask = torch.ones((seq_len // block_size, seq_len // block_size), device=self.device, dtype=torch.bool)
        if self.log_mask is None:
            self.log_mask = gen_log_mask_shrinked(
                self.device,
                seq_len,
                self.video_token_num,
                self.num_frame,
                block_size=block_size,
                sparse_type=sparse_type,
                decay_factor=decay_factor,
            )
        block_bound = self.video_token_num // block_size
        log_mask[:block_bound, :block_bound] = self.log_mask[:block_bound, :block_bound]
        return log_mask

    def get_signature(self):
        return self.create_signature(self.video_token_num, self.num_frame)

    @staticmethod
    def create_signature(video_token_num, num_frame):
        return (video_token_num, num_frame)
