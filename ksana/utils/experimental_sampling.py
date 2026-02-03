"""
Experimental Sampling Utilities for Video Diffusion Models.

This module contains various experimental techniques to improve video generation quality,
reduce artifacts, or speed up sampling. These are referenced from WanVideoWrapper and
other research implementations.
"""

import torch


class ExperimentalSamplingUtils:
    """
    A collection of experimental sampling optimization techniques for video diffusion models.

    These techniques can be used to:
    - Reduce CFG-induced artifacts (CFG-Zero-Star, TCFG, FreSca)
    - Improve temporal consistency (TSR)
    - Adaptively adjust guidance (RAAG)

    All methods are static and can be called without instantiation.
    """

    @staticmethod
    def compute_cfg_zero_star_alpha(noise_pred_cond: torch.Tensor, noise_pred_uncond: torch.Tensor) -> torch.Tensor:
        """
        CFG-Zero-Star: Compute optimal scaling factor for unconditional prediction.

        This technique scales the unconditional prediction to better align with
        the conditional prediction direction, reducing oversaturation artifacts
        that can occur with high CFG values.

        Reference: https://github.com/WeichenFan/CFG-Zero-star

        Args:
            noise_pred_cond: Conditional (positive prompt) noise prediction
            noise_pred_uncond: Unconditional (negative prompt) noise prediction

        Returns:
            Optimal scaling factor alpha for uncond prediction
        """
        batch_size = noise_pred_cond.shape[0]
        cond_flat = noise_pred_cond.float().view(batch_size, -1)
        uncond_flat = noise_pred_uncond.float().view(batch_size, -1)

        # Compute optimal scale: alpha = (cond · uncond) / (uncond · uncond)
        dot_product = (cond_flat * uncond_flat).sum(dim=-1, keepdim=True)
        uncond_norm_sq = (uncond_flat * uncond_flat).sum(dim=-1, keepdim=True)

        # Avoid division by zero
        alpha = dot_product / (uncond_norm_sq + 1e-8)
        alpha = alpha.clamp(min=0.0, max=2.0)  # Reasonable range

        # Reshape to broadcast with noise predictions
        alpha = alpha.to(dtype=noise_pred_uncond.dtype)
        if noise_pred_cond.ndim == 5:
            return alpha.view(batch_size, 1, 1, 1, 1)
        else:
            return alpha.view(batch_size, 1, 1, 1)

    @staticmethod
    def tangential_projection(noise_pred_cond: torch.Tensor, noise_pred_uncond: torch.Tensor) -> torch.Tensor:
        """
        TCFG: Tangential Classifier-Free Guidance.

        Projects the unconditional prediction onto the tangent plane perpendicular
        to the conditional prediction. This reduces CFG-induced color shifts and
        artifacts while preserving the guidance direction.

        Reference: https://arxiv.org/abs/2503.18137

        Args:
            noise_pred_cond: Conditional noise prediction
            noise_pred_uncond: Unconditional noise prediction

        Returns:
            Tangentially projected unconditional prediction
        """
        batch_size = noise_pred_cond.shape[0]
        cond_flat = noise_pred_cond.view(batch_size, -1)
        uncond_flat = noise_pred_uncond.view(batch_size, -1)

        # Compute projection of uncond onto cond direction
        cond_norm_sq = (cond_flat * cond_flat).sum(dim=-1, keepdim=True)
        dot_product = (cond_flat * uncond_flat).sum(dim=-1, keepdim=True)

        # Project uncond onto tangent plane: uncond - (uncond · cond_hat) * cond_hat
        projection = (dot_product / (cond_norm_sq + 1e-8)) * cond_flat
        tangent = uncond_flat - projection

        return tangent.view_as(noise_pred_uncond)

    @staticmethod
    def compute_raag_guidance(
        noise_pred_cond: torch.Tensor,
        noise_pred_uncond: torch.Tensor,
        cfg_scale: float,
        raag_alpha: float,
    ) -> float:
        """
        RAAG: Ratio-Aware Adaptive Guidance.

        Adaptively adjusts CFG scale based on the ratio between conditional and
        unconditional prediction magnitudes. This can help prevent oversaturation
        in regions where the model is already confident.

        Args:
            noise_pred_cond: Conditional noise prediction
            noise_pred_uncond: Unconditional noise prediction
            cfg_scale: Original CFG scale
            raag_alpha: RAAG alpha parameter (strength of adaptation)

        Returns:
            Adjusted CFG scale
        """
        cond_norm = noise_pred_cond.norm()
        uncond_norm = noise_pred_uncond.norm()

        # Compute ratio and adjust CFG
        ratio = cond_norm / (uncond_norm + 1e-8)
        adjusted_cfg = cfg_scale * (1.0 + raag_alpha * (1.0 - ratio.clamp(0.5, 2.0)))

        return float(adjusted_cfg.item())

    @staticmethod
    def fourier_filter(
        x: torch.Tensor,
        scale_low: float,
        scale_high: float,
        freq_cutoff: int,
    ) -> torch.Tensor:
        """
        FreSca: Frequency-domain scaling for CFG.

        Applies different scaling factors to low and high frequency components
        of the input tensor. This can reduce high-frequency artifacts while
        preserving structural details.

        Reference: https://github.com/WikiChao/FreSca

        Args:
            x: Input tensor (noise prediction difference: cond - uncond)
            scale_low: Scaling factor for low frequencies
            scale_high: Scaling factor for high frequencies
            freq_cutoff: Frequency threshold separating low and high

        Returns:
            Frequency-scaled tensor
        """
        # Apply FFT along spatial dimensions
        orig_dtype = x.dtype
        x = x.to(torch.float32)

        # FFT on last two dimensions (H, W)
        x_fft = torch.fft.fft2(x)
        x_fft = torch.fft.fftshift(x_fft, dim=(-2, -1))

        # Create frequency mask
        h, w = x.shape[-2:]
        cy, cx = h // 2, w // 2

        # Create distance grid from center
        y = torch.arange(h, device=x.device).view(-1, 1) - cy
        x_coord = torch.arange(w, device=x.device).view(1, -1) - cx
        dist = torch.sqrt(y.float() ** 2 + x_coord.float() ** 2)

        # Create scaling mask: low freq gets scale_low, high freq gets scale_high
        scale_mask = torch.where(dist <= freq_cutoff, scale_low, scale_high)
        if x.ndim == 5:
            scale_mask = scale_mask.view(1, 1, 1, h, w)
        else:
            scale_mask = scale_mask.view(1, 1, h, w)

        # Apply scaling
        x_fft = x_fft * scale_mask

        # Inverse FFT
        x_fft = torch.fft.ifftshift(x_fft, dim=(-2, -1))
        x = torch.fft.ifft2(x_fft).real

        return x.to(orig_dtype)

    @staticmethod
    def temporal_score_rescaling(
        noise_pred: torch.Tensor,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        tsr_k: float,
        tsr_sigma: float,
    ) -> torch.Tensor:
        """
        TSR: Temporal Score Rescaling.

        Rescales noise prediction based on temporal statistics to improve
        frame-to-frame consistency in video generation.

        Reference: https://github.com/temporalscorerescaling/TSR

        Args:
            noise_pred: Current noise prediction
            latent: Current latent state
            timestep: Current timestep
            tsr_k: Temperature parameter (lower = stronger rescaling)
            tsr_sigma: Sigma parameter (how early TSR influences sampling)

        Returns:
            Rescaled noise prediction
        """
        if noise_pred.ndim != 5:
            # Only apply to video tensors with temporal dimension
            return noise_pred

        # Compute temporal statistics
        # noise_pred shape: [B, C, T, H, W]
        temporal_mean = noise_pred.mean(dim=2, keepdim=True)
        temporal_std = noise_pred.std(dim=2, keepdim=True) + 1e-8

        # Normalize and rescale
        normalized = (noise_pred - temporal_mean) / temporal_std
        rescaled = normalized * (temporal_std * tsr_k) + temporal_mean

        # Blend based on sigma (higher sigma = more original, lower = more rescaled)
        return tsr_sigma * noise_pred + (1 - tsr_sigma) * rescaled


# Convenience aliases for direct function access
compute_cfg_zero_star_alpha = ExperimentalSamplingUtils.compute_cfg_zero_star_alpha
tangential_projection = ExperimentalSamplingUtils.tangential_projection
compute_raag_guidance = ExperimentalSamplingUtils.compute_raag_guidance
fourier_filter = ExperimentalSamplingUtils.fourier_filter
temporal_score_rescaling = ExperimentalSamplingUtils.temporal_score_rescaling


@torch.compiler.disable()
def compute_feta_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    num_frames: int,
    feta_weight: float = 2.0,
) -> torch.Tensor:
    """
    Compute FETA (Frame-Enhanced Temporal Attention) enhancement scores.

    FETA calculates cross-frame attention scores and uses them to modulate
    self-attention outputs for improved temporal consistency.

    Args:
        query: Query tensor with shape [B, S, N, C] where:
               B = batch size, S = sequence length (F*H*W),
               N = num_heads, C = head_dim
        key: Key tensor with same shape as query
        num_frames: Number of video frames (F)
        feta_weight: Enhancement weight (higher = stronger temporal smoothing)

    Returns:
        Enhancement score scalar to multiply with attention output
    """
    batch_size, seq_len, num_heads, head_dim = query.shape

    # Calculate spatial dimension (H * W)
    spatial_dim = seq_len // num_frames
    if spatial_dim * num_frames != seq_len:
        # Cannot reshape properly, return neutral score
        return torch.ones(1, device=query.device, dtype=query.dtype)

    # Reshape to separate spatial and temporal dimensions
    # [B, S, N, C] -> [B, spatial_dim, num_frames, N, C]
    query_image = query.reshape(batch_size, spatial_dim, num_frames, num_heads, head_dim)
    key_image = key.reshape(batch_size, spatial_dim, num_frames, num_heads, head_dim)

    # Reshape to [(B * spatial_dim), N, num_frames, C] for attention computation
    query_image = query_image.reshape(batch_size * spatial_dim, num_heads, num_frames, head_dim)
    key_image = key_image.reshape(batch_size * spatial_dim, num_heads, num_frames, head_dim)

    # Compute temporal attention scores
    scale = head_dim**-0.5
    query_scaled = query_image * scale

    # [B*spatial, N, T, C] @ [B*spatial, N, C, T] -> [B*spatial, N, T, T]
    attn_temp = query_scaled @ key_image.transpose(-2, -1)
    attn_temp = attn_temp.to(torch.float32)
    attn_temp = attn_temp.softmax(dim=-1)

    # Reshape to [batch_size * spatial_dim * num_heads, num_frames, num_frames]
    attn_temp = attn_temp.reshape(-1, num_frames, num_frames)

    # Create a mask for diagonal elements (self-attention, frame to itself)
    diag_mask = torch.eye(num_frames, device=attn_temp.device, dtype=torch.bool)
    diag_mask = diag_mask.unsqueeze(0).expand(attn_temp.shape[0], -1, -1)

    # Zero out diagonal elements to focus on cross-frame attention
    attn_wo_diag = attn_temp.masked_fill(diag_mask, 0)

    # Calculate mean attention score across all off-diagonal elements
    # Number of off-diagonal elements per matrix is T*T - T
    num_off_diag = num_frames * num_frames - num_frames
    mean_scores = attn_wo_diag.sum(dim=(1, 2)) / num_off_diag

    # Compute final enhancement score
    # Higher cross-frame attention -> higher enhancement
    enhance_scores = mean_scores.mean() * (num_frames + feta_weight)
    enhance_scores = enhance_scores.clamp(min=1.0)

    return enhance_scores
