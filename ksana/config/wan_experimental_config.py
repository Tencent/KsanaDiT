from dataclasses import dataclass, field


@dataclass(frozen=True)
class KsanaSLGConfig:
    """
    Skip Layer Guidance (SLG) Configuration.

    Source: ComfyUI-WanVideoWrapper/nodes.py:WanVideoSLG

    SLG is an optimization technique that skips the unconditional (negative prompt)
    inference pass on specified transformer blocks during CFG (Classifier-Free Guidance)
    computation. This can significantly speed up sampling while maintaining quality.

    The idea is that not all transformer blocks contribute equally to the CFG signal,
    so we can skip uncond computation on less important blocks.

    Attributes:
        blocks: List of transformer block indices (0-indexed) to skip uncond inference.
                For Wan2.1/2.2 14B model, typically blocks like [9, 10, 11] work well.
                Default is empty (disabled).
        start_percent: Start percentage of sampling steps to enable SLG (0.0 = beginning).
                       Default 0.1 means SLG starts after 10% of steps complete.
        end_percent: End percentage of sampling steps to enable SLG (1.0 = end).
                     Default 1.0 means SLG stays enabled until the final step.
    """

    blocks: list[int] = field(default_factory=list)
    start_percent: float = 0.1
    end_percent: float = 1.0


@dataclass(frozen=True)
class KsanaFETAConfig:
    """
    Enhance-A-Video (FETA) Configuration.

    Source: ComfyUI-WanVideoWrapper/nodes.py:WanVideoEnhanceAVideo
            ComfyUI-WanVideoWrapper/enhance_a_video/enhance.py

    FETA (Frame-Enhanced Temporal Attention) improves video temporal consistency
    by computing attention scores across frames and using them to modulate
    the self-attention output. This helps reduce flickering and improves
    frame-to-frame coherence.

    Reference: https://github.com/NUS-HPC-AI-Lab/Enhance-A-Video

    The algorithm works by:
    1. Computing temporal attention matrix between frames for each spatial position
    2. Calculating off-diagonal attention scores (cross-frame attention)
    3. Using these scores to scale the attention output, emphasizing temporal consistency

    Attributes:
        weight: Enhancement weight multiplier. Higher values increase temporal smoothing.
                Default 2.0. Range typically 0.0-10.0.
        start_percent: Start percentage of sampling steps to enable FETA.
                       Default 0.0 means FETA starts from the beginning.
        end_percent: End percentage of sampling steps to enable FETA.
                     Default 1.0 means FETA stays enabled until the end.
    """

    weight: float = 2.0
    start_percent: float = 0.0
    end_percent: float = 1.0


@dataclass(frozen=True)
class KsanaExperimentalConfig:
    """
    Experimental Sampling Optimizations Configuration.

    Source: ComfyUI-WanVideoWrapper/nodes.py:WanVideoExperimentalArgs
            ComfyUI-WanVideoWrapper/nodes_sampler.py
            ComfyUI-WanVideoWrapper/utils.py:temporal_score_rescaling

    This configuration groups various experimental techniques for improving
    video generation quality during the sampling/denoising process.

    Each technique targets different aspects of the generation:
    - CFG optimization: cfg_zero_star, raag_alpha
    - Frequency domain: use_fresca
    - Temporal consistency: temporal_score_rescaling, bidirectional_sampling
    - Attention control: video_attention_split_steps

    Attributes:
        cfg_zero_star: Enable CFG-Zero-Star optimization.
                       Reference: https://github.com/WeichenFan/CFG-Zero-star

        use_zero_init: When cfg_zero_star is enabled, whether to return zero noise
                       prediction for the first few steps. Default True.

        zero_star_steps: Number of initial steps to apply zero initialization. Default 0.

        use_fresca: Enable FreSca (Frequency Scaling) optimization.
                    Reference: https://github.com/WikiChao/FreSca

        fresca_scale_low: Low frequency scale factor for FreSca. Default 1.0.
        fresca_scale_high: High frequency scale factor for FreSca. Default 1.25.
        fresca_freq_cutoff: Frequency cutoff threshold for FreSca. Default 20.

        use_tcfg: Enable TCFG (Tangential Classifier-Free Guidance).
                  Reference: https://arxiv.org/abs/2503.18137

        raag_alpha: RAAG alpha value. 0.0 = disabled, typical range 0.5-2.0. Default 0.0.

        bidirectional_sampling: Enable bidirectional temporal sampling.
                                Reference: https://github.com/ff2416/WanFM

        temporal_score_rescaling: Enable TSR (Temporal Score Rescaling).
                                  Reference: https://github.com/temporalscorerescaling/TSR

        tsr_k: TSR temperature parameter. Default 0.95.
        tsr_sigma: TSR sigma parameter. Range [0, 1]. Default 1.0.

        video_attention_split_steps: Comma-separated step indices for attention split. Default "".
    """

    # CFG-Zero-Star: Optimize CFG scaling to reduce artifacts
    cfg_zero_star: bool = False
    use_zero_init: bool = False
    zero_star_steps: int = 0

    # FreSca: Frequency-domain scaling for CFG
    use_fresca: bool = False
    fresca_scale_low: float = 1.0
    fresca_scale_high: float = 1.25
    fresca_freq_cutoff: int = 20

    # TCFG: Tangential CFG to reduce color shifts
    use_tcfg: bool = False

    # RAAG: Ratio-aware adaptive guidance
    raag_alpha: float = 0.0

    # Bidirectional sampling for temporal consistency
    bidirectional_sampling: bool = False

    # TSR: Temporal score rescaling
    temporal_score_rescaling: bool = False
    tsr_k: float = 0.95
    tsr_sigma: float = 1.0

    # Attention split for multi-prompt transitions
    video_attention_split_steps: str = ""
