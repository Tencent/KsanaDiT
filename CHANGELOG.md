# Changelog

All notable changes to KsanaDiT will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [v0.2.3](https://github.com/Tencent/KsanaDiT/compare/v0.2.2...v0.2.3) - 2026-03-04

### Added
- Smart caching strategies: MagCache, TeaCache, EasyCache with YAML configuration support (!186)
- XPU (Intel GPU) platform detection and adaptation, EmptyCache node for cross-platform memory release (!203)
- Qwen Image Edit 2511 model support with multimodal text encoder (!191)
- Multi-GPU VAE decoding for improved performance on high-resolution/long-video scenarios (!198)
- NPU Laser Attention backend for performance optimization (!187)
- Inference metrics monitoring and reporting (!180)
- 160GB CPU memory limit for CI/test environments (!194)
- Apache 2.0 open-source license (!184)

### Changed
- Removed `@singleton` decorator, replaced with `get_default()` classmethod for better instance lifecycle management (!199)
- Optimized Wan2.1-Vace model code architecture for better maintainability (!183)
- Changed ComfyUI from git submodule to directly included folder (!185)
- Temporarily disabled QKV Fusion globally to ensure stability (!204)
- Upgraded protobuf to fix NPU Ray multi-GPU Init serialization failure (!205)

### Fixed
- NPU communication occasional out-of-order issue (!202)
- NPU pipeline occasional numerical difference on repeated runs (!195)
- Vace model and LoRA loading logic issues (!190)
- Cross-platform `empty_cache` unified interface (!188)
- Local test run improvements (!200)

---

## [v0.2.2](https://github.com/Tencent/KsanaDiT/compare/v0.2.1...v0.2.2) - 2026-01-30

### Added
- Turbo Diffusion inference acceleration with Sage-SLA backend (!153)
- Integrated KsanaDiT Attention Op into Qwen-Image model (!169)
- Pinned Memory Manager to resolve OOM issues during large model inference (!166)
- I2V (Image-to-Video) example and test cases (!153)
- Multiple unit tests and integration tests (!167, !168)

### Changed
- Removed `easydict` dependency to reduce external dependency complexity (!175)
- Restructured test directory for better maintainability (!168)

### Fixed
- Development install script (!173)
- Code typos (!172)
- Cleaned up TODO items and improved README (!171)

---

## [v0.2.1](https://github.com/Tencent/KsanaDiT/compare/v0.2.0...v0.2.1) - 2026-01-27

### Added
- NPU multi-hardware platform support (!148)

---

## [v0.2.0](https://github.com/Tencent/KsanaDiT/compare/v0.1.5...v0.2.0) - 2026-01-26

### Added
- DPM++ sampling solver support (!162)
- Abstracted generate template for unified generation interface (!154)
- Linear backend tests (!159)
- Empty Dockerfile for future containerization (!160)

### Changed
- Optimized test suite structure (!158)
- Optimized pipeline input validation (!146)

### Fixed
- Batching multi-step shape issue (!163)
- LoRA alpha missing issue and non-float32 LoRA support (!161)
- Qwen Image batching (!156)
- Qwen model loading (!151)
- LoRA and cache issues (!149)
- `migrate_model_to_pinned_memory` and path issues (!145)
- Reverted cherry-pick to fix merge LoRA and pin memory causing high CPU memory (!141)
- Unit test fixes (!155, !150, !147)
- Test value fixes (!160)

---

## [v0.1.5](https://github.com/Tencent/KsanaDiT/compare/v0.1.4...v0.1.5) - 2026-01-08

### Added
- Radial Sage Attention backend support (!107)
- More local node tests (!126)

### Changed
- Renamed whl package (!127)
- Restructured ComfyUI to not call pipeline directly, optimized Generator (!122)
- Shortened daily CI run time (!124)
- Fixed local ksana nodes for ComfyUI node tests, fixed config errors and single model run issues (!123)
- Fixed `__all__` exports (!125)

---

## [v0.1.4](https://github.com/Tencent/KsanaDiT/compare/v0.1.3...v0.1.4) - 2026-01-07

### Added
- Model switching node tests (!118)

### Fixed
- Node name fix (!121)
- LoRA merge OOM issue (!119)
- Optimized ComfyUI tests (!117)
- Style check fixes (!116)

---

## [v0.1.3](https://github.com/Tencent/KsanaDiT/compare/v0.1.2...v0.1.3) - 2026-01-06

### Changed
- Refactored Attention and Linear backend interfaces (!113)
- Optimized Attention backend code structure (!108)
- Separated ComfyUI node definitions from whl package installation for CI pipeline (!106)
- Optimized code and restructured multi-GPU decorators (!104)
- Removed rank ID return in multi-GPU mode (!105)

### Fixed
- ComfyUI I2V to T2V switching failure (!115)
- Memory leak on repeated runs (!114)
- ComfyUI benchmark modifications (!112)
- Workflow test width/height node classtype fix (!109)
- Loading failure causing pop exception (!111)
- VAE OOM issue (!110)

---

## [v0.1.2](https://github.com/Tencent/KsanaDiT/compare/v0.1.1...v0.1.2) - 2025-12-30

### Added
- Hybrid cache and custom cache support (!97)
- Support for generating multiple videos/images from one prompt (!94)
- ComfyUI rope function support (!88)

### Changed
- Separated ComfyUI nodes into submodule (!102)
- Auto version management (!103)

### Fixed
- ComfyUI loader node clearing all other models (!101)
- VAE node model loading error (!100)
- Examples fixes and daily CI additions (!98)

---

## [v0.1.1](https://github.com/Tencent/KsanaDiT/compare/v0.1.0...v0.1.1) - 2025-12-25

### Added
- Torchao FP8 dynamic quantization support (!71)

### Fixed
- Last frame issue in video generation (!90)

---

## [v0.1.0](https://github.com/Tencent/KsanaDiT/releases/tag/v0.1.0) - 2025-12-24

### Added
- **Initial release** of KsanaDiT inference framework
- Wan2.2 T2V/I2V model support with torch.compile optimization
- FP8 quantization: FP8 model loading, FP8 GEMM computation, FP8 scale support
- Multiple Attention backends: Flash Attention, Sage Attention, Torch SDPA
- ComfyUI integration: custom nodes, LoRA nodes, progress bar, multi-GPU support
- Distributed inference: local multi-GPU, Ray multi-GPU, Model Pool management
- DBCache caching strategy
- Batch inference: merged cond/uncond, batch size > 1
- Custom sigma scheduling support
- VAE encoder/decoder with I2V ComfyUI support
- Ramdisk model loading optimization for multi-GPU
- Pipeline configuration: KsanaModelConfig, KsanaPipelineConfig
- Comprehensive test framework with ComfyUI workflow tests

### Changed
- Unified load interface: `from_pretrained` → `from_models`
- Unified attention and operation usage patterns
- WAN memory optimization: dynamic offload mechanism, Time Embedding memory fix

### Fixed
- I2V first frame whitening issue
- Multi-GPU occasional hang issues
- DCCache 720p OOM
- L20 local LoRA OOM and torch compile precision issues
- ComfyUI deadlock issues
- Multiple multi-GPU return value and process cleanup fixes
