# install
```
pip install -e .
```

```python
pip install --force-reinstall .
```

# 环境变量配置

```bash
export KSANA_LOGGER_LEVEL=debug          # 日志级别: debug/info/warn/error
export KSANA_MEMORY_PROFILER=1           # 启用内存分析
```

详细说明参考 [ENV_CONFIG.md](ENV_CONFIG.md)

# contribute

code style check

```
pip install pre-commit
pip install black ruff
pre-commit install
```

# Thanks

we learn from
- [Wan-Video](https://github.com/Wan-Video/Wan2.2)
- [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)
- [fastvideo](https://github.com/hao-ai-lab/FastVideo)
- [nunchaku](https://github.com/nunchaku-tech/nunchaku)
