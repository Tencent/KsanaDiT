# KsanaDiT 环境变量配置

## 环境变量列表

- `KSANA_LOGGER_LEVEL`: 日志级别，`debug` / `info` (默认) / `warn` / `error` / `debug`
- `KSANA_MEMORY_PROFILER`: 启用内存分析器，`1` (启用) / `0` (默认关闭)，启用后会记录详细内存使用并生成 CSV 文件

## 使用方式

```bash
export KSANA_MEMORY_PROFILER=1
export KSANA_LOGGER_LEVEL=debug
python your_script.py
```
