# 异常处理规范

## 禁止裸 `except Exception`

**规则**: 非必要情况下，不要使用 `except Exception` 捕获过于宽泛的异常。应使用具体的异常类型。

### 正确做法

```python
# ✅ 使用具体异常类型
try:
    img = Image.open(path).convert("RGB")
except OSError:
    log.warning(f"failed to load image: {path}")

# ✅ 多个具体异常
try:
    data = json.loads(raw)
except (json.JSONDecodeError, UnicodeDecodeError):
    log.warning("invalid JSON data")

# ✅ 文件操作用 OSError（涵盖 FileNotFoundError, PermissionError 等）
try:
    with open(path) as f:
        content = f.read()
except OSError:
    log.warning(f"cannot read file: {path}")
```

### 错误做法

```python
# ❌ 过于宽泛，会吞掉 KeyboardInterrupt 以外的所有异常
try:
    img = Image.open(path).convert("RGB")
except Exception:
    log.warning(f"failed to load image: {path}")
```

### 确实需要 `except Exception` 的场景

在极少数情况下（如顶层错误兜底、插件系统调用第三方代码），确实需要捕获 `Exception` 时，**必须**添加 pylint 禁用注释：

```python
try:
    plugin.execute()
except Exception:  # pylint: disable=broad-except
    log.error("plugin execution failed", exc_info=True)
```

### 常见异常类型速查

| 场景 | 推荐异常类型 |
|------|-------------|
| 文件读写 / 图片加载 (PIL) | `OSError` |
| JSON 解析 | `json.JSONDecodeError` |
| YAML 解析 | `yaml.YAMLError` |
| 类型转换 | `ValueError`, `TypeError` |
| 字典/列表索引 | `KeyError`, `IndexError` |
| 网络请求 | `requests.RequestException` / `urllib.error.URLError` |
| torch 操作 | `RuntimeError` |
| 导入模块 | `ImportError` |

### Lint 工具配置

- 如需豁免，使用行内注释 `# pylint: disable=broad-except`
