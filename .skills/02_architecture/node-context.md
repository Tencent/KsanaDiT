# NodeContext — Node 间传递的上下文

> 源文件：[`kdit/nodes/core/node_context.py`](../../kdit/nodes/core/node_context.py)

**职责**：携带配置、元数据和设备信息，可安全跨 Ray 边界序列化。

```python
@dataclass
class NodeContext:
    prompt: str | list[str] = None
    negative_prompt: str | list[str] = None
    img_path: str | list[str] | list[list[str]] = None
    sample_config: SampleConfig = None
    runtime_config: RuntimeConfig = None
    cache_config: list = None
    metadata: dict = field(default_factory=dict)
    device: DeviceInfo = None  # Executor 自动注入，禁止手动设置
```

## 原则事项

- **禁止包含任何 tensor**（`__post_init__` 校验 `metadata` 字段和一层 dict values 不含 `torch.Tensor`）
- **device 字段由 Executor 自动注入**，Pipeline 层构建时留 `None`
- 可安全跨 Ray 边界序列化
- 额外配置通过 `metadata` dict 传递

## 相关文档

- 设备信息 → [`device-info.md`](device-info.md)
- 架构总览 → [`overview.md`](overview.md)
- Node 设计 → [`node.md`](node.md)
