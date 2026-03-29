# DeviceInfo — 设备信息

> 源文件：[`kdit/nodes/core/device_info.py`](../../kdit/nodes/core/device_info.py)

```python
@dataclass(frozen=True)
class DeviceInfo:
    compute_device: torch.device  # 计算设备
    offload_device: torch.device  # 卸载设备
    rank_id: int                # 当前卡的 rank
    world_size: int             # 总卡数
```

## 关键规则

- DeviceInfo 嵌入 `NodeContext.device` 字段
- **由 Executor 自动注入，禁止用户手动设置**
- Pipeline 层构建 NodeContext 时 `device` 字段为 `None`
- Executor 在调用 Node 前自动注入：`context.device = self.device_info`
- Node 中通过 `context.device.compute_device`、`context.device.offload_device` 等访问

## 相关文档

- NodeContext → [`node-context.md`](node-context.md)
- 架构总览 → [`overview.md`](overview.md)
