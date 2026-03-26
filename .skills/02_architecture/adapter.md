# Adapter / ComfyUI

adapter 为了让 kdit 的 Node 可以适配多种其他工具的一个适配层。目前只有 ComfyUI，未来或许还有别的。

本质上，adapter 只需要一点点的适配层，就可以直接调用 kdit 的 Node 达到实现适配的目的。

语义上，ComfyUI 的 workflow 是一个 JSON 文件，包含了 Node 的组合以及 Node 之间的连接关系。这个理论上就应该可以通过Node描述成一个 PipelineDef后在本地运行。
本质 ComfyUI 的 workflow 应该和 Pipeline 在一个层级。

**依赖方向**：
`kdit/adapter/comfyui/` → `kdit/` ✅；
`kdit/` → `kdit/adapter/` ❌ 禁止

---

## 包结构

- [`kdit/adapter/comfyui/nodes/`](../../kdit/adapter/comfyui/nodes/) 是 ComfyUI 插件节点，未来需要重构到`kdit/adapter/comfyui`
- [`kdit/nodes/`](../../kdit/nodes/) 是 kdit 内部节点
- 两者 API 完全不同，不要混淆，adapter就是负责两者接口的薄适配
