---
name: kdit-architecture
description: >
  kDiT 架构知识库：全局架构图、数据流、Node/Pipeline/Generator/Adapter 子系统设计、
  PinHub 沙箱机制、PoolKey 间接寻址、DeviceInfo/NodeContext 规范。
  关键词：architecture、架构、Node、Pipeline、Generator、Adapter、Engine、Executor、PinHub、Pool。
---

# kDiT 架构知识库

## 适用场景

- 理解模块关系、评估改动范围
- 架构讨论与设计评审
- 新功能涉及跨模块交互时

## 文档索引

| 文件 | 说明 |
|------|------|
| [`overview.md`](overview.md) | 架构总览、组件关系、数据流、Ownership |
| [`pipeline.md`](pipeline.md) | Pipeline 编排层、DAG、ExtraInputs、ContextBuilder |
| [`generator.md`](generator.md) | Generator 去噪引擎、BaseLatent/AuxLatent、Handler |
| [`node.md`](node.md) | Node 计算单元、Def/Pin、dispatch_policy |
| [`node-context.md`](node-context.md) | NodeContext 上下文、metadata 禁止含 tensor |
| [`pin-hub.md`](pin-hub.md) | PinHub 沙箱化数据访问器 |
| [`pool-key.md`](pool-key.md) | TensorPool/ModelPool、PoolKey、引用计数 |
| [`adapter.md`](adapter.md) | ComfyUI 适配层、依赖方向 |
| [`device-info.md`](device-info.md) | DeviceInfo 设备信息 |

## 相关 skill

| Skill | 何时调用 |
|-------|---------|
| `/kdit-standards` | 编写代码时确认规范 |
| `/kdit-quality` | 测试设计、格式检查 |
