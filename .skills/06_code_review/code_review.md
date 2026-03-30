# kDiT Code Review 详细检查清单

---

## P0：架构合规（最高优先级）

基于 `.skills/02_architecture/` 和 `.skills/03_standards/` 的 14 项架构约束：

### 依赖方向

1. **adapter → kdit 单向依赖** — `kdit/adapter/comfyui/` 可依赖 `kdit/`，反向禁止
2. **Node 不直接依赖 TensorPool** — 必须通过 PinHub 访问 tensor
3. **Generator 内部 tensor 不经过 DAG** — noise/denoise step 等在 Generator 内部流转

### Node 签名

4. **InferNode 使用 PinDef 声明输入输出** — 不允许硬编码 tensor key
5. **IONode 不参与 PinHub 沙箱** — 直接操作 pool
6. **NodeDef 必须包含 node_type + model_key** — 不允许省略

### Key 类型体系

7. **ModelKey / PipelineKey / TensorKey 使用正确** — 不混用、不自创 key
8. **PoolKey 间接寻址** — 跨 Node 传递必须用 TensorPoolKey / ModelPoolKey

### Import 与命名

9. **Import 风格符合方案 B** — 同子包相对导入，跨子包绝对导入
10. **禁止三级相对导入（`from ...xxx`）**
11. **类型注解使用 Python 3.10+ 原生语法** — `X | Y`、`list[str]`，不用 `Optional`/`Union`
12. **kdit/ 内部类不加 `Ksana` 前缀** — 仅 adapter 层 Node 加 `KDiT` 前缀

### 异常处理

13. **禁止裸 `except Exception`** — 必须捕获具体异常类型
14. **禁止 `except: pass` 吞异常** — 至少加日志

---

## P1：逻辑正确性

### 数据流

- [ ] tensor 生命周期是否正确（feed → run → get → clear）
- [ ] 中间节点是否误 clear tensor
- [ ] 最终消费者是否在 finally 中 clear_all_tensors

### 边界条件

- [ ] None 值是否正确处理
- [ ] 空列表/空 dict 是否有兜底
- [ ] batch_size = 1 和 batch_size > 1 是否都能工作

### 资源管理

- [ ] GPU 内存是否在异常路径也能释放
- [ ] 文件句柄是否用 with 管理
- [ ] Ray actor 是否正确清理

### 并发安全

- [ ] 类变量/全局变量是否有竞态风险
- [ ] singleton 是否线程安全

---

## P2：安全漏洞

- [ ] 命令注入（用户输入拼接到 shell 命令）
- [ ] 路径遍历（用户输入用于文件路径）
- [ ] 不安全的反序列化（pickle.load 不可信数据）
- [ ] 敏感信息泄漏（日志中打印 token/密钥）
- [ ] 依赖注入（动态 import 不可信模块）

---

## P3：代码质量

### 格式规范

- [ ] black 120 字符行宽
- [ ] ruff 无新增警告
- [ ] 无多余的 `# pylint: disable` 注释

### 测试覆盖

- [ ] 新功能是否有对应测试
- [ ] 测试文件命名 `*_test.py`，放在 `tests/kdit/` 镜像目录
- [ ] 测试不依赖外部状态（模型文件、网络、特定 GPU）

### 复杂度

- [ ] 函数不超过 50 行（超过应拆分）
- [ ] 嵌套层级不超过 4 层
- [ ] 无重复代码块（3 处以上相同逻辑应提取）

---

## 评审输出格式

```markdown
## Code Review 结果

### P0 架构合规
- [文件:行号] 问题描述 → 修复建议

### P1 逻辑正确性
- [文件:行号] 问题描述 → 修复建议

### P2 安全漏洞
- （无 / 具体问题）

### P3 代码质量
- [文件:行号] 问题描述 → 修复建议

### 总结
- 必须修复：X 项（P0/P1）
- 建议修复：Y 项（P2/P3）
```
