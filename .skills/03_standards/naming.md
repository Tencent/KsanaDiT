# 类命名规范

---

## 规则

- `kdit/` 包内的自定义类名**不加** `Ksana` 前缀（冗余），例如 `Engine` 而非 `KsanaEngine`
- `kdit/adapter/comfyui/` 中的类**可以保留** `Ksana` 前缀（对外标识）
- `KSANA_` 开头的常量**保留**

## 验证

```bash
# 检查 kdit/ 包内是否还有 Ksana 前缀的类（排除 adapter）
grep -rn "class Ksana" kdit/ --include="*.py" | grep -v __pycache__ | grep -v "adapter/comfyui/"
```

预期输出为空（重命名正在进行中。当前残留集中在 `kdit/models/`、`kdit/config/`、`kdit/cache/`、`kdit/operations/`、`kdit/utils/`、`kdit/scheduler/` 等模块，需逐步清理）。
