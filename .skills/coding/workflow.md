# 工作流规范：测试、格式检查与文档同步

> 本文件从 [`.skills/coding.md`](../coding.md) 拆分，包含 §13、§14、§15。

---

## 13. 新增功能必须同步单元测试

### 规则

新增或修改功能代码时，**必须**同步添加或更新对应的单元测试。

### 要求

1. **新增功能**：必须在 `tests/` 目录下添加对应的测试文件或测试方法
2. **修改功能**：如果修改了已有功能的行为，必须更新对应的测试用例以覆盖新行为
3. **Bug 修复**：修复 bug 时应添加回归测试，防止问题复现

### 测试文件位置

测试文件位于 `tests/kdit/`，镜像 `kdit/` 的目录结构：

```
kdit/generators/base_generator.py    →  tests/kdit/generators/test_wan_generator.py
kdit/utils/factory.py                →  tests/kdit/utils/factory_test.py
kdit/pipelines/pipeline.py           →  tests/kdit/pipelines/wan2_2_t2v_test.py
```

### 运行方式

```bash
# 运行全部单测
pytest -s -v tests/kdit

# 运行单个测试文件
pytest -s -v tests/kdit/generators/test_wan_generator.py

# 运行单个测试方法
pytest -s -v tests/kdit/generators/test_wan_generator.py::TestClassName::test_method
```

---

## 14. 代码格式检查规范

### 规则

代码格式检查**不依赖**手动运行 `black`、`ruff` 等工具。格式检查通过 `pre-commit` git hooks 在 `git commit` 时**自动执行**。

### 设置方式

```bash
# 一次性安装（项目初始化时执行）
pre-commit install
```

安装后，每次 `git commit` 会自动运行以下检查并自动修复：

| Hook | 作用 |
|------|------|
| `trailing-whitespace` | 删除行尾空格 |
| `end-of-file-fixer` | 确保文件以换行符结束 |
| `check-yaml` | 校验 YAML 语法 |
| `black` | Python 代码格式化（行长度 120 字符） |
| `ruff` | Lint 检查 + 自动修复（规则：`I,E,F,W`，行长度 120 字符） |

### 工作流

```bash
# 正常开发流程 — 格式检查自动执行
git add .
git commit -m "feat: add new feature"
# → pre-commit 自动运行，如有格式问题会自动修复并阻止提交
# → 修复后重新 git add && git commit 即可

# 手动全量检查（可选，用于 CI 或批量修复）
pre-commit run --all-files
```

### 禁止事项

- ❌ **禁止**跳过 pre-commit 检查（`git commit --no-verify`），除非有明确的临时理由
- ❌ **禁止**手动运行 `black` 或 `ruff` 来替代 pre-commit — 以 pre-commit 配置为准，确保团队一致性

---

## 15. 代码修改必须同步更新文档

### 规则

修改代码行为时，**必须**同步更新所有受影响的文档。

### 文档范围

| 文档位置 | 内容 | 更新时机 |
|---------|------|---------|
| `.skills/coding.md` | 开发规范和编码约定 | 新增/修改编码规范、架构模式、API 约定时 |
| `AGENTS.md` | Agent 工作指南（概要级） | 新增/修改关键模式、测试约定、配置系统时 |
| `.roo/rules-*/AGENTS.md` | 各模式特定规则 | 新增/修改与特定模式相关的规则时 |
| `docs/` | 用户文档和 API 文档 | 新增/修改公开 API、架构变更时 |
| `README.md` | 项目概览 | 新增支持的模型、重大功能变更时 |

### 要求

1. **API 变更**：修改公开 API 签名或行为时，更新 `docs/api/` 下对应的文档和 Python docstring
2. **架构变更**：修改核心架构（如 Node 调度策略、Pipeline 定义方式）时，更新 `docs/architecture*.md` 和 `.skills/coding.md`
3. **新增模型支持**：添加新模型时，更新 `docs/guide/supported-models*.md` 和相关 Pipeline 文档
4. **规则变更**：修改编码规范时，同步更新 `.skills/coding.md`、`AGENTS.md`、`.roo/rules-*/AGENTS.md`
