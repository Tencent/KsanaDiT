# 交付验收：测试、格式检查与文档同步

---

## 新增功能必须同步单元测试

### 规则

新增或修改功能代码时，**必须**同步添加或更新对应的单元测试。

### 要求

1. **新增功能**：必须在 `tests/` 目录下添加对应的测试文件或测试方法
2. **修改功能**：如果修改了已有功能的行为，必须更新对应的测试用例以覆盖新行为
3. **Bug 修复**：修复 bug 时应添加回归测试，防止问题复现

### 测试文件命名规范

测试文件**统一使用 `*_test.py` 后缀**（而非 `test_*.py` 前缀）。

```
✅  wan_generator_test.py
✅  factory_test.py
✅  context_builder_test.py
❌  test_wan_generator.py    ← 禁止
❌  test_factory.py          ← 禁止
```

验证命令：
```bash
# 检查是否存在违规的 test_*.py 文件
find tests/kdit -name "test_*.py" -not -name "__*"
# 预期输出为空
```

> **pytest 兼容性**：`pyproject.toml` 中 `python_files = ["*_test.py"]`，仅识别 `*_test.py` 后缀。
> 项目约定统一使用 `*_test.py`，禁止 `test_*.py` 前缀。

### 测试文件位置

测试文件位于 `tests/kdit/`，镜像 `kdit/` 的目录结构（部分模块测试放在 `tests/kdit/infra/` 下）：

```
kdit/utils/factory.py                →  tests/kdit/utils/factory_test.py
kdit/executor/executor.py            →  tests/kdit/infra/executor_run_node_test.py
kdit/generators/generator_def.py     →  tests/kdit/infra/generators/generator_def_test.py
```

### 运行方式

```bash
# 运行全部单测
pytest -s -v tests/kdit

# 运行单个测试文件
pytest -s -v tests/kdit/infra/generators/generator_def_test.py

# 运行单个测试方法
pytest -s -v tests/kdit/infra/generators/generator_def_test.py::TestClassName::test_method
```

---

## 代码格式检查规范

### 规则

格式检查通过 `pre-commit` git hooks 在 `git commit` 时**自动执行**。**不依赖**手动运行 `black`、`ruff` 等工具。

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

### pytest 与 pre-commit 的检查维度差异

> **关键认知**：pytest 和 pre-commit 检查的是**完全不同的维度**，两者缺一不可。仅靠 pytest 全绿不能认为代码正确。

| 检查工具 | 检查方式 | 能发现的问题 | 不能发现的问题 |
|---------|---------|-------------|-------------|
| pytest | **运行时**执行代码 | 逻辑错误、功能缺陷、运行时异常 | 类型注解引用未定义名称（F821）、格式问题、unused import |
| pre-commit (ruff) | **静态**分析源码 | F821、格式问题、import 顺序、未使用变量 | 逻辑错误、运行时行为 |

**典型陷阱**：`from __future__ import annotations`（PEP 563）使所有类型注解变成惰性字符串，运行时不求值。这意味着注解中引用的未导入名称在 pytest 中**永远不会报错**，只有 ruff 等静态工具才能发现。

```python
# 示例：pytest 通过但 ruff 报 F821
from __future__ import annotations

class Foo:
    def bar(self) -> UndefinedType:  # F821: ruff 报错，pytest 不报错
        return 42
```

**实操要求**：实现完成后，必须先 `git commit`（触发 pre-commit）检查静态问题，再运行 pytest 检查运行时行为。两步都通过才算完成。

---

## 代码修改必须同步更新文档

### 规则

修改代码行为时，**必须**同步更新所有受影响的文档。

### 文档范围

| 文档位置 | 内容 | 更新时机 |
|---------|------|---------|
| `.skills/` | 架构和编码规范 | 新增/修改编码规范、架构模式、API 约定时 |
| `AGENTS.md` | Agent 工作指南（概要级） | 新增/修改关键模式、测试约定、配置系统时 |
| `.roo/rules-*/AGENTS.md` | 各模式特定规则 | 新增/修改与特定模式相关的规则时 |
| `docs/` | 用户文档和 API 文档 | 新增/修改公开 API、架构变更时 |
| `README.md` | 项目概览 | 新增支持的模型、重大功能变更时 |

### 要求

1. **API 变更**：修改公开 API 签名或行为时，更新 `docs/api/` 下对应的文档和 Python docstring
2. **架构变更**：修改核心架构（如 Node 调度策略、Pipeline 定义方式）时，更新 `.skills/02_architecture/` `.skills/03_standards/` 和 `AGENTS.md`
3. **新增模型支持**：添加新模型时，更新 `docs/guide/supported-models*.md` 和相关 Pipeline 文档
4. **规则变更**：修改编码规范时，同步更新 `.skills/03_standards/`、`AGENTS.md`、`.roo/rules-*/AGENTS.md`
