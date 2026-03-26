# Skills 安装指南

## 原则

`.skills/` 是 kDiT 的必备技能目录，涵盖代码规范、架构设计、工作流规则等。所有工具（Roo Code、Claude Code）通过**符号链接**引用，
**禁止拷贝**。修改只需在 `.skills/` 中进行，所有工具自动同步。

---

## Roo Code

### 背景：Roo Code 的 Skill 加载机制

Roo Code 通过以下目录发现和加载 Skill：

| 目录 | 作用域 | 说明 |
|------|--------|------|
| `.roo/skills/<skill-name>/SKILL.md` | **全模式通用** | 所有模式（code、architect、ask、debug 等）均可触发 |
| `.roo/skills-<mode>/<skill-name>/SKILL.md` | **特定模式** | 仅在对应模式下可见，如 `skills-code/` 仅 Code 模式可用 |

每个 Skill 目录必须包含一个 `SKILL.md` 文件，其 YAML frontmatter 中的 `name` 和 `description` 用于 Skill 匹配和触发。

### 我们的 Skill 清单

`.skills/` 目录中包含两类 Skill：

| Skill 名称 | 入口文件 | 适用模式 | 说明 |
|------------|---------|---------|------|
| `kdit-skill` | `.skills/SKILL.md` | **全模式**（code、architect、ask、debug） | 项目知识库：架构、规范、API 等。任何模式下理解/修改 kDiT 代码时都需要 |
| `development_spec` | `.skills/01_development_spec/development_spec.md` | **Code + Architect 模式** | 先设计后编码的 7 步工作流，在编码实现和架构讨论/设计评审时触发 |

### 目录结构总览

安装完成后，`.roo/` 目录应呈现如下结构：

```
.roo/
├── skills/                          # 全模式通用 Skill
│   └── kdit-skill/
│       └── SKILL.md                 # → 符号链接到 .skills/SKILL.md
│
├── skills-code/                     # Code 模式专属 Skill
│   └── development_spec/
│       └── SKILL.md                 # → 符号链接到 .skills/01_development_spec/development_spec.md
│
├── skills-architect/                # Architect 模式专属 Skill
│   └── development_spec/
│       └── SKILL.md                 # → 符号链接到 .skills/01_development_spec/development_spec.md
│
├── rules-code/                      # 各模式的 Rules（AGENTS.md）
│   └── AGENTS.md
├── rules-architect/
│   └── AGENTS.md
├── rules-ask/
│   └── AGENTS.md
├── rules-debug/
│   └── AGENTS.md
└── ...
```

### 安装步骤

从项目根目录执行：

```bash
# ── 1. 全模式通用 Skill：kdit-skill ──
# 所有模式（code/architect/ask/debug）都能触发
mkdir -p .roo/skills/kdit-skill
rm -f .roo/skills/kdit-skill/SKILL.md
ln -s ../../../.skills/SKILL.md .roo/skills/kdit-skill/SKILL.md

# ── 2. Code 模式 Skill：development_spec ──
# Code 模式下触发（先设计后编码工作流）
mkdir -p .roo/skills-code/development_spec
rm -f .roo/skills-code/development_spec/SKILL.md
ln -s ../../../.skills/01_development_spec/development_spec.md .roo/skills-code/development_spec/SKILL.md

# ── 3. Architect 模式 Skill：development_spec ──
# Architect 模式下触发（架构讨论与设计评审）
mkdir -p .roo/skills-architect/development_spec
rm -f .roo/skills-architect/development_spec/SKILL.md
ln -s ../../../.skills/01_development_spec/development_spec.md .roo/skills-architect/development_spec/SKILL.md
```

> **重要**：每个 SKILL.md 的 YAML frontmatter 中的 `name` 字段必须与 `.roo/` 下的目录名一致。
> 例如 `.skills/SKILL.md` 的 `name` 应为 `kdit-skill`，对应目录 `.roo/skills/kdit-skill/`；
> `.skills/01_development_spec/development_spec.md` 的 `name` 应为 `development_spec`，对应目录 `.roo/skills-code/development_spec/` 和 `.roo/skills-architect/development_spec/`。

> **注意**：Skill 的符号链接是**文件级**（链接 `SKILL.md` 文件本身），而非目录级。
> 这是因为 `.skills/` 的目录结构（按编号分类）与 Roo Code 要求的目录结构（按 skill-name 命名）不同。
> Skill 内部通过相对路径引用其他 `.skills/` 文件时，需要确保路径从链接目标位置出发仍然有效。

### Skill 内部的相对路径问题

`SKILL.md` 中通常会引用 `.skills/` 下的其他文档（如 `../02_architecture/overview.md`）。
由于符号链接的目标仍在 `.skills/` 目录内，这些相对路径**天然有效**，无需额外处理。

但如果 Skill 需要通过 `read_file` 工具读取关联文件，Agent 会从 SKILL.md 的**实际位置**（即 `.skills/` 内）解析路径，
因此只要 `.skills/` 目录结构完整，所有引用均可正常工作。

### 新增 Skill 的步骤

当 `.skills/` 中新增一个 Skill 时，需要同步更新 Roo Code 配置：

1. **确定适用模式**：该 Skill 应在哪些模式下可用？
   - 全模式 → 放入 `.roo/skills/<skill-name>/`
   - 特定模式 → 放入 `.roo/skills-<mode>/<skill-name>/`
2. **创建目录并建立符号链接**：
   ```bash
   # 示例：新增一个仅 architect 模式可用的 Skill
   mkdir -p .roo/skills-architect/my-new-skill
   ln -s ../../../.skills/path/to/my-skill.md .roo/skills-architect/my-new-skill/SKILL.md
   ```
3. **更新本文档**（`.skills/00_install.md`）的 Skill 清单表格
4. **更新 `.skills/SKILL.md`** 的知识文件索引（如果新 Skill 属于知识库的一部分）

### 验证安装

```bash
# 检查所有 Skill 符号链接是否有效
echo "=== 全模式 Skills ==="
ls -la .roo/skills/*/SKILL.md 2>/dev/null

echo "=== Code 模式 Skills ==="
ls -la .roo/skills-code/*/SKILL.md 2>/dev/null

echo "=== Architect 模式 Skills ==="
ls -la .roo/skills-architect/*/SKILL.md 2>/dev/null

# 检查链接目标是否存在
find .roo/skills .roo/skills-code .roo/skills-architect -name "SKILL.md" -type l -exec sh -c \
  'test -e "$1" && echo "✅ $1" || echo "❌ BROKEN: $1"' _ {} \; 2>/dev/null
```

---

## Claude Code

### 背景：Claude Code 的加载机制

Claude Code 有三层配置，各司其职：

| 层 | 路径 | 加载时机 | 类比 Roo Code |
|----|------|---------|--------------|
| **持久指令** | `CLAUDE.md`（项目根目录） | 会话开始即加载，始终在上下文中 | `rules-*/AGENTS.md` |
| **按需 Skill** | `.claude/skills/<skill-name>/SKILL.md` | 描述在会话开始加载；完整内容在被触发时加载 | `.roo/skills/` / `.roo/skills-<mode>/` |
| **运行时配置** | `.claude/settings.json` | 会话启动时读取 | 无直接对应 |

> **重要区别**：Roo Code 通过 `skills-<mode>/` 区分模式，Claude Code **没有模式概念**。
> 所有 Skill 对所有场景可见，通过 SKILL.md 的 `description` 字段和 `When to Use` 内容来控制触发时机。

### 我们的 Skill 映射

`.skills/` 中的两个 Skill 在 Claude Code 中的映射方式：

| 源文件 | Claude Code Skill 名称 | 注册路径 | 触发方式 |
|--------|----------------------|---------|---------|
| `.skills/SKILL.md` | `kdit-skill` | `.claude/skills/kdit-skill/SKILL.md` | 自动触发（理解/修改 kDiT 代码时） + 手动 `/kdit-skill` |
| `.skills/01_development_spec/development_spec.md` | `development_spec` | `.claude/skills/development_spec/SKILL.md` | 自动触发（编码实现、架构讨论时） + 手动 `/development_spec` |

### CLAUDE.md — 持久指令

`CLAUDE.md` 放在项目根目录，内容始终存在于 Claude 的上下文中（类似 Roo Code 的 `AGENTS.md`）。
适合放入**精简的、始终需要的**指令，例如：

- 项目简介（一句话）
- 核心编码约定（import 风格、命名规则等的摘要）
- Skill 使用提示（告诉 Claude 存在哪些 Skill、何时应触发）
- 测试/格式化命令

> **注意**：CLAUDE.md 的内容会持续消耗上下文 token。
> 详细的架构文档、工作流步骤等应放在 Skill 中按需加载，不要全部塞进 CLAUDE.md。

CLAUDE.md 支持 `@path/to/file` 语法导入外部文件，但为保持简洁，建议仅在 CLAUDE.md 中写摘要，
详细内容通过 Skill 机制按需加载。

### 目录结构总览

安装完成后，Claude Code 相关目录应呈现如下结构：

```
项目根/
├── CLAUDE.md                               # 持久指令（精简版项目规范）
├── .claude/
│   ├── settings.json                       # 运行时配置（插件、权限等）
│   └── skills/                             # 按需 Skill
│       ├── kdit-skill/
│       │   └── SKILL.md                    # → 符号链接到 ../../.skills/SKILL.md
│       └── development_spec/
│           └── SKILL.md                    # → 符号链接到 ../../.skills/01_development_spec/development_spec.md
```

### 安装步骤

从项目根目录执行：

```bash
# ── 1. Skill：kdit-skill（项目知识库） ──
# Claude 在理解/修改 kDiT 代码时自动触发，也可手动 /kdit-skill
mkdir -p .claude/skills/kdit-skill
rm -f .claude/skills/kdit-skill/SKILL.md
ln -s ../../../.skills/SKILL.md .claude/skills/kdit-skill/SKILL.md

# ── 2. Skill：development_spec（先设计后编码工作流） ──
# Claude 在编码实现或架构讨论时自动触发，也可手动 /development_spec
mkdir -p .claude/skills/development_spec
rm -f .claude/skills/development_spec/SKILL.md
ln -s ../../../.skills/01_development_spec/development_spec.md .claude/skills/development_spec/SKILL.md

# ── 3. CLAUDE.md（持久指令） ──
# 需要手动创建，内容见下方模板
# touch CLAUDE.md
```

> **注意**：与 Roo Code 一样，符号链接是**文件级**（链接 `SKILL.md` 文件本身），而非目录级。
> 这是因为 `.skills/` 的目录结构（按编号分类）与 Claude Code 要求的目录结构（按 skill-name 命名）不同。

### Skill 内部的相对路径问题

与 Roo Code 相同。`SKILL.md` 中引用的 `.skills/` 下其他文档的相对路径，
由于符号链接的目标仍在 `.skills/` 目录内，这些相对路径**天然有效**。

Claude Code 通过 `Read` 工具读取关联文件时，会从 SKILL.md 的**实际位置**（即 `.skills/` 内）解析路径，
因此只要 `.skills/` 目录结构完整，所有引用均可正常工作。

### CLAUDE.md 模板

以下是推荐的 `CLAUDE.md` 内容（精简，避免占用过多上下文）：

```markdown
# kDiT

kDiT 是一个分布式视频生成推理框架，基于 DAG 编排 + Ray 分布式执行。

## 可用 Skill

本项目配置了以下 Skill，Claude 应在合适时机自动触发：

- **kdit-skill**（`/kdit-skill`）：项目知识库，包含架构、编码规范、API 参考。在理解或修改 kDiT 代码时触发。
- **development_spec**（`/development_spec`）：先设计后编码的工作流。在实现新功能、重大修改或架构讨论与设计评审时触发。

## 核心约定

- Import 风格：同目录/同子包用相对导入，跨子包（3层+）用 `from kdit.xxx` 绝对导入
- 类型注解：Python 3.10+ 原生语法（`X | Y`、`list[str]`），不用 `TYPE_CHECKING`
- 测试文件命名：`*_test.py`（不是 `test_*.py`），镜像 `kdit/` 结构放在 `tests/kdit/`
- 代码格式：`pre-commit` 管理（black 120字符、ruff），禁止 `--no-verify`
- 异常处理：禁止裸 `except Exception`
- Node 命名：`kdit/` 内不加 `Ksana` 前缀
```

### 新增 Skill 的步骤

当 `.skills/` 中新增一个 Skill 时，需要同步更新 Claude Code 配置：

1. **创建目录并建立符号链接**：
   ```bash
   # 示例：新增名为 my-new-skill 的 Skill
   mkdir -p .claude/skills/my-new-skill
   ln -s ../../../.skills/path/to/my-skill.md .claude/skills/my-new-skill/SKILL.md
   ```
2. **更新 `CLAUDE.md`** 的「可用 Skill」列表
3. **更新本文档**（`.skills/00_install.md`）的 Skill 映射表格

### 验证安装

```bash
# 检查所有 Skill 符号链接是否有效
echo "=== Claude Code Skills ==="
ls -la .claude/skills/*/SKILL.md 2>/dev/null

# 检查链接目标是否存在
find .claude/skills -name "SKILL.md" -type l -exec sh -c \
  'test -e "$1" && echo "✅ $1" || echo "❌ BROKEN: $1"' _ {} \; 2>/dev/null

# 检查 CLAUDE.md 是否存在
test -f CLAUDE.md && echo "✅ CLAUDE.md exists" || echo "⚠️  CLAUDE.md not found"
```

### settings.json — 运行时配置

当前已配置的插件（`.claude/settings.json`）：

```json
{
  "enabledPlugins": {
    "superpowers@claude-plugins-official": true,
    "code-review@claude-plugins-official": true,
    "frontend-design@claude-plugins-official": true
  }
}
```

这些插件提供额外的 Skill（如 `/superpowers:brainstorming`、`/code-review:code-review` 等），
与我们自定义的 `kdit-skill` 和 `development_spec` Skill 互补，不冲突。

如需添加权限白名单或 hooks，也在此文件中配置。详见 [Claude Code Settings 文档](https://docs.anthropic.com/en/docs/claude-code/settings)。
