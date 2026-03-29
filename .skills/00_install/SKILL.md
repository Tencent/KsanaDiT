---
name: kdit-install
description: >
  kDiT Skills 安装指南：配置 Roo Code 和 Claude Code 的 skill 符号链接、CLAUDE.md 模板、settings.json 插件配置。
  关键词：install、安装、配置、symlink、符号链接、Roo Code、Claude Code。
---

# Skills 安装指南入口

> 本文件是 install skill 的入口。完整安装指南请阅读 [`install.md`](install.md)。

## 适用场景

- 首次配置 kDiT 的 skill 系统（Roo Code / Claude Code）
- 新增 skill 后需要注册到工具链
- 排查 skill 未生效的配置问题

## 不适用场景

- 编写代码或修改架构（走 `/kdit-development-spec`）

## 核心内容

- **Roo Code**：`skills/` 目录符号链接配置
- **Claude Code**：`.claude/skills/` 符号链接 + CLAUDE.md 模板 + settings.json 插件
- **新增 Skill 步骤**：两个工具的注册流程

详见 → [`install.md`](install.md)

## 相关 skill

| Skill | 何时调用 |
|-------|---------|
| `/kdit-skill` | 查看所有 skill 的完整索引 |
