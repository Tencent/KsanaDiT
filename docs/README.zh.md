# kDiT 文档

本目录包含 kDiT 文档站点的源文件，使用 [MkDocs](https://www.mkdocs.org/) 和 [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) 构建。

## 前置条件

### Python 依赖包

安装文档构建所需的 Python 包：

```bash
pip install mkdocs-material mkdocstrings[python] mkdocs-mermaid2-plugin mkdocs-static-i18n
```

| 包名 | 版本 | 说明 |
|------|------|------|
| `mkdocs` | 1.6.1 | 项目文档静态站点生成器 |
| `mkdocs-material` | 9.7.5 | MkDocs 的 Material Design 主题 |
| `mkdocstrings` | 1.0.3 | 从 Python docstring 自动生成 API 文档 |
| `mkdocstrings-python` | 2.0.3 | mkdocstrings 的 Python 处理器 |
| `mkdocs-mermaid2-plugin` | 1.2.3 | Mermaid 图表渲染支持 |
| `mkdocs-static-i18n` | 1.3.1 | 静态国际化（i18n）插件 |

## 快速开始

### 构建文档站点

```bash
# 在项目根目录执行
mkdocs build
```

生成的站点位于 `site/` 目录（已加入 `.gitignore`）。

### 本地开发服务器（实时热重载）

```bash
mkdocs serve
```

然后在浏览器中打开 [http://127.0.0.1:8000](http://127.0.0.1:8000)。修改任何 `.md` 文件都会自动触发重新构建和浏览器刷新。

### 部署到 GitHub Pages

```bash
mkdocs gh-deploy
```

## 国际化（i18n）

文档支持**英文**（默认）和**中文**两种语言。

### 文件命名约定

| 语言 | 后缀 | 示例 |
|------|------|------|
| 英文（默认） | `.md` | `index.md` |
| 中文 | `.zh.md` | `index.zh.md` |

- 英文文件不使用语言后缀（作为默认语言）。
- 中文文件使用 `.zh.md` 后缀。
- 同一目录下必须同时存在两个文件，页面才会在两种语言中都显示。
- 语言切换器会自动出现在导航栏中。

### 添加新页面

1. 创建英文版本：`docs/guide/my-page.md`
2. 创建中文版本：`docs/guide/my-page.zh.md`
3. 在 `mkdocs.yml` 的 `nav` 部分添加页面：
   ```yaml
   nav:
     - User Guide:
         - My Page: guide/my-page.md
   ```
4. 在 `mkdocs.yml` 的 `plugins.i18n.languages` 中对应 `zh` 的 `nav_translations` 下添加中文翻译：
   ```yaml
   nav_translations:
     My Page: 我的页面
   ```

### 翻译指南

- **API 参考页面**（`.zh.md`）：翻译标题和描述文字；保持 `:::` 自动文档指令不变。
- **Mermaid 图表**：保持图表源代码为英文，不翻译节点标签和边标签。
- **LaTeX 公式**：保持所有 `$...$` 和 `$$...$$` 块不变。

## 目录结构

```
docs/
├── README.md              # 英文版构建说明
├── README.zh.md           # 中文版构建说明（本文件）
├── index.md               # 首页（英文）
├── index.zh.md            # 首页（中文）
├── architecture.md        # 架构概览（英文）
├── architecture.zh.md     # 架构概览（中文）
├── guide/                 # 用户指南
│   ├── index.md / .zh.md
│   ├── local-usage.md / .zh.md
│   ├── comfyui-usage.md / .zh.md
│   ├── comfyui-nodes.md / .zh.md
│   └── supported-models.md / .zh.md
└── api/                   # API 参考（从源码自动生成）
    ├── index.md / .zh.md
    ├── engine.md / .zh.md
    ├── accelerator.md / .zh.md
    ├── sample_solvers.md / .zh.md
    ├── scheduler.md / .zh.md
    ├── memory.md / .zh.md
    ├── utils.md / .zh.md
    ├── pipeline/          # 流水线子系统
    ├── config/            # 配置类
    ├── nodes/             # InferNode 系统
    ├── models/            # 模型定义
    ├── generators/        # 生成器实现
    ├── executor/          # 本地与分布式执行器
    ├── operations/        # 底层算子
    ├── cache/             # 步级缓存策略
    └── tensor/            # TensorPool 与 TensorKey
```

## 配置说明

主配置文件为项目根目录的 [`mkdocs.yml`](../mkdocs.yml)。关键配置段：

- **`theme`**：Material 主题，支持亮色/暗色模式切换
- **`plugins`**：search、i18n、mkdocstrings、mermaid2
- **`markdown_extensions`**：提示框、代码高亮、MathJax 数学公式、Mermaid 图表围栏
- **`nav`**：完整导航树（中英文共享；中文标签通过 `nav_translations` 定义）

## 常见问题

### `mkdocstrings` 找不到模块

确保 kDiT 已以开发模式安装：

```bash
pip install -e .
```

### MathJax 公式不渲染

MathJax CDN 脚本通过 `mkdocs.yml` 中的 `extra_javascript` 加载。查看构建后的站点时需要联网，或配置本地 MathJax 安装。

### Mermaid 图表不渲染

Mermaid 渲染需要 `mermaid2` 插件和 `pymdownx.superfences` 中的自定义围栏配置。两者均已在 `mkdocs.yml` 中预配置。
