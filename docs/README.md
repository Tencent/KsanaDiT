# kDiT Documentation

This directory contains the source files for the kDiT documentation site, built with [MkDocs](https://www.mkdocs.org/) and [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/).

## Prerequisites

### Python Packages

Install the required documentation packages:

```bash
pip install mkdocs-material mkdocstrings[python] mkdocs-mermaid2-plugin mkdocs-static-i18n
```

| Package | Version | Description |
|---------|---------|-------------|
| `mkdocs` | 1.6.1 | Static site generator for project documentation |
| `mkdocs-material` | 9.7.5 | Material Design theme for MkDocs |
| `mkdocstrings` | 1.0.3 | Auto-generate API docs from Python docstrings |
| `mkdocstrings-python` | 2.0.3 | Python handler for mkdocstrings |
| `mkdocs-mermaid2-plugin` | 1.2.3 | Mermaid diagram rendering support |
| `mkdocs-static-i18n` | 1.3.1 | Static internationalization (i18n) plugin |

## Quick Start

### Build the documentation site

```bash
# From the project root directory
mkdocs build
```

The generated site will be in the `site/` directory (already in `.gitignore`).

### Local development server (live reload)

```bash
mkdocs serve
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser. Changes to any `.md` file will trigger an automatic rebuild and browser refresh.

### Deploy to GitHub Pages

```bash
mkdocs gh-deploy
```

## Internationalization (i18n)

The documentation supports **English** (default) and **Chinese**.

### File naming convention

| Language | Suffix | Example |
|----------|--------|---------|
| English (default) | `.md` | `index.md` |
| Chinese | `.zh.md` | `index.zh.md` |

- English files use no language suffix (they are the default).
- Chinese files use the `.zh.md` suffix.
- Both files must exist in the same directory for a page to appear in both languages.
- The language switcher appears automatically in the navigation bar.

### Adding a new page

1. Create the English version: `docs/guide/my-page.md`
2. Create the Chinese version: `docs/guide/my-page.zh.md`
3. Add the page to the `nav` section in `mkdocs.yml`:
   ```yaml
   nav:
     - User Guide:
         - My Page: guide/my-page.md
   ```
4. Add the Chinese nav translation in `mkdocs.yml` under `plugins.i18n.nav_translations.zh`:
   ```yaml
   nav_translations:
     zh:
       My Page: 我的页面
   ```

### Translation guidelines

- **API Reference pages** (`.zh.md`): Translate the title and description; keep the `:::` autodoc directives unchanged.
- **Mermaid diagrams**: Keep diagram source code in English; do not translate node labels or edge labels.
- **LaTeX formulas**: Keep all `$...$` and `$$...$$` blocks unchanged.

## Directory Structure

```
docs/
├── README.md              # This file (build instructions)
├── index.md               # Homepage (English)
├── index.zh.md            # Homepage (Chinese)
├── architecture.md        # Architecture overview (English)
├── architecture.zh.md     # Architecture overview (Chinese)
├── guide/                 # User guides
│   ├── index.md / .zh.md
│   ├── local-usage.md / .zh.md
│   ├── comfyui-usage.md / .zh.md
│   ├── comfyui-nodes.md / .zh.md
│   └── supported-models.md / .zh.md
└── api/                   # API reference (auto-generated from source)
    ├── index.md / .zh.md
    ├── engine.md / .zh.md
    ├── accelerator.md / .zh.md
    ├── sample_solvers.md / .zh.md
    ├── scheduler.md / .zh.md
    ├── memory.md / .zh.md
    ├── utils.md / .zh.md
    ├── pipeline/          # Pipeline subsystem
    ├── config/            # Configuration classes
    ├── nodes/             # InferNode system
    ├── models/            # Model definitions
    ├── generators/        # Generator implementations
    ├── executor/          # Local & distributed executors
    ├── operations/        # Low-level operators
    ├── cache/             # Step-level caching strategies
    └── tensor/            # TensorPool & TensorKey
```

## Configuration

The main configuration file is [`mkdocs.yml`](../mkdocs.yml) in the project root. Key sections:

- **`theme`**: Material theme with light/dark mode toggle
- **`plugins`**: search, i18n, mkdocstrings, mermaid2
- **`markdown_extensions`**: admonition, code highlighting, MathJax, Mermaid fences
- **`nav`**: Full navigation tree (shared by both languages; Chinese labels defined via `nav_translations`)

## Troubleshooting

### `mkdocstrings` cannot find a module

Ensure kDiT is installed in development mode:

```bash
pip install -e .
```

### MathJax formulas not rendering

The MathJax CDN scripts are loaded via `extra_javascript` in `mkdocs.yml`. Ensure you have internet access when viewing the built site, or configure a local MathJax installation.

### Mermaid diagrams not rendering

Mermaid rendering requires the `mermaid2` plugin and the custom fence configuration in `pymdownx.superfences`. Both are pre-configured in `mkdocs.yml`.
