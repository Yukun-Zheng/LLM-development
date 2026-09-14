# 自动化配置 / GitHub Automation

本目录存放教材仓库的 GitHub Actions 与自动化配置。

当前主要工作流：

- [`workflows/normalize-markdown-math.yml`](workflows/normalize-markdown-math.yml)：**Markdown 数学公式自动规范化**。调用 `工具-scripts/数学公式规范化-normalize_math.py`，自动修复 GitHub 预览中的数学兼容问题。
- [`workflows/localize-chapter-headings.yml`](workflows/localize-chapter-headings.yml)：**章节标题中文优先化**。调用 `工具-scripts/章节标题中文化-localize_headings.py`。
- [`workflows/localize-file-tree.yml`](workflows/localize-file-tree.yml)：**仓库文件树中文化**。当前作为手动维护工具，调用 `工具-scripts/文件树中文化-localize_file_tree.py`。

仓库的读者可见路径采用“**中文优先 + 英文保留**”的双语命名，例如 `教材-book/03-对齐与ChatGPT-alignment-chatgpt.md`。`.github/` 本身保持原名，因为这是 GitHub Actions 要求的特殊目录。

自动化的目标是让教材作者可以专注于内容，而不是反复手工处理 Markdown、MathJax、目录命名与章节格式。
