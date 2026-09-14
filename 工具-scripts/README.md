# 教材维护脚本 / Maintenance Scripts

本目录存放教材仓库的自动化维护工具。它们服务于**排版一致性、GitHub 兼容性与长期维护**，不是教材正文的一部分。

## 当前脚本

| 文件 | 中文说明 | 作用 |
|---|---|---|
| [`数学公式规范化-normalize_math.py`](数学公式规范化-normalize_math.py) | **GitHub 数学公式规范化器** | 统一数学定界符，修复 GitHub MathJax 不兼容宏，并处理数学环境中的关系符号 |

当前数学规范化器会自动处理：

- `$...$` → `$...$`
- `$$...$$` → `$$...$$`
- `\mathrm{...}` → `\mathrm{...}`
- 数学环境中的 `<`、`>`、`<=`、`>=` → GitHub 更稳定的数学写法
- fenced code block（Python、Mermaid、shell、text 等）保持原样，不进行误替换

对应自动化工作流位于 [`.github/workflows/`](../.github/workflows/)。
