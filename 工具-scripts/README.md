# 教材维护脚本 / Maintenance Scripts

本目录存放教材仓库的自动化维护工具。它们服务于**排版一致性、GitHub 兼容性、证据可追溯性与长期维护**，不是教材正文的一部分。

## 当前脚本

| 文件 | 中文说明 | 作用 |
|---|---|---|
| [`数学公式规范化-normalize_math.py`](数学公式规范化-normalize_math.py) | **GitHub 数学公式规范化器** | 统一数学定界符，修复 GitHub MathJax 不兼容宏，并处理数学环境中的关系符号 |
| [`章节标题中文化-localize_headings.py`](章节标题中文化-localize_headings.py) | **章节标题中文化** | 保证卷/篇标题采用中文优先、英文保留的形式 |
| [`文件树中文化-localize_file_tree.py`](文件树中文化-localize_file_tree.py) | **文件树中文化迁移工具** | 将读者可见文件树迁移为中文优先双语命名，并同步更新内部引用 |
| [`原始资料审计-source_audit.py`](原始资料审计-source_audit.py) | **一手资料覆盖审计** | 扫描教材章节中的原论文、官方技术报告、官方代码、模型卡/系统卡等一手资料链接，发现缺乏原始证据的章节 |

## 数学规范化器

当前数学规范化器会自动处理：

- LaTeX / GitHub 数学定界符兼容；
- `\operatorname{...}` 等 GitHub MathJax 不兼容宏；
- 数学环境中的 `<`、`>`、`<=`、`>=`；
- fenced code block（Python、Mermaid、shell、text 等）保持原样，不进行误替换。

## 原始资料审计

运行：

```bash
python 工具-scripts/原始资料审计-source_audit.py
```

严格模式：

```bash
python 工具-scripts/原始资料审计-source_audit.py --strict
```

它会统计每一章：

- external links；
- likely primary-source links；
- 是否包含“原始资料 / 原始资料包”章节；
- 原论文、官方代码、system card、model card、独立复现等 evidence markers。

这只是**覆盖率审计**，不是自动判断论文解释是否正确。真正的科学审校仍然必须回到原始资料逐段核对。

对应自动化工作流位于 [`.github/workflows/`](../.github/workflows/)。
