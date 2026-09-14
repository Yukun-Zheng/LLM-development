# 自动化配置 / GitHub Automation

本目录存放教材仓库的 GitHub Actions 与自动化配置。

当前主要工作流：

- [`workflows/normalize-markdown-math.yml`](workflows/normalize-markdown-math.yml)：**Markdown 数学公式自动规范化**。每次相关 Markdown 或规范化脚本更新后，自动检查并修复 GitHub 预览中的数学兼容问题。

自动化的目标是让教材作者可以专注于内容，而不是反复手工处理 Markdown / MathJax 的渲染差异。
