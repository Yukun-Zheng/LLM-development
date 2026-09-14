# 自动化配置 / GitHub Automation

本目录存放教材仓库的 GitHub Actions 与自动化配置。自动化不只是“格式化”，还承担**实现正确性、真实 checkpoint parity、Agent harness 回归与教材证据质量**的持续检查。

## 当前工作流

| Workflow | 作用 | 是否阻塞正确性 |
|---|---|---:|
| [`capstone-tests.yml`](workflows/capstone-tests.yml) | **终极工程 Fast CI**：CPU-only PyTorch、pytest、Ruff correctness lint | 是 |
| [`smollm2-parity.yml`](workflows/smollm2-parity.yml) | **真实公开 checkpoint parity**：SmolLM2-135M raw weights → 我们自己的 runtime → HF reference logits | 是 |
| [`content-audit.yml`](workflows/content-audit.yml) | **教材内容审计**：Source-First 覆盖报告 + 严格内部链接检查 + artifact | 内部链接是；来源覆盖先报告 |
| [`normalize-markdown-math.yml`](workflows/normalize-markdown-math.yml) | GitHub Markdown / MathJax 公式自动规范化 | 自动修复 |
| [`localize-chapter-headings.yml`](workflows/localize-chapter-headings.yml) | 章节标题中文优先化 | 维护 |
| [`localize-file-tree.yml`](workflows/localize-file-tree.yml) | 文件树中文优先迁移工具 | 手动维护 |

---

# Fast CI

`capstone-tests.yml`：

```text
checkout
→ Python 3.12
→ pip cache
→ CPU PyTorch
→ editable install
→ pytest
→ Ruff correctness lint
```

加入 `codex_harness.py` 与四个专门状态机测试后的真实 CI：

```text
30 passed, 1 warning in 3.02s
Ruff correctness lint: All checks passed
```

当前 Codex harness regression 覆盖：

```text
tool → observation → follow-up → final
approval deny → tool body not executed
approval allow → execute
step limit → explicit TURN_STOPPED
```

普通 CI 保持 CPU-only；GPU/kernel benchmark 后续独立执行。

---

# 真实 checkpoint parity

`smollm2-parity.yml` 下载并缓存：

```text
HuggingFaceTB/SmolLM2-135M
```

对比：

```text
raw config + safetensors
      ↓
our key mapper
      ↓
our DecoderOnlyTransformer
      ↓
our logits

vs

Hugging Face reference eager implementation
```

受测固定输入、CPU float32：

```json
{"max_abs": 0.0, "mean_abs": 0.0, "argmax_agreement": 1.0}
```

---

# 教材内容审计

`content-audit.yml` 运行两个不同性质的检查。

## Source-First coverage

调用：

```text
工具-scripts/原始资料审计-source_audit.py
```

现在同时扫描：

```text
教材-book/
智能体-agent/
Codex源码解剖-codex-anatomy/
```

最新一次真实 workflow 结果：

```text
43 chapters audited
2 chapters with no likely primary-source links
```

两个未含 external primary-like link 的文件是教材方法论/全书蓝图类元文档；**C00–C07 Codex 源码解剖技术章节现在全部有官方一手来源入口**。

报告包括外部链接、primary-like source、原始资料区、evidence markers 与无一手来源章节数量，并上传 `primary-source-coverage` artifact。来源覆盖当前是**质量报告**，等历史章节格式进一步收敛后再切 `--strict`。

## Internal-link audit

调用：

```text
工具-scripts/内部链接审计-link_audit.py
```

最新一次真实 workflow：

```text
74 Markdown files audited
Internal Markdown links: PASS
```

该检查不联网，只验证仓库内所有 Markdown 相对链接实际存在，并严格阻塞 main 上的确定性导航错误。

---

# Codex Source-First 证据链

Codex 相关内容使用额外的工业源码链：

```text
openai/codex pinned public source
→ Codex源码解剖-codex-anatomy/
→ Source Card / Claim Ledger
→ codex_harness.py clean-room implementation
→ pytest behavioral regression
→ later protocol / sandbox / thread parity
```

官方源码索引：[`../参考文献-references/03-OpenAI-Codex官方源码索引-codex-source-map.md`](../参考文献-references/03-OpenAI-Codex官方源码索引-codex-source-map.md)

---

# 路径与命名

读者可见路径继续采用：

> **中文优先 + 英文保留**

例如：

```text
教材-book/03-对齐与ChatGPT-alignment-chatgpt.md
智能体-agent/09-Agent协议与互操作-agent-protocols.md
Codex源码解剖-codex-anatomy/01-核心AgentLoop-agent-loop.md
```

`.github/` 保持原名，因为这是 GitHub 特殊目录。

自动化最终目标是建立：

> **原始论文 / 官方规范 / 官方源码 → 理论/状态机 → clean-room 源码 → 单元测试 → parity → benchmark**

的连续证据链。