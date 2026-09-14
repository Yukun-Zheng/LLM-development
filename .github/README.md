# 自动化配置 / GitHub Automation

本目录存放教材仓库的 GitHub Actions 与自动化配置。自动化不只是“格式化”，还承担**实现正确性、真实 checkpoint parity 与教材证据质量**的持续检查。

## 当前工作流

| Workflow | 作用 | 是否阻塞正确性 |
|---|---|---:|
| [`capstone-tests.yml`](workflows/capstone-tests.yml) | **终极工程 Fast CI**：CPU-only PyTorch、pytest、Ruff correctness lint | 是 |
| [`smollm2-parity.yml`](workflows/smollm2-parity.yml) | **真实公开 checkpoint parity**：SmolLM2-135M raw weights → 我们自己的 runtime → 对比 HF reference logits | 是 |
| [`content-audit.yml`](workflows/content-audit.yml) | **教材内容审计**：Source-First 覆盖报告 + 严格内部链接检查 + artifact | 内部链接是；来源覆盖先报告 |
| [`normalize-markdown-math.yml`](workflows/normalize-markdown-math.yml) | GitHub Markdown / MathJax 公式自动规范化 | 自动修复 |
| [`localize-chapter-headings.yml`](workflows/localize-chapter-headings.yml) | 章节标题中文优先化 | 维护 |
| [`localize-file-tree.yml`](workflows/localize-file-tree.yml) | 文件树中文优先迁移工具 | 手动维护 |

后续将继续加入：

- 自动目录生成；
- Theory ↔ Code ↔ Paper 映射一致性检查；
- 文档构建 / PDF / 网站发布；
- optional GPU parity / kernel benchmark。

---

# Fast CI

`capstone-tests.yml` 刻意使用 CPU-only PyTorch：

```text
checkout
→ Python 3.12
→ pip cache
→ CPU PyTorch
→ editable install
→ pytest
→ Ruff correctness lint
```

当前已验证：

```text
26 passed, 1 warning
Ruff correctness lint: All checks passed
```

这样普通代码改动不再无意义下载整套 CUDA wheel；GPU/kernel benchmark 后续放到独立可选 workflow。

---

# 真实 checkpoint parity

`smollm2-parity.yml` 下载并缓存公开模型：

```text
HuggingFaceTB/SmolLM2-135M
```

然后比较：

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

当前固定输入、CPU float32 下的真实 CI 结果：

```json
{
  "max_abs": 0.0,
  "mean_abs": 0.0,
  "argmax_agreement": 1.0
}
```

这个 workflow 的意义是：防止模型运行时“看起来对”，但某个 RoPE layout、GQA、RMSNorm 或 weight mapping 实际与公开模型不一致。

---

# 教材内容审计

`content-audit.yml` 运行两个不同性质的检查。

## Source-First coverage

调用：

```text
工具-scripts/原始资料审计-source_audit.py
```

当前会同时扫描：

```text
教材-book/
智能体-agent/
```

并生成：

- 外部链接数量；
- primary-like source 数量；
- 是否有原始资料区；
- evidence marker 数量；
- 没有一手来源的章节数量。

报告上传为 `primary-source-coverage` artifact。当前这一检查先作为**质量报告**，不因为历史章节格式不统一而阻塞 main；以后 source coverage 收敛后可以切成 `--strict`。

最近一次审计：

```text
35 chapters audited
2 chapters with no primary-like external links
```

这两个目前是方法论/全书蓝图性质文档，不是具体技术机制章节。

## Internal-link audit

调用：

```text
工具-scripts/内部链接审计-link_audit.py
```

它不联网，只检查仓库所有 Markdown 的相对链接是否真实存在。

最近一次：

```text
60 Markdown files audited
Internal Markdown links: PASS
```

该检查是严格阻塞的，因为内部导航损坏是确定性错误。

---

# 路径与命名

仓库的读者可见路径采用：

> **中文优先 + 英文保留**

例如：

```text
教材-book/03-对齐与ChatGPT-alignment-chatgpt.md
智能体-agent/09-Agent协议与互操作-agent-protocols.md
```

`.github/` 本身保持原名，因为这是 GitHub Actions 要求的特殊目录。

自动化的最终目标不是追求 CI 数量，而是建立：

> **原始资料 → 理论 → 源码 → 单元测试 → 真实 parity → benchmark**

的连续证据链。