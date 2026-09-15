# 自动化配置 / GitHub Automation v3

本目录承担的不只是格式化，而是项目的**持续证据链**：实现正确性、真实 checkpoint parity、教材 Source-First、机器 capability manifest 与 frontier drift 都要被自动检查。

---

# 当前工作流

| Workflow | 作用 | 约束 |
|---|---|---|
| [`capstone-tests.yml`](workflows/capstone-tests.yml) | CPU-only PyTorch + 全部 unit/parity/reference-system tests + Ruff | 阻塞 |
| [`smollm2-parity.yml`](workflows/smollm2-parity.yml) | SmolLM2 raw weights → own runtime → HF reference logits | 阻塞 |
| [`content-audit.yml`](workflows/content-audit.yml) | Source coverage + internal links + capability manifest + frontier snapshot | 链接/manifest/schema 阻塞 |
| [`normalize-markdown-math.yml`](workflows/normalize-markdown-math.yml) | GitHub MathJax 兼容规范化 | 自动修复 |
| [`localize-chapter-headings.yml`](workflows/localize-chapter-headings.yml) | 中文优先章节标题 | 维护 |
| [`localize-file-tree.yml`](workflows/localize-file-tree.yml) | 中文优先真实路径迁移 | 手动维护 |

---

# Fast CI 当前真实证据

最新已核验的完整回归（run 66）：

```text
61 passed, 1 warning in 3.67s
Ruff correctness lint: All checks passed
```

这已经覆盖：

```text
Tokenizer / Transformer / real checkpoint primitives
contiguous KV / paged KV / scheduler / batched executor
SFT / DPO / group-relative RL objective
Tool / Coding / MCP
Codex-style Turn harness
Thread replay / fork / cancellation
Work lease / reclaim
Integrated durable runtime
Context provenance
Permission enforcement
Trajectory / benchmark / repository final-state grading
```

普通 CI 使用 CPU-only PyTorch，避免为 unit/reference tests 下载整套 CUDA runtime。GPU/kernel benchmark 后续独立执行。

---

# 真实 checkpoint parity

独立 workflow 下载并缓存：

```text
HuggingFaceTB/SmolLM2-135M
```

对比：

```text
raw config + safetensors
→ our mapper
→ our DecoderOnlyTransformer
→ our logits

vs

HF eager reference
```

当前固定受测 CPU float32：

```json
{"max_abs": 0.0, "mean_abs": 0.0, "argmax_agreement": 1.0}
```

不能把一次模型/输入 parity 外推成“支持所有 Llama-family”。

---

# Content Audit

`content-audit.yml` 运行四层检查。

## 1. Source-First coverage

```text
工具-scripts/原始资料审计-source_audit.py
```

扫描：

```text
教材-book/
智能体-agent/
Codex源码解剖-codex-anatomy/
```

这是 coverage heuristic，不等于 citation scientific validity checker。

## 2. Strict Internal Links

```text
工具-scripts/内部链接审计-link_audit.py
```

所有 Markdown relative link 必须真实存在。确定性导航损坏直接阻塞。

## 3. Capability Manifest

```text
工具-scripts/能力清单审计-capability_audit.py
```

检查：

```text
unique capability id
valid track/status/maturity
all theory/code/test paths exist
validated ⇒ code != empty
validated ⇒ tests != empty
validated ⇒ evidence != empty
```

因此 README 不能单方面把某项能力“宣布完成”。

## 4. Frontier Snapshot

```text
工具-scripts/前沿漂移审计-frontier_drift_audit.py
```

检查 machine-readable frontier snapshot 的 schema / dates / source / evidence boundary / freshness，并把“快速变化事实”与稳定教材正文分离。

---

# 自动化 Artifact

Content Audit 上传：

```text
source-audit.md
capability-audit.md
frontier-audit.md
```

作为 `textbook-audit-reports` artifact。

---

# 证据链标准

模型机制：

```text
Original Paper / Official Code
→ Formula / Data Flow
→ Reference Implementation
→ Numerical Parity
→ Benchmark / Profile
```

Codex/Agent：

```text
openai/codex official source / protocol spec
→ state-machine contract
→ clean-room runtime
→ behavioral/protocol tests
→ real environment benchmark
```

Serving：

```text
Reference semantics
→ optimized implementation
→ logits/distribution parity
→ memory/latency/throughput
```

Post-training：

```text
Original objective
→ hand-check invariant
→ Tensor implementation
→ gradient/optimizer test
→ training curve
→ held-out evaluation
```

---

# 已知 CI 维护项

GitHub runner 当前仍提示部分 action 版本基于已弃用 Node 20 runtime，并被强制运行在 Node 24。它目前只是 warning，没有阻塞 workflow；后续应在 upstream 新 major 稳定后统一升级 `actions/checkout` / `setup-python` / `upload-artifact`，不要为了消 warning 改动业务逻辑。

---

自动化最终目的不是得到一排绿色勾，而是持续强制：

> **Claim → Source → Code → Test → Eval / Evidence**

不能彼此脱节。
