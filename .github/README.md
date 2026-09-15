# 自动化配置 / GitHub Automation v3

自动化现在承担五类责任：**代码正确性、公开模型 parity、教材证据质量、机器能力清单一致性、前沿事实新鲜度**。

| Workflow | 作用 | 当前策略 |
|---|---|---|
| [`capstone-tests.yml`](workflows/capstone-tests.yml) | Reference System Fast CI：CPU PyTorch + pytest + Ruff | 阻塞 |
| [`smollm2-parity.yml`](workflows/smollm2-parity.yml) | public checkpoint → own runtime → HF logits parity | 阻塞 |
| [`content-audit.yml`](workflows/content-audit.yml) | Source-First + internal links + Capability Manifest + Frontier Snapshot | 结构/链接阻塞；来源与 freshness 报告 |
| [`normalize-markdown-math.yml`](workflows/normalize-markdown-math.yml) | GitHub MathJax compatibility normalization | 自动修复 |
| [`localize-chapter-headings.yml`](workflows/localize-chapter-headings.yml) | 中文优先章节标题 | 维护 |
| [`localize-file-tree.yml`](workflows/localize-file-tree.yml) | 中文优先文件树迁移 | 手动维护 |

---

# Fast CI

当前真实最新结果：

```text
36 passed, 1 warning in 2.14s
Ruff correctness lint: All checks passed
```

除了原有 model/cache/tokenizer/MCP/Codex harness 测试，v3 新增验证：

```text
Durable Thread
├─ DB close / reopen
├─ event replay
├─ running turn recovery
├─ checkpoint recovery
└─ invalid transition rejection

Permission Enforcement
├─ DENY → tool body not executed
├─ approval reject → not executed
└─ approval allow → dispatch

Evaluation
├─ trajectory metrics
├─ verifier outcome separation
└─ aggregate success / time / cost
```

普通单元测试保持 CPU-only；GPU/kernel benchmark 后续独立 workflow。

---

# 真实 Checkpoint Parity

`smollm2-parity.yml` 对：

```text
HuggingFaceTB/SmolLM2-135M raw config + safetensors
              ↓
our key mapping
              ↓
our DecoderOnlyTransformer
              ↓
our logits
```

和 HF eager reference 做受测 CPU float32 对比：

```json
{"max_abs": 0.0, "mean_abs": 0.0, "argmax_agreement": 1.0}
```

这项 CI 用于防止 RoPE layout、GQA、RMSNorm、SwiGLU 或 weight mapping “看起来合理但实际上错位”。

---

# Content Audit v3

`content-audit.yml` 现在运行四类审计。

## 1. Source-First coverage

```bash
python 工具-scripts/原始资料审计-source_audit.py
```

扫描：

```text
教材-book/
智能体-agent/
Codex源码解剖-codex-anatomy/
```

来源覆盖当前作为报告，不让历史格式差异直接阻塞 main。

## 2. Strict internal links

```bash
python 工具-scripts/内部链接审计-link_audit.py
```

仓库内部 Markdown 相对链接损坏属于确定性错误，因此严格阻塞。

## 3. Capability Manifest

```bash
python 工具-scripts/能力清单审计-capability_audit.py
```

验证根目录 [`../能力清单-CAPABILITIES.json`](../能力清单-CAPABILITIES.json)：

```text
unique capability id
valid track / status / maturity
theory/code/test paths exist
implemented/validated capability has code
validated capability has tests + evidence
```

这样 `README / STATUS / QUALITY` 未来可以逐渐由同一份机器事实源生成，而不是人工维护多个互相漂移的勾选表。

## 4. Frontier Snapshot

```bash
python 工具-scripts/前沿漂移审计-frontier_drift_audit.py
```

验证：

```text
观测站-observatory/frontier-snapshot.json
```

包括 schema、唯一 id、官方 source URL、evidence boundary 与 snapshot age。常规 CI 报告 age；发布前可以使用：

```bash
python 工具-scripts/前沿漂移审计-frontier_drift_audit.py --strict-age
```

防止网络波动把仓库误判失败，因此 CI 不逐 URL 做强制在线健康检查。

所有报告统一作为 `textbook-audit-reports` artifact 上传。

---

# Codex Source-First Chain

```text
openai/codex pinned public source
→ Codex源码解剖/
→ Source Card / Claim Ledger
→ clean-room implementation
→ behavioral/protocol tests
→ durable/security/eval subsystems
→ later real benchmark
```

Codex 是 Coding Agent 的 industrial reference，不是整个 Agent 世界的唯一标准架构。

---

# 自动化最终目标

```text
Primary / Official Source
→ Claim / Contract
→ Theory / State / Data Flow
→ Reference Implementation
→ Unit / Numerical / Protocol Parity
→ Capability / Systems Eval
→ Failure / Safety Evidence
→ Frontier Revalidation
```

CI 的目标不是 badge 数量，而是让“这项能力真的完成了吗？”尽可能有机器可检查的答案。
