# 全书质量看板 / Quality Dashboard

> **目的**：把“这本书还差什么”变成可审计问题，而不是主观感觉。  
> 这里记录的是**当前成熟度**，不是宣传等级。章节只要没有代码/实验/复现，就不会因为文字长而升到 L2/L3。

---

# 成熟度定义

| 等级 | 标准 |
|---|---|
| **L0 目录级** | 问题定义、范围、原始资料入口存在 |
| **L1 教材级** | 历史、机制、公式、图示、原始来源基本完整 |
| **L2 复现级** | 有从零代码、实验、自动测试或数值 parity |
| **L3 研究级** | 有独立复现、反例/消融、争议、最新证据和开放问题 |

---

# 评价维度

每个核心章节最终应尽量具有：

```text
Primary Sources
+ Mathematics
+ Data Flow / Shape
+ Figures
+ Hand Calculation
+ Reference Code
+ Tests
+ Reproduction
+ Counterexamples / Limitations
+ Open Questions
```

---

# 模型与系统主线

| 主题 | 原始资料 | 数学 | 图/数据流 | 从零代码 | 自动测试 | 真实复现 | 反例/争议 | 当前等级 | 下一关键动作 |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Transformer 基础 | ✓ | ✓ | ✓ | ✓ | ✓ | △ | △ | L2 | 原论文结构逐层 parity + 手算增强 |
| GPT-3 / ICL / Scaling | ✓ | ✓ | ✓ | △ | — | △ | ✓ | L1 | Scaling 数据复绘、ICL toy experiments |
| RLHF / DPO | ✓ | ✓ | ✓ | △ | △ | △ | ✓ | L1-L2 | tiny preference pipeline |
| RoPE / RMSNorm / GQA / SwiGLU | ✓ | ✓ | ✓ | ✓ | ✓ | △ | △ | L2 | 真实公开 checkpoint parity |
| LLaMA / LoRA / Quantization | ✓ | ✓ | △ | △ | △ | △ | ✓ | L1 | LoRA / QLoRA / quant reference labs |
| MoE | ✓ | ✓ | ✓ | △ | — | △ | ✓ | L1 | 从零 router + load balance lab |
| Reasoning / RLVR | ✓ | ✓ | ✓ | △ | — | △ | ✓ | L1 | verifier + toy RL trajectory lab |
| Tokenization / Data | ✓ | ✓ | ✓ | ✓ | ✓ | △ | ✓ | L2 | tokenizer fertility / dedup experiments |
| Optimization Dynamics | ✓ | ✓ | △ | △ | — | △ | ✓ | L1 | optimizer trajectory labs |
| Interpretability | ✓ | ✓ | ✓ | △ | — | △ | ✓ | L1 | activation patch / SAE lab |
| Training Systems | ✓ | ✓ | ✓ | △ | — | △ | ✓ | L1 | distributed simulation + communication accounting |
| Inference Systems | ✓ | ✓ | ✓ | ✓ | ✓ | △ | ✓ | L2 | paged KV / batching / speculative decoding |
| Hardware / Kernels | ✓ | ✓ | ✓ | △ | — | △ | ✓ | L1 | roofline / bandwidth measurement labs |
| Post-Transformer | ✓ | ✓ | ✓ | — | — | — | ✓ | L1 | SSM / linear attention reference code |
| Diffusion LM | ✓ | ✓ | ✓ | — | — | — | ✓ | L1 | tiny masked diffusion LM |

---

# 智能体主线

| 主题 | 原始资料 | 形式化 | 从零代码 | 自动测试 | 真实环境 | Verifier | 安全 | 当前等级 | 下一关键动作 |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Agent Foundations | ✓ | ✓ | ✓ | ✓ | △ | ✓ | △ | L2 | 加入 trajectory metrics |
| Tool / Environment | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | △ | L2 | permission layer / sandbox |
| Planning / Replanning | ✓ | ✓ | ✓ | ✓ | △ | ✓ | △ | L2 | dynamic replan + scheduler |
| Memory / Context | ✓ | ✓ | ✓ | ✓ | △ | △ | △ | L2 | compaction + checkpoint/resume |
| Coding Agent | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | △ | L2 | tree-sitter / LSP / SWE-bench |
| Multi-Agent | ✓ | ✓ | ✓ | ✓ | △ | △ | △ | L2 | parallel workers + reviewer/merge |
| Agent Evaluation | ✓ | ✓ | △ | △ | △ | ✓ | ✓ | L1 | WebArena / OSWorld / SWE-bench harness |
| Agentic RL | ✓ | ✓ | — | — | — | △ | △ | L1 | trajectory dataset + offline toy RL |
| Agent Protocols | ✓ | ✓ | △ | △ | — | △ | ✓ | L1 | minimal MCP → stdio/HTTP → A2A |
| Computer Use | ✓ | ✓ | — | — | — | △ | ✓ | L1 | DOM adapter → screenshot → grounding |

---

# Capstone 工程验收

## 已完成

- [x] byte tokenizer / BPE
- [x] modern decoder-only Transformer
- [x] RMSNorm / RoPE / GQA / SwiGLU
- [x] KV Cache
- [x] cached-decode parity against full recomputation
- [x] sampling
- [x] safetensors raw loader
- [x] structured tool calls
- [x] filesystem / shell / Git / repo map / exact edit
- [x] persistent event memory
- [x] typed task DAG
- [x] external verifier
- [x] coordinator primitive
- [x] Git worktree primitive

## P0：必须优先完成

- [ ] **公开真实 checkpoint logits parity**
- [ ] minimal MCP client/server + tests
- [ ] 统一 fast CPU CI
- [ ] 自动生成目录/质量数据，消除 README 漂移

## P1：模型系统

- [ ] SDPA / FlashAttention backend parity
- [ ] Paged KV Cache
- [ ] Prefix Cache
- [ ] Continuous Batching
- [ ] Chunked Prefill
- [ ] Speculative Decoding
- [ ] Grammar-level Constrained Decoding

## P1：Coding / Agent

- [ ] task checkpoint / resume
- [ ] permission manager
- [ ] container / process sandbox
- [ ] tree-sitter
- [ ] LSP definitions / references
- [ ] unified diff / semantic patch
- [ ] test-selection verifier
- [ ] parallel worktree workers
- [ ] reviewer / merge agent

## P2：General Agent

- [ ] JS-capable browser automation
- [ ] DOM / Accessibility Tree observation
- [ ] screenshots
- [ ] mouse / keyboard actions
- [ ] visual grounding
- [ ] Computer Use verifier
- [ ] indirect prompt-injection defenses

## P2：评测

- [ ] SWE-bench harness
- [ ] WebArena harness
- [ ] OSWorld-style harness
- [ ] long-horizon failure/recovery benchmark
- [ ] cost / latency / token / tool-call accounting

---

# 教材基础设施缺口

- [ ] 自动 TOC 生成器
- [ ] Theory ↔ Code ↔ Paper 三向索引
- [ ] 每篇论文的 paper card / claim ledger
- [ ] source-audit 自动进入 CI
- [ ] broken-link checker
- [ ] Markdown / formula / citation linter
- [ ] MkDocs / Docusaurus / Quarto 网站版
- [ ] PDF / print build
- [ ] 双语术语表
- [ ] 索引（Index）
- [ ] 章节习题答案分离版

---

# 决策原则

当“继续加新章节”和“把已有核心机制做到 L2/L3”发生冲突时，优先：

```text
真实 parity
> 自动测试
> 可复现实验
> 原始资料核验
> 新章节数量
```

全书真正的质量不是 Markdown 文件数量，而是：

> **能否从原论文追到公式，从公式追到代码，从代码追到实验，再从实验追到真实系统。**