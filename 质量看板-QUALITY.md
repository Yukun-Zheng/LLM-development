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
| Transformer 基础 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | △ | L2 | 更多逐层 activation parity + 手算增强 |
| GPT-3 / ICL / Scaling | ✓ | ✓ | ✓ | △ | — | △ | ✓ | L1 | Scaling 数据复绘、ICL toy experiments |
| RLHF / DPO | ✓ | ✓ | ✓ | △ | △ | △ | ✓ | L1-L2 | tiny preference pipeline |
| RoPE / RMSNorm / GQA / SwiGLU | ✓ | ✓ | ✓ | ✓ | ✓ | **✓** | △ | **L2** | 扩展到更多公开模型族 |
| LLaMA-family runtime | ✓ | ✓ | ✓ | **✓** | **✓** | **✓** | △ | **L2** | tokenizer parity、更多 checkpoints |
| LLaMA / LoRA / Quantization | ✓ | ✓ | △ | △ | △ | △ | ✓ | L1 | LoRA / QLoRA / quant reference labs |
| MoE | ✓ | ✓ | ✓ | △ | — | △ | ✓ | L1 | 从零 router + load balance lab |
| Reasoning / RLVR | ✓ | ✓ | ✓ | △ | — | △ | ✓ | L1 | verifier + toy RL trajectory lab |
| Tokenization / Data | ✓ | ✓ | ✓ | ✓ | ✓ | △ | ✓ | L2 | tokenizer fertility / dedup experiments |
| Optimization Dynamics | ✓ | ✓ | △ | △ | — | △ | ✓ | L1 | optimizer trajectory labs |
| Interpretability | ✓ | ✓ | ✓ | △ | — | △ | ✓ | L1 | activation patch / SAE lab |
| Training Systems | ✓ | ✓ | ✓ | △ | — | △ | ✓ | L1 | distributed simulation + communication accounting |
| Inference Systems | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | L2 | paged KV / batching / speculative decoding |
| Hardware / Kernels | ✓ | ✓ | ✓ | △ | — | △ | ✓ | L1 | roofline / bandwidth measurement labs |
| Post-Transformer | ✓ | ✓ | ✓ | — | — | — | ✓ | L1 | SSM / linear attention reference code |
| Diffusion LM | ✓ | ✓ | ✓ | — | — | — | ✓ | L1 | tiny masked diffusion LM |

## 真实模型 parity 里程碑

当前已对 `HuggingFaceTB/SmolLM2-135M` 做真实公开 checkpoint 验证。raw safetensors 经我们自己的 key mapping 与 `DecoderOnlyTransformer` forward 后，与 Hugging Face reference eager implementation 在固定输入、CPU float32 下得到：

```json
{
  "max_abs": 0.0,
  "mean_abs": 0.0,
  "argmax_agreement": 1.0
}
```

因此“我们的 runtime 能否真正运行公开现代 Llama-family 权重”已经从计划项变成有 CI 证据的完成项。这个结论只限当前受测模型和设置，不外推到所有模型族。

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
| Agent Protocols | ✓ | ✓ | **✓** | **✓** | △ | △ | ✓ | **L2** | MCP stdio/HTTP → auth → minimal A2A |
| Computer Use | ✓ | ✓ | — | — | — | △ | ✓ | L1 | DOM adapter → screenshot → grounding |

## Agent protocol 里程碑

当前已经从零实现一个可读的 MCP 教学子集：

```text
JSON-RPC 2.0
→ stateless server/discover
→ tools/list
→ tools/call
→ ToolRegistry adapter
→ in-process transport
→ client wrapper
→ unit tests
```

它不是生产 SDK；stdio、HTTP、auth、resources/prompts/extensions 和 A2A 仍按未完成处理。

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
- [x] **公开真实 SmolLM2-135M checkpoint logits parity**
- [x] structured tool calls
- [x] filesystem / shell / Git / repo map / exact edit
- [x] persistent event memory
- [x] typed task DAG
- [x] external verifier
- [x] coordinator primitive
- [x] Git worktree primitive
- [x] **minimal MCP client/server + tests**
- [x] **Fast CPU CI：CPU-only PyTorch + pytest + Ruff correctness lint**
- [x] **Source-First coverage audit + artifact**
- [x] **严格内部 Markdown 链接审计**
- [x] **Theory ↔ Code ↔ Paper 学习地图**

## 自动化证据

最新 Fast CI 已实际通过：

```text
26 passed, 1 warning
Ruff correctness lint: All checks passed
```

内容审计也已实际通过：

```text
35 chapters audited
2 methodology/blueprint documents have no primary-like external links
60 Markdown files audited
Internal Markdown links: PASS
```

Source coverage 当前先作为报告 artifact，不对历史格式差异做硬阻塞；内部链接检查严格阻塞。

## P0：当前基础设施优先项

- [ ] 自动生成目录/质量数据，彻底消除 README 漂移
- [ ] 把 Theory ↔ Code ↔ Paper 学习地图改为可自动校验的 manifest
- [ ] 建立 paper card / claim ledger 模板并覆盖核心论文

## P1：模型系统

- [ ] SDPA / FlashAttention backend parity
- [ ] Paged KV Cache
- [ ] Prefix Cache
- [ ] Continuous Batching
- [ ] Chunked Prefill
- [ ] Speculative Decoding
- [ ] Grammar-level Constrained Decoding
- [ ] 更多公开模型族 checkpoint parity matrix

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
- [ ] MCP stdio / HTTP / auth
- [ ] minimal A2A runtime

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
- [x] Theory ↔ Code ↔ Paper 三向学习地图（当前手工维护）
- [ ] 自动校验 cross-index manifest
- [ ] 每篇论文的 paper card / claim ledger
- [x] source-audit 进入 CI
- [x] internal broken-link checker 进入 CI
- [ ] 外部链接健康检查（需要避免把网络波动当内容错误）
- [ ] Markdown / formula / citation linter 进一步统一
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