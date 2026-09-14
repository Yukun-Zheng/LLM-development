# 全书质量看板 / Quality Dashboard

> **目的**：把“这本书还差什么”变成可审计问题，而不是主观感觉。  
> 这里记录的是**当前成熟度**，不是宣传等级。章节只要没有代码/实验/复现，就不会因为文字长而升到 L2/L3。

---

# 成熟度定义

| 等级 | 标准 |
|---|---|
| **L0 目录级** | 问题定义、范围、原始资料入口存在 |
| **L1 教材级** | 历史、机制、公式/状态机、图示、原始来源基本完整 |
| **L2 复现级** | 有从零代码、实验、自动测试或数值/行为 parity |
| **L3 研究级** | 有独立复现、反例/消融、争议、最新证据和开放问题 |

核心主题最终尽量具备：Primary Sources、Mathematics/Formalization、Data/State Flow、Figures、Hand Calculation、Reference Code、Tests、Reproduction、Counterexamples/Limitations、Open Questions。

---

# 模型与系统主线

| 主题 | 原始资料 | 数学 | 图/数据流 | 从零代码 | 自动测试 | 真实复现 | 反例/争议 | 当前等级 | 下一关键动作 |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Transformer 基础 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | △ | L2 | 更多 activation parity + 手算 |
| GPT-3 / ICL / Scaling | ✓ | ✓ | ✓ | △ | — | △ | ✓ | L1 | Scaling 数据复绘、ICL toy experiments |
| RLHF / DPO | ✓ | ✓ | ✓ | △ | △ | △ | ✓ | L1-L2 | tiny preference pipeline |
| RoPE / RMSNorm / GQA / SwiGLU | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | △ | L2 | 更多公开模型族 |
| LLaMA-family runtime | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | △ | L2 | tokenizer parity、更多 checkpoints |
| LLaMA / LoRA / Quantization | ✓ | ✓ | △ | △ | △ | △ | ✓ | L1 | LoRA / QLoRA / quant labs |
| MoE | ✓ | ✓ | ✓ | △ | — | △ | ✓ | L1 | 从零 router + load-balance lab |
| Reasoning / RLVR | ✓ | ✓ | ✓ | △ | — | △ | ✓ | L1 | verifier + toy RL trajectory |
| Tokenization / Data | ✓ | ✓ | ✓ | ✓ | ✓ | △ | ✓ | L2 | fertility / dedup experiments |
| Optimization Dynamics | ✓ | ✓ | △ | △ | — | △ | ✓ | L1 | optimizer trajectory labs |
| Interpretability | ✓ | ✓ | ✓ | △ | — | △ | ✓ | L1 | activation patch / SAE lab |
| Training Systems | ✓ | ✓ | ✓ | △ | — | △ | ✓ | L1 | communication simulator |
| Inference Systems | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | L2 | paged KV / batching / speculative decoding |
| Hardware / Kernels | ✓ | ✓ | ✓ | △ | — | △ | ✓ | L1 | roofline / bandwidth labs |
| Post-Transformer | ✓ | ✓ | ✓ | — | — | — | ✓ | L1 | SSM / linear attention reference code |
| Diffusion LM | ✓ | ✓ | ✓ | — | — | — | ✓ | L1 | tiny masked diffusion LM |

## 真实模型 parity 里程碑

`HuggingFaceTB/SmolLM2-135M` raw safetensors → 我们自己的 `DecoderOnlyTransformer`，与 HF eager reference 在固定输入、CPU float32 下：

```json
{"max_abs": 0.0, "mean_abs": 0.0, "argmax_agreement": 1.0}
```

只对当前受测模型与设置成立，不外推到所有模型族。

---

# 智能体主线

| 主题 | 原始资料 | 形式化 | 从零代码 | 自动测试 | 真实环境 | Verifier | 安全 | 当前等级 | 下一关键动作 |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Agent Foundations | ✓ | ✓ | ✓ | ✓ | △ | ✓ | △ | L2 | trajectory metrics |
| Tool / Environment | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | △ | L2 | permission + sandbox |
| Planning / Replanning | ✓ | ✓ | ✓ | ✓ | △ | ✓ | △ | L2 | dynamic replan + scheduler |
| Memory / Context | ✓ | ✓ | ✓ | ✓ | △ | △ | △ | L2 | compaction + checkpoint/resume |
| Coding Agent | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | △ | L2 | tree-sitter / LSP / SWE-bench |
| Multi-Agent | ✓ | ✓ | ✓ | ✓ | △ | △ | △ | L2 | AgentGraph + parallel worktrees |
| Agent Evaluation | ✓ | ✓ | △ | △ | △ | ✓ | ✓ | L1 | WebArena / OSWorld / SWE-bench harness |
| Agentic RL | ✓ | ✓ | — | — | — | △ | △ | L1 | trajectory dataset + toy RL |
| Agent Protocols | ✓ | ✓ | ✓ | ✓ | △ | △ | ✓ | L2 | MCP stdio/HTTP/auth + A2A |
| Computer Use | ✓ | ✓ | — | — | — | △ | ✓ | L1 | DOM → screenshot → grounding |
| **OpenAI Codex Source Anatomy** | **✓ official source** | **✓ state machine** | **✓ v1** | **✓** | △ | △ | △ | **L2 partial** | thread/event/compaction/sandbox/App Server parity |

## Codex source-anatomy 里程碑

已经以 `openai/codex` 官方仓库为一手资料建立 C00–C07，并固定当前核验快照。当前 clean-room `codex_harness.py` 已重建：

```text
TurnSettings
→ model sampling
→ tool call
→ approval request/decision
→ tool execution
→ observation
→ follow-up sampling
→ completion / explicit stop
```

四个专门测试证明 approval deny 时真实 tool body 不执行，并验证 allow、follow-up 与 step-limit event semantics。**这里仍未实现真正 OS sandbox、durable thread、App Server 或完整 MCP。**

---

# Capstone 工程验收

## 已完成

- [x] byte tokenizer / BPE
- [x] modern decoder-only Transformer
- [x] RMSNorm / RoPE / GQA / SwiGLU
- [x] KV Cache + cached-decode parity
- [x] sampling + safetensors raw loader
- [x] 公开真实 SmolLM2-135M checkpoint logits parity
- [x] structured tool calls
- [x] filesystem / shell / Git / repo map / exact edit
- [x] persistent event memory
- [x] typed task DAG / verifier / coordinator / worktree primitive
- [x] minimal MCP client/server + tests
- [x] **MiniCodex clean-room v1：Turn / Approval / Event / Tool follow-up**
- [x] Fast CPU CI：CPU-only PyTorch + pytest + Ruff correctness lint
- [x] Source-First coverage audit + artifact
- [x] strict internal Markdown-link audit
- [x] Theory ↔ Code ↔ Paper learning map
- [x] Transformer + Codex Harness Source Card / Claim Ledger 起步

## 自动化证据

最新包含 Codex harness tests 的 Fast CI 已真实通过：

```text
30 passed, 1 warning in 3.02s
Ruff correctness lint: All checks passed
```

内容审计继续由 `.github/workflows/content-audit.yml` 自动执行，且现在把 `Codex源码解剖-codex-anatomy/` 纳入 Source-First coverage。内部链接检查严格阻塞；来源覆盖先产出报告 artifact。

---

# 下一优先级

## P0：Codex Harness / Agent Runtime

- [ ] async Submission Queue / Event Queue
- [ ] ThreadId / TurnId / persistent ThreadStore
- [ ] resume / fork / pending steering
- [ ] scoped AGENTS.md resolver + provenance
- [ ] context budget + compaction + rollover continuation
- [ ] permission profiles + async approval broker
- [ ] **enforced** process/container sandbox
- [ ] App Server JSON-RPC control plane
- [ ] MCP stdio/HTTP/auth/resources/prompts/elicitation
- [ ] AgentGraph / mailbox / nested spawn
- [ ] parallel worktree workers + reviewer / merge
- [ ] rollout trace / artifact store

## P1：模型与推理系统

- [ ] SDPA / FlashAttention backend parity
- [ ] Paged KV / Prefix Cache / Chunked Prefill
- [ ] Continuous Batching
- [ ] Speculative Decoding
- [ ] Grammar-Constrained Decoding
- [ ] 更多公开模型族 checkpoint parity matrix

## P1：Coding Intelligence

- [ ] tree-sitter
- [ ] LSP definitions / references
- [ ] unified diff / semantic patch
- [ ] test-selection / coverage-guided verifier
- [ ] SWE-bench harness

## P2：General Agent

- [ ] JS-capable browser automation
- [ ] DOM / Accessibility Tree
- [ ] screenshots / mouse / keyboard
- [ ] visual grounding
- [ ] Computer Use verifier
- [ ] indirect prompt-injection defenses
- [ ] WebArena / OSWorld-style eval
- [ ] long-horizon failure/recovery benchmark

---

# 教材基础设施缺口

- [ ] 自动 TOC 生成器
- [x] Theory ↔ Code ↔ Paper 三向学习地图（当前手工维护）
- [ ] 自动校验 cross-index manifest
- [ ] 扩大 paper/source card + claim ledger 覆盖
- [x] source-audit 进入 CI
- [x] internal broken-link checker 进入 CI
- [ ] external-link health checker
- [ ] Markdown / formula / citation linter 进一步统一
- [ ] MkDocs / Docusaurus / Quarto 网站版
- [ ] PDF / print build
- [ ] 双语术语表 / Index
- [ ] 章节习题答案分离版

---

# 决策原则

当“继续加新章节”和“把已有核心机制做到 L2/L3”冲突时，优先：

```text
真实 parity
> 自动测试
> 可复现实验
> 官方源码 / 原始资料核验
> 新章节数量
```

全书真正的质量不是 Markdown 文件数量，而是：

> **能否从原始 claim 追到公式/状态机，从状态机追到源码，从源码追到测试与真实系统，再清楚写出我们还没实现什么。**