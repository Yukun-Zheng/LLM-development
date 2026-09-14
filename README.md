# 大语言模型发展史、系统工程与智能体：从原始论文到从零实现

> **版本**：v2 · 2026-09-14  
> **定位**：中文主导、英文术语对照、Source-First、Mechanism-First、Reproducibility-First 的大语言模型、LLM Systems 与 Agent 教材工程。  
> **最终目标**：从空目录开始，把现代 LLM runtime、推理系统、Coding Agent、Browser / Computer Use、多智能体与评测系统逐层亲手写出来。训练 frontier 权重本身不作为要求。

---

# 一、四个互相约束的入口

```text
                         LLM Development
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
   模型科学主线             系统工程主线            智能体主线
  Model Science            LLM Systems          Agentic Systems
        │                      │                      │
        │                      │               ┌──────┴────────┐
        │                      │               ▼               ▼
        │                      │         Agent 通识       Codex 源码解剖
        │                      │                         OpenAI/Codex
        │                      │                              │
        └──────────────────────┼──────────────────────────────┘
                               ▼
                   从零 Frontier 系统总工程
                  Astra-class / Codex-class
```

## 1. 模型科学

数学 / 概率 → Tokenization / Data → Transformer / MoE → Scaling → Post-training → Reasoning → Multimodality → Post-Transformer → Diffusion LM。

入口：[`教材-book/README.md`](教材-book/README.md)

## 2. 系统工程

GPU / HBM / Kernel → Distributed Training → FlashAttention → KV Cache → Serving / Scheduling / Runtime / Compiler。

## 3. 智能体系统

Tool / Environment → Planning → Memory → Verification → Coding → Browser / Computer → Protocols → Multi-Agent → Agentic RL。

入口：[`智能体-agent/README.md`](智能体-agent/README.md)

## 4. OpenAI Codex 源码解剖

OpenAI 已公开 `openai/codex` 的 CLI / agent harness / runtime 源码，所以 Coding Agent 不再只按论文和产品行为推测。本仓库新增独立的 Source Anatomy 课程，按官方源码建立：

```text
官方源码
→ 行为契约 / 状态机
→ clean-room 教学重写
→ unit / protocol / behavioral parity
→ gap report
```

入口：[`Codex源码解剖-codex-anatomy/README.md`](Codex源码解剖-codex-anatomy/README.md)  
官方源码地图：[`参考文献-references/03-OpenAI-Codex官方源码索引-codex-source-map.md`](参考文献-references/03-OpenAI-Codex官方源码索引-codex-source-map.md)

**边界**：这里直接研究的是公开的 Codex CLI / harness / runtime；不能因此声称 frontier Codex 模型权重、训练 recipe 或 OpenAI 全部云端生产基础设施已经开源。

---

# 二、仓库结构

| 路径 | 作用 |
|---|---|
| [`教材-book/`](教材-book/) | 模型科学与系统工程正文 |
| [`智能体-agent/`](智能体-agent/) | Agent 通识独立主线 |
| [`Codex源码解剖-codex-anatomy/`](Codex源码解剖-codex-anatomy/) | **OpenAI Codex 官方源码 → clean-room 重建 → parity** |
| [`代码-code/`](代码-code/) | 从零实现代码与终极工程 |
| [`原始论文-paper-notes/`](原始论文-paper-notes/) | Paper/Source Card + Claim Ledger |
| [`参考文献-references/`](参考文献-references/) | 原论文、官方源码、规范、Model/System Card 的证据地图 |
| [`附录-appendices/`](附录-appendices/) | 数学、MiniGPT、后训练数学等附录 |
| [`图表-figures/`](图表-figures/) | 作者重绘机制图、数据流、时间线 |
| [`工具-scripts/`](工具-scripts/) | 数学、来源、链接与教材维护工具 |
| [`学习地图-MAP.md`](学习地图-MAP.md) | Theory ↔ Code ↔ Paper ↔ Test 索引 |
| [`质量看板-QUALITY.md`](质量看板-QUALITY.md) | 全书成熟度与缺口审计 |
| [`.github/`](.github/) | 自动测试、真实 parity、内容审计 |

---

# 三、模型与系统教材正文

| 篇 | 主题 |
|---|---|
| 00 | 导读、方法论与证据标准 |
| 01 | Seq2Seq → Attention → Transformer → GPT/BERT |
| 02 | Scaling Laws、GPT-3、ICL、Chinchilla |
| 03 | Instruction Tuning、RLHF、ChatGPT、DPO |
| 04 | RoPE、RMSNorm、SwiGLU、GQA、FlashAttention、KV Cache |
| 05 | LLaMA、Mistral、LoRA、QLoRA、量化 |
| 06 | RAG、多模态、工具调用与 Agent 历史交汇 |
| 07 | MoE 与中国大模型路线 |
| 08 | CoT、Test-Time Compute、RLVR、Reasoning Models |
| 09 | 大模型训练系统 |
| 10 | 大模型推理与服务系统 |
| 11 | 评测、安全、幻觉、数据污染 |
| 12 | 2026 前沿 |
| 13 | Tokenization 与数据工程 |
| 14 | 优化器、数值精度与训练动力学 |
| 15 | 可解释性与机制研究 |
| 16 | 硬件、Kernel 与基础设施 |
| 17 | 闭源前沿模型与证据边界 |
| 18 | 从零构建 Astra-class / Codex-class 系统 |
| 19 | Post-Transformer：SSM、Mamba、RWKV、Neural Memory、Hybrid |
| 20 | Diffusion Language Models |

详细目录：[`教材-book/README.md`](教材-book/README.md)

---

# 四、Agent 与 Codex 课程

Agent 通识已经覆盖 A1–A10：Foundations、Tools、Planning、Memory、Coding Agent、Multi-Agent、Evaluation/Safety、Agentic RL、MCP/A2A、Browser/Computer Use。

Codex 源码解剖当前覆盖 C00–C07：

```text
C00 官方仓库地图
C01 Agent Loop：Task / Turn / Sampling / Tool Follow-up
C02 Session / Thread / Event / App Server
C03 Tools / Exec / Patch / Approval / Sandbox
C04 AGENTS.md / Context / Compaction / Memory
C05 MCP / App Server / Interoperability
C06 Multi-Agent / Agent Graph / Worktree
C07 Clean-room MiniCodex / Parity
```

对应 Source Card：[`原始论文-paper-notes/2025-2026-OpenAI-Codex-Harness.md`](原始论文-paper-notes/2025-2026-OpenAI-Codex-Harness.md)

---

# 五、终极代码工程

入口：[`代码-code/从零构建Astra与Codex级系统/`](代码-code/从零构建Astra与Codex级系统/)

```text
Byte Tokenizer / BPE
→ RMSNorm / RoPE / GQA / SwiGLU
→ Decoder-only Transformer
→ Raw Safetensors / Public Checkpoint Mapping
→ KV Cache / Prefill / Decode / Sampling
→ Structured Tool Runtime
→ Filesystem / Shell / Git / Edit / Repo Map
→ Agent Loop / Planning / Memory / Verification
→ Minimal MCP
→ Codex-style Turn / Approval / Event Harness
→ Coding Agent / Worktree / Coordinator
→ [next] durable Thread / Sandbox / App Server / Agent Graph
→ [next] Browser / Computer Use
→ [next] Astra-class general agent
```

## 已跨过的三个关键门槛

### 1. 真实公开模型 checkpoint parity

`HuggingFaceTB/SmolLM2-135M` raw weights → 我们自己的 Transformer runtime，在受测 CPU float32 / HF eager reference 条件下：

```json
{"max_abs": 0.0, "mean_abs": 0.0, "argmax_agreement": 1.0}
```

### 2. Minimal MCP

已经从零实现 JSON-RPC 教学 envelope、`server/discover`、`tools/list`、`tools/call`、ToolRegistry adapter、in-process transport 与 tests。

### 3. MiniCodex clean-room v1

新增 `codex_harness.py`，直接把从 OpenAI 官方公开 `Task / Turn / tool follow-up / approval / event` 契约提炼成独立 Python 状态机。当前自动测试覆盖：

```text
tool → observation → follow-up → final
approval deny → tool body never executes
approval allow → tool executes
model-step limit → explicit TURN_STOPPED
```

注意：当前 `SandboxPolicy` 只是状态契约，**没有冒充真实 OS sandbox**；强制隔离仍是下一阶段独立 subsystem。

---

# 六、Source-First 证据链

最高写作规范：[`教材-book/00-教材方法论与证据标准-source-first.md`](教材-book/00-教材方法论与证据标准-source-first.md)

主要证据地图：

- [`参考文献-references/00-原始资料总索引-primary-sources.md`](参考文献-references/00-原始资料总索引-primary-sources.md)
- [`参考文献-references/01-智能体原始资料-agent-sources.md`](参考文献-references/01-智能体原始资料-agent-sources.md)
- [`参考文献-references/02-PostTransformer与扩散语言模型原始资料.md`](参考文献-references/02-PostTransformer与扩散语言模型原始资料.md)
- [`参考文献-references/03-OpenAI-Codex官方源码索引-codex-source-map.md`](参考文献-references/03-OpenAI-Codex官方源码索引-codex-source-map.md)

全书要求：

```text
原始论文 / 官方规范 / 官方源码
        ↓
Claim / Mechanism
        ↓
数学 / State / Data Flow
        ↓
我们的 clean-room 源码
        ↓
Unit / Parity / Reproduction
        ↓
Limitations / Gap
```

---

# 七、自动化质量证据

当前 Fast CPU CI 最新已真实通过：

```text
30 passed, 1 warning
Ruff correctness lint: All checks passed
```

同时保留真实 SmolLM2 checkpoint parity、Source-First coverage audit、严格内部 Markdown link audit 与 MathJax normalization。

质量总看板：[`质量看板-QUALITY.md`](质量看板-QUALITY.md)  
跨层学习地图：[`学习地图-MAP.md`](学习地图-MAP.md)

---

# 八、学习与毕业标准

每个核心机制都走：

```text
Understand
→ Calculate / Formalize
→ Implement
→ Reproduce / Parity
→ Attack / Research
```

最终给一个空目录，不依赖 LangChain 等高层 Agent framework，也不把模型核心藏进 `AutoModelForCausalLM`，能够逐层构建：

```text
Tokenizer
→ Model Runtime
→ Weight Loader
→ Inference Engine
→ Structured Generation
→ Thread / Turn / Event Runtime
→ Tool / Protocol Runtime
→ Approval / Enforced Sandbox
→ Context / Memory / Compaction / Resume
→ Coding Agent
→ MCP / App Server / Multi-Agent / Worktree
→ Browser / Computer Agent
→ Evaluation / Verification
```

并能解释每一层的数学、数据流、状态机、源码、一手资料、实验和系统代价。对于 Coding Agent，还要求能与 `openai/codex` 的**公开可验证 harness 行为**逐项对照；然后再扩展到 Astra-class general agent。