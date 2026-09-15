# 大语言模型、系统工程与智能体：从原始论文到可执行研究平台

> **版本**：v3 · 2026-09-15  
> **定位**：中文主导、英文术语对照、Source-First、Mechanism-First、Reproducibility-First。  
> **终点**：不是收集最多的模型名，而是让读者能从空目录逐层实现现代 LLM runtime、推理系统、post-training 教学闭环、持久 Agent OS、Coding / Browser / Computer Agent、多智能体与评测系统。Frontier 权重训练本身不作为要求。

---

# 一、v3：项目不再只是“教材”

```text
                    LLM-development
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
     TEXTBOOK         REFERENCE SYSTEM      OBSERVATORY
       教科书              可执行系统           前沿观测站
        │                  │                  │
 Theory / Math        Model Runtime        Frontier Snapshot
 History / Sources    Inference Engine     Reproduction
 Mechanism            Agent Runtime        Evaluation
 Exercises            Security             Drift Audit
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ▼
                     Research Platform
```

核心约束：

$$
\boxed{\text{Claim}\leftrightarrow\text{Code}\leftrightarrow\text{Evidence}}
$$

一个能力只有写了代码但没有测试，不算 `validated`；一个章节只有文字没有实验，也不会因为篇幅长而自动成为 L2/L3。

总体蓝图：[`教材-book/99-全书架构蓝图-v3.md`](教材-book/99-全书架构蓝图-v3.md)

---

# 二、七个技术层

```text
A  Model Science
B  Inference Systems
C  Post-training Systems
D  Agent Kernel
E  Environment / Protocol / Security
F  Multi-Agent
G  Evaluation / Observatory
```

## A　Model Science

数学 / 概率 → Tokenization / Data → Transformer / MoE → Scaling → Reasoning → Multimodality → Efficient / Hybrid Sequence Architectures → Diffusion LM。

入口：[`教材-book/README.md`](教材-book/README.md)

## B　Inference Systems

Attention / Kernel → KV / State → Paged KV → Prefix Cache → Prefill / Decode → Continuous Batching → Speculation → Distributed / Disaggregated Serving。

## C　Post-training Systems

不训练 frontier-scale 权重，不等于不实现训练算法。项目最终至少运行：

```text
Tiny Pretraining
→ SFT
→ Preference Data
→ DPO
→ Reward / Verifier
→ RLVR / GRPO-style Toy Loop
→ Agent Trajectory Learning
```

## D　Agent Kernel

核心对象从单次 `model -> tool` 提升为：

```text
Submission
→ Thread
→ Turn
→ Event
→ Event Store
→ Replay
→ Checkpoint
→ Resume / Fork
```

## E　Environment / Protocol / Security

Tool / Filesystem / Shell / Git / Browser / Computer，与 MCP / A2A / App-Server-like protocol、Permission / Approval / Sandbox / Credentials 分层实现。

## F　Multi-Agent

Task Graph → Scheduler → Worker → Mailbox → Artifact Handoff → Worktree / State Isolation → Reviewer / Merge，并用单 Agent baseline 做成本与成功率对照。

## G　Evaluation / Observatory

pytest、真实 capability benchmark 与 frontier evidence 三者分离。入口：[`评测-eval/`](评测-eval/) 与 [`观测站-observatory/`](观测站-observatory/)。

---

# 三、仓库结构

| 路径 | 作用 |
|---|---|
| [`教材-book/`](教材-book/) | 模型科学、系统工程、post-training 理论主线 |
| [`智能体-agent/`](智能体-agent/) | Agent 通识：state/action/environment/planning/memory/eval/safety |
| [`Codex源码解剖-codex-anatomy/`](Codex源码解剖-codex-anatomy/) | **Industrial Reference #1：OpenAI Codex 官方源码 → clean-room → parity** |
| [`代码-code/`](代码-code/) | 从零实现的 Reference System |
| [`评测-eval/`](评测-eval/) | **独立能力评测层：success / time / cost / steps / verifier** |
| [`观测站-observatory/`](观测站-observatory/) | **机器可读前沿快照与 drift audit** |
| [`能力清单-CAPABILITIES.json`](能力清单-CAPABILITIES.json) | **Theory ↔ Code ↔ Test ↔ Evidence 机器可读事实源** |
| [`原始论文-paper-notes/`](原始论文-paper-notes/) | Paper / Source Card + Claim Ledger |
| [`参考文献-references/`](参考文献-references/) | 原论文、官方源码、规范、Model/System Card 证据地图 |
| [`附录-appendices/`](附录-appendices/) | 数学、MiniGPT、后训练数学 |
| [`图表-figures/`](图表-figures/) | 作者重绘机制图、shape/data-flow、时间线 |
| [`工具-scripts/`](工具-scripts/) | 数学、来源、链接、manifest、frontier 审计 |
| [`学习地图-MAP.md`](学习地图-MAP.md) | Theory ↔ Code ↔ Paper ↔ Test ↔ Eval 导航 |
| [`质量看板-QUALITY.md`](质量看板-QUALITY.md) | 当前成熟度与 P0/P1/P2 缺口 |
| [`.github/`](.github/) | 自动测试、真实 parity、内容/manifest/drift 审计 |

---

# 四、教材覆盖范围

主干当前覆盖 00–20：Transformer、Scaling、Alignment、现代架构、开放权重与高效微调、RAG/多模态/Tool、MoE、中国模型、Reasoning、训练系统、推理系统、Evaluation/Safety、2026 Frontier、Data、Optimization、Interpretability、Hardware、Closed Frontier Models、Capstone、Post-Transformer/Hybrid、Diffusion LM。

详细目录：[`教材-book/README.md`](教材-book/README.md)

从 v3 开始，第 19 篇逐渐从营销式“Post-Transformer”问题转成更可测量的 **Efficient / Hybrid Sequence Architectures**：比较 state/KV bytes、FLOPs/token、bandwidth、retrieval precision 与 parallelism。

---

# 五、OpenAI Codex：工业参考实现，而不是唯一 Agent 答案

OpenAI 已公开 `openai/codex` 的 CLI / agent harness / runtime 源码，因此 Coding Agent 可以直接研究真实工业实现，而不只根据产品行为猜测。

路线：

```text
Official Source
→ State / Protocol Contract
→ Clean-room Implementation
→ Unit / Behavioral / Protocol Parity
→ Gap Report
```

入口：[`Codex源码解剖-codex-anatomy/README.md`](Codex源码解剖-codex-anatomy/README.md)

但 Codex 被定义为：

> **Industrial Reference Implementation #1：Coding Agent Runtime**

而不是“所有 Agent 系统唯一正确的架构”。后续同样会研究 vLLM/SGLang 作为 inference runtime reference、MCP/A2A 作为 protocol reference、公开模型仓库作为 architecture reference。

---

# 六、终极代码工程当前真实状态

入口：[`代码-code/从零构建Astra与Codex级系统/`](代码-code/从零构建Astra与Codex级系统/)

已经落地：

```text
Byte Tokenizer / BPE
→ RMSNorm / RoPE / GQA / SwiGLU
→ Decoder-only Transformer
→ Raw Safetensors / Public Checkpoint Mapping
→ KV Cache / Prefill / Decode / Sampling
→ Structured Tool Runtime
→ Filesystem / Shell / Git / Edit / Repo Map
→ Planning / Verification / Persistent Memory
→ Minimal MCP
→ Codex-style Turn / Approval / Event Harness
→ Worktree / Coordinator
→ Durable Thread / Event Store / Replay / Resume   [v3]
→ Pre-dispatch Permission Enforcement              [v3]
→ Agent Trajectory Metrics                         [v3]
```

## 真实模型 parity

`HuggingFaceTB/SmolLM2-135M` raw weights → 我们自己的 Transformer runtime，在当前受测 CPU float32 / HF eager reference 条件下：

```json
{"max_abs": 0.0, "mean_abs": 0.0, "argmax_agreement": 1.0}
```

这证明的是当前受测配置的 forward parity，不外推到所有模型族。

## Durable Runtime v0

新增 event-sourced SQLite Thread Store。当前状态由事件 replay 得到，而不是依赖进程内对象；测试覆盖数据库关闭/重新打开后恢复 running turn、checkpoint、pause/resume 与 terminal transition。

## Permission v0

新增 `proposal → permission policy → optional approval → dispatch` 的 application-layer enforcement。DENY / approval reject 时底层 tool body 不执行。它仍**不是 OS/container sandbox**；真实 process/network/secret isolation 继续单列 P0。

## Evaluation v0

新增统一 trajectory metrics：success、verifier、model steps、tool calls、tool failures、approvals、wall time、tokens、cost，为后续 SWE-bench / browser / OS benchmark 适配建立同一记录格式。

实时事实源：[`能力清单-CAPABILITIES.json`](能力清单-CAPABILITIES.json)

---

# 七、v3 P0：现在优先做什么

```text
1. Capability Manifest / CI             ← 已开始
2. Durable Thread / Event Store / Replay← 已开始
3. Evaluation Harness                   ← 已开始
4. Permission Enforcement               ← 已开始
5. Enforced OS / Container Sandbox      ← 未完成
6. Context Provenance / Notes / Index   ← 未完成
7. Paged KV Reference + Parity          ← 未完成
8. Request Scheduler / Metrics           ← 未完成
```

接下来不再以“新增 Markdown 数量”作为主要进展。

---

# 八、Source-First 与 Observatory

稳定知识：回到原论文、官方代码、官方规范、system/model card。

快速变化事实：进入 [`观测站-observatory/frontier-snapshot.json`](观测站-observatory/frontier-snapshot.json)，记录：

```text
as_of
release/version
official source
facts
evidence boundary
revalidation window
```

这样第 12/17 篇不会单独承担“今天世界最新状态”的维护压力。

---

# 九、自动化质量链

CI 现在覆盖：

```text
Fast CPU tests + Ruff
Real public-checkpoint parity
Source-First coverage
Strict internal Markdown links
Capability Manifest validation
Frontier snapshot schema/freshness report
MathJax normalization
```

目标证据链：

```text
Primary Source / Official Source
→ Claim / Contract
→ Theory / State / Data Flow
→ Our Code
→ Unit / Parity
→ Capability Eval
→ Limitations / Gap
```

---

# 十、最终毕业标准 v3

给一个空目录，不依赖 LangChain 等高层 Agent framework，也不把模型核心藏进 `AutoModelForCausalLM`，能够解释并逐层实现：

```text
Model
├─ tokenizer
├─ architecture
├─ checkpoint loader
└─ parity

Inference
├─ KV/state
├─ paged cache
├─ scheduler/batching
├─ speculation
└─ metrics

Post-training
├─ SFT
├─ preference/DPO
└─ verifier/RL loop

Agent OS
├─ durable thread
├─ turn/event/replay
├─ context/memory
├─ planning/verifier
├─ permission/sandbox
├─ MCP/A2A
├─ coding/browser/computer
└─ multi-agent

Evaluation
├─ deterministic tests
├─ real benchmark adapters
├─ cost/latency accounting
└─ failure/safety analysis
```

并且对任意重要结论都能回答：

> **原始证据在哪里？代码在哪里？测试在哪里？真实能力证据在哪里？适用边界在哪里？失败时怎样被观测到？**

这才是本项目的终点。
