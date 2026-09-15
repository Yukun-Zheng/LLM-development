# 大语言模型、系统工程与智能体：从原始论文到可执行研究平台

> **版本**：v3 · 2026-09-15  
> **定位**：中文主导、英文术语对照、Source-First、Mechanism-First、Reproducibility-First。  
> **终点**：不是收集最多模型名，而是让读者从空目录逐层实现现代 LLM runtime、推理/Serving、post-training、持久 Agent Runtime、Coding / Browser / Computer Agent、多智能体与评测系统。Frontier-scale 权重训练本身不作为要求。

---

# 一、项目现在是三位一体

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
 Mechanism            Post-training       Evaluation
 Exercises            Agent Runtime       Drift Audit
                      Security
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ▼
                     Research Platform
```

核心约束：

$$
\boxed{\text{Claim}\leftrightarrow\text{Code}\leftrightarrow\text{Evidence}}
$$

一个章节文字很多但没有实现/实验，不会因此升到 L2/L3；一个 capability 有源码但没有测试/evidence，也不能叫 `validated`。

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

模型主线覆盖 Transformer/MoE/Scaling/Reasoning/Multimodality/Hybrid Architecture/Diffusion；系统主线覆盖 KV/Serving/Kernels；Agent 主线覆盖 Thread/Event/Context/Tools/Protocols/Security/Multi-Agent；Evaluation 与 Observatory 独立承担实验事实和前沿事实。

教材入口：[`教材-book/README.md`](教材-book/README.md)  
智能体入口：[`智能体-agent/README.md`](智能体-agent/README.md)  
学习地图：[`学习地图-MAP.md`](学习地图-MAP.md)

---

# 三、仓库结构

| 路径 | 作用 |
|---|---|
| [`教材-book/`](教材-book/) | 模型科学、系统、后训练理论正文 |
| [`智能体-agent/`](智能体-agent/) | Agent 通识主线 |
| [`Codex源码解剖-codex-anatomy/`](Codex源码解剖-codex-anatomy/) | **Industrial Reference #1：OpenAI Codex 官方源码 → clean-room → parity** |
| [`代码-code/`](代码-code/) | 从零实现 Reference System |
| [`评测-eval/`](评测-eval/) | 独立能力评测层 |
| [`观测站-observatory/`](观测站-observatory/) | 机器可读 frontier snapshot / drift audit |
| [`能力清单-CAPABILITIES.json`](能力清单-CAPABILITIES.json) | Theory ↔ Code ↔ Test ↔ Evidence 事实源 |
| [`原始论文-paper-notes/`](原始论文-paper-notes/) | Paper/Source Card + Claim Ledger |
| [`参考文献-references/`](参考文献-references/) | 原论文、官方源码、规范、Model/System Card |
| [`附录-appendices/`](附录-appendices/) | 数学、MiniGPT、后训练数学 |
| [`图表-figures/`](图表-figures/) | 机制图、shape/data-flow、时间线 |
| [`工具-scripts/`](工具-scripts/) | 来源/链接/manifest/frontier/math 审计 |
| [`质量看板-QUALITY.md`](质量看板-QUALITY.md) | 当前成熟度与缺口 |
| [`.github/`](.github/) | CI、checkpoint parity、内容与 drift 审计 |

---

# 四、真实代码工程已经走到哪里

入口：[`代码-code/从零构建Astra与Codex级系统/`](代码-code/从零构建Astra与Codex级系统/)

当前纵向链已经扩展为：

```text
Text / Tokenizer
→ Own Transformer Runtime
→ Raw Public Checkpoint
→ Real Logits Parity
→ KV Cache
→ Reference Paged KV
→ Request Scheduler
→ Actual Batched Model Forward

SFT
→ DPO
→ Group-relative Verifier-RL Objective Primitive

Tool Runtime
→ Codex-style Turn Executor
→ Durable Thread / Event Replay
→ Fork / Cancellation
→ Durable Work Queue / Lease
→ Integrated Thread→Queue→Turn Runtime
→ Context Provenance / Compaction
→ Permission / Approval Gate

Generic Benchmark Harness
→ Repository Final-state Grader
```

## 真实模型 parity

`HuggingFaceTB/SmolLM2-135M` raw weights → 自己写的 Transformer runtime，固定受测 CPU float32 / HF eager reference：

```json
{"max_abs": 0.0, "mean_abs": 0.0, "argmax_agreement": 1.0}
```

这只证明当前受测配置，不外推全部模型。

---

# 五、Inference 已不再只是 `generate()`

现在已有：

```text
Contiguous KV
→ Reference Paged KV
→ Request Scheduler
→ Homogeneous Batched Executor
```

`batch_executor.py` 真正把多个 request 放进一次 model forward，并把 batched K/V 拆回各自 decode state；自动测试证明 batched prefill/decode 与 individual/full recomputation 数值一致。

当前边界：decode batch 要求 equal cached sequence length。真正 variable-length continuous batching、physical block allocator、prefix sharing、chunked/disaggregated prefill 与 speculative decoding 仍未完成。

实验：[`代码-code/从零构建Astra与Codex级系统/教程-lessons/05-PagedKV与调度-serving-runtime.md`](代码-code/从零构建Astra与Codex级系统/教程-lessons/05-PagedKV与调度-serving-runtime.md)

---

# 六、Post-training 第一次真正进入 Tensor / Gradient

现在不是只讲 RLHF/DPO/GRPO 数学。

```text
SFT:
causal shift + prompt mask + CE + backward + optimizer

DPO:
policy/reference chosen/rejected sequence logp
→ implicit reward margin
→ DPO loss
→ update policy, freeze reference

Group-relative RL primitive:
verifier rewards [B,G]
→ within-group advantages
→ policy ratio
→ clipped surrogate
→ optional reference KL
```

自动不变量包括：

```text
uniform logits SFT CE = log(V)
policy == reference → DPO loss = log(2)
group-relative advantages have zero group mean
```

实验：[`Lab 04`](代码-code/从零构建Astra与Codex级系统/教程-lessons/04-从SFT到DPO-post-training.md)、[`Lab 07`](代码-code/从零构建Astra与Codex级系统/教程-lessons/07-从Verifier到GroupRelativeRL-rlvr.md)。

完整 RLVR pipeline 仍缺真实 rollouts / verifier / old-policy snapshot / tiny-model policy update / held-out evaluator。

---

# 七、Agent 已从 while-loop 走向 Durable Runtime

现在核心不是：

```text
model → tool → model
```

而逐渐变成：

```text
Submission
→ Durable Thread
→ Durable Work Queue
→ Worker Lease
→ Turn Executor
→ Event / Checkpoint
→ Context Builder
→ Action / Tool
→ Verifier
→ ACK / Replan / Finish
```

当前已经验证：

```text
process reopen → replay thread state
thread fork → independent child event stream
cancel child → parent unchanged
lease expires → another worker can reclaim
terminal thread → pending work not executed
step limit → failed work + failed thread
```

`runtime.py` 已把 ThreadStore、WorkQueue 与 Codex-style TurnExecutor 接成第一版持久 pipeline。

重要未解问题：如果 worker 在 side-effecting tool 中途死亡，不能直接 at-least-once 重跑，否则可能产生重复副作用。下一阶段要做 durable tool-execution record / idempotency semantics。

实验：[`Lab 06`](代码-code/从零构建Astra与Codex级系统/教程-lessons/06-从AgentLoop到DurableRuntime-durable-agent.md)

---

# 八、Evaluation 已经开始看最终环境，而不是听模型自述

现在有：

```text
BenchmarkCase
→ Executor
→ Trajectory
→ Grader
→ Grade
→ AggregateMetrics
```

并新增 deterministic repository fixture：固定 initial files / goal / verify argv / expected final state。

自动测试故意让 Agent 只说：

```text
"I fixed it and all tests pass."
```

但不编辑文件，结果独立 final-state grader 判 **FAIL**；真实 edit + independent verification 才 **PASS**。

入口：[`评测-eval/README.md`](评测-eval/README.md)

---

# 九、OpenAI Codex 的位置

OpenAI 已公开 `openai/codex` CLI / harness / runtime 源码，因此本项目直接用官方源建立：

```text
Official Source
→ Runtime / Protocol Contract
→ Clean-room Teaching Implementation
→ Behavioral / Protocol Parity
→ Gap Report
```

入口：[`Codex源码解剖-codex-anatomy/README.md`](Codex源码解剖-codex-anatomy/README.md)

Codex 是 **Industrial Reference #1：Coding Agent Runtime**，不是整个 Agent 学科唯一架构答案。Frontier Codex model weights / full training recipe / all cloud infra 并未因此开源。

---

# 十、最新自动化硬证据

Fast CPU CI run 66：

```text
61 passed, 1 warning in 3.67s
Ruff correctness lint: All checks passed
```

Content CI 继续检查：

```text
Source-First coverage
Strict internal Markdown links
Capability Manifest
Frontier snapshot freshness/schema
Math normalization
```

实时事实源：[`能力清单-CAPABILITIES.json`](能力清单-CAPABILITIES.json)  
质量总览：[`质量看板-QUALITY.md`](质量看板-QUALITY.md)

---

# 十一、当前最重要的未完成工作

```text
Inference:
variable-length continuous batching
physical Paged-KV allocator / prefix cache
chunked/disaggregated prefill
speculative decoding
real throughput/memory/fairness benchmark

Post-training:
dataset/checkpoint training loops
verifier rollouts
actual tiny-policy RLVR update
held-out evaluator

Agent OS:
durable tool execution / idempotency
active-turn crash recovery
pending live steering
worker heartbeat
App Server / artifact store

Security:
real OS/container isolation
filesystem/network/secret boundaries
escape tests

Evaluation:
multi-file hidden-test repo suite
SWE-bench adapter
Browser environment
Computer/OS environment

General Agent:
JS browser / DOM/A11y
screenshot / visual grounding
mouse/keyboard / state verifier
```

从现在开始，项目的主要进展单位不是“新增多少页”，而是：

> **又有多少 claim 从“讲过”变成“亲手实现、自动验证、真实评测，并且明确知道失败边界”。**
