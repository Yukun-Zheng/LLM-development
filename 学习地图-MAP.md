# Theory ↔ Code ↔ Source ↔ Test ↔ Eval 学习地图 v3

> **用途**：每个核心主题都必须能从理论追到一手资料，从一手资料追到自己的 reference code，再追到 tests / parity / benchmark。  
> 机器状态：[`能力清单-CAPABILITIES.json`](能力清单-CAPABILITIES.json)。

`src/...` 默认位于 `代码-code/从零构建Astra与Codex级系统/src/astra_codex/`。

---

# 1　Model Core

| 主题 | 理论 | Primary / Official | Reference Code | Evidence | 下一研究层 |
|---|---|---|---|---|---|
| Tokenization / BPE | [`教材 13`](教材-book/13-词元化与数据工程-tokenization-data.md) | BPE / SentencePiece | `tokenizer.py` | tokenizer tests | fertility / multilingual/code |
| Transformer | [`教材 01`](教材-book/01-Transformer基础-foundations-transformer.md) | Vaswani 2017 + [`Source Card`](原始论文-paper-notes/2017-Attention-Is-All-You-Need.md) | `model.py` | model/cache tests | activation parity |
| RMSNorm / RoPE / GQA / SwiGLU | [`教材 04`](教材-book/04-现代LLM架构-modern-architecture.md) | original papers | `model.py` | real checkpoint parity | second architecture family |
| Weight Mapping | [`18A`](教材-book/18A-模型运行时源码导读-runtime-walkthrough.md) | public config/safetensors | `weights.py`, `public_checkpoint.py` | strict load / shape audit | more mappings |
| **SmolLM2 real runtime** | [`18A`](教材-book/18A-模型运行时源码导读-runtime-walkthrough.md) | `HuggingFaceTB/SmolLM2-135M` | own Transformer | **max_abs=0, mean_abs=0, argmax=1.0** | tokenizer/generation parity |

---

# 2　Inference / Serving

代码实验：[`Lab 05`](代码-code/从零构建Astra与Codex级系统/教程-lessons/05-PagedKV与调度-serving-runtime.md)

| 主题 | 理论 | Source | Code | Evidence | 下一步 |
|---|---|---|---|---|---|
| Contiguous KV | [`教材 10`](教材-book/10-推理服务系统-inference-systems.md) | serving implementations | `cache.py`, `engine.py` | cached/full logits parity | bytes/token |
| Reference Paged KV | [`教材 10`](教材-book/10-推理服务系统-inference-systems.md) | PagedAttention / vLLM | `paged_cache.py` | paged/full logits parity | physical allocator / fragmentation |
| Request Scheduler | [`教材 10`](教材-book/10-推理服务系统-inference-systems.md) | vLLM/SGLang concepts | `scheduler.py` | admission/decode/cancel/TTFT/TPOT | fairness / actual throughput |
| **Batched Model Executor** | [`教材 10`](教材-book/10-推理服务系统-inference-systems.md) | serving runtimes | `batch_executor.py` | **one real batched forward; prefill/decode == individual recomputation** | variable-length block-table batching |
| Prefix Cache | [`教材 10`](教材-book/10-推理服务系统-inference-systems.md) | serving systems | — | todo | hit rate / TTFT |
| Chunked / Disaggregated Prefill | [`教材 10`](教材-book/10-推理服务系统-inference-systems.md) | serving systems | — | todo | TTFT / TPOT |
| Speculative Decoding | [`教材 10`](教材-book/10-推理服务系统-inference-systems.md) | original papers | — | todo | distribution parity / speed |
| Flash / SDPA | [`教材 16`](教材-book/16-硬件内核与基础设施-hardware-kernels.md) | FlashAttention | explicit attention | backend parity todo | bandwidth / latency |

统一最终指标：`TTFT / TPOT / ITL / throughput / memory / queueing / batch occupancy / fairness`。

---

# 3　Post-training

实验：

- [`Lab 04：SFT → DPO`](代码-code/从零构建Astra与Codex级系统/教程-lessons/04-从SFT到DPO-post-training.md)
- [`Lab 07：Verifier → Group-relative RL`](代码-code/从零构建Astra与Codex级系统/教程-lessons/07-从Verifier到GroupRelativeRL-rlvr.md)

| 主题 | 理论 | Primary Source | Code | 自动不变量 | 下一闭环 |
|---|---|---|---|---|---|
| SFT | [`教材 03`](教材-book/03-对齐与ChatGPT-alignment-chatgpt.md) | FLAN/T0/InstructGPT | `posttraining.py` | masked CE hand-check + optimizer update | dataset/packing/checkpoint |
| DPO | [`教材 03`](教材-book/03-对齐与ChatGPT-alignment-chatgpt.md) | DPO paper | `posttraining.py` | `policy==reference → loss=log(2)`；ref frozen | pairwise dataset / held-out preference eval |
| **Group-relative verifier RL objective** | [`教材 08`](教材-book/08-推理模型时代-reasoning-era.md), [`A8`](智能体-agent/08-环境学习与AgenticRL-agentic-rl.md) | DeepSeekMath / R1 | `rlvr.py` | centered advantages；gradient direction；clipping/KL | sampled rollouts + real tiny-policy update |
| Full RLVR | 同上 | original/official reports | — | todo | verifier environment + old policy + held-out eval |

原则：**不训练 frontier-scale weights ≠ 不实现训练算法。**

---

# 4　Hybrid / Post-Transformer

| 主题 | 理论 | Source | Current Code | 下一验证 |
|---|---|---|---|---|
| S4 | [`教材 19`](教材-book/19-Post-Transformer架构-post-transformer.md) | S4 | — | recurrent ↔ matrix parity |
| Mamba / Mamba-2 | [`教材 19`](教材-book/19-Post-Transformer架构-post-transformer.md) | official papers/code | — | scan / matrix / chunked equivalence |
| RWKV / xLSTM | [`教材 19`](教材-book/19-Post-Transformer架构-post-transformer.md) | papers/repos | — | state / latency lab |
| Neural Memory | [`教材 19`](教材-book/19-Post-Transformer架构-post-transformer.md) | Titans | — | test-time memory |
| Diffusion LM | [`教材 20`](教材-book/20-扩散语言模型-diffusion-language-models.md) | discrete diffusion / LLaDA | — | corruption → denoise → remask |

---

# 5　Durable Agent Kernel

实验：[`Lab 06`](代码-code/从零构建Astra与Codex级系统/教程-lessons/06-从AgentLoop到DurableRuntime-durable-agent.md)

| Capability | Theory / Source | Code | Evidence | 下一层 |
|---|---|---|---|---|
| Minimal Agent Loop | [`A1`](智能体-agent/01-智能体基础-agent-foundations.md) / ReAct | `agent.py` | scripted tests | baseline only |
| Codex-style Turn | [`C01`](Codex源码解剖-codex-anatomy/01-核心AgentLoop-agent-loop.md) / openai/codex | `codex_harness.py` | follow-up/approval/stop | event stream / steering |
| Durable Thread | [`C02`](Codex源码解剖-codex-anatomy/02-会话线程与事件协议-session-thread-events.md) | `durable.py` | restart replay / fork / cancel | active-turn crash recovery |
| Durable Work Queue | [`C02`](Codex源码解剖-codex-anatomy/02-会话线程与事件协议-session-thread-events.md) | `runtime_queue.py` | lease exclusion / expiry reclaim / ACK/cancel | heartbeat / async pool |
| **Integrated Durable Runtime** | C01+C02 | `runtime.py` | **submission→work→turn→checkpoint→ACK; failure semantics** | durable tool execution/idempotency |
| Context Provenance | [`A4`](智能体-agent/04-记忆与上下文-memory-context.md), [`C04`](Codex源码解剖-codex-anatomy/04-指令上下文压缩与记忆-context-memory.md) | `context.py` | compaction lineage retains raw evidence | notes/index/token budget |
| Planning | [`A3`](智能体-agent/03-规划反思与验证-planning-reflection-verification.md) | `planning.py` | DAG/cycle/failure tests | dynamic replanning |
| Verifier | [`A3`](智能体-agent/03-规划反思与验证-planning-reflection-verification.md) | `verification.py` | file/command/composite | artifact graders |
| Coding Agent | [`A5`](智能体-agent/05-编程智能体-coding-agents.md) | `coding.py`, tools/edit/repo-map | repo primitives | semantic code intelligence |

当前关键研究边界：进程在 side-effecting tool 中途死亡时，简单 retry 可能产生 duplicate side effects；需要 durable tool execution / idempotency 设计。

---

# 6　Environment / Protocol / Security

| 主题 | 理论 / Spec | Code | 证据 | 下一步 |
|---|---|---|---|---|
| Filesystem / Shell / Git | [`A2`](智能体-agent/02-工具与环境-tool-use-environments.md) | `tools.py` | rooted path / command tests | sandbox integration |
| Permission / Approval | [`A7`](智能体-agent/07-评测安全与开放问题-agent-evaluation-safety.md), [`C03`](Codex源码解剖-codex-anatomy/03-工具执行审批与沙箱-tools-approval-sandbox.md) | `security.py` | deny/reject prevents dispatch | credentials + real sandbox |
| OS / Container Sandbox | C03 | — | — | process/fs/network/secret isolation |
| MCP | [`A9`](智能体-agent/09-Agent协议与互操作-agent-protocols.md) | `mcp.py` subset | discover/list/call | transports/auth/tasks/extensions |
| A2A | A9 | — | — | AgentCard/Task/Message/Artifact |
| Browser | [`A10`](智能体-agent/10-浏览器与计算机使用-browser-computer-use.md) | HTTP-only `general_tools.py` | no GUI claim | JS/DOM/A11y |
| Computer Use | A10 | — | — | screenshot/grounding/actions/verifier |

---

# 7　Evaluation

独立实验层：[`评测-eval/README.md`](评测-eval/README.md)

| 层 | Code | Current Evidence | 下一步 |
|---|---|---|---|
| Trajectory Metrics | `evaluation.py` | success/verifier/steps/tools/time/cost aggregation | production instrumentation |
| Generic Case/Grader | `benchmark.py` | executor/grader separation | metadata/versioning |
| **Repository Final-state Eval** | `repository_eval.py` | **false success claim → FAIL；real edit + independent verification → PASS** | hidden/multi-file fixtures |
| SWE-bench | — | — | adapter |
| Browser / Web | — | — | WebArena-style adapter |
| Computer / OS | — | — | OSWorld-style adapter |
| Recovery | durable/runtime primitives | unit invariants | kill/restart/duplicate-side-effect benchmark |

统一证据：

```text
trajectory
+ final environment state
+ artifact
+ grader
+ cost/latency
```

---

# 8　Multi-Agent

| 主题 | Theory/Source | Code | 当前 | 下一研究问题 |
|---|---|---|---|---|
| Coordinator | [`A6`](智能体-agent/06-多智能体与协调-multi-agent.md) | `multi_agent.py` | deterministic | scheduling policy |
| Worktree | [`C06`](Codex源码解剖-codex-anatomy/06-多智能体与工作树-multi-agent-worktree.md) | `worktree.py` | primitive | concurrent repo safety |
| AgentGraph / Mailbox | C06 | — | todo | lifecycle / messaging |
| Reviewer / Merge | coding track | — | todo | verifier-driven merge |
| Scale-out Agent Eval | [`评测`](评测-eval/README.md) | metrics substrate | todo | **1/2/4/8 agents: success-time-cost-conflict** |

---

# 9　Frontier Observatory

[`观测站-observatory/frontier-snapshot.json`](观测站-observatory/frontier-snapshot.json)：

```text
Official Source
→ machine-readable fact
→ evidence boundary
→ freshness age
→ drift audit
```

审计：[`工具-scripts/前沿漂移审计-frontier_drift_audit.py`](工具-scripts/前沿漂移审计-frontier_drift_audit.py)。

---

# 10　固定学习闭环

```text
Theory
→ Primary / Official Source
→ Reference Implementation
→ Numerical / Behavioral / Protocol Tests
→ Real Evaluation
→ Ablation / Failure Case
```

Fast CI run 66 已达到：

```text
61 passed, 1 warning in 3.67s
Ruff correctness lint: All checks passed
```

后续增长目标不是“更多文件”，而是让越来越多行从 **Theory only** 变成 **Reference + Evidence + Eval**。
