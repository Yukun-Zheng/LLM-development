# Theory ↔ Code ↔ Source ↔ Test ↔ Eval 学习地图 v3

> **用途**：把教材、原论文/官方源码、Reference System、自动测试与真实能力评测连接起来。  
> 机器可读状态：[`能力清单-CAPABILITIES.json`](能力清单-CAPABILITIES.json)。  
> 下表中的 `src/...` 默认位于 `代码-code/从零构建Astra与Codex级系统/src/astra_codex/`。

---

# 1　模型核心

| 主题 | 理论 | Source | 源码 | Test / Parity | 下一 Eval |
|---|---|---|---|---|---|
| Tokenization / BPE | [`教材 13`](教材-book/13-词元化与数据工程-tokenization-data.md) | Sennrich / SentencePiece | `tokenizer.py` | tokenizer tests | 中文/代码 fertility |
| Transformer | [`教材 01`](教材-book/01-Transformer基础-foundations-transformer.md) | Vaswani 2017 + [`Source Card`](原始论文-paper-notes/2017-Attention-Is-All-You-Need.md) | `model.py` | model/cache tests | activation parity |
| RMSNorm / RoPE / GQA / SwiGLU | [`教材 04`](教材-book/04-现代LLM架构-modern-architecture.md) | original papers | `model.py` | SmolLM2 parity | architecture matrix |
| Public Checkpoint Mapping | [`18A`](教材-book/18A-模型运行时源码导读-runtime-walkthrough.md) | public config / safetensors | `weights.py`, `public_checkpoint.py` | strict load + shape audit | more model families |
| **SmolLM2 real runtime** | [`18A`](教材-book/18A-模型运行时源码导读-runtime-walkthrough.md) | `HuggingFaceTB/SmolLM2-135M` | `public_checkpoint.py` | **max_abs=0, mean_abs=0, argmax=1.0** | generation/tokenizer parity |

---

# 2　Inference Systems

| 主题 | 理论 | Source | 当前代码 | 正确性证据 | 下一系统指标 |
|---|---|---|---|---|---|
| Contiguous KV | [`教材 10`](教材-book/10-推理服务系统-inference-systems.md) | inference implementations | `cache.py`, `engine.py` | cached-vs-full logits parity | bytes/token |
| Paged KV | [`教材 10`](教材-book/10-推理服务系统-inference-systems.md) | vLLM / PagedAttention | — | todo | fragmentation / HBM |
| Prefix Cache | [`教材 10`](教材-book/10-推理服务系统-inference-systems.md) | serving systems | — | todo | hit rate / TTFT |
| Continuous Batching | [`教材 10`](教材-book/10-推理服务系统-inference-systems.md) | serving runtimes | — | todo | throughput / fairness |
| Chunked / Disaggregated Prefill | [`教材 10`](教材-book/10-推理服务系统-inference-systems.md) | serving runtimes | — | todo | TTFT / TPOT |
| Speculative Decoding | [`教材 10`](教材-book/10-推理服务系统-inference-systems.md) | original papers | — | todo | accepted tokens / speedup |
| Flash / SDPA | [`教材 16`](教材-book/16-硬件内核与基础设施-hardware-kernels.md) | FlashAttention 1/2/3 | explicit reference attention | backend parity todo | bandwidth / latency |

最终统一记录：`TTFT / TPOT / ITL / throughput / memory / scheduler fairness`。

---

# 3　Post-training

| 主题 | 理论 | Source | 当前实现 | 下一最小闭环 |
|---|---|---|---|---|
| SFT | [`教材 03`](教材-book/03-对齐与ChatGPT-alignment-chatgpt.md) | FLAN/T0/InstructGPT | — | tiny conversation SFT |
| Reward / PPO | [`教材 03`](教材-book/03-对齐与ChatGPT-alignment-chatgpt.md), [`附录 C`](附录-appendices/C-后训练数学-post-training-math.md) | Christiano/PPO/InstructGPT | verifier primitive | reward model + toy PPO |
| DPO | [`教材 03`](教材-book/03-对齐与ChatGPT-alignment-chatgpt.md) | DPO paper | — | pairwise dataset + loss parity |
| GRPO / RLVR | [`教材 08`](教材-book/08-推理模型时代-reasoning-era.md) | DeepSeekMath / R1 | verifier infrastructure | verifiable reward + policy update |
| Agentic RL | [`A8`](智能体-agent/08-环境学习与AgenticRL-agentic-rl.md) | environment/RL sources | — | trajectory store + toy loop |

**不做 frontier-scale training，不等于不实现 training algorithms。**

---

# 4　Hybrid Sequence Architecture

| 主题 | 理论 | Source | Reference Code | 下一验证 |
|---|---|---|---|---|
| S4 | [`教材 19`](教材-book/19-Post-Transformer架构-post-transformer.md) | S4 | — | recurrent ↔ matrix parity |
| Mamba / Mamba-2 | [`教材 19`](教材-book/19-Post-Transformer架构-post-transformer.md) | original + official code | — | scan / matrix / chunked equivalence |
| RWKV / xLSTM | [`教材 19`](教材-book/19-Post-Transformer架构-post-transformer.md) | papers/repos | — | state size / latency |
| Neural Memory | [`教材 19`](教材-book/19-Post-Transformer架构-post-transformer.md) | Titans | — | test-time memory lab |
| Diffusion LM | [`教材 20`](教材-book/20-扩散语言模型-diffusion-language-models.md) | discrete diffusion / LLaDA | — | corruption → denoise → remask |

比较目标：state/KV bytes、FLOPs/token、bandwidth、parallelism、retrieval/retention 与 decode complexity。

---

# 5　Agent Kernel

| Capability | 理论 / Industrial Source | 源码 | 自动验证 | 下一层 |
|---|---|---|---|---|
| Minimal Agent Loop | [`A1`](智能体-agent/01-智能体基础-agent-foundations.md) / ReAct | `agent.py` | scripted backend | baseline only |
| Codex-style Turn | [`C01`](Codex源码解剖-codex-anatomy/01-核心AgentLoop-agent-loop.md) | `codex_harness.py` | follow-up / approval / stop | async submissions/events |
| **Durable Thread / Event Store** | [`C02`](Codex源码解剖-codex-anatomy/02-会话线程与事件协议-session-thread-events.md) | `durable.py` | **DB reopen replay + transition tests** | fork / cancel / lease |
| Planning DAG | [`A3`](智能体-agent/03-规划反思与验证-planning-reflection-verification.md) | `planning.py` | dependency/cycle/failure | dynamic replan |
| Verifier | [`A3`](智能体-agent/03-规划反思与验证-planning-reflection-verification.md) | `verification.py` | file/command/composite | artifact graders |
| Persistent Memory | [`A4`](智能体-agent/04-记忆与上下文-memory-context.md) | `memory.py` | SQLite tests | notes/index/provenance/compaction |
| Coding Agent | [`A5`](智能体-agent/05-编程智能体-coding-agents.md) | `coding.py`, tools/edit/repo-map | primitives | tree-sitter/LSP/SWE-bench |

---

# 6　Environment / Protocol / Security

| 主题 | 理论 / Spec | 当前代码 | 证据 | 下一步 |
|---|---|---|---|---|
| Filesystem / Shell / Git | [`A2`](智能体-agent/02-工具与环境-tool-use-environments.md) | `tools.py` | rooted path / command tests | sandbox integration |
| **Permission / Approval** | [`A7`](智能体-agent/07-评测安全与开放问题-agent-evaluation-safety.md), [`C03`](Codex源码解剖-codex-anatomy/03-工具执行审批与沙箱-tools-approval-sandbox.md) | `security.py` | **deny/reject prevents dispatch** | credentials + OS sandbox |
| OS / Container Sandbox | [`C03`](Codex源码解剖-codex-anatomy/03-工具执行审批与沙箱-tools-approval-sandbox.md) | — | — | process/fs/network/secret isolation |
| MCP | [`A9`](智能体-agent/09-Agent协议与互操作-agent-protocols.md) | `mcp.py` subset | discover/list/call tests | 2026 semantics / transport/auth/tasks |
| A2A | [`A9`](智能体-agent/09-Agent协议与互操作-agent-protocols.md) | — | — | AgentCard / Message / Task / Artifact |
| Browser | [`A10`](智能体-agent/10-浏览器与计算机使用-browser-computer-use.md) | `general_tools.py` HTTP only | no GUI claim | JS + DOM/A11y |
| Computer Use | [`A10`](智能体-agent/10-浏览器与计算机使用-browser-computer-use.md) | — | — | screenshot/grounding/actions/verifier |

---

# 7　Multi-Agent

| 主题 | 理论 / Source | 源码 | 当前证据 | 真正研究问题 |
|---|---|---|---|---|
| Coordinator | [`A6`](智能体-agent/06-多智能体与协调-multi-agent.md) | `multi_agent.py` | deterministic tests | scheduler policy |
| Worktree | [`C06`](Codex源码解剖-codex-anatomy/06-多智能体与工作树-multi-agent-worktree.md) | `worktree.py` | primitive tests | concurrent repo safety |
| AgentGraph / Mailbox | [`C06`](Codex源码解剖-codex-anatomy/06-多智能体与工作树-multi-agent-worktree.md) | — | — | parent/child lifecycle |
| Reviewer / Merge | Coding track | — | — | verifier-based merge |
| Scaling Agents | [`评测实验室`](评测-eval/README.md) | metrics exists | — | **1/2/4/8 Agent success-time-cost comparison** |

Agent 数量本身不是能力证据。

---

# 8　Evaluation

独立事实层：[`评测-eval/README.md`](评测-eval/README.md)。

| Capability | 源码 | Test | 下一 Benchmark |
|---|---|---|---|
| **Trajectory Metrics** | `evaluation.py` | summary + aggregation tests | all Agent evals |
| Coding Eval | — | — | SWE-bench adapter |
| Browser Eval | — | — | WebArena-style adapter |
| Computer Eval | — | — | OSWorld-style adapter |
| Recovery Eval | `durable.py` | restart replay | crash/timeout/retry tasks |
| Cost / Latency | fields exist | aggregate test | real instrumentation |

统一证据：`trajectory + environment state + artifact + grader + cost`。

---

# 9　Frontier Observatory

快速变化事实进入 [`观测站-observatory/frontier-snapshot.json`](观测站-observatory/frontier-snapshot.json)：

```text
Official Source
→ machine-readable fact
→ evidence boundary
→ freshness age
→ drift report
```

审计：[`工具-scripts/前沿漂移审计-frontier_drift_audit.py`](工具-scripts/前沿漂移审计-frontier_drift_audit.py)。

---

# 10　Industrial Reference Implementations

Codex 被定义为：**Industrial Reference #1 — Coding Agent Runtime**，不是 Agent 世界唯一答案。

后续采用同一种 Source Anatomy 方法：

```text
vLLM / SGLang   → Inference Runtime
Qwen / DeepSeek → Public Model Architecture
MCP / A2A       → Protocol Runtime
```

---

# 11　固定学习闭环

```text
Read Theory
→ Read Primary / Official Source
→ Rebuild Reference Code
→ Unit / Numerical / Protocol Parity
→ Capability / Systems Evaluation
→ Failure Cases / Ablation
```

如果后半段没有发生，就还没有从“读懂”走到“研究级掌握”。
