# Theory ↔ Code ↔ Paper/Source ↔ Test 学习地图

> **用途**：把教材章节、原始论文/官方源码、我们的 clean-room 实现与自动验证连接成同一张图。  
> 真正掌握一个主题，不应停在任意一列；至少要能从原始资料追到机制，再从机制追到实现和证据。

源码表中的 `src/...` 默认指：`代码-code/从零构建Astra与Codex级系统/src/astra_codex/`。

---

# 1　模型核心

| 主题 | 理论入口 | 原始资料 | 我们的源码 | 验证 |
|---|---|---|---|---|
| Tokenization / BPE | [`教材 13`](教材-book/13-词元化与数据工程-tokenization-data.md) | Sennrich / SentencePiece，见 [`Primary Sources`](参考文献-references/00-原始资料总索引-primary-sources.md) | `tokenizer.py` | `tests/test_tokenizer.py` |
| Transformer | [`教材 01`](教材-book/01-Transformer基础-foundations-transformer.md) | Vaswani et al. 2017 + [`Source Card`](原始论文-paper-notes/2017-Attention-Is-All-You-Need.md) | `model.py` | `tests/test_model_and_cache.py` |
| RMSNorm | [`教材 04`](教材-book/04-现代LLM架构-modern-architecture.md) | Zhang & Sennrich 2019 | `model.py::RMSNorm` | real checkpoint parity |
| RoPE | [`教材 04`](教材-book/04-现代LLM架构-modern-architecture.md) | RoFormer 2021 | `model.py::RotaryEmbedding` | half-split/interleaved tests |
| GQA | [`教材 04`](教材-book/04-现代LLM架构-modern-architecture.md) | Ainslie et al. 2023 | `model.py::GroupedQueryAttention` | cache + SmolLM2 parity |
| SwiGLU | [`教材 04`](教材-book/04-现代LLM架构-modern-architecture.md) | Shazeer 2020 | `model.py::SwiGLU` | SmolLM2 parity |
| Decoder-only LM | [`教材 01`](教材-book/01-Transformer基础-foundations-transformer.md) | GPT-1/2/3 | `model.py::DecoderOnlyTransformer` | full-forward tests |
| KV Cache | [`教材 10`](教材-book/10-推理服务系统-inference-systems.md), [`18A`](教材-book/18A-模型运行时源码导读-runtime-walkthrough.md) | inference systems sources | `cache.py`, `engine.py` | cached vs full parity |
| Sampling | [`教材 10`](教材-book/10-推理服务系统-inference-systems.md) | decoding literature | `sampling.py` | unit tests |
| Public checkpoint mapping | [`教材 18`](教材-book/18-从零构建Astra与Codex级系统-capstone.md) | SmolLM2 public config/weights | `weights.py`, `public_checkpoint.py` | strict load + shape audit |
| **真实 Llama-family runtime** | [`18A`](教材-book/18A-模型运行时源码导读-runtime-walkthrough.md) | `HuggingFaceTB/SmolLM2-135M` | `public_checkpoint.py` | **max_abs=0, mean_abs=0, argmax=1.0** |

---

# 2　后训练与推理

| 主题 | 理论入口 | 原始资料 | 当前代码状态 | 下一复现 |
|---|---|---|---|---|
| Instruction Tuning | [`教材 03`](教材-book/03-对齐与ChatGPT-alignment-chatgpt.md) | FLAN / T0 / InstructGPT | 理论为主 | tiny SFT |
| RLHF / PPO | [`教材 03`](教材-book/03-对齐与ChatGPT-alignment-chatgpt.md), [`附录 C`](附录-appendices/C-后训练数学-post-training-math.md) | Christiano / PPO / InstructGPT | 数学推导 | toy reward model + PPO |
| DPO | [`教材 03`](教材-book/03-对齐与ChatGPT-alignment-chatgpt.md) | Rafailov et al. | 数学推导 | pairwise preference lab |
| GRPO / RLVR | [`教材 08`](教材-book/08-推理模型时代-reasoning-era.md) | DeepSeekMath / R1 | verifier primitive | verifiable RL trajectory |
| Test-Time Compute | [`教材 08`](教材-book/08-推理模型时代-reasoning-era.md) | CoT / self-consistency / test-time compute | verifier 可复用 | search-budget experiments |

---

# 3　系统工程

| 主题 | 理论入口 | 原始资料 | 当前代码 | 下一实现 |
|---|---|---|---|---|
| FlashAttention | [`教材 16`](教材-book/16-硬件内核与基础设施-hardware-kernels.md) | FlashAttention 1/2/3 | explicit attention reference | SDPA parity → Flash backend |
| Distributed Training | [`教材 09`](教材-book/09-训练系统-training-systems.md) | ZeRO / Megatron / FSDP | 未从零实现 | communication simulator |
| Paged KV | [`教材 10`](教材-book/10-推理服务系统-inference-systems.md) | vLLM / PagedAttention | contiguous KV | page allocator + parity |
| Continuous Batching | [`教材 10`](教材-book/10-推理服务系统-inference-systems.md) | serving systems | single-request engine | request scheduler |
| Speculative Decoding | [`教材 10`](教材-book/10-推理服务系统-inference-systems.md) | original papers | 未实现 | draft/target acceptance lab |
| Hardware Roofline | [`教材 16`](教材-book/16-硬件内核与基础设施-hardware-kernels.md) | hardware/system originals | 未实现 | bytes/FLOPs profiler |

---

# 4　Post-Transformer 与 Diffusion

| 主题 | 理论入口 | 原始资料 | 当前代码 | 下一实现 |
|---|---|---|---|---|
| S4 | [`教材 19`](教材-book/19-Post-Transformer架构-post-transformer.md) | [`资料 02`](参考文献-references/02-PostTransformer与扩散语言模型原始资料.md) | 未实现 | recurrent/matrix parity |
| Mamba | [`教材 19`](教材-book/19-Post-Transformer架构-post-transformer.md) | original + official code | 未实现 | selective scan |
| Mamba-2 / SSD | [`教材 19`](教材-book/19-Post-Transformer架构-post-transformer.md) | Mamba-2 | 未实现 | recurrent/matrix/chunked equivalence |
| RWKV | [`教材 19`](教材-book/19-Post-Transformer架构-post-transformer.md) | paper + official repo | 未实现 | recurrence lab |
| xLSTM | [`教材 19`](教材-book/19-Post-Transformer架构-post-transformer.md) | paper + code | 未实现 | sLSTM/mLSTM toy blocks |
| Titans / Neural Memory | [`教材 19`](教材-book/19-Post-Transformer架构-post-transformer.md) | Titans | 未实现 | test-time memory lab |
| Diffusion LM | [`教材 20`](教材-book/20-扩散语言模型-diffusion-language-models.md) | discrete diffusion / LLaDA | 未实现 | corruption → denoise → remask |

---

# 5　Agent 通识核心

| 主题 | 理论入口 | 原始资料 | 我们的源码 | 验证 |
|---|---|---|---|---|
| Agent Loop | [`A1`](智能体-agent/01-智能体基础-agent-foundations.md), [`18B`](教材-book/18B-智能体运行时源码导读-agent-walkthrough.md) | ReAct | `agent.py` | scripted backend tests |
| Structured Actions | [`A2`](智能体-agent/02-工具与环境-tool-use-environments.md) | Toolformer / tool-use originals | `structured.py` | schema/tool tests |
| Filesystem / Shell / Git | [`A2`](智能体-agent/02-工具与环境-tool-use-environments.md) | environment interfaces | `tools.py` | rooted path + exec tests |
| Safe Editing | [`A5`](智能体-agent/05-编程智能体-coding-agents.md) | SWE-agent | `editing.py` | ambiguity-safe tests |
| Repo Map | [`A5`](智能体-agent/05-编程智能体-coding-agents.md) | SWE-agent / coding work | `repo_map.py` | AST/Markdown tests |
| Planning DAG | [`A3`](智能体-agent/03-规划反思与验证-planning-reflection-verification.md) | planning literature | `planning.py` | DAG/cycle/failure tests |
| Verifier | [`A3`](智能体-agent/03-规划反思与验证-planning-reflection-verification.md) | verifier/coding work | `verification.py` | file/command/composite tests |
| Persistent Memory | [`A4`](智能体-agent/04-记忆与上下文-memory-context.md) | MemGPT / Generative Agents | `memory.py` | SQLite tests |
| Coding Agent | [`A5`](智能体-agent/05-编程智能体-coding-agents.md) | SWE-bench / SWE-agent | `coding.py` | repo-loop tests |
| Worktree | [`A6`](智能体-agent/06-多智能体与协调-multi-agent.md) | Git/worktree engineering | `worktree.py` | primitive tests |
| Coordinator | [`A6`](智能体-agent/06-多智能体与协调-multi-agent.md) | CAMEL / AutoGen + explicit DAG | `multi_agent.py` | coordinator tests |

---

# 6　Agent Protocols 与 Computer Use

| 主题 | 理论入口 | 原始资料 | 当前源码 | 下一实现 |
|---|---|---|---|---|
| JSON-RPC / MCP | [`A9`](智能体-agent/09-Agent协议与互操作-agent-protocols.md) | [`Agent Sources`](参考文献-references/01-智能体原始资料-agent-sources.md) | `mcp.py`: discover/list/call/in-process | stdio → HTTP → auth |
| A2A | [`A9`](智能体-agent/09-Agent协议与互操作-agent-protocols.md) | A2A spec | 未实现 | AgentCard/Message/Task/Artifact |
| Browser | [`A10`](智能体-agent/10-浏览器与计算机使用-browser-computer-use.md) | WebArena | `general_tools.py`: HTTP text only | JS + DOM/A11y |
| Computer Use | [`A10`](智能体-agent/10-浏览器与计算机使用-browser-computer-use.md) | OSWorld | 未实现 | screenshot + grounding + actions |

---

# 7　OpenAI Codex：官方源码 ↔ Clean-room 重建

官方源码索引：[`参考文献 03`](参考文献-references/03-OpenAI-Codex官方源码索引-codex-source-map.md)  
课程入口：[`Codex 源码解剖`](Codex源码解剖-codex-anatomy/README.md)  
Source Card：[`OpenAI Codex Harness`](原始论文-paper-notes/2025-2026-OpenAI-Codex-Harness.md)

| 公开 Codex 机制 | 源码解剖 | 官方一手资料 | 我们的实现 | 验证/状态 |
|---|---|---|---|---|
| Task / Turn loop | [`C01`](Codex源码解剖-codex-anatomy/01-核心AgentLoop-agent-loop.md) | `core/src/session/turn.rs` | `codex_harness.py` | tool follow-up ✓ |
| Event model | [`C02`](Codex源码解剖-codex-anatomy/02-会话线程与事件协议-session-thread-events.md) | `docs/protocol_v1.md` | `HarnessEvent` | ordering tests ✓; async queue todo |
| Approval | [`C03`](Codex源码解剖-codex-anatomy/03-工具执行审批与沙箱-tools-approval-sandbox.md) | `tools/approvals.rs` | `ApprovalDecision` / `TurnSettings` | allow/deny tests ✓ |
| Sandbox | [`C03`](Codex源码解剖-codex-anatomy/03-工具执行审批与沙箱-tools-approval-sandbox.md) | `tools/sandboxing.rs`, sandbox crates | metadata enum only | **OS enforcement todo** |
| AGENTS.md / scoped instructions | [`C04`](Codex源码解剖-codex-anatomy/04-指令上下文压缩与记忆-context-memory.md) | base instructions | — | resolver todo |
| Context compaction | [`C04`](Codex源码解剖-codex-anatomy/04-指令上下文压缩与记忆-context-memory.md) | `compact*` | event memory only | compact/resume todo |
| MCP integration | [`C05`](Codex源码解剖-codex-anatomy/05-MCP与AppServer-mcp-app-server.md) | `codex-mcp`, `rmcp-client` | `mcp.py` subset | in-process ✓; transport/auth todo |
| App Server | [`C02`](Codex源码解剖-codex-anatomy/02-会话线程与事件协议-session-thread-events.md), [`C05`](Codex源码解剖-codex-anatomy/05-MCP与AppServer-mcp-app-server.md) | app-server protocol | — | todo |
| Multi-Agent semantics | [`C06`](Codex源码解剖-codex-anatomy/06-多智能体与工作树-multi-agent-worktree.md) | `multi_agents.rs` | `multi_agent.py` primitive | persistent AgentGraph todo |
| Worktree | [`C06`](Codex源码解剖-codex-anatomy/06-多智能体与工作树-multi-agent-worktree.md) | `worktree/` | `worktree.py` | primitive ✓; parallel merge todo |
| Clean-room parity program | [`C07`](Codex源码解剖-codex-anatomy/07-CleanRoom重建与Parity-clean-room-parity.md) | pinned `openai/codex` source | current capstone | **Fast CI 30 passed** |

这个表明确区分 `✓ / partial / todo`。存在同名 class 绝不等于已经复刻生产能力。

---

# 8　安全、评测与可解释性

| 主题 | 理论入口 | 代码/实验状态 |
|---|---|---|
| LLM Evaluation | [`教材 11`](教材-book/11-评测与安全-evaluation-safety.md) | benchmark harness 待扩 |
| Agent Evaluation | [`A7`](智能体-agent/07-评测安全与开放问题-agent-evaluation-safety.md) | SWE-bench/WebArena/OSWorld harness 待实现 |
| Interpretability | [`教材 15`](教材-book/15-可解释性与机制研究-interpretability.md) | activation patch / SAE lab 待实现 |
| Agent Safety | [`A7`](智能体-agent/07-评测安全与开放问题-agent-evaluation-safety.md), [`C03`](Codex源码解剖-codex-anatomy/03-工具执行审批与沙箱-tools-approval-sandbox.md) | approval v1 ✓; enforced sandbox todo |

---

# 9　怎么使用这张图

```text
1. Read Theory / Source Anatomy
      ↓
2. Read Original Paper / Official Source
      ↓
3. Rebuild Code
      ↓
4. Run Test / Reproduction / Parity
      ↓
5. Record Gap / Counterexample
```

如果第四步不存在，质量看板通常不会把它标到 L2。

总成熟度入口：[`质量看板-QUALITY.md`](质量看板-QUALITY.md)