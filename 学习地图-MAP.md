# Theory ↔ Code ↔ Paper 学习地图

> **用途**：把“教材章节、原始论文、源码实现、自动测试”连接成同一张图。  
> 学习一个主题时，不应该只停在其中任意一列；真正掌握至少要能从理论跳到源码，再回到原始工作和实验。

---

# 1　模型核心

| 主题 | 理论入口 | 原始资料 | 我们的源码 | 验证 |
|---|---|---|---|---|
| Tokenization / BPE | [`教材 13`](教材-book/13-词元化与数据工程-tokenization-data.md) | Sennrich et al.; SentencePiece，见 [`Primary Sources`](参考文献-references/00-原始资料总索引-primary-sources.md) | `代码-code/从零构建Astra与Codex级系统/src/astra_codex/tokenizer.py` | `tests/test_tokenizer.py` |
| Transformer | [`教材 01`](教材-book/01-Transformer基础-foundations-transformer.md) | Vaswani et al. 2017 | `src/astra_codex/model.py` | `tests/test_model_and_cache.py` |
| RMSNorm | [`教材 04`](教材-book/04-现代LLM架构-modern-architecture.md) | Zhang & Sennrich 2019 | `model.py::RMSNorm` | real checkpoint parity |
| RoPE | [`教材 04`](教材-book/04-现代LLM架构-modern-architecture.md) | RoFormer 2021 | `model.py::RotaryEmbedding` | Llama/HF half-split + interleaved tests |
| GQA | [`教材 04`](教材-book/04-现代LLM架构-modern-architecture.md) | Ainslie et al. 2023 | `model.py::GroupedQueryAttention` | cache parity + SmolLM2 parity |
| SwiGLU | [`教材 04`](教材-book/04-现代LLM架构-modern-architecture.md) | Shazeer 2020 | `model.py::SwiGLU` | SmolLM2 parity |
| Decoder-only LM | [`教材 01`](教材-book/01-Transformer基础-foundations-transformer.md) | GPT-1/2/3 | `model.py::DecoderOnlyTransformer` | full forward tests |
| KV Cache | [`教材 10`](教材-book/10-推理服务系统-inference-systems.md), [`源码导读 18A`](教材-book/18A-模型运行时源码导读-runtime-walkthrough.md) | Transformer inference implementations / systems papers | `cache.py`, `engine.py` | cached decode vs full recomputation parity |
| Sampling | [`教材 10`](教材-book/10-推理服务系统-inference-systems.md) | decoding literature | `sampling.py` | unit tests / generation examples |
| Safetensors / Weight Mapping | [`教材 18`](教材-book/18-从零构建Astra与Codex级系统-capstone.md) | public checkpoint format / model config | `weights.py`, `public_checkpoint.py` | shape audit + strict load |
| **真实 Llama-family Runtime** | [`源码导读 18A`](教材-book/18A-模型运行时源码导读-runtime-walkthrough.md) | `HuggingFaceTB/SmolLM2-135M` public config/weights | `public_checkpoint.py` | **CI logits: max_abs=0, mean_abs=0, argmax=1.0** |

源码路径表中的 `src/...` 默认指：

`代码-code/从零构建Astra与Codex级系统/src/astra_codex/`

---

# 2　后训练与推理

| 主题 | 理论入口 | 原始资料 | 当前代码状态 | 下一复现 |
|---|---|---|---|---|
| Instruction Tuning | [`教材 03`](教材-book/03-对齐与ChatGPT-alignment-chatgpt.md) | FLAN / T0 / InstructGPT | 理论为主 | tiny SFT pipeline |
| RLHF / PPO | [`教材 03`](教材-book/03-对齐与ChatGPT-alignment-chatgpt.md), [`附录 C`](附录-appendices/C-后训练数学-post-training-math.md) | Christiano / PPO / InstructGPT | 数学推导 | toy reward model + PPO |
| DPO | [`教材 03`](教材-book/03-对齐与ChatGPT-alignment-chatgpt.md) | Rafailov et al. | 数学推导 | tiny pairwise preference lab |
| GRPO / RLVR | [`教材 08`](教材-book/08-推理模型时代-reasoning-era.md) | DeepSeekMath / R1 | verifier primitive 已有 | toy verifiable RL trajectory pipeline |
| Test-Time Compute | [`教材 08`](教材-book/08-推理模型时代-reasoning-era.md) | CoT / self-consistency / Snell et al. | Agent verifier 可复用 | search-budget controlled experiments |

---

# 3　系统工程

| 主题 | 理论入口 | 原始资料 | 当前代码 | 下一实现 |
|---|---|---|---|---|
| FlashAttention | [`教材 16`](教材-book/16-硬件内核与基础设施-hardware-kernels.md) | FlashAttention 1/2/3 | explicit attention reference | PyTorch SDPA parity → Flash backend |
| Distributed Training | [`教材 09`](教材-book/09-训练系统-training-systems.md) | ZeRO / Megatron / FSDP | 尚未从零实现 | communication simulator |
| Paged KV | [`教材 10`](教材-book/10-推理服务系统-inference-systems.md) | vLLM / PagedAttention | contiguous KV | page allocator + parity |
| Continuous Batching | [`教材 10`](教材-book/10-推理服务系统-inference-systems.md) | serving systems | single-request engine | request scheduler |
| Speculative Decoding | [`教材 10`](教材-book/10-推理服务系统-inference-systems.md) | speculative decoding originals | 未实现 | draft/target acceptance lab |
| Hardware Roofline | [`教材 16`](教材-book/16-硬件内核与基础设施-hardware-kernels.md) | hardware/system originals | 未实现 | bytes/FLOPs profiler |

---

# 4　Post-Transformer 与 Diffusion

| 主题 | 理论入口 | 原始资料 | 当前代码 | 下一实现 |
|---|---|---|---|---|
| S4 | [`教材 19`](教材-book/19-Post-Transformer架构-post-transformer.md) | [`资料 02`](参考文献-references/02-PostTransformer与扩散语言模型原始资料.md) | 未实现 | recurrent vs matrix form parity |
| Mamba | [`教材 19`](教材-book/19-Post-Transformer架构-post-transformer.md) | Mamba original + official code | 未实现 | selective scan reference |
| Mamba-2 / SSD | [`教材 19`](教材-book/19-Post-Transformer架构-post-transformer.md) | Mamba-2 paper | 未实现 | recurrent / matrix / chunked equivalence |
| RWKV | [`教材 19`](教材-book/19-Post-Transformer架构-post-transformer.md) | RWKV paper + official repo | 未实现 | RWKV-like recurrence |
| xLSTM | [`教材 19`](教材-book/19-Post-Transformer架构-post-transformer.md) | xLSTM paper + code | 未实现 | sLSTM / mLSTM toy blocks |
| Titans / Neural Memory | [`教材 19`](教材-book/19-Post-Transformer架构-post-transformer.md) | Titans | 未实现 | tiny test-time memory lab |
| Diffusion LM | [`教材 20`](教材-book/20-扩散语言模型-diffusion-language-models.md) | discrete diffusion / LLaDA | 未实现 | corruption → denoise → remask |

---

# 5　智能体核心

| 主题 | 理论入口 | 原始资料 | 我们的源码 | 验证 |
|---|---|---|---|---|
| Agent Loop | [`A1`](智能体-agent/01-智能体基础-agent-foundations.md), [`18B`](教材-book/18B-智能体运行时源码导读-agent-walkthrough.md) | ReAct | `agent.py` | scripted backend tests |
| Structured Actions | [`A2`](智能体-agent/02-工具与环境-tool-use-environments.md) | Toolformer / tool-use originals | `structured.py` | schema/tool tests |
| Filesystem / Shell / Git | [`A2`](智能体-agent/02-工具与环境-tool-use-environments.md) | environment/tool interfaces | `tools.py` | rooted path + execution tests |
| Safe Editing | [`A5`](智能体-agent/05-编程智能体-coding-agents.md) | SWE-agent / coding interfaces | `editing.py` | ambiguity-safe edit tests |
| Repo Map | [`A5`](智能体-agent/05-编程智能体-coding-agents.md) | SWE-agent / code-agent work | `repo_map.py` | AST/Markdown tests |
| Planning DAG | [`A3`](智能体-agent/03-规划反思与验证-planning-reflection-verification.md) | ToT / planning literature | `planning.py` | DAG/cycle/failure propagation tests |
| Verifier | [`A3`](智能体-agent/03-规划反思与验证-planning-reflection-verification.md) | verifier/reasoning/coding work | `verification.py` | file/command/composite tests |
| Persistent Memory | [`A4`](智能体-agent/04-记忆与上下文-memory-context.md) | MemGPT / Generative Agents | `memory.py` | SQLite persistence/search tests |
| Coding Agent | [`A5`](智能体-agent/05-编程智能体-coding-agents.md) | SWE-bench / SWE-agent | `coding.py` | repo-loop tests |
| Worktree | [`A6`](智能体-agent/06-多智能体与协调-multi-agent.md) | Git primitive / multi-agent engineering | `worktree.py` | primitive tests |
| Coordinator | [`A6`](智能体-agent/06-多智能体与协调-multi-agent.md) | CAMEL / AutoGen + our explicit task-DAG design | `multi_agent.py` | deterministic coordinator tests |

---

# 6　Agent Protocols 与 Computer Use

| 主题 | 理论入口 | 原始资料 | 当前源码 | 下一实现 |
|---|---|---|---|---|
| JSON-RPC | [`A9`](智能体-agent/09-Agent协议与互操作-agent-protocols.md) | JSON-RPC / MCP wire semantics | `mcp.py` | request/response tests |
| MCP | [`A9`](智能体-agent/09-Agent协议与互操作-agent-protocols.md) | [`Agent Sources`](参考文献-references/01-智能体原始资料-agent-sources.md) | `mcp.py`: discover/list/call/in-process | stdio → HTTP → auth |
| A2A | [`A9`](智能体-agent/09-Agent协议与互操作-agent-protocols.md) | A2A official spec | 未实现 | AgentCard / Message / Task / Artifact |
| Browser | [`A10`](智能体-agent/10-浏览器与计算机使用-browser-computer-use.md) | WebArena | `general_tools.py`: HTTP text fetch only | JS runtime + DOM/A11y tree |
| Computer Use | [`A10`](智能体-agent/10-浏览器与计算机使用-browser-computer-use.md) | OSWorld | 未实现 | screenshot + grounding + actions + verifier |

---

# 7　安全、评测与可解释性

| 主题 | 理论入口 | 代码/实验状态 |
|---|---|---|
| LLM Evaluation | [`教材 11`](教材-book/11-评测与安全-evaluation-safety.md) | 需扩 benchmark harness |
| Agent Evaluation | [`A7`](智能体-agent/07-评测安全与开放问题-agent-evaluation-safety.md) | SWE-bench / WebArena / OSWorld harness 待实现 |
| Interpretability | [`教材 15`](教材-book/15-可解释性与机制研究-interpretability.md) | activation patch / SAE lab 待实现 |
| Agent Safety | [`A7`](智能体-agent/07-评测安全与开放问题-agent-evaluation-safety.md), [`A10`](智能体-agent/10-浏览器与计算机使用-browser-computer-use.md) | sandbox / permission / approval gate 待实现 |

---

# 8　怎么使用这张图

对于任意知识点，推荐固定走四步：

```text
1. Read Theory
      ↓
2. Read Original Source
      ↓
3. Read / Rebuild Code
      ↓
4. Run Test / Reproduction
```

如果第四步不存在，质量看板通常不会把它标到 L2。

总成熟度入口：[`质量看板-QUALITY.md`](质量看板-QUALITY.md)