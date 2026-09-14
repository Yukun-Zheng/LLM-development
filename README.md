# 大语言模型发展史、系统工程与智能体：从原始论文到从零实现

> **版本**：v2 · 2026-09-14  
> **定位**：中文主导、英文术语对照、Source-First、Mechanism-First、Reproducibility-First 的大语言模型与智能体教材工程。  
> **最终目标**：不是“看懂几十篇论文”，而是从空目录开始，把现代 LLM runtime、推理系统、工具运行时、Coding Agent、Computer Use、多智能体与评测系统逐层亲手写出来。训练 frontier 权重本身不作为要求。

---

# 一、三条主线

这不是一条越来越长的时间线，而是三条相互约束的技术主线：

```text
                         LLM Development
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
   模型科学主线             系统工程主线            智能体主线
  Model Science            LLM Systems          Agentic Systems
        │                      │                      │
数学 / 概率 / 表示       GPU / Kernel / 通信     Tool / Environment
Tokenizer / Data         Distributed Training     Planning / Memory
Transformer / SSM       KV Cache / Serving       Verification
Scaling / Post-train    Quantization / Runtime   Coding / Computer Use
Reasoning / Diffusion   Scheduling / Compiler    Multi-Agent / Agentic RL
        │                      │                      │
        └──────────────────────┼──────────────────────┘
                               ▼
                  从零 Frontier 系统总工程
                 Astra-class / Codex-class
```

## 1. 模型科学主线

研究“模型本身为什么能工作、能力从哪里来”：

- 数学、概率、信息论、优化；
- Tokenization、数据工程、预训练目标；
- Transformer、现代注意力、MoE；
- Scaling、Instruction Tuning、RLHF/DPO/RLVR；
- Reasoning、Multimodality；
- Post-Transformer、SSM、RWKV、neural memory；
- Diffusion Language Models。

入口：[`教材-book/README.md`](教材-book/README.md)

## 2. 系统工程主线

研究“理论上的模型怎样在真实硬件上训练与服务”：

- 数值精度与训练稳定性；
- GPU / HBM / SRAM / Tensor Core；
- DP / TP / PP / EP、ZeRO、FSDP、NCCL；
- FlashAttention；
- KV Cache、Paged KV、Prefix Cache；
- Continuous Batching、Speculative Decoding；
- Runtime、Serving、Scheduler、Compiler。

## 3. 智能体主线

研究“模型怎样进入环境、持续行动并完成真实任务”：

- Tool Use 与 Environment；
- Planning、Reflection、Verification、Recovery；
- Memory 与 Context Management；
- Coding Agent；
- Browser / Computer Use；
- Multi-Agent；
- MCP / A2A 等协议与互操作；
- Agent Evaluation / Safety；
- Agentic RL。

入口：[`智能体-agent/README.md`](智能体-agent/README.md)

---

# 二、仓库结构

| 路径 | 作用 |
|---|---|
| [`教材-book/`](教材-book/) | **模型科学与系统工程教材正文** |
| [`智能体-agent/`](智能体-agent/) | **智能体独立主线** |
| [`代码-code/`](代码-code/) | **从零实现代码与终极工程** |
| [`附录-appendices/`](附录-appendices/) | 数学、MiniGPT、后训练数学等附录 |
| [`参考文献-references/`](参考文献-references/) | 原论文、官方技术报告、代码、Model/System Card 的证据地图 |
| [`图表-figures/`](图表-figures/) | 作者重绘的数据流、机制图、时间线 |
| [`工具-scripts/`](工具-scripts/) | 数学排版、来源审计、教材维护工具 |
| [`质量看板-QUALITY.md`](质量看板-QUALITY.md) | **全书成熟度与缺口审计** |
| [`.github/`](.github/) | 自动测试与格式检查 |

---

# 三、教材正文总览

当前主干已经覆盖：

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
| 19 | **Post-Transformer：SSM、RWKV、Neural Memory 与混合架构** |
| 20 | **Diffusion Language Models：非自回归语言建模** |

详细目录：[`教材-book/README.md`](教材-book/README.md)

---

# 四、智能体课程总览

| 模块 | 主题 |
|---|---|
| A1 | 智能体第一性原理 |
| A2 | Tool Use 与 Environment |
| A3 | Planning、Reflection、Verification、Recovery |
| A4 | Memory 与 Context Management |
| A5 | Coding Agent |
| A6 | Multi-Agent 与协调 |
| A7 | Agent Evaluation / Safety |
| A8 | Agentic RL |
| A9 | **MCP、A2A 与 Agent Protocols** |
| A10 | **Browser / Computer Use 与 GUI Agent** |

入口：[`智能体-agent/README.md`](智能体-agent/README.md)

---

# 五、终极代码工程

核心工程：

[`代码-code/从零构建Astra与Codex级系统/`](代码-code/从零构建Astra与Codex级系统/)

目前已经从零实现并自动测试：

```text
UTF-8 Byte Tokenizer / BPE
        ↓
Embedding + RMSNorm + RoPE
        ↓
Causal GQA + SwiGLU
        ↓
Decoder-only Transformer
        ↓
KV Cache / Prefill / Decode
        ↓
Sampling / Structured Tool Call
        ↓
Filesystem / Shell / Git / Repo Map / Edit
        ↓
Agent Loop / Memory / Planning / Verification
        ↓
Coding Agent / Worktree / Coordinator
```

下一批关键里程碑：

1. **公开真实 checkpoint logits parity**；
2. minimal MCP client/server；
3. Paged KV / Continuous Batching / Speculative Decoding；
4. tree-sitter / LSP / semantic patch；
5. sandbox；
6. Browser / Computer Use；
7. parallel worktree workers + reviewer/merge；
8. SWE-bench / long-horizon agent eval。

---

# 六、Source-First：每个重要结论回到原始资料

最高写作规范：[`教材-book/00-教材方法论与证据标准-source-first.md`](教材-book/00-教材方法论与证据标准-source-first.md)

统一原始资料地图：

- [`参考文献-references/00-原始资料总索引-primary-sources.md`](参考文献-references/00-原始资料总索引-primary-sources.md)
- [`参考文献-references/01-智能体原始资料-agent-sources.md`](参考文献-references/01-智能体原始资料-agent-sources.md)

正文区分：

```text
原始工作声称什么
        ↓
原始公式 / 代码 / 实验是什么
        ↓
后续复现是否支持
        ↓
哪些结论被修正
        ↓
今天较稳健的理解是什么
```

对闭源模型则明确写“公开证据支持到哪里”，不把社区猜测伪装成架构事实。

---

# 七、学习标准

任何核心概念最终都要求能经过五层：

```text
Understand → Calculate → Implement → Reproduce → Research
```

例如 KV Cache 不是“知道能加速”就算学完，而是应该做到：

1. 推导缓存的 tensor shape 和显存占用；
2. 写 full recomputation；
3. 写 incremental decode；
4. 做 logits parity；
5. 再研究 paged KV、prefix cache 与 production serving。

智能体同理：

> “我完成了任务”不是证据；真实 environment observation 与 verifier 才是证据。

---

# 八、最终毕业标准

给一个空目录，不依赖 LangChain 等 Agent 高层框架，也不把模型核心藏进 `AutoModelForCausalLM`，能够逐层构建：

```text
Tokenizer
→ Model Runtime
→ Weight Loader
→ Inference Engine
→ Structured Generation
→ Tool Runtime
→ Context / Memory
→ Planning / Verification
→ Coding Agent
→ Browser / Computer Agent
→ Multi-Agent Runtime
→ Evaluation
```

并且能解释每一层的数学、数据流、源码、原始论文、实验与系统代价。

这才是本项目的终点。