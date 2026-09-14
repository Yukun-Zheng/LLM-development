# 教材正文 / Book

本目录是《大语言模型发展史与技术原理》的模型科学与系统工程主线。正文坚持：**中文解释 + 英文术语对照 + 原始资料优先 + 理论最终落到代码与实验**。

> **最高写作规范**：[`00-教材方法论与证据标准-source-first.md`](00-教材方法论与证据标准-source-first.md)  
> **智能体独立主线**：[`../智能体-agent/README.md`](../智能体-agent/README.md)  
> **最终工程终点**：[`18-从零构建Astra与Codex级系统-capstone.md`](18-从零构建Astra与Codex级系统-capstone.md)  
> **全书扩展蓝图**：[`99-全书扩展蓝图-v2.md`](99-全书扩展蓝图-v2.md)

---

# 中文目录 / Contents

| 篇章 | 章节标题 | 关键主题 |
|---|---|---|
| 方法论 | [`教材方法论与证据标准`](00-教材方法论与证据标准-source-first.md) | Source-First、Mechanism-First、Reproducibility |
| 导读 | [`怎样用这本书真正学会大语言模型`](00-导读-preface.md) | 学习方法、路线、实验标准 |
| 第一篇 | [`从 Seq2Seq 到 Transformer`](01-Transformer基础-foundations-transformer.md) | RNN、Attention、Transformer、GPT、BERT |
| 第二篇 | [`规模定律、GPT-3、ICL 与 Chinchilla`](02-规模定律与GPT3-scaling-gpt3.md) | Scaling、Few-shot、Compute-Optimal |
| 第三篇 | [`指令微调、RLHF、ChatGPT 与 DPO`](03-对齐与ChatGPT-alignment-chatgpt.md) | Alignment、Preference Learning |
| 第四篇 | [`现代 LLM 架构组件`](04-现代LLM架构-modern-architecture.md) | RoPE、RMSNorm、SwiGLU、GQA、KV Cache |
| 第五篇 | [`开放权重模型与高效微调`](05-开源与高效微调-open-source-efficient-ft.md) | LLaMA、LoRA、QLoRA、Quantization |
| 第六篇 | [`RAG、多模态、工具调用与 Agent 的历史交汇`](06-RAG多模态与智能体-rag-multimodal-agent.md) | Retrieval、Multimodal、Tool Use |
| 第七篇 | [`MoE 与中国大模型路线`](07-MoE与中国大模型-moe-china.md) | GLM、Qwen、DeepSeek、Kimi、MoE |
| 第八篇 | [`推理模型时代`](08-推理模型时代-reasoning-era.md) | CoT、Test-Time Compute、GRPO、RLVR |
| 第九篇 | [`大模型训练系统`](09-训练系统-training-systems.md) | Data、ZeRO、FSDP、Megatron、Parallelism |
| 第十篇 | [`大模型推理与服务系统`](10-推理服务系统-inference-systems.md) | vLLM、PagedAttention、Batching、Speculative Decoding |
| 第十一篇 | [`评测、安全、幻觉与数据污染`](11-评测与安全-evaluation-safety.md) | Evaluation、Safety、Contamination |
| 第十二篇 | [`2026 前沿`](12-2026前沿-frontier-2026.md) | Native Agent、Long Context、Frontier Trends |
| 第十三篇 | [`词元化、语料与数据工程`](13-词元化与数据工程-tokenization-data.md) | BPE、SentencePiece、FineWeb、Dedup、Mixture |
| 第十四篇 | [`优化器、数值精度与训练动力学`](14-优化器与训练动力学-optimization-dynamics.md) | AdamW、BF16、Warmup、μP、Stability |
| 第十五篇 | [`可解释性与机制研究`](15-可解释性与机制研究-interpretability.md) | Circuits、Activation Patching、SAE、Editing |
| 第十六篇 | [`硬件、Kernel 与基础设施`](16-硬件内核与基础设施-hardware-kernels.md) | GPU、HBM、Tensor Core、NCCL、FlashAttention |
| 第十七篇 | [`闭源前沿模型与证据边界`](17-闭源前沿模型与证据边界-frontier-closed-models.md) | Model/System Card、Evidence Boundary |
| 第十八篇 | [`从零构建 Astra-class / Codex-class 系统`](18-从零构建Astra与Codex级系统-capstone.md) | Runtime、Tools、Coding Agent、Computer Use、Multi-Agent |
| 第十九篇 | [`Post-Transformer 架构`](19-Post-Transformer架构-post-transformer.md) | SSM、Mamba、Mamba-2、RWKV、xLSTM、Neural Memory |
| 第二十篇 | [`Diffusion Language Models`](20-扩散语言模型-diffusion-language-models.md) | Masked Diffusion、Denoising、Parallel Generation |
| 源码导读 A | [`模型运行时源码导读`](18A-模型运行时源码导读-runtime-walkthrough.md) | Token ID → Logits → KV Cache |
| 源码导读 B | [`智能体运行时源码导读`](18B-智能体运行时源码导读-agent-walkthrough.md) | Tool Call → Agent Loop → Coding Loop |
| 蓝图 | [`全书扩展蓝图 v2`](99-全书扩展蓝图-v2.md) | Full LLM Stack、Labs、L2/L3 |

---

# 三条学习路线

## 路线 A：模型科学

```text
00 → 01 → 13 → 04 → 02 → 14 → 03 → 05 → 07 → 08 → 15 → 19 → 20
```

目标：理解模型从表示、架构、数据、Scaling、后训练到 Post-Transformer / Diffusion 的完整机制演进。

## 路线 B：系统工程

```text
01 → 04 → 09 → 16 → 10 → 18A → 18
```

目标：从数学函数一路追到 kernel、显存、通信、缓存、调度和真实 runtime。

## 路线 C：智能体

```text
06 → ../智能体-agent/A1-A10 → 18B → 18
```

目标：从 Tool Use 进入环境、规划、记忆、验证、Coding、Computer Use、Multi-Agent、Agentic RL。

---

# 为什么把 Post-Transformer 与 Diffusion 单独成篇

如果把现代语言模型默认等同于“decoder-only Transformer + autoregressive decoding”，教材会把一个历史选择误写成自然定律。

因此本书明确比较至少三类序列生成范式：

```text
Autoregressive Transformer
State / Recurrent / Hybrid Sequence Models
Diffusion Language Models
```

研究重点不是“谁取代谁”，而是：

- 历史怎样被表示；
- token 之间怎样交换信息；
- cache/state 如何增长；
- 训练是否并行；
- 推理是否并行；
- 长程记忆放在哪里；
- reasoning / tool use 对架构提出什么新要求。

---

# 智能体为什么不再塞在第六篇里

第六篇只保留历史交汇：RAG、多模态、工具调用让模型第一次系统性“走出参数”。完整 Agent 学科已经拆到：

[`../智能体-agent/`](../智能体-agent/)

因为智能体已经拥有独立的：

- state / observation / action / policy 形式化；
- Tool / Environment 协议；
- Planning / Replanning；
- Verification / Recovery；
- Memory / Context；
- Coding / Browser / Computer Use；
- Multi-Agent；
- Protocol Interoperability；
- Agent Evaluation / Safety；
- Agentic RL。

---

# 原始资料入口

- [`../参考文献-references/00-原始资料总索引-primary-sources.md`](../参考文献-references/00-原始资料总索引-primary-sources.md)
- [`../参考文献-references/01-智能体原始资料-agent-sources.md`](../参考文献-references/01-智能体原始资料-agent-sources.md)
- [`../参考文献-references/README.md`](../参考文献-references/README.md)

---

# 章节成熟度

全书使用统一标准：

- **L0 目录级**：有问题定义和资料入口；
- **L1 教材级**：机制、公式、历史、引用完整；
- **L2 复现级**：有代码、实验、自动测试；
- **L3 研究级**：有反例、争议、后续证据、开放问题。

总看板：[`../质量看板-QUALITY.md`](../质量看板-QUALITY.md)

核心章节最终目标是 L2–L3，而不是“有一篇 Markdown”。