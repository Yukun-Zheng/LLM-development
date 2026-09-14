# 教材正文 / Book

本目录存放《大语言模型发展史与技术原理》的**主体章节**。正文采用“中文解释 + 英文术语对照”的写法；读者可见路径采用“中文优先 + 英文保留”的双语命名。

> **最高写作规范**：[`00-教材方法论与证据标准-source-first.md`](00-教材方法论与证据标准-source-first.md)  
> 本书采用 **Source-First / Mechanism-First / Reproducibility-First** 原则：重要结论优先回到原论文、官方技术报告、官方代码、模型卡、系统卡、数据集与 benchmark 原始资料，并区分“原始主张”“后续证据”“当前较稳健理解”。
>
> **完整扩展蓝图**：[`99-全书扩展蓝图-v2.md`](99-全书扩展蓝图-v2.md)  
> 现有章节是第一版主干，最终覆盖范围按 v2 蓝图扩展到数学、数据、优化、硬件、分布式训练、后训练、reasoning、multimodality、agent、serving、interpretability、evaluation、安全、全球模型生态与 Post-Transformer。

## 中文目录 / Contents

| 篇章 | 章节标题 | 关键术语 | 文件 |
|---|---|---|---|
| 方法论 | **教材方法论与证据标准** | Source-First、Mechanism-First、Reproducibility | [`00-教材方法论与证据标准-source-first.md`](00-教材方法论与证据标准-source-first.md) |
| 导读 | **怎样用这本书真正学会大语言模型** | Learning Guide | [`00-导读-preface.md`](00-导读-preface.md) |
| 第一篇 | **从 Seq2Seq 到 Transformer：大模型真正的技术起点** | Seq2Seq、Attention、Transformer、GPT、BERT | [`01-Transformer基础-foundations-transformer.md`](01-Transformer基础-foundations-transformer.md) |
| 第二篇 | **规模定律、GPT‑3、上下文学习与 Chinchilla** | Scaling Laws、GPT‑3、In-Context Learning、Compute-Optimal Training | [`02-规模定律与GPT3-scaling-gpt3.md`](02-规模定律与GPT3-scaling-gpt3.md) |
| 第三篇 | **指令微调、对齐、RLHF、ChatGPT 与 DPO** | Instruction Tuning、Alignment、RLHF、Preference Optimization | [`03-对齐与ChatGPT-alignment-chatgpt.md`](03-对齐与ChatGPT-alignment-chatgpt.md) |
| 第四篇 | **现代大语言模型架构组件** | RoPE、RMSNorm、SwiGLU、GQA、FlashAttention、KV Cache | [`04-现代LLM架构-modern-architecture.md`](04-现代LLM架构-modern-architecture.md) |
| 第五篇 | **开源大模型革命与高效微调** | LLaMA、Mistral、LoRA、QLoRA、Quantization | [`05-开源与高效微调-open-source-efficient-ft.md`](05-开源与高效微调-open-source-efficient-ft.md) |
| 第六篇 | **检索增强、多模态、工具调用与智能体** | RAG、Multimodality、Tool Use、Agent | [`06-RAG多模态与智能体-rag-multimodal-agent.md`](06-RAG多模态与智能体-rag-multimodal-agent.md) |
| 第七篇 | **混合专家与中国大模型路线** | MoE、GLM、Qwen、DeepSeek、Kimi、MiniMax | [`07-MoE与中国大模型-moe-china.md`](07-MoE与中国大模型-moe-china.md) |
| 第八篇 | **推理模型时代** | Chain-of-Thought、Test-Time Compute、RLVR、Reasoning RL | [`08-推理模型时代-reasoning-era.md`](08-推理模型时代-reasoning-era.md) |
| 第九篇 | **大模型训练系统** | Data、Parallelism、ZeRO、FSDP、Megatron、Stability | [`09-训练系统-training-systems.md`](09-训练系统-training-systems.md) |
| 第十篇 | **大模型推理与服务系统** | vLLM、PagedAttention、Continuous Batching、Speculative Decoding | [`10-推理服务系统-inference-systems.md`](10-推理服务系统-inference-systems.md) |
| 第十一篇 | **评测、安全、幻觉与数据污染** | Evaluation、Safety、Hallucination、Contamination | [`11-评测与安全-evaluation-safety.md`](11-评测与安全-evaluation-safety.md) |
| 第十二篇 | **2026 前沿：原生智能体、多模态、超长上下文与 Transformer 之后** | Native Agents、Long Context、Multimodality、Post-Transformer | [`12-2026前沿-frontier-2026.md`](12-2026前沿-frontier-2026.md) |
| 第十三篇 | **词元化、语料与数据工程** | BPE、SentencePiece、C4、Pile、FineWeb、Dedup、Mixture | [`13-词元化与数据工程-tokenization-data.md`](13-词元化与数据工程-tokenization-data.md) |
| 第十四篇 | **优化器、数值精度与训练动力学** | AdamW、Warmup、BF16、Gradient Clipping、μP | [`14-优化器与训练动力学-optimization-dynamics.md`](14-优化器与训练动力学-optimization-dynamics.md) |
| 第十五篇 | **可解释性与机制研究** | Probing、Activation Patching、Circuits、SAE、Model Editing | [`15-可解释性与机制研究-interpretability.md`](15-可解释性与机制研究-interpretability.md) |
| 第十六篇 | **硬件、Kernel 与基础设施** | Roofline、Tensor Core、FlashAttention、NCCL、ZeRO、vLLM | [`16-硬件内核与基础设施-hardware-kernels.md`](16-硬件内核与基础设施-hardware-kernels.md) |
| 第十七篇 | **闭源前沿模型与证据边界** | System Card、Model Card、Evidence Boundary、Frontier Models | [`17-闭源前沿模型与证据边界-frontier-closed-models.md`](17-闭源前沿模型与证据边界-frontier-closed-models.md) |
| 蓝图 | **全书扩展蓝图 v2** | Full LLM Stack、Labs、L2/L3 Maturity | [`99-全书扩展蓝图-v2.md`](99-全书扩展蓝图-v2.md) |

## 为什么继续扩展到第十三至十七篇

原来的 12 篇更接近“LLM 主历史线”，但如果目标是形成真正全面的教材，还必须补上此前被低估的基础：

1. **模型输入世界**：tokenization 与数据工程；
2. **模型如何真的被训出来**：optimizer、precision、training dynamics；
3. **模型内部如何被科学研究**：mechanistic interpretability；
4. **模型为什么在现实里能跑起来**：hardware、kernel、communication 与 serving infrastructure；
5. **信息不完整时如何研究前沿**：closed-model evidence boundaries。

后续章节仍将按同一标准继续扩展，而不是把“历史时间线”误当成“完整知识体系”。

## 推荐阅读顺序

如果目标是**系统掌握大模型发展史和机制**，建议：

`00 方法论 → 00 导读 → 01 → 02 → 13 → 14 → 03 → 04 → 05 → 07 → 08 → 06 → 09 → 16 → 10 → 15 → 11 → 17 → 12`

如果目标是快速补齐某条技术线，可以按主题跳读：

- **Transformer 与架构**：01 → 04 → 07 → 16
- **GPT‑3、Scaling 与数据**：02 → 13 → 14 → 09
- **ChatGPT 与后训练**：03 → 08
- **RAG、多模态与智能体**：06 → 08 → 12
- **训练与推理系统工程**：09 → 16 → 10
- **模型内部机制**：01 → 04 → 15
- **闭源前沿模型研究**：08 → 17 → 12
- **评测、安全与可靠性**：11 → 15 → 17 → 12

## 原始资料入口

全书的统一一手资料地图见：

- [`../参考文献-references/00-原始资料总索引-primary-sources.md`](../参考文献-references/00-原始资料总索引-primary-sources.md)
- [`../参考文献-references/README.md`](../参考文献-references/README.md)

## 术语原则

正文第一次出现重要概念时，尽量采用 **中文名称（English Term, Abbreviation）**；之后根据学界习惯使用中文、英文或缩写。例如：

- 上下文学习（In-Context Learning, ICL）
- 指令微调（Instruction Tuning）
- 人类反馈强化学习（Reinforcement Learning from Human Feedback, RLHF）
- 混合专家（Mixture of Experts, MoE）
- 智能体（Agent）
- 测试时计算（Test-Time Compute）
- 词元化（Tokenization）
- 机制可解释性（Mechanistic Interpretability）

## 章节成熟度

章节可按教材方法论中的标准标记：

- **L0 目录级**：问题与资料索引；
- **L1 教材级**：机制、公式、历史与引用完整；
- **L2 复现级**：包含代码、实验与结果；
- **L3 研究级**：包含争议、反例、最新证据与开放问题。

核心章节最终目标是尽可能达到 **L2–L3**，而不是只做到“有一篇 Markdown”。
