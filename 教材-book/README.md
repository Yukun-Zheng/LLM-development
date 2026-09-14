# 教材正文 / Book

本目录存放《大语言模型发展史与技术原理》的**主体章节**。正文采用“中文解释 + 英文术语对照”的写法；文件路径保留英文，方便 Git、链接、脚本和命令行使用。

## 中文目录 / Contents

| 篇章 | 章节标题 | 关键术语 | 文件 |
|---|---|---|---|
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

## 推荐阅读顺序

如果目标是**系统掌握大模型发展史和机制**，按 `00 → 01 → 02 → … → 12` 顺序阅读。

如果目标是快速补齐某条技术线，可以按主题跳读：

- **Transformer 与架构**：01 → 04 → 07
- **GPT‑3、Scaling 与数据**：02 → 05 → 09
- **ChatGPT 与后训练**：03 → 08
- **RAG、多模态与智能体**：06 → 08 → 12
- **训练与推理系统工程**：09 → 10
- **评测、安全与可靠性**：11

## 术语原则

正文第一次出现重要概念时，尽量采用 **中文名称（English Term, Abbreviation）**；之后根据学界习惯使用中文、英文或缩写。例如：

- 上下文学习（In-Context Learning, ICL）
- 指令微调（Instruction Tuning）
- 人类反馈强化学习（Reinforcement Learning from Human Feedback, RLHF）
- 混合专家（Mixture of Experts, MoE）
- 智能体（Agent）
- 测试时计算（Test-Time Compute）
