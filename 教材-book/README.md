# 教材正文 / Book

本目录是《大语言模型发展史、系统工程与智能体》的**模型科学与系统工程主线**。正文坚持：**中文解释 + 英文术语对照 + 原始资料优先 + 理论最终落到代码、实验与评测**。

> **最高写作规范**：[`00-教材方法论与证据标准-source-first.md`](00-教材方法论与证据标准-source-first.md)  
> **智能体独立主线**：[`../智能体-agent/README.md`](../智能体-agent/README.md)  
> **OpenAI Codex 工业源码解剖**：[`../Codex源码解剖-codex-anatomy/README.md`](../Codex源码解剖-codex-anatomy/README.md)  
> **最终工程终点**：[`18-从零构建Astra与Codex级系统-capstone.md`](18-从零构建Astra与Codex级系统-capstone.md)  
> **v3 总体架构蓝图**：[`99-全书架构蓝图-v3.md`](99-全书架构蓝图-v3.md)  
> **机器可读能力清单**：[`../能力清单-CAPABILITIES.json`](../能力清单-CAPABILITIES.json)

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
| 第十九篇 | [`Post-Transformer / Hybrid Sequence Architectures`](19-Post-Transformer架构-post-transformer.md) | SSM、Mamba、RWKV、xLSTM、Linear/Hybrid、Neural Memory |
| 第二十篇 | [`Diffusion Language Models`](20-扩散语言模型-diffusion-language-models.md) | Masked Diffusion、Denoising、Parallel Generation |
| 源码导读 A | [`模型运行时源码导读`](18A-模型运行时源码导读-runtime-walkthrough.md) | Token ID → Logits → KV Cache |
| 源码导读 B | [`智能体运行时源码导读`](18B-智能体运行时源码导读-agent-walkthrough.md) | Tool Call → Turn → Agent Loop |
| 蓝图 | [`全书架构蓝图 v3`](99-全书架构蓝图-v3.md) | Textbook + Reference System + Observatory |

---

# v3：不再只有“三条阅读路线”，而是七个技术层

```text
A  Model Science
B  Inference Systems
C  Post-training Systems
D  Agent Kernel
E  Environment / Protocol / Security
F  Multi-Agent
G  Evaluation / Observatory
```

教材正文主要承载 A/B/C 的理论与系统基础；完整 D/E/F 在 [`../智能体-agent/`](../智能体-agent/) 与 [`../Codex源码解剖-codex-anatomy/`](../Codex源码解剖-codex-anatomy/) 展开；G 由 [`../评测-eval/`](../评测-eval/) 和 [`../观测站-observatory/`](../观测站-observatory/) 承担。

---

# 推荐学习路线

## 路线 A：模型科学

```text
00 → 01 → 13 → 04 → 02 → 14 → 03 → 05 → 07 → 08 → 15 → 19 → 20
```

目标不是背模型名字，而是理解：表示、数据、架构、优化、后训练、推理能力和序列状态的计算结构。

## 路线 B：模型系统

```text
01 → 04 → 09 → 16 → 10 → 18A → 18
```

要求最终能从公式一路追到 kernel、HBM、KV/state、scheduler、TTFT、TPOT 和真实 checkpoint parity。

## 路线 C：智能体与工业 Runtime

```text
06
→ ../智能体-agent/A1-A10
→ ../Codex源码解剖-codex-anatomy/C00-C07
→ 18B
→ 18
```

目标是理解并亲手实现 Thread / Turn / Event、工具、协议、上下文、记忆、验证、权限、Coding、Computer Use 与 Multi-Agent。

## 路线 D：Research / Reproduction

```text
Primary Source
→ Chapter
→ Reference Code
→ Unit / Parity
→ Evaluation
→ Failure Analysis
```

入口：[`../学习地图-MAP.md`](../学习地图-MAP.md)、[`../质量看板-QUALITY.md`](../质量看板-QUALITY.md)、[`../能力清单-CAPABILITIES.json`](../能力清单-CAPABILITIES.json)。

---

# 为什么第十九篇逐渐从“Post-Transformer”转向 Hybrid Sequence Architectures

现实中的前沿路线并不是简单的：

```text
Transformer → 被某一个新架构完全替代
```

而越来越像：

```text
Global Attention
+ Sparse / Local Attention
+ Linear / Recurrent State
+ KV Compression
+ MoE
        ↓
  Heterogeneous Hybrid Stack
```

所以我们最终比较的是：

- state / KV memory 怎样增长；
- 每 token FLOPs；
- HBM bandwidth；
- 训练并行性；
- 推理解码复杂度；
- 长程 retrieval / retention；
- 哪些层必须做精确全局交互。

这比讨论“Transformer 死没死”更科学。

---

# 第十二篇不再独自承担“最新世界状态”

稳定理论应该写进教材；快速变化的模型、协议和系统事实进入：

[`../观测站-observatory/frontier-snapshot.json`](../观测站-observatory/frontier-snapshot.json)

观测站记录 `as_of`、官方来源、公开事实和 evidence boundary，并由 CI 做 schema / freshness report。这样教材不需要靠手工维护一张永远会漂移的前沿表格。

---

# 原始资料入口

- [`../参考文献-references/00-原始资料总索引-primary-sources.md`](../参考文献-references/00-原始资料总索引-primary-sources.md)
- [`../参考文献-references/01-智能体原始资料-agent-sources.md`](../参考文献-references/01-智能体原始资料-agent-sources.md)
- [`../参考文献-references/03-OpenAI-Codex官方源码索引-codex-source-map.md`](../参考文献-references/03-OpenAI-Codex官方源码索引-codex-source-map.md)
- [`../原始论文-paper-notes/`](../原始论文-paper-notes/)

---

# 章节成熟度

全书继续使用：

- **L0 目录级**：问题定义和资料入口；
- **L1 教材级**：机制、公式、历史、来源基本完整；
- **L2 复现级**：有代码、实验、自动测试或 parity；
- **L3 研究级**：有独立复现、反例/消融、争议与开放问题。

但从 v3 开始，具体 capability 是否真的完成，以根目录 [`../能力清单-CAPABILITIES.json`](../能力清单-CAPABILITIES.json) 为机器可读依据；`validated` 必须能追到代码、测试与证据。
