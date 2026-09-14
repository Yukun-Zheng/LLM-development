# 原始资料总索引：Primary-Source Map

> **用途**：这是全书的“证据地图”。它不是普通 bibliography，而是告诉读者：每个技术主题应该回到哪些原论文、官方代码、模型卡、系统卡和数据集论文。
>
> **原则**：核心技术结论优先使用一手来源。二手资料只用于背景组织，不用于替代原始证据。
>
> **时间边界**：更新至 2026-09-14。

---

# A　序列建模、Attention 与 Transformer

## A1　Seq2Seq 与 Attention

- Sutskever, Vinyals & Le (2014), **Sequence to Sequence Learning with Neural Networks**  
  https://arxiv.org/abs/1409.3215
- Bahdanau, Cho & Bengio (2014), **Neural Machine Translation by Jointly Learning to Align and Translate**  
  https://arxiv.org/abs/1409.0473
- Luong, Pham & Manning (2015), **Effective Approaches to Attention-based Neural Machine Translation**  
  https://arxiv.org/abs/1508.04025

## A2　Transformer

- Vaswani et al. (2017), **Attention Is All You Need**  
  https://arxiv.org/abs/1706.03762
- 官方历史实现：Google Research / Tensor2Tensor 及相关代码历史。

## A3　Normalization、Activation 与位置编码

- Ba, Kiros & Hinton (2016), **Layer Normalization**  
  https://arxiv.org/abs/1607.06450
- Zhang & Sennrich (2019), **Root Mean Square Layer Normalization**  
  https://arxiv.org/abs/1910.07467
- Xiong et al. (2020), **On Layer Normalization in the Transformer Architecture**  
  https://arxiv.org/abs/2002.04745
- Shazeer (2020), **GLU Variants Improve Transformer**  
  https://arxiv.org/abs/2002.05202
- Su et al. (2021), **RoFormer: Enhanced Transformer with Rotary Position Embedding**  
  https://arxiv.org/abs/2104.09864
- Press, Smith & Lewis (2021), **Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation**  
  https://arxiv.org/abs/2108.12409

---

# B　GPT、BERT 与预训练范式

- Radford et al. (2018), **Improving Language Understanding by Generative Pre-Training**  
  https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
- Devlin et al. (2018), **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**  
  https://arxiv.org/abs/1810.04805
- Radford et al. (2019), **Language Models are Unsupervised Multitask Learners**  
  https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- Raffel et al. (2019/2020), **Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer**  
  https://arxiv.org/abs/1910.10683
- Brown et al. (2020), **Language Models are Few-Shot Learners**  
  https://arxiv.org/abs/2005.14165

---

# C　Tokenization、Vocabulary 与文本表示

## C1　Subword

- Sennrich, Haddow & Birch (2015/2016), **Neural Machine Translation of Rare Words with Subword Units**  
  https://arxiv.org/abs/1508.07909
- Kudo (2018), **Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates**  
  https://arxiv.org/abs/1804.10959
- Kudo & Richardson (2018), **SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing**  
  https://arxiv.org/abs/1808.06226
- Provilkov, Emelianenko & Voita (2019), **BPE-Dropout**  
  https://arxiv.org/abs/1910.13267

## C2　Tokenizer 工程

- GPT-2 tokenizer 定义与 vocabulary：OpenAI GPT-2 官方仓库 / 模型文件。
- OpenAI `tiktoken` 官方实现：  
  https://github.com/openai/tiktoken
- Hugging Face `tokenizers` 官方实现：  
  https://github.com/huggingface/tokenizers

---

# D　数据、清洗、去重与语料工程

- Raffel et al. (2019/2020), C4 数据处理，见 T5 原论文  
  https://arxiv.org/abs/1910.10683
- Gao et al. (2020), **The Pile: An 800GB Dataset of Diverse Text for Language Modeling**  
  https://arxiv.org/abs/2101.00027
- Lee et al. (2021), **Deduplicating Training Data Makes Language Models Better**  
  https://arxiv.org/abs/2107.06499
- Soldaini et al. (2024), **Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research**  
  https://arxiv.org/abs/2402.00159
- Penedo et al. (2024), **The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale**  
  https://arxiv.org/abs/2406.17557
- Li et al. (2024), **DataComp-LM: In Search of the Next Generation of Language Model Pretraining Datasets**  
  https://arxiv.org/abs/2406.11794

重点不是“哪个数据集最大”，而是：采集、语言识别、质量过滤、去重、污染控制、版权/隐私处理、mixture 权重和 curriculum 如何改变模型。

---

# E　Scaling Laws 与 Compute-Optimal Training

- Kaplan et al. (2020), **Scaling Laws for Neural Language Models**  
  https://arxiv.org/abs/2001.08361
- Henighan et al. (2020), **Scaling Laws for Autoregressive Generative Modeling**  
  https://arxiv.org/abs/2010.14701
- Hoffmann et al. (2022), **Training Compute-Optimal Large Language Models**  
  https://arxiv.org/abs/2203.15556
- Muennighoff et al. (2023), **Scaling Data-Constrained Language Models**  
  https://arxiv.org/abs/2305.16264

---

# F　优化器、数值精度与训练动力学

- Kingma & Ba (2014), **Adam: A Method for Stochastic Optimization**  
  https://arxiv.org/abs/1412.6980
- Loshchilov & Hutter (2017/2019), **Decoupled Weight Decay Regularization (AdamW)**  
  https://arxiv.org/abs/1711.05101
- You et al. (2019), **Large Batch Optimization for Deep Learning: Training BERT in 76 minutes** (LAMB)  
  https://arxiv.org/abs/1904.00962
- Micikevicius et al. (2017/2018), **Mixed Precision Training**  
  https://arxiv.org/abs/1710.03740
- Kalamkar et al. (2019), **A Study of BFLOAT16 for Deep Learning Training**  
  https://arxiv.org/abs/1905.12322
- Yang et al. (2021/2022), **Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer**（μP）  
  https://arxiv.org/abs/2203.03466

对于新优化器（如 Muon 等），正文只在能取得原始技术说明、实现与可复现实验后给出结论。

---

# G　Instruction Tuning、RLHF 与偏好学习

- Christiano et al. (2017), **Deep Reinforcement Learning from Human Preferences**  
  https://arxiv.org/abs/1706.03741
- Schulman et al. (2017), **Proximal Policy Optimization Algorithms**  
  https://arxiv.org/abs/1707.06347
- Wei et al. (2021), **Finetuned Language Models Are Zero-Shot Learners**  
  https://arxiv.org/abs/2109.01652
- Sanh et al. (2021), **Multitask Prompted Training Enables Zero-Shot Task Generalization**  
  https://arxiv.org/abs/2110.08207
- Ouyang et al. (2022), **Training language models to follow instructions with human feedback**  
  https://arxiv.org/abs/2203.02155
- Bai et al. (2022), **Constitutional AI: Harmlessness from AI Feedback**  
  https://arxiv.org/abs/2212.08073
- Rafailov et al. (2023), **Direct Preference Optimization**  
  https://arxiv.org/abs/2305.18290
- Ethayarajh et al. (2024), **KTO**  
  https://arxiv.org/abs/2402.01306
- Hong et al. (2024), **ORPO**  
  https://arxiv.org/abs/2403.07691

---

# H　Reasoning、搜索与 Test-Time Compute

- Wei et al. (2022), **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models**  
  https://arxiv.org/abs/2201.11903
- Kojima et al. (2022), **Large Language Models are Zero-Shot Reasoners**  
  https://arxiv.org/abs/2205.11916
- Wang et al. (2022), **Self-Consistency Improves Chain of Thought Reasoning in Language Models**  
  https://arxiv.org/abs/2203.11171
- Yao et al. (2023), **Tree of Thoughts**  
  https://arxiv.org/abs/2305.10601
- Lightman et al. (2023), **Let's Verify Step by Step**  
  https://arxiv.org/abs/2305.20050
- Snell et al. (2024), **Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters**  
  https://arxiv.org/abs/2408.03314
- Shao et al. (2024), **DeepSeekMath**（GRPO）  
  https://arxiv.org/abs/2402.03300
- DeepSeek-AI (2025), **DeepSeek-R1**  
  https://arxiv.org/abs/2501.12948
- OpenAI (2024), **Learning to reason with LLMs**  
  https://openai.com/index/learning-to-reason-with-llms/
- OpenAI (2024), **OpenAI o1 System Card**  
  https://openai.com/index/openai-o1-system-card/

---

# I　现代注意力、KV Cache 与 Kernel

- Shazeer (2019), **Fast Transformer Decoding: One Write-Head is All You Need**（MQA）  
  https://arxiv.org/abs/1911.02150
- Ainslie et al. (2023), **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints**  
  https://arxiv.org/abs/2305.13245
- Dao et al. (2022), **FlashAttention**  
  https://arxiv.org/abs/2205.14135
- Dao (2023), **FlashAttention-2**  
  https://arxiv.org/abs/2307.08691
- Shah et al. (2024), **FlashAttention-3**  
  https://arxiv.org/abs/2407.08608

---

# J　MoE 与稀疏计算

- Shazeer et al. (2017), **Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer**  
  https://arxiv.org/abs/1701.06538
- Lepikhin et al. (2020), **GShard**  
  https://arxiv.org/abs/2006.16668
- Fedus et al. (2021), **Switch Transformers**  
  https://arxiv.org/abs/2101.03961
- Jiang et al. (2024), **Mixtral of Experts**  
  https://arxiv.org/abs/2401.04088
- Dai et al. (2024), **DeepSeekMoE**  
  https://arxiv.org/abs/2401.06066
- DeepSeek-AI (2024), **DeepSeek-V2**  
  https://arxiv.org/abs/2405.04434
- DeepSeek-AI (2024/2025), **DeepSeek-V3 Technical Report**  
  https://arxiv.org/abs/2412.19437

---

# K　开放权重模型

- Touvron et al. (2023), **LLaMA**  
  https://arxiv.org/abs/2302.13971
- Touvron et al. (2023), **Llama 2**  
  https://arxiv.org/abs/2307.09288
- Grattafiori et al. (2024), **The Llama 3 Herd of Models**  
  https://arxiv.org/abs/2407.21783
- Meta (2025), **Llama 4 official model resources / model cards**  
  https://ai.meta.com/llama/get-started/
- Jiang et al. (2023), **Mistral 7B**  
  https://arxiv.org/abs/2310.06825
- Jiang et al. (2024), **Mixtral of Experts**  
  https://arxiv.org/abs/2401.04088
- Mistral AI (2024), **Mistral Large 2 / Large Enough**  
  https://mistral.ai/news/mistral-large-2407/
- Gemma Team (2024), **Gemma**  
  https://arxiv.org/abs/2403.08295
- Gemma Team (2025), **Gemma 3 Technical Report**  
  https://arxiv.org/abs/2503.19786

---

# L　中国大模型技术路线

## L1　GLM

- GLM, https://arxiv.org/abs/2103.10360
- GLM-130B, https://arxiv.org/abs/2210.02414
- ChatGLM family report, https://arxiv.org/abs/2406.12793

## L2　Qwen

- Qwen Technical Report, https://arxiv.org/abs/2309.16609
- Qwen2, https://arxiv.org/abs/2407.10671
- Qwen2.5, https://arxiv.org/abs/2412.15115
- Qwen3, https://arxiv.org/abs/2505.09388

## L3　DeepSeek

- DeepSeek LLM, https://arxiv.org/abs/2401.02954
- DeepSeekMoE, https://arxiv.org/abs/2401.06066
- DeepSeekMath, https://arxiv.org/abs/2402.03300
- DeepSeek-V2, https://arxiv.org/abs/2405.04434
- DeepSeek-V3, https://arxiv.org/abs/2412.19437
- DeepSeek-R1, https://arxiv.org/abs/2501.12948

## L4　Kimi / Moonshot AI

- Moonshot AI (2025), **Kimi K2: Open Agentic Intelligence**  
  https://arxiv.org/abs/2507.20534

## L5　MiniMax

- MiniMax (2026), **The MiniMax-M2 Series: Mini Activations Unleashing Max Real-World Intelligence**  
  https://arxiv.org/abs/2605.26494

InternLM、Baichuan、Yi 等模型按其官方论文、技术报告、GitHub 与模型卡补充，不使用转载 benchmark 代替一手资料。

---

# M　RAG 与外部知识

- Lewis et al. (2020), **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**  
  https://arxiv.org/abs/2005.11401
- Karpukhin et al. (2020), **Dense Passage Retrieval**  
  https://arxiv.org/abs/2004.04906
- Borgeaud et al. (2021/2022), **RETRO**  
  https://arxiv.org/abs/2112.04426
- Izacard et al. (2022), **Atlas**  
  https://arxiv.org/abs/2208.03299

---

# N　Tool Use 与 Agent

- Nakano et al. (2021), **WebGPT: Browser-assisted question-answering with human feedback**  
  https://arxiv.org/abs/2112.09332
- Yao et al. (2022/2023), **ReAct**  
  https://arxiv.org/abs/2210.03629
- Schick et al. (2023), **Toolformer**  
  https://arxiv.org/abs/2302.04761
- Shinn et al. (2023), **Reflexion**  
  https://arxiv.org/abs/2303.11366
- Jimenez et al. (2024), **SWE-bench**  
  https://arxiv.org/abs/2310.06770

Agent 章节必须把 **base model、scaffold、工具、环境、retry budget、test-time compute** 分开评测。

---

# O　多模态与视觉语言模型

- Radford et al. (2021), **Learning Transferable Visual Models From Natural Language Supervision (CLIP)**  
  https://arxiv.org/abs/2103.00020
- Alayrac et al. (2022), **Flamingo**  
  https://arxiv.org/abs/2204.14198
- Li et al. (2023), **BLIP-2**  
  https://arxiv.org/abs/2301.12597
- Liu et al. (2023), **Visual Instruction Tuning (LLaVA)**  
  https://arxiv.org/abs/2304.08485
- OpenAI (2023), **GPT-4 Technical Report**  
  https://arxiv.org/abs/2303.08774

对于闭源原生多模态模型，只写官方公开到的能力与系统设计边界。

---

# P　训练系统与并行

- Shoeybi et al. (2019), **Megatron-LM**  
  https://arxiv.org/abs/1909.08053
- Rajbhandari et al. (2019/2020), **ZeRO**  
  https://arxiv.org/abs/1910.02054
- Narayanan et al. (2021), **Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM**  
  https://arxiv.org/abs/2104.04473
- PyTorch FSDP 官方文档与实现：  
  https://pytorch.org/docs/stable/fsdp.html
- DeepSpeed 官方实现：  
  https://github.com/microsoft/DeepSpeed
- Megatron-LM 官方实现：  
  https://github.com/NVIDIA/Megatron-LM

---

# Q　推理系统

- Kwon et al. (2023), **Efficient Memory Management for Large Language Model Serving with PagedAttention**（vLLM）  
  https://arxiv.org/abs/2309.06180
- Leviathan et al. (2023), **Fast Inference from Transformers via Speculative Decoding**  
  https://arxiv.org/abs/2211.17192
- Chen et al. (2023), **Accelerating Large Language Model Decoding with Speculative Sampling**  
  https://arxiv.org/abs/2302.01318
- vLLM 官方实现：  
  https://github.com/vllm-project/vllm

---

# R　Interpretability、Mechanistic Interpretability 与 Model Editing

- Olah et al. (2020), **Zoom In: An Introduction to Circuits**  
  https://distill.pub/2020/circuits/zoom-in/
- Elhage et al. (2021), **A Mathematical Framework for Transformer Circuits**  
  https://transformer-circuits.pub/2021/framework/index.html
- Olsson et al. (2022), **In-context Learning and Induction Heads**  
  https://arxiv.org/abs/2209.11895
- Meng et al. (2022), **Locating and Editing Factual Associations in GPT (ROME)**  
  https://arxiv.org/abs/2202.05262
- Anthropic (2024), **Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet**  
  https://transformer-circuits.pub/2024/scaling-monosemanticity/
- TransformerLens 官方实现：  
  https://github.com/TransformerLensOrg/TransformerLens

必须区分“可视化相关性”“causal intervention”和“真正机制解释”。

---

# S　评测、数据污染与记忆

- Hendrycks et al. (2020), **MMLU**  
  https://arxiv.org/abs/2009.03300
- Srivastava et al. (2022), **BIG-bench**  
  https://arxiv.org/abs/2206.04615
- Liang et al. (2022), **HELM**  
  https://arxiv.org/abs/2211.09110
- Rein et al. (2023), **GPQA**  
  https://arxiv.org/abs/2311.12022
- Jimenez et al. (2024), **SWE-bench**  
  https://arxiv.org/abs/2310.06770

评测章节还需直接检查 benchmark 官方仓库、版本和 contamination policy。

---

# T　前沿闭源模型：只写公开证据边界

## T1　OpenAI

- OpenAI (2023), **GPT-4 Technical Report**  
  https://arxiv.org/abs/2303.08774
- OpenAI (2024), **o1 System Card**  
  https://openai.com/index/openai-o1-system-card/
- OpenAI (2025), **GPT-5 System Card**  
  https://openai.com/index/gpt-5-system-card/
- OpenAI (2026), **GPT-5.4 Thinking System Card**  
  https://openai.com/index/gpt-5-4-thinking-system-card/
- OpenAI (2026), **GPT-5.5 System Card**  
  https://openai.com/index/gpt-5-5-system-card/
- OpenAI (2026), **GPT-5.6 System Card**  
  https://deploymentsafety.openai.com/gpt-5-6

注意：system card 不是完整架构论文。它能支撑安全、能力、部分训练范式和系统级描述，但不能支撑未公开的内部层数、参数量或具体结构猜测。

## T2　Anthropic

- Anthropic 官方 Model System Cards 总入口：  
  https://www.anthropic.com/system-cards
- Anthropic Transparency Hub：  
  https://www.anthropic.com/transparency
- Constitutional AI 原论文：  
  https://arxiv.org/abs/2212.08073

## T3　Google / Gemini

使用 Google DeepMind 官方技术报告、模型卡与安全白皮书。对于不同 Gemini 世代，章节必须注明具体型号和发布日期，避免把产品名当成固定架构。

## T4　xAI

- xAI (2025), **Grok 4 Model Card**  
  https://data.x.ai/2025-08-20-grok-4-model-card.pdf
- xAI (2025), **Grok 4** 官方发布  
  https://x.ai/news/grok-4
- xAI Safety / model card 总入口  
  https://x.ai/safety

---

# U　硬件与计算机系统基础

- Jouppi et al. (2017), **In-Datacenter Performance Analysis of a Tensor Processing Unit**  
  https://arxiv.org/abs/1704.04760
- Williams, Waterman & Patterson (2009), **Roofline: An Insightful Visual Performance Model for Multicore Architectures**  
  https://dl.acm.org/doi/10.1145/1498765.1498785
- NVIDIA CUDA Programming Guide、GPU architecture whitepapers、NCCL documentation 作为 GPU / communication 原始资料。

硬件章节不能停在“GPU 很快”，必须解释 HBM、SRAM/register、tensor core、memory bandwidth、interconnect、collective communication 和 kernel fusion。

---

# V　如何使用本索引

对于任何技术，按以下顺序阅读：

```text
原论文 / 技术报告
        ↓
官方代码 / checkpoint / config
        ↓
数据集与 benchmark 原始资料
        ↓
独立复现 / 消融
        ↓
综述与历史材料
```

如果原论文与官方代码冲突，正文要指出冲突；如果官方 benchmark 与独立复现差异明显，也要分别保留。

**全书不追求“所有地方都有引用符号”，而追求“所有重要事实都有可追溯的一手证据链”。**
