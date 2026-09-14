# 核心论文、技术报告与官方资料索引

> **用途**：这是全书的“来源脊柱”，不是替代正文的独立 bibliography。正文中的关键事实仍应在首次出现处就近引用。  
> **优先级**：原论文 / 官方技术报告 / 官方模型卡 / 官方代码仓库 > 高质量综述 > 新闻与二手解读。  
> **时间边界**：教材技术状态更新至 2026-09-14；本索引持续增补，不声称穷尽所有 LLM 论文。

---

# 1　Transformer 前传与基础架构

1. Bahdanau, Cho & Bengio, 2014. **Neural Machine Translation by Jointly Learning to Align and Translate**. https://arxiv.org/abs/1409.0473
2. Sutskever, Vinyals & Le, 2014. **Sequence to Sequence Learning with Neural Networks**. https://arxiv.org/abs/1409.3215
3. Vaswani et al., 2017. **Attention Is All You Need**. https://arxiv.org/abs/1706.03762
4. Ba, Kiros & Hinton, 2016. **Layer Normalization**. https://arxiv.org/abs/1607.06450
5. Zhang & Sennrich, 2019. **Root Mean Square Layer Normalization**. https://arxiv.org/abs/1910.07467
6. Xiong et al., 2020. **On Layer Normalization in the Transformer Architecture**. https://arxiv.org/abs/2002.04745
7. Shazeer, 2020. **GLU Variants Improve Transformer**. https://arxiv.org/abs/2002.05202
8. Su et al., 2021. **RoFormer: Enhanced Transformer with Rotary Position Embedding**. https://arxiv.org/abs/2104.09864
9. Shazeer, 2019. **Fast Transformer Decoding: One Write-Head is All You Need** (MQA). https://arxiv.org/abs/1911.02150
10. Ainslie et al., 2023. **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints**. https://arxiv.org/abs/2305.13245

---

# 2　GPT、BERT 与预训练范式

1. Radford et al., 2018. **Improving Language Understanding by Generative Pre-Training** (GPT-1). https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
2. Devlin et al., 2018. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**. https://arxiv.org/abs/1810.04805
3. Radford et al., 2019. **Language Models are Unsupervised Multitask Learners** (GPT-2). https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
4. Brown et al., 2020. **Language Models are Few-Shot Learners** (GPT-3). https://arxiv.org/abs/2005.14165
5. Raffel et al., 2019/2020. **Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer** (T5). https://arxiv.org/abs/1910.10683

---

# 3　Scaling Laws、数据规模与 Compute-Optimal Training

1. Kaplan et al., 2020. **Scaling Laws for Neural Language Models**. https://arxiv.org/abs/2001.08361
2. Hoffmann et al., 2022. **Training Compute-Optimal Large Language Models** (Chinchilla). https://arxiv.org/abs/2203.15556
3. Henighan et al., 2020. **Scaling Laws for Autoregressive Generative Modeling**. https://arxiv.org/abs/2010.14701
4. Muennighoff et al., 2023. **Scaling Data-Constrained Language Models**. https://arxiv.org/abs/2305.16264
5. OpenAI, 2023. **GPT-4 Technical Report**. https://arxiv.org/abs/2303.08774

阅读时尤其注意：scaling law 是在特定数据、模型族和训练 recipe 下的经验规律，不应被当作跨条件不变的物理定律。

---

# 4　Instruction Tuning、RLHF 与 Preference Optimization

1. Wei et al., 2021. **Finetuned Language Models Are Zero-Shot Learners** (FLAN). https://arxiv.org/abs/2109.01652
2. Sanh et al., 2021. **Multitask Prompted Training Enables Zero-Shot Task Generalization** (T0). https://arxiv.org/abs/2110.08207
3. Christiano et al., 2017. **Deep Reinforcement Learning from Human Preferences**. https://arxiv.org/abs/1706.03741
4. Schulman et al., 2017. **Proximal Policy Optimization Algorithms**. https://arxiv.org/abs/1707.06347
5. Ouyang et al., 2022. **Training language models to follow instructions with human feedback** (InstructGPT). https://arxiv.org/abs/2203.02155
6. Bai et al., 2022. **Constitutional AI: Harmlessness from AI Feedback**. https://arxiv.org/abs/2212.08073
7. Rafailov et al., 2023. **Direct Preference Optimization: Your Language Model is Secretly a Reward Model**. https://arxiv.org/abs/2305.18290
8. Azar et al., 2023/2024. **A General Theoretical Paradigm to Understand Learning from Human Preferences** (IPO). https://arxiv.org/abs/2310.12036
9. Ethayarajh et al., 2024. **KTO: Model Alignment as Prospect Theoretic Optimization**. https://arxiv.org/abs/2402.01306
10. Hong et al., 2024. **ORPO: Monolithic Preference Optimization without Reference Model**. https://arxiv.org/abs/2403.07691

---

# 5　Chain-of-Thought、搜索与推理

1. Wei et al., 2022. **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models**. https://arxiv.org/abs/2201.11903
2. Kojima et al., 2022. **Large Language Models are Zero-Shot Reasoners**. https://arxiv.org/abs/2205.11916
3. Wang et al., 2022. **Self-Consistency Improves Chain of Thought Reasoning in Language Models**. https://arxiv.org/abs/2203.11171
4. Yao et al., 2023. **Tree of Thoughts: Deliberate Problem Solving with Large Language Models**. https://arxiv.org/abs/2305.10601
5. Lightman et al., 2023. **Let’s Verify Step by Step**. https://arxiv.org/abs/2305.20050
6. Snell et al., 2024. **Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters**. https://arxiv.org/abs/2408.03314
7. Shao et al., 2024. **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models** (GRPO). https://arxiv.org/abs/2402.03300
8. DeepSeek-AI, 2025. **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning**. https://arxiv.org/abs/2501.12948

---

# 6　开源 / 开放权重大模型

1. Touvron et al., 2023. **LLaMA: Open and Efficient Foundation Language Models**. https://arxiv.org/abs/2302.13971
2. Touvron et al., 2023. **Llama 2: Open Foundation and Fine-Tuned Chat Models**. https://arxiv.org/abs/2307.09288
3. Grattafiori et al., 2024. **The Llama 3 Herd of Models**. https://arxiv.org/abs/2407.21783
4. Jiang et al., 2023. **Mistral 7B**. https://arxiv.org/abs/2310.06825
5. Jiang et al., 2024. **Mixtral of Experts**. https://arxiv.org/abs/2401.04088
6. Gemma Team, 2024. **Gemma: Open Models Based on Gemini Research and Technology**. https://arxiv.org/abs/2403.08295
7. Gemma Team, 2025. **Gemma 3 Technical Report**. https://arxiv.org/abs/2503.19786

本书使用“open-weight / 开放权重”和“open source / 开源”时会尽量区分许可证、训练数据透明度、代码开放程度与权重可得性，不把它们混为一个概念。

---

# 7　LoRA、QLoRA、量化与高效微调

1. Hu et al., 2021. **LoRA: Low-Rank Adaptation of Large Language Models**. https://arxiv.org/abs/2106.09685
2. Dettmers et al., 2023. **QLoRA: Efficient Finetuning of Quantized LLMs**. https://arxiv.org/abs/2305.14314
3. Dettmers et al., 2022. **LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale**. https://arxiv.org/abs/2208.07339
4. Frantar et al., 2022/2023. **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers**. https://arxiv.org/abs/2210.17323
5. Lin et al., 2023. **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration**. https://arxiv.org/abs/2306.00978

---

# 8　FlashAttention 与注意力系统优化

1. Dao et al., 2022. **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**. https://arxiv.org/abs/2205.14135
2. Dao, 2023. **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning**. https://arxiv.org/abs/2307.08691
3. Shah et al., 2024. **FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision**. https://arxiv.org/abs/2407.08608

关键辨析：FlashAttention 的核心贡献主要是 IO-aware exact attention kernel；它并不是简单提出一个新的 attention 数学定义。

---

# 9　MoE 与稀疏计算

1. Shazeer et al., 2017. **Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer**. https://arxiv.org/abs/1701.06538
2. Lepikhin et al., 2020. **GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding**. https://arxiv.org/abs/2006.16668
3. Fedus, Zoph & Shazeer, 2021. **Switch Transformers**. https://arxiv.org/abs/2101.03961
4. Jiang et al., 2024. **Mixtral of Experts**. https://arxiv.org/abs/2401.04088
5. Dai et al., 2024. **DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models**. https://arxiv.org/abs/2401.06066
6. DeepSeek-AI, 2024. **DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model**. https://arxiv.org/abs/2405.04434
7. DeepSeek-AI, 2024/2025. **DeepSeek-V3 Technical Report**. https://arxiv.org/abs/2412.19437

---

# 10　中国大模型与代表技术路线

这一节优先列公开技术报告，不以厂商宣传页替代论文。

## GLM / ChatGLM

1. Du et al., 2021/2022. **GLM: General Language Model Pretraining with Autoregressive Blank Infilling**. https://arxiv.org/abs/2103.10360
2. Zeng et al., 2022. **GLM-130B: An Open Bilingual Pre-trained Model**. https://arxiv.org/abs/2210.02414
3. GLM Team, 2024. **ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools**. https://arxiv.org/abs/2406.12793

## Qwen

4. Bai et al., 2023. **Qwen Technical Report**. https://arxiv.org/abs/2309.16609
5. Yang et al., 2024. **Qwen2 Technical Report**. https://arxiv.org/abs/2407.10671
6. Qwen Team, 2024. **Qwen2.5 Technical Report**. https://arxiv.org/abs/2412.15115
7. Yang et al., 2025. **Qwen3 Technical Report**. https://arxiv.org/abs/2505.09388

## DeepSeek

8. DeepSeek-AI, 2024. **DeepSeek LLM: Scaling Open-Source Language Models with Longtermism**. https://arxiv.org/abs/2401.02954
9. Dai et al., 2024. **DeepSeekMoE**. https://arxiv.org/abs/2401.06066
10. Shao et al., 2024. **DeepSeekMath**. https://arxiv.org/abs/2402.03300
11. DeepSeek-AI, 2024. **DeepSeek-V2**. https://arxiv.org/abs/2405.04434
12. DeepSeek-AI, 2024/2025. **DeepSeek-V3 Technical Report**. https://arxiv.org/abs/2412.19437
13. DeepSeek-AI, 2025. **DeepSeek-R1**. https://arxiv.org/abs/2501.12948

## Kimi / Moonshot

14. Moonshot AI, 2025. **Kimi K2: Open Agentic Intelligence** / technical report. https://arxiv.org/abs/2507.20534

## 其他需要持续跟踪的中国模型族

正文还讨论或索引 InternLM、Baichuan、Yi、MiniMax 等路线。对快速迭代型号，优先在对应章节使用其当期官方技术报告/模型卡，并注明版本日期；不把不同版本的宣传 benchmark 混在一个无控制表格中横向排名。

---

# 11　RAG、检索与外部知识

1. Lewis et al., 2020. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**. https://arxiv.org/abs/2005.11401
2. Karpukhin et al., 2020. **Dense Passage Retrieval for Open-Domain Question Answering**. https://arxiv.org/abs/2004.04906
3. Borgeaud et al., 2021/2022. **Improving Language Models by Retrieving from Trillions of Tokens** (RETRO). https://arxiv.org/abs/2112.04426
4. Izacard et al., 2022. **Atlas: Few-shot Learning with Retrieval Augmented Language Models**. https://arxiv.org/abs/2208.03299

RAG 不是“把 PDF 塞进 prompt”这么简单。完整系统至少涉及 chunking、embedding、index、retrieval、reranking、context construction、generation 与 attribution/evaluation。

---

# 12　Tool Use、Agent 与环境交互

1. Yao et al., 2022/2023. **ReAct: Synergizing Reasoning and Acting in Language Models**. https://arxiv.org/abs/2210.03629
2. Schick et al., 2023. **Toolformer: Language Models Can Teach Themselves to Use Tools**. https://arxiv.org/abs/2302.04761
3. Nakano et al., 2021. **WebGPT: Browser-assisted question-answering with human feedback**. https://arxiv.org/abs/2112.09332
4. Shinn et al., 2023. **Reflexion: Language Agents with Verbal Reinforcement Learning**. https://arxiv.org/abs/2303.11366
5. Wang et al., 2023. **Voyager: An Open-Ended Embodied Agent with Large Language Models**. https://arxiv.org/abs/2305.16291

阅读 agent 工作时必须拆开：base model、prompt/scaffold、tools、memory、planner、environment、training signal、evaluation protocol。否则容易把系统工程进步误认为 base model 架构进步。

---

# 13　多模态基础

1. Radford et al., 2021. **Learning Transferable Visual Models From Natural Language Supervision** (CLIP). https://arxiv.org/abs/2103.00020
2. Alayrac et al., 2022. **Flamingo: a Visual Language Model for Few-Shot Learning**. https://arxiv.org/abs/2204.14198
3. Li et al., 2023. **BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models**. https://arxiv.org/abs/2301.12597
4. Liu et al., 2023. **Visual Instruction Tuning** (LLaVA). https://arxiv.org/abs/2304.08485
5. Gemini Team, 2023. **Gemini: A Family of Highly Capable Multimodal Models**. https://arxiv.org/abs/2312.11805

---

# 14　长上下文与位置外推

1. Press et al., 2021/2022. **Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation** (ALiBi). https://arxiv.org/abs/2108.12409
2. Su et al., 2021. **RoFormer / RoPE**. https://arxiv.org/abs/2104.09864
3. Chen et al., 2023. **Extending Context Window of Large Language Models via Positional Interpolation**. https://arxiv.org/abs/2306.15595
4. Peng et al., 2023. **YaRN: Efficient Context Window Extension of Large Language Models**. https://arxiv.org/abs/2309.00071

必须区分：模型 API 允许的 context window、训练时真正见过的长度、位置编码可外推的长度、以及模型实际上能有效利用信息的长度。

---

# 15　训练系统与大规模并行

1. Shoeybi et al., 2019. **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism**. https://arxiv.org/abs/1909.08053
2. Rajbhandari et al., 2019/2020. **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models**. https://arxiv.org/abs/1910.02054
3. Narayanan et al., 2021. **Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM**. https://arxiv.org/abs/2104.04473
4. Zhao et al., 2023. **PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel**. https://arxiv.org/abs/2304.11277

核心并行维度：

```text
Data Parallel (DP)
Tensor Parallel (TP)
Pipeline Parallel (PP)
Sequence / Context Parallel
Expert Parallel (EP)
```

现实中的大模型训练通常是多维并行的组合，而不是单选题。

---

# 16　推理系统

1. Kwon et al., 2023. **Efficient Memory Management for Large Language Model Serving with PagedAttention** (vLLM). https://arxiv.org/abs/2309.06180
2. Leviathan, Kalman & Matias, 2022/2023. **Fast Inference from Transformers via Speculative Decoding**. https://arxiv.org/abs/2211.17192
3. Chen et al., 2023. **Accelerating Large Language Model Decoding with Speculative Sampling**. https://arxiv.org/abs/2302.01318
4. Dao et al., 2022. **FlashAttention**. https://arxiv.org/abs/2205.14135

推理优化要分 prefill 与 decode。前者更像大矩阵计算，后者往往受 KV Cache、memory bandwidth、batching 和自回归依赖限制。

---

# 17　评测、数据污染与能力测量

1. Hendrycks et al., 2020/2021. **Measuring Massive Multitask Language Understanding** (MMLU). https://arxiv.org/abs/2009.03300
2. Srivastava et al., 2022. **Beyond the Imitation Game: Quantifying and Extrapolating the Capabilities of Language Models** (BIG-bench). https://arxiv.org/abs/2206.04615
3. Liang et al., 2022. **Holistic Evaluation of Language Models** (HELM). https://arxiv.org/abs/2211.09110
4. Cobbe et al., 2021. **Training Verifiers to Solve Math Word Problems** (GSM8K). https://arxiv.org/abs/2110.14168
5. Hendrycks et al., 2021. **Measuring Mathematical Problem Solving With the MATH Dataset**. https://arxiv.org/abs/2103.03874
6. Jimenez et al., 2023/2024. **SWE-bench: Can Language Models Resolve Real-World GitHub Issues?**. https://arxiv.org/abs/2310.06770

benchmark 分数必须同时检查：测试污染、prompt protocol、sampling budget、工具权限、test-time tokens、pass@k、judge model 和版本日期。

---

# 18　Tokenizer 与数据处理

1. Sennrich et al., 2016. **Neural Machine Translation of Rare Words with Subword Units** (BPE for NMT). https://arxiv.org/abs/1508.07909
2. Kudo & Richardson, 2018. **SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing**. https://arxiv.org/abs/1808.06226
3. OpenAI GPT-2 repository. https://github.com/openai/gpt-2

训练数据研究还应持续关注：deduplication、quality filtering、language balance、code/math mixture、synthetic data、contamination 与 licensing。它们常常比一次很小的架构修改更影响最终模型。

---

# 19　如何使用这个索引

不要按 100 篇论文从上到下硬啃。建议按教材章节进入：

```text
教材正文
  ↓ 遇到关键结论
打开对应原论文
  ↓
先看 abstract / architecture figure / method
  ↓
回教材继续建立上下文
  ↓
真正重要的论文再完整精读
  ↓
用代码或手算复现一个最小版本
```

第一轮必须精读的“主干论文”可以压缩为：

1. Attention Is All You Need
2. GPT-1
3. BERT
4. GPT-2
5. Scaling Laws
6. GPT-3
7. Chinchilla
8. InstructGPT
9. Chain-of-Thought
10. LLaMA
11. LoRA / QLoRA
12. FlashAttention
13. DPO
14. ReAct
15. Mixtral / DeepSeekMoE
16. DeepSeek-V2 / V3
17. DeepSeekMath
18. DeepSeek-R1
19. Qwen2.5 / Qwen3
20. vLLM / PagedAttention

读完这条主干，再扩展到具体模型族，学习效率远高于按厂商名称堆论文。

---

# 20　引用规范

教材正文建议统一采用：

```markdown
……GPT-3 系统展示了 large-scale autoregressive LM 的 few-shot / in-context behavior。[Brown et al., 2020](https://arxiv.org/abs/2005.14165)
```

对于厂商自报结果：

```markdown
官方技术报告称……[Company/Team, year](官方报告链接)
```

不要改写成无条件事实。

对于快速变化的产品能力：

```markdown
截至 2026-09-14，官方文档显示……
```

并保留日期。

对于重绘图：

```text
图 X：作者重绘；据 Vaswani et al. (2017) 与某模型技术报告整理。
```

这样可以让“历史事实”“论文结论”“厂商自报”“本书解释”和“作者推断”始终保持可区分。