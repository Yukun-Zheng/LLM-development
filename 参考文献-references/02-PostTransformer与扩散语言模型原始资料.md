# Post-Transformer 与 Diffusion Language Models 原始资料索引

> **用途**：为第十九篇与第二十篇提供一手证据地图。这里只列原始论文、作者/官方代码与必要的原始技术材料；二手解读不作为核心机制证据。

---

# A　State Space Models

## A1　S4

- Gu, Goel & Ré (2021), **Efficiently Modeling Long Sequences with Structured State Spaces**  
  https://arxiv.org/abs/2111.00396
- 作者代码 / State Spaces 项目：  
  https://github.com/state-spaces/s4

研究重点：

- continuous-time SSM 如何离散化；
- structured state matrix；
- convolution form 与 recurrent form；
- 长序列计算复杂度。

## A2　Mamba

- Gu & Dao (2023), **Mamba: Linear-Time Sequence Modeling with Selective State Spaces**  
  https://arxiv.org/abs/2312.00752
- 官方代码：  
  https://github.com/state-spaces/mamba

研究重点：

- input-dependent selective parameters；
- selective scan；
- hardware-aware algorithm；
- training parallelism 与 recurrent decode 的关系。

## A3　Mamba-2 / Structured State Space Duality

- Dao & Gu (2024), **Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality**  
  https://arxiv.org/abs/2405.21060
- 实现仍以 State Spaces / Mamba 官方仓库为主要入口：  
  https://github.com/state-spaces/mamba

教材中不能把标题简化成“Transformer 就是 SSM”；必须回到论文中 SSD 对特定结构化序列变换的精确定义。

---

# B　Recurrent / RNN-like Language Models

## B1　RWKV

- Peng et al. (2023), **RWKV: Reinventing RNNs for the Transformer Era**  
  https://arxiv.org/abs/2305.13048
- 官方项目：  
  https://github.com/BlinkDL/RWKV-LM

研究重点：

- parallel training representation；
- recurrent inference state；
- time mixing / channel mixing；
- fixed recurrent state 与 KV cache 的系统差异。

## B2　xLSTM

- Beck et al. (2024), **xLSTM: Extended Long Short-Term Memory**  
  https://arxiv.org/abs/2405.04517
- 官方代码：  
  https://github.com/NX-AI/xlstm

研究重点：exponential gating、sLSTM、mLSTM、matrix memory、并行性限制。

---

# C　Linear / Efficient Attention

- Katharopoulos et al. (2020), **Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention**  
  https://arxiv.org/abs/2006.16236

研究时必须区分：

```text
exact attention kernel optimization
vs
changing the mathematical attention function
```

FlashAttention 属于前者；很多 linear attention 方法属于后者。

---

# D　Test-Time Neural Memory

- Behrouz et al. (2025), **Titans: Learning to Memorize at Test Time**  
  https://arxiv.org/abs/2501.00663

研究重点：

- short-term attention memory；
- learnable long-term neural memory；
- test-time update；
- neural memory 与外部 Agent memory 的边界。

教材明确区分：

```text
模型内部 neural state / memory
!=
Agent 外部 event log / database / retrieval memory
```

---

# E　Diffusion Language Models 前史

## E1　BERT / Masked Language Modeling

- Devlin et al. (2018), **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**  
  https://arxiv.org/abs/1810.04805

BERT 是理解 masked conditioning 的重要前史，但不是完整 iterative diffusion generation algorithm。

## E2　离散扩散基础

- Austin et al. (2021), **Structured Denoising Diffusion Models in Discrete State-Spaces**  
  https://arxiv.org/abs/2107.03006

研究重点：离散状态空间的 forward corruption 与 reverse denoising 如何定义。

- Hoogeboom et al. (2021), **Argmax Flows and Multinomial Diffusion: Learning Categorical Distributions**  
  https://arxiv.org/abs/2102.05379

---

# F　Large Language Diffusion Models

## F1　LLaDA

- Nie et al. (2025), **Large Language Diffusion Models**  
  https://arxiv.org/abs/2502.09992

研究时要检查：

- pretraining corruption schedule；
- masked diffusion objective；
- instruction tuning；
- denoising schedule；
- confidence / remasking strategy；
- 与 autoregressive baseline 是否控制参数、数据与 compute。

教材不把“多个 token 并行更新”直接等价成“wall-clock 必然更快”。

---

# G　必须自己复现的最小实验

## G1　SSM

```text
same input
→ recurrent form
→ matrix / convolution form
→ compare numerical output
```

## G2　Linear Attention

```text
explicit pairwise form
vs
reassociated state-summary form
```

在数学条件满足时做 parity。

## G3　Tiny Diffusion LM

```text
clean sequence
→ corruption
→ masked loss
→ iterative denoising
→ confidence remasking
```

## G4　AR vs Diffusion

控制 backbone 和 toy data，比较：

- objective；
- number of forwards；
- wall-clock；
- sequence consistency；
- ability to revise earlier positions。

---

# H　对应教材

- [`../教材-book/19-Post-Transformer架构-post-transformer.md`](../教材-book/19-Post-Transformer架构-post-transformer.md)
- [`../教材-book/20-扩散语言模型-diffusion-language-models.md`](../教材-book/20-扩散语言模型-diffusion-language-models.md)

后续若新增模型，只在确认原始论文/官方代码和版本边界后加入本索引。