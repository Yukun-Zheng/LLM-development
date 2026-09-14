# Paper Card：2017《Attention Is All You Need》

> **论文**：Vaswani et al., *Attention Is All You Need*  
> https://arxiv.org/abs/1706.03762  
> **会议**：NeurIPS 2017  
> **教材位置**：[`../教材-book/01-Transformer基础-foundations-transformer.md`](../教材-book/01-Transformer基础-foundations-transformer.md)  
> **我们的实现**：[`../代码-code/从零构建Astra与Codex级系统/src/astra_codex/model.py`](../代码-code/从零构建Astra与Codex级系统/src/astra_codex/model.py)  
> **最后核验日期**：2026-09-14

---

# 1　原始问题

2017 年主流 sequence transduction 模型仍大量依赖 recurrent 或 convolutional structure。RNN/LSTM 在时间维递归：

$$
h_t=f(h_{t-1},x_t),
$$

使同一序列内部的时间步存在显式串行依赖。Attention 在 Transformer 之前已经用于 encoder-decoder 对齐，但通常附着在 recurrent backbone 上。

论文要回答的是：

> **能否完全去掉 recurrence 与 convolution，仅依赖 attention 构建高质量 sequence transduction model？**

这比后来常见的“attention 能建模长距离依赖”表述更接近论文原始历史问题。

---

# 2　原始架构

原论文是 encoder-decoder Transformer，不是今天最常见的 decoder-only GPT。

```text
Source Tokens
    ↓
Embedding + Positional Encoding
    ↓
N × Encoder Layer
    ├─ Multi-Head Self-Attention
    └─ Position-wise FFN
          ↓
      Encoder Memory
          ↓
Target Tokens
    ↓
Embedding + Positional Encoding
    ↓
N × Decoder Layer
    ├─ Masked Multi-Head Self-Attention
    ├─ Encoder-Decoder Attention
    └─ Position-wise FFN
          ↓
Linear + Softmax
```

原始论文每个 sublayer 使用 residual connection + LayerNorm。

---

# 3　Scaled Dot-Product Attention

原公式：

$$
\mathrm{Attention}(Q,K,V)
=
\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

shape：

```text
Q: [T_q, d_k]
K: [T_k, d_k]
V: [T_k, d_v]

QK^T: [T_q, T_k]
Attention(Q,K,V): [T_q, d_v]
```

对于 batched multi-head implementation，我们教材扩展为：

```text
Q: [B, H, T_q, D_h]
K: [B, H, T_k, D_h]
V: [B, H, T_k, D_h]
```

但这后一个 shape 是现代实现表达，不是论文排版本身。

---

# 4　为什么除以 $\sqrt{d_k}$

论文解释：当 $d_k$ 较大时，dot product magnitude 增长，会把 softmax 推入梯度很小的区域，因此进行 scaling。

如果 query/key 分量均独立、均值 0、方差 1，则：

$$
q\cdot k=\sum_{i=1}^{d_k}q_i k_i,
$$

其方差约为：

$$
\mathrm{Var}(q\cdot k)=d_k.
$$

除以 $\sqrt{d_k}$ 后把量级重新稳定到 $O(1)$。

这里要区分：方差推导是对论文直觉的数学展开；论文原文并没有把整个训练稳定性问题归结成一个严格概率定理。

---

# 5　Multi-Head Attention

原公式：

$$
\mathrm{MultiHead}(Q,K,V)
=
\mathrm{Concat}(head_1,\ldots,head_h)W^O,
$$

其中：

$$
head_i=
\mathrm{Attention}(QW_i^Q,KW_i^K,VW_i^V).
$$

核心不是简单重复同一个 attention $h$ 次，而是每个 head 有独立 projection subspace。

---

# 6　Claim Ledger

## TRF-2017-C01

- **类型**：method
- **精确转述**：论文提出一种完全基于 attention、无需 recurrence 与 convolution 的 sequence transduction architecture。
- **原始证据**：Abstract、Section 1、Section 3。
- **适用条件**：指论文所定义 Transformer 架构，不意味着后来所有 attention-based model 都完全没有 convolution/recurrent component。
- **后续证据**：Transformer 成为后续 BERT、GPT、T5 等大量预训练模型的基础架构。
- **状态**：supported / historical foundation
- **教材位置**：教材 01。

## TRF-2017-C02

- **类型**：systems / architecture
- **精确转述**：去掉序列维 recurrent dependency，使训练中的 token representations 可以更高度并行计算。
- **原始证据**：Section 1 与 Section 4 complexity discussion。
- **适用条件**：self-attention 本身仍有 $T^2$ pairwise score structure；“可并行”不等于“计算量恒定”。
- **后续证据**：GPU/TPU 时代的大规模 Transformer training 广泛验证了并行训练优势。
- **状态**：supported with systems conditions
- **教材位置**：教材 01、09、16。

## TRF-2017-C03

- **类型**：empirical
- **精确转述**：论文报告 Transformer 在 WMT 2014 English-to-German / English-to-French translation 上获得当时有竞争力/领先的质量，并显著降低训练成本。
- **原始证据**：Table 2、Section 6。
- **适用条件**：原论文特定数据、训练 recipe、hardware 与 BLEU 设置。
- **后续证据**：后续 Transformer 系列广泛取代 recurrent NMT backbone，但不能把原 BLEU 数字跨现代设置直接比较。
- **状态**：historical empirical result
- **教材位置**：教材 01。

## TRF-2017-C04

- **类型**：architecture / complexity
- **精确转述**：self-attention 对任意两个位置的 path length 为常数级，而 recurrent layer 的 sequential path length 随 sequence length 增长。
- **原始证据**：Section 4 / Table 1。
- **适用条件**：指计算图中的 path length，不意味着 long-range information 一定能被训练过程无损利用。
- **后续证据**：大量长上下文工作表明，“理论 path length 短”与“真实长上下文利用可靠”是两个问题。
- **状态**：supported but often overinterpreted
- **教材位置**：教材 01、10、12。

---

# 7　原始实验设置：需要记住什么

原论文使用机器翻译任务而非通用 next-token pretraining。

代表配置包括 Transformer Base / Big；训练使用 Adam、warmup 与特定 learning-rate schedule，并使用 label smoothing。

因此不能把 2017 Transformer 的实验直接描述成：

> “证明 decoder-only next-token prediction 是最优语言模型。”

论文根本没有做这个实验。

---

# 8　后来 GPT 改了什么

Transformer 原版：

```text
Encoder
+
Decoder
+
Cross Attention
```

GPT 路线逐渐收敛到：

```text
Decoder-only
+
Causal Self-Attention
+
Next-token objective
```

所以今天写一个 GPT-style decoder block 时，我们是在继承 Transformer 的 attention / FFN / residual backbone，但**不是逐字复现 2017 原架构**。

---

# 9　我们的代码对应关系

`model.py` 当前实现的是现代 decoder-only variant：

```text
RMSNorm
→ GQA
→ RoPE
→ Causal Mask
→ Residual
→ RMSNorm
→ SwiGLU
→ Residual
```

它与 2017 原版不同：

| 机制 | 2017 原 Transformer | 当前我们的 runtime |
|---|---|---|
| 总体结构 | Encoder-Decoder | Decoder-only |
| Norm | LayerNorm | RMSNorm |
| Position | sinusoidal / learned positional embedding | RoPE |
| Attention heads | MHA | GQA |
| FFN activation | ReLU | SwiGLU |
| Objective | NMT sequence transduction | causal LM forward |

这个表非常重要：**“从零实现现代 Transformer”不等于“复刻 2017 Transformer config”。**

---

# 10　后续工作修正了哪些默认设计

Transformer 的主干保留，但大量组件被替换：

```text
Post-Norm → Pre-Norm
LayerNorm → RMSNorm
Sinusoidal Position → RoPE / ALiBi
MHA → MQA / GQA / MLA
ReLU FFN → GELU / SwiGLU
Dense FFN → MoE
naive attention kernel → FlashAttention
full attention everywhere → local / sparse / hybrid / SSM
```

所以 Transformer 更像一个不断演化的架构家族，而不是 2017 configuration 被原样放大到今天。

---

# 11　不能从原论文推出什么

不能推出：

1. Transformer 永远优于 RNN / SSM；
2. attention 的 $O(T^2)$ 成本不重要；
3. 短 path length 自动保证无限长记忆；
4. decoder-only 是论文证明的最佳架构；
5. sinusoidal positional encoding 是今天仍然最佳的位置表示；
6. 原论文 benchmark 可以直接与 2026 LLM benchmark 横向比较。

这些都是后来时代的不同问题。

---

# 12　我们自己的复现状态

| 项目 | 状态 | 证据 |
|---|---|---|
| Scaled dot-product attention | ✅ | `model.py` |
| Causal self-attention | ✅ | `model.py` |
| Multi-head family | ✅ GQA 特例/推广 | `GroupedQueryAttention` |
| KV incremental inference | ✅ | cache parity tests |
| 现代 Llama-family forward | ✅ | SmolLM2-135M real checkpoint logits parity |
| 原版 Encoder-Decoder Transformer | ☐ | 后续专门实验 |
| 原 WMT translation experiment | ☐ | 未复现 |

---

# 13　当前较稳健的理解

1. Transformer 的历史突破首先是**去 recurrence 的 attention-only sequence transduction architecture**。
2. Attention 允许直接 token-to-token interaction，但也引入 $T^2$ score structure 与推理 KV memory 问题。
3. 今天的 LLM 继承 Transformer 骨架，但 normalization、position、attention sharing、FFN、systems kernel 已发生巨大变化。
4. “Transformer 成功”既是算法问题，也是硬件并行与系统工程问题。
5. 真正理解 Transformer 的标准不是能画经典框图，而是能解释**矩阵、shape、mask、复杂度、训练并行、decode cache，以及现代变体为什么修改它**。