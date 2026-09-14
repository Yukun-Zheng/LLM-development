# Lab 00　从字符串到 Logits：把语言模型真的走一遍

> 对应源码：`src/astra_codex/tokenizer.py`、`config.py`、`model.py`

这不是“调用一个模型”的实验，而是把现代 decoder-only LLM 的前向数据流逐层展开。

## 1　字符串不是神经网络的输入

文本首先要变成离散整数：

```text
"robot"
  ↓ UTF-8
[114, 111, 98, 111, 116]
  ↓ tokenizer / BPE
[token_id_0, token_id_1, ...]
```

本项目先实现 `ByteTokenizer`，保证任意 UTF-8 文本无损可逆；随后实现一个最小 byte-level BPE，使“为什么需要 subword tokenizer”能够通过真实 token 数量观察，而不是停留在名词解释。

原始资料：

- Sennrich et al., 2015/2016, BPE for NMT: https://arxiv.org/abs/1508.07909
- Kudo & Richardson, 2018, SentencePiece: https://arxiv.org/abs/1808.06226

## 2　Embedding：查表，不是魔法

输入：

```text
input_ids [B,T]
```

Embedding 权重：

$$
E\in\mathbb{R}^{V\times C}.
$$

查表以后：

$$
X=E[\text{input\_ids}]\in\mathbb{R}^{B\times T\times C}.
$$

代码对应 `DecoderOnlyTransformer.embed_tokens`。

## 3　RMSNorm

源码不是调用高层 Transformer block，而是直接实现：

$$
\mathrm{RMS}(x)=\sqrt{\frac{1}{d}\sum_i x_i^2+\epsilon},
$$

$$
\mathrm{RMSNorm}(x)=g\odot\frac{x}{\mathrm{RMS}(x)}.
$$

原始资料：Zhang & Sennrich, 2019, https://arxiv.org/abs/1910.07467

## 4　GQA：Q 头多，KV 头少

源码中：

```text
Q [B,Hq,T,D]
K [B,Hkv,T,D]
V [B,Hkv,T,D]
```

若 $H_q/H_{kv}=r$，每个 KV 头被 $r$ 个 query heads 共享。教学实现用 `repeat_interleave` 显式展开，工业实现可以避免物理复制。

原始资料：Ainslie et al., 2023, https://arxiv.org/abs/2305.13245

## 5　RoPE

RoPE 不再把位置向量简单加到 embedding 上，而是在 Q/K 子空间中施加位置相关旋转。代码对应 `RotaryEmbedding`。

原始资料：Su et al., 2021, https://arxiv.org/abs/2104.09864

## 6　Attention

教学实现明确写出：

```python
scores = q @ k.transpose(-2, -1)
scores = scores / sqrt(head_dim)
scores = scores.masked_fill(~causal_mask, -inf)
probs = softmax(scores)
context = probs @ v
```

因此你可以逐行对应：

$$
\mathrm{Attention}(Q,K,V)=
\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V.
$$

原始资料：Vaswani et al., 2017, https://arxiv.org/abs/1706.03762

## 7　SwiGLU

$$
\mathrm{SwiGLU}(x)=
W_d\left(\mathrm{SiLU}(W_gx)\odot W_ux\right).
$$

原始资料：Shazeer, 2020, https://arxiv.org/abs/2002.05202

## 8　LM Head

最终 hidden state：

```text
[B,T,C]
```

乘词表投影得到：

```text
logits [B,T,V]
```

logits 还不是概率；softmax / sampling 属于推理策略层。

## 9　实验

运行：

```bash
pip install -e '.[dev]'
pytest tests/test_model_and_cache.py -q
```

第一项验收不是“模型生成得像不像 ChatGPT”，而是所有 tensor contract 正确、不同路径算出的结果一致。
