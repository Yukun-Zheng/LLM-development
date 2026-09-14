# 附录 B　从零实现 MiniGPT：把整本书压缩成一个可运行程序

> 对应代码：[`code/minigpt.py`](../code/minigpt.py)  
> 目标：不用 Transformers、Trainer、DeepSpeed 或 vLLM，只依赖 PyTorch，把 decoder-only Transformer 的训练与自回归推理完整串起来。

---

## B.1　为什么一定要自己写一次

如果只会调用 `AutoModelForCausalLM.from_pretrained(...)`，你很容易把“模型”理解成一个黑盒。本附录希望把黑盒拆成一条明确的数据流：

```text
UTF-8 文本
  ↓ tokenizer
整数 token ids                         [B, T]
  ↓ embedding
连续向量 X                            [B, T, C]
  ↓ N 个 Transformer block
隐藏状态 H                            [B, T, C]
  ↓ LM head
logits                                [B, T, V]
  ↓ 与下一个 token 对齐
cross entropy
  ↓ backward
参数梯度
  ↓ optimizer.step()
更新参数
```

这里：

- `B`：batch size；
- `T`：sequence length；
- `C`：hidden dimension / embedding dimension；
- `V`：vocabulary size。

真正的大模型会把每一层都换成更复杂、更高效的工程实现，但最基本的数据流并没有变。

---

## B.2　Tokenizer：先把“语言”变成整数

教材代码故意使用最简单的字符级 tokenizer：

```python
self.itos = sorted(set(text))
self.stoi = {ch: i for i, ch in enumerate(self.itos)}
```

于是：

$$
\text{text}\rightarrow (x_1,x_2,\ldots,x_T),\qquad x_t\in\{0,\ldots,V-1\}.
$$

真实 LLM 通常使用 BPE、SentencePiece、Unigram 或其变体，因为字符级 token 太细，序列会很长；单词级 token 又难以处理开放词表。GPT-2 使用 byte-level BPE，而后续大量模型继续沿用 subword/byte-aware tokenizer 的路线。

来源：

- Sennrich et al., 2016, *Neural Machine Translation of Rare Words with Subword Units*: https://arxiv.org/abs/1508.07909
- Kudo & Richardson, 2018, *SentencePiece*: https://arxiv.org/abs/1808.06226
- OpenAI GPT-2 repository / tokenizer implementation: https://github.com/openai/gpt-2

**实验**：把字符 tokenizer 换成 BPE 后，在相同 `block_size` 下比较“可覆盖的原始文本长度”。这会直接帮助理解 tokenizer 为什么会影响上下文的有效长度与推理成本。

---

## B.3　Embedding：离散 ID 如何进入神经网络

输入 token ids：

$$
X_{id}\in\mathbb{N}^{B\times T}.
$$

词嵌入矩阵：

$$
E\in\mathbb{R}^{V\times C}.
$$

查表后：

$$
X=E[X_{id}]\in\mathbb{R}^{B\times T\times C}.
$$

代码同时使用 learned positional embedding，使模型知道 token 的绝对位置。现代 LLM 更常使用 RoPE；这里保留 learned position embedding 是为了让“内容表示”和“位置信息”分开看得更清楚。

关于位置编码与 RoPE，见正文 Part IV。原始来源：

- Vaswani et al., 2017: https://arxiv.org/abs/1706.03762
- Su et al., 2021, RoFormer / RoPE: https://arxiv.org/abs/2104.09864

---

## B.4　RMSNorm：为什么归一化出现在每个 Block 前

代码实现：

$$
\operatorname{RMSNorm}(x)
=
\gamma\odot
\frac{x}{\sqrt{\frac{1}{C}\sum_{i=1}^{C}x_i^2+\epsilon}}.
$$

与 LayerNorm 相比，RMSNorm 不减均值，只按 root mean square 缩放。教材采用 Pre-Norm：

$$
x' = x + \operatorname{Attention}(\operatorname{Norm}(x)),
$$

$$
y = x' + \operatorname{MLP}(\operatorname{Norm}(x')).
$$

这样你能清楚看到 residual stream 是贯穿整个 Transformer 的“主干状态”，attention 与 MLP 更像不断写回主干的两个计算模块。

来源：

- Zhang & Sennrich, 2019, *Root Mean Square Layer Normalization*: https://arxiv.org/abs/1910.07467
- Xiong et al., 2020, Pre-LN analysis: https://arxiv.org/abs/2002.04745

---

## B.5　Self-Attention：最重要的 shape 变化

设输入：

$$
X\in\mathbb{R}^{B\times T\times C}.
$$

教材代码用一个线性层一次算出 Q/K/V：

$$
[Q,K,V]=XW_{QKV}.
$$

切分并 reshape：

```text
[B, T, C]
   ↓ linear
[B, T, 3C]
   ↓ chunk
3 × [B, T, C]
   ↓ split heads
3 × [B, h, T, d_h]
```

其中：

$$
d_h=C/h.
$$

注意力分数：

$$
S=\frac{QK^\top}{\sqrt{d_h}}
\in\mathbb{R}^{B\times h\times T\times T}.
$$

然后施加 causal mask：位置 $t$ 不能读取 $t+1,t+2,\ldots$。

$$
A=\operatorname{softmax}(S+M),
$$

$$
O=AV.
$$

整个最关键的数据流是：

```text
Q [B,h,T,dh]
K [B,h,T,dh]
      │
      └── QKᵀ ──> scores [B,h,T,T]
                         ↓ causal mask
                         ↓ softmax
                      A [B,h,T,T]
                         │
V [B,h,T,dh] ────────────┘
                         ↓
                    [B,h,T,dh]
                         ↓ merge heads
                    [B,T,C]
```

来源：Vaswani et al., 2017: https://arxiv.org/abs/1706.03762

### 为什么除以 $\sqrt{d_h}$

如果 Q、K 各维近似独立且方差为 1，则点积的方差会随 $d_h$ 增长。直接把大方差 score 输入 softmax 容易进入过度尖锐区间。缩放项使 score 的尺度更稳定。

---

## B.6　Causal Mask：删掉它会发生什么

语言模型训练的目标是：

$$
p(x_1,\ldots,x_T)=\prod_{t=1}^{T}p(x_t\mid x_{<t}).
$$

因此预测第 $t$ 个位置时，模型只能看到过去。

教材中的 mask 是下三角矩阵：

$$
M=
\begin{bmatrix}
0&-\infty&-\infty&\cdots\\
0&0&-\infty&\cdots\\
0&0&0&\cdots\\
\vdots&\vdots&\vdots&\ddots
\end{bmatrix}.
$$

**必须亲自做的实验**：删除 causal mask 后训练。loss 往往会异常好看，因为网络可以偷看未来 token，但这样的模型不能作为自回归生成模型正常工作。

这也是理解“训练指标低 ≠ 任务定义正确”的最好例子之一。

---

## B.7　SwiGLU：Attention 不是 Transformer 的全部

每个 Transformer block 还有逐 token 的前馈网络。教材采用：

$$
\operatorname{SwiGLU}(x)
=
W_{down}\left[\operatorname{SiLU}(W_gx)\odot(W_ux)\right].
$$

Attention 负责 token 间的信息交换；MLP/FFN 在每个位置上做非线性变换。二者在计算图中的角色不同。

来源：

- Shazeer, 2020, *GLU Variants Improve Transformer*: https://arxiv.org/abs/2002.05202
- LLaMA, 2023（采用 SwiGLU 等现代 recipe）: https://arxiv.org/abs/2302.13971

---

## B.8　从隐藏状态到下一个 token

最后一层隐藏状态：

$$
H\in\mathbb{R}^{B\times T\times C}.
$$

语言模型头：

$$
Z=HW_{vocab}\in\mathbb{R}^{B\times T\times V}.
$$

对每个位置：

$$
p(x_{t+1}\mid x_{\le t})=\operatorname{softmax}(Z_t).
$$

训练标签就是输入序列向右平移一位：

```text
input :  [x0, x1, x2, x3]
target:  [x1, x2, x3, x4]
```

loss：

$$
\mathcal L
=-\frac{1}{BT}\sum_{b,t}
\log p_\theta(x_{b,t+1}\mid x_{b,\le t}).
$$

这就是“next-token prediction”的数学核心。

---

## B.9　训练为什么能并行，而生成为什么不能完全并行

训练时，整段真实文本已知。因果 mask 保证每个位置看不到未来，所以一次 forward 就能同时得到所有位置的 next-token loss：

```text
训练：
[x0 x1 x2 x3 x4]
 ↓  ↓  ↓  ↓
p1 p2 p3 p4 p5
```

生成时，下一个输入本身就是上一步的输出：

```text
prompt
  ↓
x1
  ↓ append
x2
  ↓ append
x3
```

因此 token 维度存在自回归依赖。这也是 KV Cache、continuous batching、speculative decoding 等推理优化为何重要的根本背景。

---

## B.10　反向传播到底更新了什么

一次训练 step：

```python
logits, loss = model(x, y)
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
```

`loss.backward()` 沿整张计算图计算：

$$
\frac{\partial\mathcal L}{\partial W_Q},
\frac{\partial\mathcal L}{\partial W_K},
\frac{\partial\mathcal L}{\partial W_V},
\frac{\partial\mathcal L}{\partial W_{MLP}},
\frac{\partial\mathcal L}{\partial E},\ldots
$$

注意：模型并没有被显式告知“语法”“事实”“推理规则”分别存在哪里。所有统计结构都通过优化 next-token likelihood 间接进入参数。

这正是 GPT 路线最重要、也最值得持续研究的问题：**为什么如此朴素的目标函数，在足够数据、模型和优化规模下，会诱导出复杂的内部表示与行为？**

---

## B.11　运行教材代码

```bash
python code/minigpt.py
```

使用自己的语料：

```bash
python code/minigpt.py --text path/to/corpus.txt --steps 5000
```

代码只要求 PyTorch。它是教学实现，不追求 production throughput。

建议记录：

- 参数量；
- train/validation loss；
- 不同 `block_size`；
- 不同 `n_layer`；
- 不同 `n_head`；
- 不同 `n_embd`；
- 生成样本；
- wall-clock time 与显存。

---

## B.12　六个必须做的消融实验

| 实验 | 改动 | 你应该观察什么 |
|---|---|---|
| A | 删除 causal mask | loss 是否虚假下降、生成是否失效 |
| B | 删除 positional information | 模型是否难以利用顺序 |
| C | `n_head=1` vs 多头 | 表达/训练差异与参数量变化 |
| D | SwiGLU → ReLU FFN | loss、稳定性与速度差异 |
| E | 1 层 → 2/4/8 层 | capacity 与优化难度 |
| F | context 32 → 64 → 128 | 可利用历史与计算量变化 |

其中 attention 的朴素复杂度随序列长度近似：

$$
O(T^2C).
$$

把 `T` 翻倍时观察显存和运行时间，是理解长上下文系统问题最直观的实验。

---

## B.13　怎样一步步把 MiniGPT 变成“现代 LLM”

建议按以下顺序升级，而不是一次性复制某个大模型仓库：

```text
MiniGPT
 ↓
BPE / SentencePiece tokenizer
 ↓
RoPE
 ↓
weight tying
 ↓
GQA / MQA
 ↓
PyTorch SDPA / FlashAttention
 ↓
KV Cache
 ↓
mixed precision
 ↓
gradient accumulation
 ↓
DDP / FSDP
 ↓
LoRA
 ↓
SFT
 ↓
preference optimization
 ↓
tool-use / agent loop
```

每加一项，都要回答三个问题：

1. **数学函数变了吗？**
2. **参数/数据流变了吗？**
3. **还是仅仅系统实现更高效？**

这三个问题贯穿整本教材。

---

## B.14　最低验收标准

完成本附录后，应能不查资料解释：

- `[B,T,C] → [B,h,T,d_h] → [B,h,T,T] → [B,T,C]` 的完整 attention shape；
- causal mask 为什么只用于 decoder 自回归方向；
- 为什么训练能并行预测所有位置；
- 为什么生成仍然具有 token-by-token 依赖；
- cross entropy 如何对应最大似然训练；
- residual stream、attention、MLP、normalization 分别扮演什么角色；
- 哪些现代技术改变算法，哪些主要改变效率。

做到这些之后，再去读 LLaMA、Qwen、DeepSeek、GPT 系列的 architecture diagram，才不会只看到一堆模块名称。