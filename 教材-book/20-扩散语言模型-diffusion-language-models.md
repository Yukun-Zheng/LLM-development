# 第二十篇（Part XX）　Diffusion Language Models：从自回归因子分解走向并行去噪生成

> **本章主线**：现代大语言模型长期默认使用自回归分解，但这并不是语言建模的唯一方式。扩散语言模型（Diffusion Language Model, DLM）尝试通过逐步破坏与恢复离散 token 序列来学习生成分布，从而重新讨论双向上下文、并行解码、推理轨迹和 test-time compute。

---

# 1　先把自回归假设写清楚

经典 decoder-only LM 使用：

$$
p_\theta(x_{1:T})=\prod_{t=1}^{T}p_\theta(x_t\mid x_{\lt t}).
$$

这带来非常清晰的生成过程：

```text
x1
↓
x2 | x1
↓
x3 | x1,x2
↓
...
```

训练时可以 teacher forcing 并行计算所有位置的 next-token loss；推理时却必须沿时间维一个 token 接一个 token 生成。

因此：

> **训练并行，不代表生成并行。**

这正是非自回归语言建模重新被研究的重要原因之一。

---

# 2　什么是离散扩散语言模型

连续扩散模型通常定义 forward noising process 与 reverse denoising process。

语言 token 是离散变量，因此离散扩散需要定义：

$$
q(x_t\mid x_{t-1})
$$

如何逐步破坏 token，以及：

$$
p_\theta(x_{t-1}\mid x_t)
$$

如何逐步恢复。

语言领域一种直观形式是 **mask-based diffusion**：

```text
clean sequence
The cat sat on the mat
        ↓ corruption
The [MASK] sat [MASK] the mat
        ↓ denoise
The cat sat on the mat
```

与 BERT-style masked language modeling 的关键差别是：扩散模型不仅把 masking 当成一个训练辅助目标，而是把**多步 corruption → denoising**定义成完整生成过程。

---

# 3　LLaDA：从头训练大规模 Diffusion LM

代表性原始工作：

- Nie et al., **Large Language Diffusion Models**, 2025.  
  https://arxiv.org/abs/2502.09992

LLaDA 的重要性在于它不是只拿一个已经训练好的 AR 模型做局部改造，而是系统研究大规模 diffusion language model 的预训练与指令跟随。

研究它时应该分清：

1. forward corruption process；
2. denoising objective；
3. network 是否使用双向 attention；
4. generation schedule；
5. 每轮更新多少 token；
6. stopping condition；
7. 与 AR 模型比较时是否匹配参数、数据和计算预算。

不能把“并行生成”简单等价成“必然更快”。

---

# 4　Masked Diffusion 的一个最小形式

设序列：

$$
x=(x_1,\ldots,x_T).
$$

随机 mask 集合 $M$，得到 corrupted sequence：

$$
\tilde x_i=\begin{cases}
[\mathrm{MASK}], & i\in M,\\
x_i, & i\notin M.
\end{cases}
$$

模型根据可见 token 预测被 mask 的位置：

$$
\mathcal L=-\sum_{i\in M}\log p_\theta(x_i\mid \tilde x).
$$

与 AR loss 的差异非常直观：

```text
AR:
只允许从左边条件化

Masked Diffusion:
可以同时利用左右两侧已知 token
```

但真正 diffusion LM 还需要定义不同噪声水平/时间步下的训练分布，而不仅是固定比例 MLM。

---

# 5　为什么双向上下文可能重要

对于一个待填位置：

```text
The capital of France is [MASK] .
```

双向条件允许模型看到 mask 两边的信息。

对于代码：

```python
result = function([MASK])
assert result == 42
```

后文的 assertion 本身也可能约束缺失内容。

这给 diffusion LM 一个非常自然的“整体修订”视角：

> 不是永远把过去冻结，而是允许整段候选输出被多轮重写。

这与 Agent 中的 revise / verify / patch 有有趣的结构类比，但两者不能混为同一算法。

---

# 6　生成：从左到右变成“逐步去噪”

AR decoding：

```text
1 token → 2 tokens → 3 tokens → ... → T tokens
```

Diffusion decoding：

```text
[M][M][M][M][M]
       ↓
a[M][M]e[M]
       ↓
a b[M]e f
       ↓
a b c e f
       ↓
a b c d f
       ↓
a b c d e
```

这里必须研究两个维度：

1. **denoising steps 数量**；
2. **每一步决定哪些位置**。

所以总推理成本不能只按“token 数”衡量，而要看：

$$
\text{total forward FLOPs}
\approx
\text{number of denoise steps}\times\text{cost per forward}.
$$

如果每一步都对整段序列做 full forward，所谓并行 token 更新可能仍然非常昂贵。

---

# 7　Confidence-based Remasking

一种常见思路是：

1. 模型给所有 mask 位置预测分布；
2. 选择最有把握的位置提交；
3. 低置信度位置继续保持 mask；
4. 下一轮再预测。

概念上：

$$
c_i=\max_v p_\theta(x_i=v\mid \tilde x).
$$

然后按 $c_i$ 决定 commit / remask。

这使 decoding 变成：

> **模型不仅决定 token 是什么，还决定哪些 token 现在值得相信。**

这是一个很适合和 verifier / uncertainty 联系起来的思想。

---

# 8　Diffusion LM 与 Reasoning

Reasoning model 的一个传统形态是：

```text
prompt
→ autoregressive reasoning tokens
→ answer
```

Diffusion LM 则自然提出另一类问题：

> 推理过程是否必须严格左到右展开？

理论上，模型可以：

```text
initial rough solution
→ revise globally
→ repair contradictions
→ refine uncertain spans
→ final answer
```

但“能够全局改写”不自动等价于“更会推理”。必须通过受控实验验证：

- math；
- code；
- planning；
- long-form consistency；
- verifier-guided refinement。

---

# 9　与 BERT 的根本区别

很多人第一次看到 masked diffusion 会问：

> “这不就是 BERT 吗？”

不是。

BERT 原始目标主要用于学习 encoder representation：

$$
p(x_i\mid x_{\setminus i}).
$$

但没有把完整的 iterative denoising trajectory 定义成文本生成算法。

Diffusion LM 的关键是：

```text
corruption schedule
+ denoising model
+ iterative generation
+ decoding policy
```

形成完整生成模型。

BERT 原始资料：

- Devlin et al., 2018, https://arxiv.org/abs/1810.04805

---

# 10　与 Autoregressive LM 的统一对照

| 维度 | Autoregressive LM | Diffusion LM |
|---|---|---|
| 因子分解 | 左到右 | corruption / denoising |
| Attention | causal | 常可双向 |
| 训练目标 | next-token CE | denoising / masked objective |
| 推理状态 | prefix + KV Cache | 当前完整 noisy sequence |
| 单步更新 | 常 1 token | 可多个位置 |
| 历史 token | 一旦输出通常固定 | 可重新 mask / 修订 |
| 推理轮数 | token 数量 | denoising steps |
| Cache 机制 | KV Cache 成熟 | 与 schedule 强相关 |

这张表不是优劣排名，而是计算图差异。

---

# 11　真正的工程问题

Diffusion LM 想成为 production system，需要回答：

## 11.1　Cache 怎样做

AR 模型中，过去 prefix 固定，因此 K/V 可以缓存。

Diffusion 中，如果旧 token 可能被重新修改：

> 哪些 hidden/KV 还能复用？

这会直接影响真实延迟。

## 11.2　Dynamic batching 怎样做

不同请求可能处于不同 denoising step、不同 mask pattern。

Serving scheduler 不能简单照搬 AR continuous batching。

## 11.3　Streaming 怎样定义

AR 可以自然流式显示 token。

Diffusion 输出可能反复修订，所以 UI 应显示：

- provisional tokens？
- committed spans？
- confidence？

这是模型和产品接口共同决定的问题。

---

# 12　从零实现：Tiny Diffusion LM

最终代码需要增加：

```text
代码-code/扩散语言模型/
├── corruption.py
├── diffusion_model.py
├── scheduler.py
├── denoise.py
├── remask.py
└── compare_ar_vs_diffusion.py
```

## Lab 1：固定比例 Masked LM

先实现最简单：

```python
mask = torch.rand_like(input_ids.float()) < p
corrupted = input_ids.masked_fill(mask, mask_token_id)
```

只在 mask 位置计算 loss。

## Lab 2：噪声时间步

采样 $t$，让 mask ratio 与 $t$ 相关。

## Lab 3：迭代恢复

从全 mask 开始，多轮预测并提交高置信度 token。

## Lab 4：受控对比 AR

同一 tiny Transformer backbone：

```text
AR objective
vs
Diffusion objective
```

比较：

- loss；
- generation steps；
- wall-clock；
- sequence consistency；
- edit/revision ability。

---

# 13　教材必须防止的三个误区

### 误区 1：并行 token 更新 = 线性加速

错误。真正代价取决于每次 forward 范围和 denoising step 数量。

### 误区 2：Diffusion LM 已经证明全面优于 AR

错误。不同工作、规模、数据、计算预算和 benchmark 并不完全可控。

### 误区 3：非自回归就不需要推理系统优化

错误。它只是把系统瓶颈换了位置。

---

# 14　值得长期追踪的问题

1. Diffusion LM 在大规模 reasoning 上是否能形成与 AR 不同的优势？
2. 如何设计真正高效的 cache？
3. 是否能将 speculative / verifier ideas 与 diffusion schedule 结合？
4. 能否把不确定性直接用于 adaptive denoising compute？
5. Agent action generation 是否适合 diffusion 式全局修订？
6. 代码生成是否会受益于“后文约束前文”的双向修正？

如果这些问题被解决，未来语言模型的软件栈可能不再默认围绕“一个 token 接一个 token”设计。