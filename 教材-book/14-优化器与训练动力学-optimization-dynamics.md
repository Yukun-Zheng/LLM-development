# 第十四篇（Part XIV）　优化器、数值精度与训练动力学：为什么同一个架构能训成完全不同的模型

> **本章主线**：模型架构只定义了函数族，真正把随机初始化变成可用大模型的是优化过程。学习率、warmup、AdamW、梯度裁剪、batch size、混合精度、参数初始化、μP 等都直接决定训练是否稳定、是否高效以及 scaling 是否成立。

---

# 1　“架构相同”不等于“训练相同”

设参数为 $\theta$，训练目标为：

$$
\min_\theta \mathcal L(\theta).
$$

最朴素的 SGD：

$$
\theta_{t+1}=\theta_t-\eta g_t,
$$

其中：

$$
g_t=\nabla_\theta \mathcal L_t(\theta_t).
$$

但真实 LLM 训练里，更新还受到：

- optimizer state；
- learning-rate schedule；
- weight decay；
- gradient clipping；
- normalization；
- mixed precision；
- loss scaling；
- batch size；
- distributed reduction；
- initialization；
- sequence length；
- data order；

共同影响。

因此“我们复现了架构”远远不等于“我们复现了训练”。

---

# 2　Adam：为什么成为 Transformer 训练的默认起点

Kingma & Ba (2014) 提出的 Adam 同时维护一阶与二阶矩估计。[原论文](https://arxiv.org/abs/1412.6980)

一阶动量：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t.
$$

二阶矩：

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
$$

偏差修正：

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t}.
$$

更新：

$$
\theta_{t+1}
=
\theta_t-
\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

直觉上，Adam 会让不同参数维度根据历史梯度尺度自动调整步长。

这对深层 Transformer 这种各层、各参数组梯度尺度差异显著的网络非常实用。

---

# 3　AdamW：weight decay 为什么要与梯度更新解耦

Loshchilov & Hutter 提出的 AdamW 关键点是：对于 Adam 这类 adaptive optimizer，L2 regularization 与真正的 weight decay 不再等价。[原论文](https://arxiv.org/abs/1711.05101)

AdamW 可以写成：

$$
\theta_{t+1}
=
(1-\eta\lambda)\theta_t
-
\eta\cdot \mathrm{AdamUpdate}(g_t).
$$

即先明确衰减参数，再做 adaptive gradient update。

现代 LLM recipe 中经常还会对参数分组：

```text
weight matrices      → weight decay
bias                  → often no decay
norm scale parameters → often no decay
```

但具体规则必须回到各模型的官方 config / training code，不能把社区惯例当成 universal law。

---

# 4　Warmup：为什么训练开头不能直接全速跑

Transformer 训练常在最初若干 steps 逐渐升高 learning rate：

$$
\eta_t
=
\eta_{max}\cdot\frac{t}{T_{warmup}},
\qquad t\le T_{warmup}.
$$

原因不是“warmup 是 Transformer 的数学组成部分”，而是训练初期：

- optimizer moments 尚未稳定；
- activation / gradient statistics 尚未进入稳定区；
- 大 learning rate 更容易引起 loss spike。

原 Transformer 使用了其特定 schedule；后来的 LLM 则发展出 cosine decay、linear decay、constant-with-warmup 等多种 recipe。

写教材时必须区分：

> **原论文 schedule** 与 **后来业界常用 schedule**。

---

# 5　Cosine decay、linear decay 与训练末期

一个常见 cosine schedule：

$$
\eta_t
=
\eta_{min}
+
\frac12(\eta_{max}-\eta_{min})
\left(1+\cos\frac{\pi(t-T_w)}{T-T_w}\right).
$$

但不同 schedule 的优劣并不能脱离：

- token budget；
- batch size；
- model size；
- data mixture；
- optimizer；

孤立讨论。

训练 recipe 的很多“经验结论”只在特定 scale 下成立。

---

# 6　Gradient clipping：限制的是更新风险，不是 loss

常见 global norm clipping：

$$
g\leftarrow g\cdot
\min\left(1,\frac{c}{\lVert g\rVert_2}\right).
$$

当梯度 norm 超过阈值 $c$ 时进行缩放。

它的意义是避免偶发异常 batch 或数值不稳定把参数一次推得过远。

但 clipping 过强会把大量正常梯度都压平，改变优化动力学。

因此训练监控至少应该记录：

```text
loss
learning rate
grad norm
clip fraction
parameter norm
optimizer moments
```

只盯 loss 是不够的。

---

# 7　Batch size：更大并不总是更好

假设 global batch 含 $B$ 个序列或 token blocks。

增大 batch：

- 梯度噪声下降；
- GPU 利用率可能提高；
- data parallel 更容易扩展；

但也可能：

- 降低 update frequency；
- 改变 optimization noise；
- 需要同步调整 learning rate；
- 导致 generalization / convergence 行为变化。

LAMB 等工作专门研究大 batch 训练。[You et al., 2019](https://arxiv.org/abs/1904.00962)

但 LLM 的最优 batch 不能从 BERT 的一个结果直接照搬。

---

# 8　Gradient accumulation：显存里的 batch 与优化器看到的 batch

如果单 GPU 每次只能放 $b$ 个样本，累积 $K$ 次梯度：

$$
B_{effective}=b\times K\times N_{data\ parallel}.
$$

需要区分：

- micro batch；
- local batch；
- global batch；
- tokens per update。

很多论文只写“batch size”会导致复现误解。

---

# 9　FP32 为什么太贵

FP32 每个标量 4 bytes。

一个仅参数就有 $N$ 个参数的模型，纯权重内存约：

$$
M_{weight}=4N\ \text{bytes}.
$$

训练时还需要：

- gradients；
- optimizer moments；
- activations；
- temporary buffers。

所以训练显存远大于“参数量 × 4 bytes”。

---

# 10　Mixed Precision Training

Micikevicius et al. 系统讨论了 mixed precision training。[原论文](https://arxiv.org/abs/1710.03740)

核心思想：

```text
部分计算：FP16/BF16
部分 master states：更高精度
必要时 loss scaling
```

目标是：

- 提高 tensor core throughput；
- 降低 memory footprint；
- 尽量保持优化稳定性。

---

# 11　FP16 与 BF16 的区别为什么重要

FP16 与 BF16 都是 16-bit，但 exponent / mantissa 分配不同。

BF16 保留与 FP32 相近的 exponent range，因此对深度学习训练中的动态范围通常更友好。[Kalamkar et al., 2019](https://arxiv.org/abs/1905.12322)

非常粗略地：

```text
FP16  → 更多 mantissa precision，较小 exponent range
BF16  → 更大 exponent range，较少 mantissa precision
```

因此大模型预训练大量采用 BF16。

---

# 12　Loss scaling

FP16 下，小梯度可能 underflow。

一种技术是把 loss 乘以 $S$：

$$
\tilde{\mathcal L}=S\mathcal L.
$$

反向后：

$$
\tilde g=Sg.
$$

optimizer update 前再除回 $S$。

BF16 因 exponent range 更大，对 loss scaling 的依赖通常更小。

---

# 13　初始化：训练开始前，信号就已经被设计

深层网络初始化的目标通常包括：

- activation variance 不随深度爆炸；
- gradient variance 不随深度消失；
- residual branch 尺度合理。

Transformer 不同家族会采用不同初始化策略与 residual scaling。

不能只看到：

```python
nn.Linear(...)
```

就认为所有模型的初始化等价。

---

# 14　Pre-Norm 为什么成为大模型常见结构

原始 Transformer 采用 post-norm 形式；后续大量工作发现 pre-norm 对深层 Transformer 优化通常更稳定。

Pre-Norm：

$$
x_{l+1}=x_l+F(\mathrm{Norm}(x_l)).
$$

Post-Norm：

$$
x_{l+1}=\mathrm{Norm}(x_l+F(x_l)).
$$

Xiong et al. (2020) 对这一差异做过理论与实验分析。[原论文](https://arxiv.org/abs/2002.04745)

---

# 15　Loss spike 不是“小抖动”

大规模训练可能出现：

```text
正常下降
   ↓
突然 loss spike
   ↓
grad norm 爆炸
   ↓
optimizer state 被污染
   ↓
长时间恢复甚至训练报废
```

可能原因包括：

- 特殊 batch；
- 数值 overflow；
- learning rate；
- distributed bug；
- unstable attention logits；
- bad data shard；
- optimizer state corruption。

真正的大模型训练工程必须能 checkpoint、detect、rollback、skip data、恢复 optimizer state。

---

# 16　μP：超参数能否跨模型规模迁移

当模型从几十 M 扩大到几十 B，直接重新调所有 hyperparameters 极其昂贵。

μP（Maximal Update Parameterization）研究如何通过特定参数化，让小模型上的超参数更可靠地迁移到大模型。[Yang et al., 2022](https://arxiv.org/abs/2203.03466)

这背后的研究问题非常重要：

> **怎样让 scaling 不意味着每次都重新做一次昂贵的超参数搜索？**

---

# 17　训练动力学应该画什么图

一个严谨训练 run 至少应该长期记录：

```text
train loss
validation loss
learning rate
grad norm
parameter norm
activation statistics
throughput
MFU / hardware utilization
memory
communication time
NaN / Inf counts
```

如果做 MoE，还应增加：

```text
expert load
router entropy
capacity overflow
auxiliary loss
```

如果做 reasoning RL，还需要：

```text
reward
KL
response length
entropy
advantage statistics
verifier pass rate
```

---

# 18　优化器比较必须控制什么

“Optimizer A 比 AdamW 强”这样的结论至少应控制：

- model；
- initialization；
- data order；
- token budget；
- batch size；
- LR sweep；
- weight decay；
- warmup；
- compute budget；
- wall-clock；
- memory overhead。

否则可能只是 A 的默认 LR 比 B 更合适。

---

# 19　原始资料包

- Adam  
  https://arxiv.org/abs/1412.6980
- AdamW  
  https://arxiv.org/abs/1711.05101
- LAMB  
  https://arxiv.org/abs/1904.00962
- Mixed Precision Training  
  https://arxiv.org/abs/1710.03740
- BFLOAT16  
  https://arxiv.org/abs/1905.12322
- Pre-Norm / Post-Norm analysis  
  https://arxiv.org/abs/2002.04745
- μP  
  https://arxiv.org/abs/2203.03466

---

# 20　最小实验

推荐做五组实验：

1. Adam 与 AdamW 在相同 LR sweep 下比较；
2. 无 warmup / 短 warmup / 长 warmup；
3. FP32、FP16、BF16 的 loss 与 overflow 行为；
4. 不同 global batch size，在固定 token budget 下比较；
5. 记录 grad norm 与 loss spike，做一个自动 rollback toy trainer。

当你能看懂这些训练曲线时，才算真正开始理解：**LLM 不是被“架构”训练出来的，而是被完整 optimization system 训练出来的。**
