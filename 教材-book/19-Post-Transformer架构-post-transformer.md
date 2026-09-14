# 第十九篇（Part XIX）　Post-Transformer：状态空间、循环模型、神经记忆与混合架构

> **本章主线**：Transformer 不是序列建模的逻辑终点。本章不以“谁替代 Transformer”作为营销式问题，而是研究：Attention 的二次复杂度、KV Cache、固定上下文窗口和显式 pairwise interaction 究竟带来了什么代价？SSM、RNN-like、linear/hybrid attention 与 test-time memory 分别改变了哪一种计算结构？

---

# 0　先定义“Post-Transformer”

这里的 Post-Transformer 不等于“完全没有 attention”。更准确的分类是：

```text
Sequence Model
├─ Full Attention
│  └─ Transformer
├─ Efficient / Structured Attention
│  ├─ local / sliding-window
│  ├─ sparse
│  ├─ linear
│  └─ hybrid
├─ State Space Model
│  ├─ S4
│  ├─ Mamba
│  └─ Mamba-2 / SSD
├─ Recurrent / RNN-like LM
│  ├─ RWKV
│  └─ xLSTM
└─ Learned Test-Time Memory
   └─ Titans 等
```

所以真正的问题不是名字，而是：

> **模型怎样把过去压缩成状态，怎样让当前 token 与历史交互，计算和内存复杂度如何随序列长度增长？**

---

# 1　Transformer 的两个核心系统代价

标准 causal self-attention 对长度为 $T$ 的序列形成：

$$
A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d}}+M\right),
$$

注意力矩阵规模为：

$$
T\times T.
$$

训练时最直接的 pairwise interaction 具有 $O(T^2)$ 结构；自回归推理虽然每步只产生一个 query，但为了访问历史，需要维护不断增长的 KV Cache。

简化地说：

$$
\text{KV memory}\propto L\cdot T\cdot H_{kv}\cdot d_h.
$$

因此，Post-Transformer 的很多工作都在攻击以下至少一个问题：

1. 二次 attention 计算；
2. 随序列增长的 KV Cache；
3. 固定 context window；
4. 长程信息是否必须通过显式 token-to-token interaction 保存。

---

# 2　从经典 State Space Model 到 S4

连续时间线性状态空间可写为：

$$
\dot h(t)=Ah(t)+Bx(t),
$$

$$
y(t)=Ch(t)+Dx(t).
$$

离散后得到：

$$
h_t=\bar A h_{t-1}+\bar Bx_t,
$$

$$
y_t=Ch_t+Dx_t.
$$

关键思想是：历史不必全部作为 token cache 保存，而可以被递归压缩进固定维状态 $h_t$。

S4 将结构化状态空间用于长序列建模，并发展出高效卷积/递归计算路线。

**原始资料**：

- Gu, Goel & Ré, **Efficiently Modeling Long Sequences with Structured State Spaces**, 2021.  
  https://arxiv.org/abs/2111.00396

本书后续实验应比较：

```text
Attention history = explicit K/V tensors
SSM history       = recurrent hidden state
```

这两种“记忆”不是同一对象。

---

# 3　Mamba：Selective State Space

Mamba 的关键变化不是简单“把 Transformer 换成 RNN”，而是让状态空间参数对输入具有选择性，从而允许模型根据当前 token 决定怎样写入、遗忘和读取状态。

原始论文：

- Gu & Dao, **Mamba: Linear-Time Sequence Modeling with Selective State Spaces**, 2023.  
  https://arxiv.org/abs/2312.00752
- 官方实现：  
  https://github.com/state-spaces/mamba

概念上可写成输入依赖的：

$$
h_t=A_t h_{t-1}+B_t x_t,
$$

$$
y_t=C_t h_t.
$$

这里 $A_t,B_t,C_t$ 不再只是固定矩阵。

## 3.1　为什么 selective 很关键

如果所有输入都以同一种方式进入状态，模型难以选择：

- 哪些 token 应长期保留；
- 哪些只是局部噪声；
- 哪些信息需要覆盖旧状态。

Selective mechanism 把“内容相关的记忆控制”带回 SSM。

## 3.2　训练与推理的双重视角

一类重要系统思想是：

```text
训练：希望并行
推理：希望递归且状态固定
```

这正是 SSM 路线的重要吸引力。

---

# 4　Mamba-2：Structured State Space Duality

Mamba-2 对教材特别重要，因为它不是只提出一个新 block，而是从数学上建立 Attention 与一类 SSM 的联系。

原始论文：

- Dao & Gu, **Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality**, 2024.  
  https://arxiv.org/abs/2405.21060

核心概念：**Structured State Space Duality（SSD）**。

教材不应把结论粗暴写成“Transformer 就是 SSM”，而应准确写：论文对特定结构化矩阵与序列变换建立了统一视角，从而连接某些 attention-like 与 SSM-like computation。

## 4.1　应该亲手做的实验

最终代码需要至少有：

```text
reference_ssm.py
mamba_like_scan.py
ssd_matrix_form.py
```

并验证：

1. recurrent form；
2. matrix form；
3. chunked form；

在数值上得到相同结果。

---

# 5　RWKV：把 Transformer-like mixing 做成 recurrent inference

RWKV 是另一条非常不同的路线：希望在训练时保留类似 Transformer 的并行形式，同时在推理时使用 RNN 风格状态。

原始资料入口：

- RWKV 项目： https://github.com/BlinkDL/RWKV-LM
- RWKV-4 论文： https://arxiv.org/abs/2305.13048

研究时要区分：

```text
architecture identity
training parallelism
inference recurrence
state size
long-context retention
```

不能只比较 benchmark。

---

# 6　xLSTM：重新审视 LSTM 的记忆结构

在 Transformer 时代之后，LSTM 也出现重新设计。

原始论文：

- Beck et al., **xLSTM: Extended Long Short-Term Memory**, 2024.  
  https://arxiv.org/abs/2405.04517

这里值得研究的不是“LSTM 回来了”这种标题，而是：

- exponential gating；
- scalar / matrix memory；
- recurrent state 怎样扩大表达能力；
- 与 Transformer 在并行性上的根本差异。

---

# 7　Titans 与 Test-Time Neural Memory

长上下文通常被理解成“把更多 token 塞进 attention window”。另一种思路是：模型在推理期间维护可更新的长期记忆模块。

原始论文：

- Behrouz et al., **Titans: Learning to Memorize at Test Time**, 2024/2025.  
  https://arxiv.org/abs/2501.00663

这条路线与 Agent memory 有一个非常重要的分界：

```text
Neural memory
= 模型内部可学习状态

Agent memory
= 系统外部显式事件 / 文件 / 数据库 / 向量存储
```

二者可以结合，但绝不能混为一谈。

---

# 8　Linear Attention：把 Softmax Attention 重新因子化

标准 attention：

$$
\operatorname{softmax}(QK^\top)V
$$

难以直接把 $K,V$ 压缩为固定状态。

Linear attention 一类方法尝试使用 feature map $\phi$，使：

$$
\phi(Q)\phi(K)^\top V
$$

可以改变乘法顺序：

$$
\phi(Q)\left(\phi(K)^\top V\right).
$$

这把“历史 token”转化成一个累计统计量的可能性打开了。

代表原始资料：

- Katharopoulos et al., **Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention**, 2020.  
  https://arxiv.org/abs/2006.16236

但必须注意：换掉 softmax kernel 会改变模型函数，不是免费的系统优化。

---

# 9　Hybrid Architecture 可能比“赢家通吃”更现实

现实模型未必会在：

```text
Attention vs SSM
```

中二选一。

更可能出现：

```text
local attention
+ recurrent / state-space memory
+ occasional global attention
+ MoE
+ external retrieval
```

研究 hybrid architecture 时，应分别问：

- 哪些层做精确 pairwise interaction？
- 哪些层压缩历史？
- 状态何时重置？
- 检索何时访问外部信息？
- 推理时实际缓存了什么？

---

# 10　统一比较框架

所有替代架构都必须在同一张表里比较：

| 维度 | Transformer | SSM | Recurrent LM | Linear Attention | Hybrid |
|---|---|---|---|---|---|
| 训练并行性 | 高 | 取决实现 | 较难 | 高 | 混合 |
| Prefill 复杂度 | 常见 $O(T^2)$ | 可线性 | 线性 | 可线性 | 取决结构 |
| Decode state | KV Cache 随 $T$ 增长 | 固定/小状态 | 固定状态 | 固定统计量 | 混合 |
| 精确 token-token interaction | 强 | 间接 | 间接 | kernel-dependent | 部分 |
| 长程遗忘机制 | context/attention | state dynamics | recurrent gates | state summary | 混合 |

这张表只描述结构性质，不代表最终质量排序。

---

# 11　代码终点

本章最终不应止于论文综述，而应进入：

```text
代码-code/后Transformer/
├── reference_ssm.py
├── selective_scan.py
├── linear_attention.py
├── rwkv_like.py
├── neural_memory.py
└── benchmarks.py
```

最小验收：

1. 能画出每种架构 state 的 shape；
2. 能写 reference forward；
3. 能测 sequence length 对显存/延迟的增长；
4. 能解释训练并行与 decode recurrence 的差异；
5. 能对同参数规模 toy task 做受控实验。

---

# 12　这一领域最重要的开放问题

1. 长程记忆究竟需要精确 token-level access 还是压缩状态即可？
2. SSM 的 fixed-size state 在极长任务中会不会形成不可避免的信息瓶颈？
3. hybrid attention/SSM 的最优层比例是什么？
4. test-time neural memory 与 external retrieval 应怎样协同？
5. Agent 长期记忆应放在模型内部还是系统外部？
6. Post-Transformer 架构在 reasoning 与 tool use 上是否需要不同 inductive bias？

这些问题比“谁打败 Transformer”更值得长期研究。