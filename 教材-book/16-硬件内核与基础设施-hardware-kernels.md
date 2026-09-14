# 第十六篇（Part XVI）　硬件、Kernel 与基础设施：为什么大模型首先是一个计算系统

> **本章主线**：相同的数学公式，在不同硬件、内存层级、通信拓扑和 kernel 实现上，训练速度可能相差数倍甚至数量级。本章从 Roofline、HBM、SRAM、Tensor Core、NCCL、FlashAttention、ZeRO、Megatron 到 serving，把“大模型工程”还原成真实数据移动问题。

---

# 1　FLOPs 不是全部

很多人第一次估算大模型只看 FLOPs：

$$
C\approx 6ND,
$$

其中 $N$ 是参数量，$D$ 是训练 token 数。

这个近似对宏观 compute budgeting 很有用，但不能解释实际 wall-clock。

真实运行时间还取决于：

- memory bandwidth；
- arithmetic intensity；
- kernel launch overhead；
- communication；
- synchronization；
- padding；
- load imbalance；
- checkpointing；
- host-device transfer。

所以两个理论 FLOPs 相同的实现，可以有完全不同的速度。

---

# 2　Roofline Model：先问算力受限还是带宽受限

Williams, Waterman & Patterson (2009) 提出的 Roofline Model 给出一个非常有用的系统视角。

定义 arithmetic intensity：

$$
I=\frac{\text{FLOPs}}{\text{Bytes moved}}.
$$

可达到性能受两条上界约束：

$$
P\le P_{peak},
$$

以及：

$$
P\le I\cdot BW.
$$

因此：

$$
P\le \min(P_{peak}, I\cdot BW).
$$

如果 $I$ 很低，增加理论 tensor-core FLOPs 并不能显著提速，因为瓶颈在 memory bandwidth。

[Roofline 原论文](https://dl.acm.org/doi/10.1145/1498765.1498785)

---

# 3　GPU 内存层级

一个极度简化的 GPU memory hierarchy：

```text
registers
  ↓ fastest / tiniest
shared memory / SRAM
  ↓
L1 / L2 cache
  ↓
HBM
  ↓
host memory / storage
```

访问越靠近计算单元通常越快，但容量越小。

大模型 kernel 优化的核心问题经常不是：

> 少算一次乘法。

而是：

> 少把一块大 tensor 从 HBM 读回来一次。

---

# 4　为什么矩阵乘法适合 Tensor Core

Transformer 的主要计算高度依赖 GEMM：

$$
C=AB.
$$

例如线性层：

$$
Y=XW.
$$

GPU Tensor Core 专门优化小块矩阵乘加，再通过 tile 组合完成大矩阵。

这也是为什么：

- hidden dimension；
- head dimension；
- batch / sequence shape；

会影响硬件利用率。

数学上只写 $XW$，系统里却要考虑 layout、tile、precision、fusion。

---

# 5　Attention 为什么原本很吃显存

标准 attention：

$$
S=QK^\top,
$$

$$
A=\mathrm{softmax}(S),
$$

$$
O=AV.
$$

若显式 materialize $S$ 与 $A$：

$$
S,A\in\mathbb R^{B\times h\times T\times T}.
$$

当 $T$ 很大时，中间矩阵是二次增长。

传统实现会反复在 HBM 中读写这些中间结果。

---

# 6　FlashAttention 的核心不是近似 Attention

Dao et al. (2022) 提出的 FlashAttention 是 **IO-aware exact attention**。[原论文](https://arxiv.org/abs/2205.14135)

它并没有把：

$$
\mathrm{softmax}(QK^\top)V
$$

换成另一个近似数学函数。

核心思想是：

```text
Q,K,V 分块
   ↓
将 tile 放入更快 SRAM
   ↓
局部计算 score / softmax
   ↓
在线维护归一化统计量
   ↓
避免完整 T×T attention matrix 往返 HBM
```

因此它的创新主要发生在 **数据移动与 kernel 调度**。

---

# 7　Online Softmax 为什么允许分块

普通 softmax：

$$
\mathrm{softmax}(x_i)
=
\frac{e^{x_i}}{\sum_j e^{x_j}}.
$$

为数值稳定，通常减去最大值：

$$
\frac{e^{x_i-m}}{\sum_j e^{x_j-m}},
\qquad m=\max_j x_j.
$$

FlashAttention 通过 block-wise 地维护最大值与归一化因子，使得不必一次保存完整 score matrix。

理解这一点后你会发现：

> 算法创新有时来自“重新排列等价计算”，而不是改变最终数学函数。

---

# 8　FlashAttention-2 / 3：更细的并行与硬件适配

后续版本继续优化：

- work partition；
- warp-level parallelism；
- asynchronous execution；
- low-precision hardware。

原始资料：

- FlashAttention-2  
  https://arxiv.org/abs/2307.08691
- FlashAttention-3  
  https://arxiv.org/abs/2407.08608

教材不能把它们写成“FlashAttention 版本号升级”；要具体解释每一代在硬件执行路径上改变了什么。

---

# 9　训练显存到底去哪了

对一个参数量为 $N$ 的模型，训练时不仅存权重。

以 Adam 类优化器为例，粗略需要：

```text
weights
+ gradients
+ first moment m
+ second moment v
+ activations
+ temporary buffers
```

因此不能简单用：

$$
N\times 2\ \text{bytes}
$$

估计训练显存。

真正显存账本必须逐项计算。

---

# 10　Data Parallel：最容易理解的并行

每张 GPU 保存完整模型，处理不同 batch：

```text
GPU0: model copy + batch0
GPU1: model copy + batch1
GPU2: model copy + batch2
```

反向后做 gradient all-reduce：

$$
g=\frac1n\sum_{i=1}^n g_i.
$$

优点简单；缺点是每卡都要放完整模型和 optimizer state。

---

# 11　ZeRO：把重复状态切开

Rajbhandari et al. 提出的 ZeRO 针对 data parallel 中重复保存的训练状态做 partition。[原论文](https://arxiv.org/abs/1910.02054)

概念上逐步切分：

- optimizer states；
- gradients；
- parameters。

于是每张卡不再保存所有冗余副本。

ZeRO 的关键不是“模型并行算法”这一句话，而是：

> **哪些状态在哪个时间点需要被 gather，什么时候可以只保存 shard。**

---

# 12　Tensor Parallel：一个矩阵乘法拆到多卡

假设：

$$
Y=XW.
$$

若把 $W$ 按列切成：

$$
W=[W_1,W_2],
$$

则：

$$
Y=[XW_1,XW_2].
$$

不同 GPU 分别计算子矩阵。

Megatron-LM 系统化了 Transformer 中的 tensor model parallel。[Shoeybi et al., 2019](https://arxiv.org/abs/1909.08053)

代价是层内需要通信。

---

# 13　Pipeline Parallel：按层切模型

```text
GPU group 0 → layers 0..7
GPU group 1 → layers 8..15
GPU group 2 → layers 16..23
```

microbatches 像流水线一样通过。

问题是 pipeline bubble：某些阶段会等待。

因此要设计：

- microbatch 数；
- schedule；
- interleaving；
- activation communication。

---

# 14　Sequence / Context Parallel

长序列带来 activation 与 attention 规模压力。

一种思路是在 sequence dimension 上切分，使不同设备处理不同 token 范围。

这会引入新的 collective communication 和 attention data dependency。

长上下文训练不能只看“显存够不够”，还要看跨设备通信模式。

---

# 15　Expert Parallel：MoE 的特殊系统问题

MoE router 把 token 分配给不同 expert：

```text
tokens on GPU0
   ↓ router
some → expert on GPU2
some → expert on GPU5
```

于是出现 all-to-all communication。

如果 router 极不均衡：

- 某些 GPU 爆满；
- 某些 GPU 空闲；
- throughput 被最忙 expert 决定。

因此 MoE 的“只激活少量参数”不代表训练天然便宜。

---

# 16　NCCL 与 Collective Communication

分布式训练常见 collective：

- all-reduce；
- all-gather；
- reduce-scatter；
- all-to-all。

不同并行方式依赖不同 collective。

因此一个训练架构的真实成本应该写成：

$$
T_{step}
=
T_{compute}+T_{memory}+T_{communication}+T_{bubble}+T_{overhead}.
$$

而不是只写 FLOPs。

---

# 17　Interconnect 为什么重要

单机多卡可能通过 NVLink / NVSwitch；跨节点则依赖 InfiniBand / Ethernet 等网络。

当 model parallel communication 很频繁时，网络带宽和 latency 会直接决定 scaling efficiency。

因此：

```text
8 GPUs in one node
```

与

```text
8 GPUs across 8 nodes
```

即使 GPU 型号一样，也完全不是同一个系统。

---

# 18　Checkpointing：算力换显存

Activation checkpointing 不保存所有中间 activation，而在 backward 时重新计算。

```text
forward
save only checkpoints
      ↓
backward
recompute missing activations
```

效果：

- memory 降低；
- compute 增加。

这是典型的 compute-memory trade-off。

---

# 19　Prefill 与 Decode 是两种完全不同的 workload

LLM 推理分：

### Prefill

一次处理 prompt 的大量 token。

特点：

- 大矩阵计算；
- parallelism 高；
- 更 compute-heavy。

### Decode

每一步通常只生成少量新 token。

特点：

- 频繁读取 KV cache；
- memory-bandwidth / latency 更重要；
- batch scheduling 很关键。

把二者用同一个“tokens/s”概括会丢掉很多系统信息。

---

# 20　KV Cache 是 serving 的核心状态

每层需要保存过去 token 的 K/V。

粗略占用：

$$
M_{KV}
\propto
B\cdot T\cdot L\cdot h_{kv}\cdot d_h\cdot bytes.
$$

所以 GQA / MQA / MLA 类方法不仅是架构设计，也直接决定 serving economics。

---

# 21　PagedAttention：为什么 KV Cache 也需要“虚拟内存思维”

vLLM 的 PagedAttention 关注一个实际问题：多个请求的 KV cache 长度不断变化，连续大块内存容易碎片化。

其设计借鉴 virtual memory / paging 思想，把 KV cache 分页管理。[原论文](https://arxiv.org/abs/2309.06180)

因此 serving 可以：

- 更灵活分配 cache；
- 降低 fragmentation；
- 提高 batch capacity。

---

# 22　Continuous Batching

传统 static batching：

```text
等一批请求
一起开始
等最长请求结束
整批结束
```

LLM 请求长度高度不均匀，因此效率很差。

Continuous batching：

```text
某请求结束
立即插入新请求
```

这使 serving scheduler 本身成为关键系统组件。

---

# 23　Speculative Decoding：用便宜模型猜，大模型验证

核心思想：

```text
draft model proposes several tokens
          ↓
target model verifies in parallel
          ↓
accept prefix / correct mismatch
```

代表原始资料：

- Leviathan et al.  
  https://arxiv.org/abs/2211.17192
- Chen et al.  
  https://arxiv.org/abs/2302.01318

只要 acceptance rate 足够高，就能减少昂贵 target model 的逐 token 串行次数。

---

# 24　MFU：训练系统到底有没有吃满硬件

Model FLOPs Utilization（MFU）试图衡量：

$$
\mathrm{MFU}
=
\frac{\text{模型理论所需 FLOPs / 实际时间}}{\text{硬件理论峰值 FLOPs}}.
$$

它比单纯 tokens/s 更适合跨模型规模理解训练效率，但也依赖 FLOPs 估算定义。

---

# 25　TPU：专用硬件路线

Jouppi et al. (2017) 分析 Google TPU 的 datacenter performance。[原论文](https://arxiv.org/abs/1704.04760)

它展示了另一条核心思想：

> 深度学习 workload 可以推动专用矩阵计算硬件，而不是永远依赖通用 CPU/GPU 设计。

今天研究 LLM 系统不能只理解 CUDA，也要理解 accelerator 设计的一般原则。

---

# 26　原始资料包

### Hardware

- Roofline  
  https://dl.acm.org/doi/10.1145/1498765.1498785
- TPU  
  https://arxiv.org/abs/1704.04760

### Training systems

- Megatron-LM  
  https://arxiv.org/abs/1909.08053
- ZeRO  
  https://arxiv.org/abs/1910.02054
- Megatron large-scale training  
  https://arxiv.org/abs/2104.04473
- NVIDIA Megatron-LM  
  https://github.com/NVIDIA/Megatron-LM
- Microsoft DeepSpeed  
  https://github.com/microsoft/DeepSpeed

### Kernels / inference

- FlashAttention  
  https://arxiv.org/abs/2205.14135
- FlashAttention-2  
  https://arxiv.org/abs/2307.08691
- vLLM / PagedAttention  
  https://arxiv.org/abs/2309.06180
- vLLM repository  
  https://github.com/vllm-project/vllm

---

# 27　最小实验

真正学会这一章，至少做：

1. 对 naive attention 与 PyTorch SDPA / FlashAttention 做显存和速度 profiling；
2. 画 sequence length 增长时显存曲线；
3. 比较 prefill 与 decode 的 GPU utilization；
4. 用两卡实现最小 data parallel 与 tensor split；
5. 记录 all-reduce 时间占 step time 的比例；
6. 用 vLLM 对比 static batching 与 continuous batching throughput。

做到这里，你会真正理解：**一个大模型不是一个 `nn.Module`，而是一条贯穿算法、内存、网络、kernel 与调度器的计算流水线。**
