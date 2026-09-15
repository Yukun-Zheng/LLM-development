# Lab 05：从 KV Cache 到 Paged KV 与 Request Scheduler

> **目标**：从一个请求的自回归 decode，走到 serving system 的两个核心对象：**状态如何分配**、**请求何时执行**。

对应源码：

- [`../src/astra_codex/cache.py`](../src/astra_codex/cache.py)
- [`../src/astra_codex/paged_cache.py`](../src/astra_codex/paged_cache.py)
- [`../src/astra_codex/scheduler.py`](../src/astra_codex/scheduler.py)
- [`../src/astra_codex/engine.py`](../src/astra_codex/engine.py)

对应测试：

- [`../tests/test_model_and_cache.py`](../tests/test_model_and_cache.py)
- [`../tests/test_v3_p0_runtime.py`](../tests/test_v3_p0_runtime.py)
- [`../tests/test_scheduler_benchmark.py`](../tests/test_scheduler_benchmark.py)

理论入口：[`../../../教材-book/10-推理服务系统-inference-systems.md`](../../../教材-book/10-推理服务系统-inference-systems.md)

原始资料：

- Kwon et al., **Efficient Memory Management for Large Language Model Serving with PagedAttention**, 2023: https://arxiv.org/abs/2309.06180
- vLLM official repository: https://github.com/vllm-project/vllm

---

# 1　为什么 KV Cache 是 serving 的起点

对于 causal Transformer，第 $t$ 步只新产生一个 query，但仍然要读取过去 token 的 K/V。

单层简化 shape：

```text
K: [B, H_kv, T, D_h]
V: [B, H_kv, T, D_h]
```

随着 $T$ 增长，cache bytes 近似：

$$
M_{KV}
\propto
2L H_{kv}TD_hb,
$$

其中：

- $L$：层数；
- $H_{kv}$：KV heads；
- $D_h$：head dimension；
- $b$：每个元素 bytes。

所以 serving 的问题不只是“模型 forward 快不快”，而是：

> **大量长度不同、到达时间不同的请求，怎样共享有限显存中的动态 KV state？**

---

# 2　Contiguous Cache 的简单世界

当前 `cache.py` 把每层历史看成一个连续 tensor：

```text
Layer 0: K[0:T], V[0:T]
Layer 1: K[0:T], V[0:T]
...
```

这个 representation 非常适合验证数学正确性。

我们已经验证：

$$
\mathrm{logits}_{cached}
\approx
\mathrm{logits}_{full}.
$$

但是当请求长度不同、不断加入/退出时，“每个请求一块连续大 buffer”会引入容量预留和碎片问题。

---

# 3　Reference Paged KV：先把语义写对

`ReferencePagedKVCache` 把每层历史拆成 logical pages。

假设：

```text
page_size = 2
T = 5
```

那么单层 page table：

```text
page 0: 2 tokens
page 1: 2 tokens
page 2: 1 token
```

即：

```text
(2, 2, 1)
```

再 decode 一个 token：

```text
(2, 2, 2)
```

测试不仅检查这个 page table，还检查：

```text
Paged cache path
      vs
Full forward
      ↓
last-token logits parity
```

这一步非常重要，因为我们先定义：

> **分页以后模型函数不能变。**

---

# 4　为什么当前实现还不能叫“vLLM 复现”

当前：

```python
cache.as_past_key_values()
```

会把 page list 再 `torch.cat` 成 contiguous K/V。

所以当前真实数据流是：

```text
pages
→ concatenate
→ ordinary attention
```

而生产级 PagedAttention 更接近：

```text
logical block table
→ physical KV blocks
→ attention kernel directly follows block mapping
```

因此当前已经证明的是：

```text
page semantics ✓
allocator efficiency ?
kernel efficiency ?
HBM benefit ?
fragmentation benefit ?
```

教材必须明确这条证据边界。

---

# 5　Serving 的第二个对象：Scheduler

即使 cache 完美，如果请求管理还是：

```text
request A 跑完
→ request B
→ request C
```

GPU utilization 仍然很差。

所以 `scheduler.py` 显式定义：

```text
WAITING
→ PREFILL
→ DECODE
→ FINISHED
```

或：

```text
WAITING / DECODE
→ CANCELLED
```

这里 scheduler 不负责模型数学；它负责决定：

> 下一次 model executor 应处理哪些 request？

---

# 6　Prefill 与 Decode 为什么应该分开

Prefill：

```text
输入很多 prompt tokens
矩阵通常较大
```

Decode：

```text
每个活跃 sequence 通常只产生 1 token
KV/history 很大
```

计算形态完全不同。

当前 reference scheduler 使用 decode-first：

```text
if any active decode requests:
    schedule decode batch
else:
    admit waiting prompts under prefill-token budget
```

这不是“最优调度定律”，只是一个可检查 baseline。

以后新 scheduler 必须和它做 controlled comparison。

---

# 7　Prefill token budget

假设：

```text
max_prefill_tokens = 8
requests:
A prompt = 5
B prompt = 7
C prompt = 3
```

按 arrival order，reference scheduler 可以选：

```text
A + C = 8
```

而 B 留在 WAITING。

测试固定验证：

```text
batch.request_ids == (A, C)
batch.token_count == 8
```

所以 scheduler 行为不是模糊的“尽量 batch”，而是可复现状态机。

---

# 8　Serving Metrics

对请求 $i$，到达时间：

$$
t_i^{arrive}.
$$

第一 token 时间：

$$
t_i^{first}.
$$

则：

$$
\mathrm{TTFT}_i
=t_i^{first}-t_i^{arrive}.
$$

如果一共产生 $N$ 个 token，第一和最后 token 时间分别是 $t^{first}$ 与 $t^{last}$，当前 reference TPOT：

$$
\mathrm{TPOT}
=\frac{t^{last}-t^{first}}{N-1}.
$$

这让以后比较调度策略时不只看：

```text
"感觉快了"
```

而可以比较：

```text
TTFT
TPOT
throughput
queueing delay
batch occupancy
fairness
```

---

# 9　下一步真正进入 Continuous Batching

当前 Scheduler 输出：

```text
ScheduledBatch
```

但还没有真正把多个 request 的 state 合并成一次 batched model execution。

下一阶段：

```text
Request Scheduler
        ↓
Batch Builder
        ↓
KV Block Table
        ↓
Batched Prefill / Decode
        ↓
Scatter logits/state back to each request
        ↓
Sampling
```

这个里程碑完成后，才能说我们拥有第一个真正的 serving engine，而不仅是 generation loop + scheduling simulation。

---

# 10　运行实验

```bash
cd '代码-code/从零构建Astra与Codex级系统'
pip install -e '.[dev]'
pytest tests/test_model_and_cache.py \
       tests/test_v3_p0_runtime.py \
       tests/test_scheduler_benchmark.py -q
```

真正研究时，每一次优化都要保留：

```text
Reference
vs
Optimized
→ numerical parity
→ memory
→ latency
→ throughput
```

这是本项目推理系统主线的基本纪律。
