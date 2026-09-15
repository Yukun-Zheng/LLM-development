# Lesson 18：Page-Aware Attention 与异长 Decode

上一课已经实现：

```text
physical block allocator
→ physical K/V tensor slabs
→ physical prefix sharing
```

但那里仍有一个关键妥协：进入 Transformer attention 之前，会把某个 request 的 block table 重新 `materialize()` 成连续：

```text
[1, H_kv, T, D]
```

因此“存储是 paged 的”，但“attention 还是 contiguous 的”。这一课第一次拆掉这层妥协。

---

## 1　目标

我们希望同时 decode 两个历史长度不同的请求：

```text
A cached length = 3
B cached length = 5

next token A = 9
next token B = 10
```

而且 attention 不允许走：

```text
block table
→ torch.cat(all historical K/V)
→ ordinary attention
```

而必须直接：

```text
request A block table ─┐
                       ├→ page-aware attention
request B block table ─┘
```

对应源码：

```text
src/astra_codex/page_aware_decode.py
```

核心类：

```text
HeterogeneousPageAwareDecodeReference
```

---

## 2　为什么异长 batching 难

普通 homogeneous batch 假设每个 request 有相同历史长度：

```text
K: [B, H_kv, T, D]
V: [B, H_kv, T, D]
```

于是很容易统一做：

```text
QK^T
softmax
PV
```

但 serving 中真实请求往往是：

```text
request 0 → 127 cached tokens
request 1 → 918 cached tokens
request 2 → 31 cached tokens
request 3 → 4096 cached tokens
```

此时不存在一个自然、无浪费的统一 `T`。

把所有请求 pad 到最大长度虽然能算，但会浪费：

```text
memory bandwidth
attention FLOPs
KV movement
```

所以现代 serving runtime 需要把：

```text
logical sequence
```

和：

```text
physical K/V storage
```

分开。

---

## 3　本课的 block-table attention

物理 tensor pool：

```text
K/V slab:
[L, physical_block, H_kv, block_size, D]
```

一个 request 的 block table：

```text
[(block=3, logical_tokens=4),
 (block=8, logical_tokens=4),
 (block=1, logical_tokens=2)]
```

意味着逻辑长度：

```text
4 + 4 + 2 = 10
```

对 one-token decode，query 为：

```text
Q: [1, H_q, 1, D]
```

我们逐 block 计算：

$$
S_j = \frac{QK_j^T}{\sqrt{d}}
$$

其中每个 `K_j` 直接是 tensor slab 的 view，而不是把历史 K/V 拼起来。

然后只拼接 score：

$$
S=[S_1,S_2,\ldots,S_m,S_{new}]
$$

统一做：

$$
P=\mathrm{softmax}(S)
$$

再把概率重新按 block 切开：

$$
C=\sum_j P_jV_j + P_{new}V_{new}.
$$

注意这里：

> 可以拼 score，但不需要拼 cached K/V。

score 的最后一维只是标量 attention weight；大体积 K/V 仍留在 physical slabs 中。

---

## 4　GQA 怎么处理

物理 cache 保存的是：

```text
H_kv
```

而 query 是：

```text
H_q
```

若：

$$
H_q = r H_{kv},
$$

则每个物理 K/V block 在参与 attention 时按 head 维做逻辑 repeat：

```text
[1, H_kv, L_block, D]
→
[1, H_q, L_block, D]
```

这里的语义与原 `GroupedQueryAttention` 一致，因此最终可以和 full recomputation 做 numerical parity。

---

## 5　RoPE 为什么必须按 request 单独算位置

这批请求的历史长度不同：

```text
A past_len = 3
B past_len = 5
```

所以同一个 batched next-token 的绝对 position 分别是：

```text
A → position 3
B → position 5
```

不能简单给整个 batch 一个共享 position。

当前 reference path 因此：

```text
Q/K projection → batched
RoPE position  → per request
page attention → per request
FFN            → batched
```

这是一个很重要的分层：

> “batch API 支持异长”不等于“每一条 kernel 都已经完全 fused”。

当前实现故意保持这个边界透明。

---

## 6　当前真正 batch 的部分

对于 `B` 个请求：

```text
Embedding                ✅ batched
RMSNorm                  ✅ batched
Q/K/V projection         ✅ batched
FFN                       ✅ batched
Final norm / LM head      ✅ batched
```

而 page-aware attention 当前是：

```text
for request in batch:
    walk its own block table
    compute block score chunks
    one softmax over its logical history
    accumulate block contexts
```

因此准确名称是：

```text
Heterogeneous Page-Aware Decode Reference
```

而不是“已经实现 production fused continuous batching”。

---

## 7　新 token 什么时候写入 cache

one-token decode 的当前 token 本身也必须参与 attention。

所以每层先得到：

```text
q_new
k_new
v_new
```

attention 读取：

```text
historical physical blocks
+
k_new / v_new
```

但不会立刻把它写入 pool。

所有 Transformer layer 都成功完成后，再统一对每个 request：

```text
pool.append_delta(request_id, all_layer_new_kv)
```

这样至少保证“模型某一层中途失败”时，不会留下半层 K/V。

但当前仍有一个更高层的事务边界：

> 多 request batch 的 block capacity 尚未提前做整体 reservation。

所以若第一个 request append 成功、第二个 request 才发现 OOM，仍可能形成 batch-level partial commit。

生产 scheduler 下一步必须在执行模型之前完成：

```text
capacity planning
→ block reservation
→ execute
→ commit
```

---

## 8　验收：禁止 materialize

最关键的自动测试不是只看 logits。

测试会直接 monkeypatch：

```python
pool.materialize = forbidden
```

一旦 page-aware decode 偷偷回退到连续 K/V，就立即失败。

然后同时送入：

```text
A prefix length = 3
B prefix length = 5
```

做一次：

```text
executor.decode_batch([A,B], next_tokens)
```

分别与：

```text
model(full_A)
model(full_B)
```

比较最后 logits。

因此验收条件是：

```text
no K/V materialization
+
mixed cached lengths
+
logits numerical parity
```

---

## 9　K/V 本身也要 parity

只比较 logits 还不够，因为错误 cache 可能在当前 token 没明显暴露，而在后续 decode 累积。

第二个测试会在 page-aware decode 后重新读取物理 pool，并与：

```text
model(full_sequence, use_cache=True)
```

返回的每层：

```text
K
V
```

逐 tensor 比较。

这验证的不只是最终分类输出，而是内部状态也一致。

---

## 10　Prefix sharing 与 Page-Aware Decode 的组合

还要验证一个更接近 serving 的场景：

```text
parent prompt
      ↓ exact prefix reuse
child prompt
      ↓ shares physical blocks
page-aware decode(child, new_token)
```

如果 child 最后一块是 shared partial block，那么 append 必须触发 COW。

测试要求：

```text
child decode logits == full recomputation
parent K/V before == parent K/V after
shared full block stays shared
partial tail becomes private
```

因此：

```text
prefix cache
+ physical allocator
+ tensor slabs
+ COW
+ page-aware decode
```

第一次真正串成一条链。

---

## 11　这一层之后，还差什么才叫 Continuous Batching

这一课解决了：

```text
✅ mixed cached lengths
✅ direct physical-block K/V reads
✅ batch QKV/FFN
✅ numerical parity
```

但还缺：

```text
scheduler request lifecycle integration
block reservation before execution
request arrival while previous batch is decoding
finished request immediate removal
new request immediate admission
preemption / swap
GPU fused block-table kernel
fairness / starvation policy
TTFT / TPOT / throughput benchmark under live arrivals
```

因此下一阶段不是再造一种 cache，而是把已有组件真正闭环：

```text
Request Scheduler
      ↓
Block Allocator
      ↓
Prefix Cache
      ↓
Heterogeneous Page-Aware Decode
      ↓
Sampling
      ↓
request state update
      ├→ FINISHED → free blocks
      └→ DECODE   → next iteration
```

这才是 Continuous Batching Engine 的主体。

---

## 12　最终要替换掉什么

现在 page-aware reference attention 仍在 Python 层遍历 request/block。

未来 production path 应把：

```text
block_table
sequence_lengths
Q
physical K/V slabs
```

直接交给 GPU kernel，避免 Python loops 和零散 block matmul。

但先写 reference path 很重要，因为未来任何 Triton/CUDA kernel 都必须满足同一个 oracle：

```text
optimized_page_attention(...)
≈
reference_page_attention(...)
```

换句话说：

> **今天写的是未来高性能 kernel 的正确性基准，不是最终性能实现。**
