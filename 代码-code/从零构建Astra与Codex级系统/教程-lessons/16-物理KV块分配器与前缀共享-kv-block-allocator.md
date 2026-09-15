# Lesson 16　物理 KV 块分配器与前缀共享

前面的 `ReferencePagedKVCache` 已经证明了一个重要语义：KV 历史可以被拆成 page，再按 page table 重新构造成连续历史，且 logits 与完整重算保持一致。

但那还没有回答 serving 系统真正困难的问题：

> 当同时有几百、几千个 request 时，谁拥有哪些 KV block？block 从哪里来？结束后怎么回收？共享前缀时如何避免复制？共享的最后一个半满 block 又怎样继续写？显存不够时应该在什么时候失败？

这一课把问题从：

```text
Tensor storage layout
```

推进到：

```text
Physical block ownership / lifecycle
```

源码：

```text
src/astra_codex/kv_block_allocator.py
```

---

## 1　为什么连续 KV 会产生 allocator 问题

对单个 request，连续 KV 很自然：

```text
K: [H_kv, T, D]
V: [H_kv, T, D]
```

但 serving 中 request 长度不同，而且会动态增长：

```text
A:  73 tokens
B: 901 tokens
C:  14 tokens
D: 287 tokens
```

如果每个 request 都预留最大长度：

```text
allocated capacity >> actually used tokens
```

会形成大量内部浪费。

如果频繁重新分配连续大 tensor，又会带来：

```text
copy
fragmentation
allocator pressure
synchronization
```

PagedAttention 的核心工程思想之一，就是把“一个 request 的逻辑 KV 序列”与“显存中的物理 block”解耦。

原始论文：

- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023 / arXiv:2309.06180  
  https://arxiv.org/abs/2309.06180
- 官方 vLLM：  
  https://github.com/vllm-project/vllm

---

## 2　两个地址空间

我们的 reference allocator 明确区分：

```text
logical request sequence
```

和：

```text
physical KV block pool
```

例如 block size = 4：

```text
Physical pool
block 0
block 1
block 2
block 3
...
```

request A 有 6 个 token：

```text
A logical sequence:
0 1 2 3 | 4 5

A block table:
[(block 0, 4),
 (block 1, 2)]
```

这里：

```text
sequence_length(A) = 6
physical capacity   = 8 slots
```

最后一个 block 只用了 2/4，所以当前内部碎片是 2 slots。

---

## 3　为什么 block table 不能只存 block id

reference implementation 使用：

```python
BlockTableEntry(
    block_id,
    logical_tokens,
)
```

而不是只有：

```text
[block_id_0, block_id_1, ...]
```

原因是一个 physical block 的内容长度，与某个 request 对它的**逻辑可见长度**可能不同。

例如两个 request 共享 block：

```text
physical block 7 contains 3 tokens

request A view: 3 tokens
request B view: 2 tokens
```

B 的逻辑序列不能因为共享了同一个 physical block，就自动看到 A 的第 3 个 token。

这就是：

```text
physical contents
!=
logical view
```

---

## 4　有限 block pool 与 free list

初始化：

```python
allocator = KVBlockAllocator(
    total_blocks=1024,
    block_size=16,
)
```

它建立：

```text
PhysicalBlock[0 ... 1023]
free list
request tables
```

一个空闲 block 必须满足：

```text
refcount = 0
used_tokens = 0
block_id ∈ free_list
```

一个正在使用的 block 必须满足：

```text
refcount > 0
block_id ∉ free_list
```

这些不是注释约定，而是 `assert_invariants()` 会逐项检查的 runtime invariant。

---

## 5　Append 为什么必须先做容量预检

假设：

```text
total free blocks = 1
request currently uses 3/4 tokens in final block
append 6 tokens
```

它需要：

```text
1 token fills current block
+ 4 tokens need one new block
+ 1 token needs another block
```

也就是 2 个新 block。

最糟糕的实现会：

```text
先填当前 block
→ allocate one block
→ 再发现 OOM
```

于是一次失败的 append 已经部分改变 request state。

reference allocator 先计算：

```text
required_free_blocks
```

只有满足：

```text
required <= len(free_list)
```

才真正修改 block table。

自动测试专门验证：

```text
append raises MemoryError
→ block table unchanged
→ metrics unchanged
→ invariants still hold
```

这是一种最小的 transaction-like allocator semantics。

---

## 6　Prefix sharing 为什么需要 refcount

假设 request A：

```text
[A0 A1 A2 A3] [A4 A5]
   block 0      block 1
```

fork request B：

```text
A block table ─┐
               ├→ block 0, block 1
B block table ─┘
```

此时：

```text
refcount(block 0) = 2
refcount(block 1) = 2
```

逻辑 token 总数已经是：

```text
6 + 6 = 12
```

但物理 KV 内容仍只有：

```text
6 tokens
```

这就是 prefix sharing 的直接内存收益来源。

allocator metrics 同时记录：

```text
logical_tokens
physical_token_slots_used
shared_block_references
```

因此教材可以直接观察：

```text
logical workload growth
vs
physical storage growth
```

而不是只讲一句“prefix cache 可以省显存”。

---

## 7　共享半满 block 的 Copy-on-Write

这是整个 allocator 里最值得理解的一个 corner case。

A/B 共享：

```text
block 1 = [t4, t5, _, _]
refcount = 2
```

现在 B 想 append 一个新 token：

```text
B: t6
```

如果直接原地写 block 1：

```text
block 1 = [t4, t5, t6, _]
```

A 也会“看到”这个物理修改。

因此 append 前执行：

```text
shared partial final block
        ↓
allocate private clone
        ↓
copy logical prefix metadata
        ↓
decrement old refcount
        ↓
append into private block
```

也就是 Copy-on-Write（COW）。

测试验收：

```text
before:
A.last.block_id == B.last.block_id

B append

then:
A.last.block_id != B.last.block_id
A logical length unchanged
B logical length increased
```

---

## 8　为什么 block-aligned prefix 最适合共享

source：

```text
[0 1 2 3] [4 5 6 7] [8 9]
```

现在 target 只复用前 6 token：

```text
[0 1 2 3] [4 5]
```

第一个 block 可以直接共享。

第二个 source block 实际包含：

```text
[4 5 6 7]
```

而 target 只应该看到：

```text
[4 5]
```

reference allocator 对这个 partial prefix 选择：

```text
full block → refcount share
partial tail → private clone
```

这与实际 block-hash prefix cache 的一个基本工程事实一致：

> full blocks 最自然地成为可复用/cacheable 单元；partial block 往往需要特殊处理。

---

## 9　Truncate 为什么也会碰到 shared block

如果：

```text
A/B share [x0 x1 x2]
```

B truncate 到 2 tokens：

```text
A view = 3
B view = 2
physical block still = 3
```

这是合法的，因为 B 的 block-table entry 可以记录：

```text
logical_tokens = 2
```

如果 B 之后再 append，则进入 COW。

所以：

```text
truncate shared view
```

不应该破坏另一个 request 的 physical contents。

---

## 10　Release 与引用计数

A/B 共享两个 blocks。

释放 A：

```text
refcount: 2 → 1
```

block 不能回 free list。

再释放 B：

```text
refcount: 1 → 0
used_tokens → 0
block_id → free_list
```

测试会验证：

```text
release parent
→ child still owns blocks

release child
→ all blocks reclaimed
```

---

## 11　Allocator Metrics

当前 metrics：

```text
total_blocks
allocated_blocks
free_blocks
logical_requests
logical_tokens
physical_token_slots_used
physical_token_capacity
internal_fragmentation_slots
shared_block_references
utilization
```

其中：

$$
\mathrm{utilization}
=
\frac{\mathrm{physical\ token\ slots\ used}}
{\mathrm{allocated\ blocks}\times\mathrm{block\ size}}.
$$

注意另一个很有用但不同的量：

$$
\mathrm{sharing\ gain}
\approx
\frac{\mathrm{logical\ tokens}}
{\mathrm{physical\ token\ slots\ used}}.
$$

当大量 request 共用 system prompt / repository context 时，这个比例可以明显大于 1。

---

## 12　这一版还没有做什么

`kv_block_allocator.py` 现在只是：

```text
physical block metadata allocator
```

它**没有**声称已经实现：

```text
GPU KV slab
block-table attention kernel
paged FlashAttention
heterogeneous batch kernel
GPU allocator concurrency
cross-device block migration
prefix hash index
LRU/clock eviction
preemption/swap
```

因此当前路径是：

```text
ReferencePagedKVCache
        ↓
KVBlockAllocator metadata semantics
        ↓
TensorBlockPool
        ↓
block-table attention
        ↓
heterogeneous continuous batching
```

---

## 13　为什么这一步比“直接调 vLLM API”更重要

如果只会写：

```python
engine = LLM(...)
```

你知道 vLLM 能服务模型，但并不知道：

```text
request lifecycle
→ logical blocks
→ physical allocation
→ prefix sharing
→ refcount
→ COW
→ reclaim
→ OOM
```

如何发生。

这一课的目标是让这些 serving 系统对象变成你亲手写过、能画出状态转移、能构造反例的代码。

---

## 14　自动测试覆盖

`tests/test_kv_block_allocator.py` 专门覆盖：

```text
basic allocation + fragmentation
exact fork + refcount sharing
COW on shared partial final block
partial-prefix clone
truncate shared logical view
release/reclaim
multi-block OOM atomicity
COW OOM atomicity
```

这组测试的意义不是性能，而是锁死 allocator contract。

---

## 15　下一步实验

下一阶段不再继续堆 metadata API，而要把 allocator 接上真正的 tensor pool：

```text
PhysicalBlock ID
      ↓
[layer, block, H_kv, block_size, D]
K/V tensor slabs
      ↓
request block table
      ↓
attention gather / page-aware kernel
```

然后才能真正比较：

```text
contiguous KV
vs
reference paged materialization
vs
physical block tensor pool
```

指标：

```text
HBM bytes/request
internal fragmentation
prefix-sharing gain
allocation latency
TTFT
TPOT
throughput
```

这才是从“理解 PagedAttention”走向“自己实现 serving engine”的下一道门槛。
