# Lesson 19：Continuous Batching 调度闭环

前面几课已经分别有：

```text
Request Scheduler
Physical KV Block Allocator
Physical K/V Tensor Pool
Physical Prefix Cache
Heterogeneous Page-Aware Decode
```

但只要它们彼此独立，仍然不能称为一个 serving engine。

这一课第一次把 request lifecycle 和真实模型执行接起来。

对应源码：

```text
src/astra_codex/continuous_batching.py
src/astra_codex/scheduler.py
src/astra_codex/physical_prefix_cache.py
src/astra_codex/page_aware_decode.py
```

---

## 1　Continuous batching 到底连续在哪里

最简单的静态 batching 是：

```text
collect N requests
→ wait until batch ready
→ generate them together until all finish
→ accept next batch
```

问题是不同请求长度不同：

```text
A finishes at step 5
B finishes at step 80
```

如果 A 的 slot 一直空到 B 结束，GPU 利用率会下降。

Continuous batching 的核心不是“batch 很大”，而是：

> **batch membership 在生成过程中持续变化。**

理想状态：

```text
t0: [A, B]
t1: [A, B]
t2: A finished
    [C, B]      ← 新请求补进来
t3: [C, B, D]   ← 又有新请求进入
```

---

## 2　本课的 Reference Engine

当前类：

```text
ContinuousBatchingReferenceEngine
```

维护：

```text
ReferenceRequestScheduler
        │
        ├→ PhysicalPrefixCacheEngine
        │
        └→ HeterogeneousPageAwareDecodeReference
```

每次 `step()` 有两个 lane：

```text
1. DECODE lane
   所有当前活跃 request
   → heterogeneous page-aware one-token decode

2. PREFILL lane
   当前 WAITING request
   → 在本轮 decode 后立即 admission/prefill
```

因此一个新请求不必等旧请求全部结束才进入系统。

---

## 3　为什么先 decode 再 prefill

当前 scheduler 延续 decode-first 策略：

```text
active decode latency
优先于
new prefill throughput
```

这样可以避免长 prompt prefill 把已经处于交互生成阶段的请求拖住太久。

但如果永远：

```text
有 DECODE → 不做 PREFILL
```

新请求会一直等待到所有旧请求结束。

所以本课把 scheduler API 拆成：

```text
next_decode_batch()
next_prefill_batch()
```

高层 engine 一轮中可以：

```text
decode old requests
→ prefill waiting requests
```

这是 interleaved scheduling 的最小 reference。

未来 chunked prefill 会让二者进一步共享 token budget。

---

## 4　生成 token 与 K/V 的时间关系

这是实现 serving engine 时非常容易搞混的一点。

假设 prompt：

```text
[x1, x2, x3]
```

prefill 后模型输出 logits，sample：

```text
y1
```

此时 cache 里只有：

```text
[x1, x2, x3]
```

`y1` 已经返回用户，但还没有作为输入经过 Transformer。

下一轮 decode 才做：

```text
input y1
+ cached x1..x3
→ cache becomes x1..x3,y1
→ sample y2
```

因此 runtime 对每个请求维护：

```text
pending_token
```

含义是：

> 已经发给用户、但下一轮才要进入 KV cache 的 token。

---

## 5　Request lifecycle

一个请求现在走：

```text
submit
 ↓
WAITING
 ↓ prefill
DECODE
 ↓ emit y1
pending_token = y1
 ↓ next iteration
page-aware decode(y1)
 ↓ emit y2
...
 ↓ EOS / max_new_tokens
FINISHED
 ↓
release physical KV blocks
```

取消则是：

```text
WAITING / DECODE
→ CANCELLED
→ if admitted: release physical cache
```

这开始把 scheduler state 与真正的 memory ownership 联系起来。

---

## 6　异长请求为什么现在可以在同一 decode lane

例如：

```text
A prompt length = 3
B prompt length = 6
```

prefill 完以后：

```text
A physical cache length = 3
B physical cache length = 6
```

下一轮：

```text
page_aware_decode([A,B], [pending_A,pending_B])
```

不会 pad 两个 KV cache，也不会把二者先 materialize 到相同长度。

QKV projection / FFN 走 batch；attention 根据各自 block table 读取。

因此 scheduler 第一次真正能把：

```text
不同历史长度
```

的 request 放进同一个 decode API。

---

## 7　新请求如何在旧请求仍解码时进入

自动测试构造：

```text
t=0   submit old

t=1   step
      prefill old
      old → DECODE

t=1.5 submit new

t=2   step
      decode old
      prefill new
```

验收要求同一 iteration 明确记录：

```text
decode_request_ids  = (old,)
prefill_request_ids = (new,)
```

并且两者都能在这轮产生 token。

这证明 admission 已不再被“只要有老 decode 就永远阻塞 prefill”的策略卡死。

---

## 8　完成后立即回收 block

当请求达到：

```text
EOS
or
max_new_tokens
```

它最后刚 sample 出来的 token 不需要再进入下一次 model step，因此可以立刻：

```text
request → FINISHED
pending_token → None
PhysicalPrefixCacheEngine.release_request(...)
```

allocator 的 refcount 决定物理 block 是否真的回 free list。

如果 block 与另一个 request 共享：

```text
refcount 2 → 1
```

不会误释放。

如果没有其他 owner：

```text
refcount 1 → 0
→ free list
→ tensor slab zeroed
```

---

## 9　TTFT / TPOT / total latency 开始对应真实执行

Scheduler 原本已经定义：

```text
arrival_time
prefill_time
first_token_time
last_token_time
finish_time
```

现在这些时间点不再是纯状态机模拟，而开始由真实 serving loop 驱动。

因此：

$$
TTFT=t_{first}-t_{arrival}
$$

$$
TPOT=\frac{t_{last}-t_{first}}{N_{generated}-1}
$$

$$
Latency=t_{finish}-t_{arrival}.
$$

测试会用确定的 synthetic clock 检查这些数值，同时确认请求结束后 physical blocks 已经全部释放。

---

## 10　这一版为什么还不能直接叫 production Continuous Batching

虽然 lifecycle 已经闭环，但当前仍有几处重要简化。

第一，prefill lane 当前逐请求执行：

```text
for waiting request:
    prefix_cache.prefill(...)
```

它不是 variable-length batched/chunked prefill。

第二，一轮中 decode 和 prefill 是两个显式 model phases：

```text
decode call
then
prefill call(s)
```

还不是统一 token-budget scheduler。

第三，page-aware attention 是 Python reference：

```text
for request
  for physical block
```

没有 fused Triton/CUDA kernel。

第四，batch-level capacity reservation 尚未完成。一个 production scheduler 应在 model execution 前：

```text
estimate new blocks
→ reserve
→ execute model
→ commit
```

避免跨 request 的 partial commit。

---

## 11　接下来真正需要做的 serving 系统

现在链条已经是：

```text
Request
→ Scheduler
→ Prefix Discovery
→ Physical Block Allocation
→ Physical K/V Slabs
→ Page-Aware Heterogeneous Decode
→ Token Emission
→ Request State Update
→ Block Release
```

下一阶段不再需要继续发明 cache abstraction，而是进入 production-serving 问题：

```text
chunked prefill
batch-wide block reservation
preemption / swap
prefix cache eviction/admission
fused page-attention kernel
live-arrival workload generator
TTFT / TPOT / throughput / fairness curves
```

到这里，项目已经从“理解 PagedAttention”开始转向真正写一个自己的小型 serving runtime。
