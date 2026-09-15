# Lesson 19：Continuous Batching 调度闭环

前面几课已经分别有：

```text
Request Scheduler
Physical KV Block Allocator
Physical K/V Tensor Pool
Physical Prefix Cache
Heterogeneous Page-Aware Decode
```

但只要它们彼此独立，仍然不能称为一个 serving engine。这一课第一次把 request lifecycle、真实模型执行和物理内存所有权接起来。

对应源码：

```text
src/astra_codex/continuous_batching.py
src/astra_codex/scheduler.py
src/astra_codex/physical_prefix_cache.py
src/astra_codex/page_aware_decode.py
src/astra_codex/kv_block_allocator.py
src/astra_codex/kv_tensor_pool.py
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
    [C, B]
t3: [C, B, D]
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
                 │
                 └→ concrete physical-block reservation
```

每次 `step()` 有两个 lane：

```text
1. DECODE lane
   active requests
   → heterogeneous page-aware one-token decode

2. PREFILL lane
   WAITING requests
   → decode 后立即 admission / prefill
```

因此一个新请求不必等旧请求全部结束才进入系统。

---

## 3　为什么先 decode 再 prefill

当前 scheduler 延续 decode-first 思路：

```text
active decode latency
优先于
new prefill throughput
```

但如果永远“只要有 DECODE 就不 PREFILL”，新请求会被饿死。所以 scheduler 暴露：

```text
next_decode_batch()
next_prefill_batch()
```

高层 engine 一轮中可以：

```text
decode old requests
→ prefill waiting requests
```

这是 interleaved scheduling 的最小 reference。未来 chunked prefill 会让二者进一步共享统一 token budget。

---

## 4　生成 token 与 K/V 的时间关系

假设 prompt：

```text
[x1, x2, x3]
```

prefill 后 sample：

```text
y1
```

此时 cache 里仍只有：

```text
[x1, x2, x3]
```

`y1` 已返回用户，但还没有作为输入经过 Transformer。下一轮 decode 才做：

```text
input y1
+ cached x1..x3
→ cache becomes x1..x3,y1
→ sample y2
```

因此 runtime 对每个请求维护 `pending_token`：已经发给用户、但下一轮才要进入 KV cache 的 token。

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

这把 scheduler state 与真正的 memory ownership 联系起来。

---

## 6　异长请求为什么现在可以在同一 decode lane

例如：

```text
A prompt length = 3
B prompt length = 6
```

下一轮：

```text
page_aware_decode([A,B], [pending_A,pending_B])
```

不会把两个 KV cache pad 到同一长度，也不会先 materialize 成连续历史 K/V。QKV projection / FFN 走 batch；attention 根据各自 block table 直接读 physical slabs。

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

这证明 admission 不再被旧 decode 请求完全阻塞。

---

## 8　完成后立即回收 block

达到 EOS 或 `max_new_tokens` 后，最后刚 sample 的 token 不再需要进入下一次 model step，可以立刻：

```text
request → FINISHED
pending_token → None
PhysicalPrefixCacheEngine.release_request(...)
```

allocator 的 refcount 决定 block 是否真正回 free list。共享 block 只会从 `2 → 1`；独占 block 才会从 `1 → 0 → free list`，tensor slab 同时清零。

---

## 9　TTFT / TPOT / total latency 对应真实执行

Scheduler 定义：

```text
arrival_time
prefill_time
first_token_time
last_token_time
finish_time
```

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

测试用 synthetic clock 检查这些数值，并确认 terminal request 不留下 physical block leak。

---

## 10　为什么只做 OOM preflight 还不够

最初 page-aware decode 只做：

```text
estimate batch block demand
→ check free_blocks >= demand
→ run model
→ append K/V
```

这可以提前发现容量不足，但存在典型 TOCTOU 问题：

```text
check 时 block 是 free
        ↓
模型计算期间别的 allocator path 拿走它
        ↓
commit 时容量消失
```

所以现在又增加了：

```text
KVBlockReservation
```

真正从 allocator 的 free list 中拿走**具体 block ID**。

---

## 11　Concrete block reservation

page-aware batch 在 model math 之前执行：

```text
estimate COW + new-block demand
        ↓
reserve_blocks(N)
        ↓
free list 中移除具体 block ids
        ↓
model / page-aware attention
        ↓
append_batch_delta(..., reservation)
        ↓
consume reserved block ids
        ↓
release unused reservation
```

因此在同一个 allocator 对象内，其他请求走普通：

```text
append_tokens(...)
```

时根本看不到这些 reserved ids。

自动负测试故意：

```text
pool free blocks = 2
batch reserves both
unrelated request tries append
→ MemoryError
```

释放 reservation 后 unrelated request 才能正常拿到 block。

---

## 12　模型执行失败时 reservation 怎么处理

测试在 reservation 已创建以后，故意让：

```text
Embedding.forward
→ RuntimeError
```

`finally` 必须释放所有**未消费** block ids。

验收比较失败前后：

```text
block tables
sequence lengths
free/reserved counts
K slabs
V slabs
```

全部保持一致。

这意味着 capacity fencing 不会因为 model exception 自己造成 block leak。

---

## 13　当前 reservation 仍不等于数据库事务

这一版解决的是：

```text
✅ concrete block ids 从 free list 被隔离
✅ model execution 期间同 allocator 的其他普通 allocation 不能偷走
✅ normal OOM 在 model math 前发现
✅ model failure 会归还未消费 reservation
✅ successful batch consumes reservation and leaves no reserved-block leak
```

但没有声称：

```text
❌ arbitrary tensor-write failure 后完整 rollback
❌ multi-process shared allocator locking
❌ cross-node reservation
❌ distributed fencing token
❌ GPU kernel transaction
```

如果某个异常发生在部分 request 已完成 metadata/tensor commit 之后，当前还没有完整 undo log。这是下一层真正的 transactional commit 问题。

---

## 14　这一版为什么还不能叫 production Continuous Batching

当前仍有四个主要简化。

第一，prefill lane 仍逐请求执行，不是 variable-length batched/chunked prefill。

第二，decode 和 prefill 是两个显式 model phase，还不是统一 token-budget scheduler。

第三，page-aware attention 仍是 Python request/block loop，没有 fused Triton/CUDA kernel。

第四，reservation 是进程内 allocator ownership，不是分布式 serving control plane。

---

## 15　硬证据

Fast CPU CI：

```text
run 189 → continuous-batching lifecycle
          145 passed, 14 skipped

run 195 → batch-wide OOM preflight
          148 passed, 14 skipped

run 199 → concrete physical-block reservation/fencing
          150 passed, 14 skipped
          Ruff correctness lint: All checks passed
```

run 199 新增验证：

```text
reserved ids disappear from normal free list
unrelated allocation cannot steal reserved blocks
model failure returns unconsumed reservation
successful batch consumes reservation
reservation count returns to zero
```

---

## 16　接下来真正需要做的 serving 系统

现在链条已经是：

```text
Request
→ Scheduler
→ Prefix Discovery
→ Physical Block Allocation
→ Physical K/V Slabs
→ Concrete Capacity Reservation
→ Page-Aware Heterogeneous Decode
→ Token Emission
→ Request State Update
→ Block Release
```

下一阶段进入：

```text
chunked prefill + unified token budget
transactional rollback / scheduler-owned reservation
preemption / swap
prefix cache eviction/admission
fused page-attention kernel
live-arrival workload generator
TTFT / TPOT / throughput / fairness curves
```

到这里，项目已经从“理解 PagedAttention”进入真正写自己的小型现代 LLM serving runtime。
