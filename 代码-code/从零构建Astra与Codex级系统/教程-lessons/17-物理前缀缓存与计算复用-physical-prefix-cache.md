# Lesson 17：物理前缀缓存与计算复用

这一课把此前分开的三件事真正接起来：**前缀匹配、物理 KV 块共享、只计算未命中的 suffix**。

前面的 `prefix_cache.py` 已经说明“相同前缀不必重复计算”这个思想；`kv_block_allocator.py` 与 `kv_tensor_pool.py` 又把 KV 从逻辑 Python tuple 推进成有限物理 block pool。现在的问题是：如何让一个新请求真正从另一个活跃请求继承物理 K/V，并少跑一部分 Transformer？

---

## 1　问题定义

假设已经服务过请求 A：

```text
A = [7, 8, 9, 10, 11]
```

随后来了请求 B：

```text
B = [7, 8, 9, 20, 21, 22]
```

二者最长公共前缀长度为 3。

朴素做法：

```text
B 全部 6 token
→ 再做一次完整 prefill
```

物理前缀缓存希望做到：

```text
A 已有物理 KV
        │
        ├── 共享/复制 prefix [7,8,9]
        │
B       └── 只 forward suffix [20,21,22]
```

因此 B 的 prefill 计算量从 6 token 降到 3 token。

---

## 2　为什么“找到相同字符串”还不够

真实 serving runtime 中至少同时存在三个问题：

1. **发现问题**：哪个现存 request 与新 prompt 有最长公共前缀？
2. **存储问题**：前缀对应哪些 physical blocks？可以直接共享哪些？
3. **计算问题**：哪些 token 已经有 K/V 与 causal logits，哪些 suffix 仍必须执行模型？

所以本课实现拆成：

```text
TokenPrefixIndex
      ↓
PrefixMatch(source_request_id, prefix_tokens)
      ↓
PhysicalKVTensorPool.fork_request(...)
      ↓
materialize shared prefix
      ↓
forward unmatched suffix only
      ↓
ingest new K/V into physical pool
```

源码：

- `src/astra_codex/physical_prefix_cache.py`
- `src/astra_codex/kv_tensor_pool.py`
- `src/astra_codex/kv_block_allocator.py`

---

## 3　Reference Prefix Index

当前使用一个逐 token trie：

```text
root
 └─ 7
    └─ 8
       └─ 9
          ├─ 10 ... request A
          └─ 20 ... request B
```

每个 prefix node 记录拥有这个前缀的 live request。

查询：

```text
input tokens
→ 从 root 逐 token 下行
→ 记录最后一个仍有 owner 的节点
→ 得到最长可复用 prefix
```

复杂度上，这不是生产级索引。它的目的只是把语义做透明：

```text
longest prefix discovery
!=
KV ownership
!=
model computation
```

未来会把它替换成 block hash / radix index，而不改变后两层接口。

---

## 4　为什么完整 block 可以共享，partial tail 不能总是共享

设 block size = 2。

A：

```text
[7,8] [9,10] [11]
 B0     B1      B2
```

B 只共享前三个 token：

```text
[7,8] [9]
```

第一块 `[7,8]` 可以安全共享同一个 physical block id：

```text
A ──┐
    ├── block 0 refcount = 2
B ──┘
```

但第二块不能简单让 B 指向 A 的完整 `[9,10]`，因为 B 逻辑上并不拥有 token `10`。

因此 allocator 采用：

```text
完整块 → retain / refcount++
部分尾块 → private clone
```

这条规则非常重要，因为 block table 表示的是**逻辑可见序列**，不能因为物理存储方便而让请求看到额外 suffix。

---

## 5　Exact hit 为什么可以做到 0 次新 forward

假设 B 与 A 完全相同：

```text
A = [1,2,3,4]
B = [1,2,3,4]
```

Transformer 是 causal 的，所以位置 `t` 的 logits 只依赖：

$$
x_{\le t}
$$

而不依赖后续 token。

因此如果 A 已经保存每个位置的 logits，那么 B 的最后一个 prompt position logits 可以直接复用。

当前 reference engine 对 exact hit 验证：

```text
prefill(A) → 1 model forward
prefill(B) → 0 additional model forward
```

同时 B 的 block table 与 A 指向同一组 physical block ids。

这比“命中缓存但还是重新跑模型”更接近 prefix caching 真正要优化的对象。

---

## 6　Partial hit 的数据流

B 与 A 共享 3 token：

```text
A tokens
[7,8,9,10,11]

B tokens
[7,8,9,20,21,22]
```

执行过程：

```text
1. PrefixIndex → match(A, 3)

2. fork_request(A → B, prefix_tokens=3)

3. B physical cache:
   [7,8] shared
   [9]   private partial tail

4. model(
     suffix=[20,21,22],
     past_key_values=materialize(B-prefix)
   )

5. returned present KV
   → ingest only new suffix

6. final B cache length = 6
```

测试再拿：

```text
model([7,8,9,20,21,22])
```

做完整 recomputation，并要求最终 logits 数值对齐。

因此验收不是“看起来用了 cache”，而是：

```text
reused computation
+ physical sharing semantics
+ numerical parity
```

三者同时成立。

---

## 7　共享 partial block 后为什么必须 Copy-on-Write

Exact hit 时，最后一个物理块可能没有填满。

例如 block size = 4：

```text
A = [1,2,3,4] [5,6]
B = [1,2,3,4] [5,6]
```

初始时最后一块可以共享。

如果 B 随后 decode `7`：

```text
B wants [5,6,7]
```

直接写共享块会污染 A。

所以 append 前必须：

```text
shared partial tail
→ allocate private block
→ copy [5,6]
→ decrement old refcount
→ append 7 into private block
```

测试会保存 A 的完整 K/V，再让 B decode，最后逐 tensor 验证 A 完全没有变化。

---

## 8　Source request 被释放后为什么 child 仍必须可用

共享缓存不能隐式依赖 source request 永远活着。

正确的所有权关系是：

```text
request
  ↓
block-table reference
  ↓
physical block refcount
```

而不是：

```text
child request → parent Python object
```

所以：

```text
A shares blocks with B
→ release A
→ refcount 2 → 1
→ physical blocks remain alive
→ B continues decode normally
```

只有 refcount 变成 0 时，物理块才真正进入 free list，并且当前 tensor pool 会清零该 block 的 K/V storage。

---

## 9　这一版已经证明了什么

Fast CPU CI run 179：

```text
136 passed, 14 skipped, 1 warning
Ruff correctness lint: All checks passed
```

其中本课新增测试证明：

```text
TokenPrefixIndex longest live prefix
exact hit → zero additional model forward
partial hit → only suffix is computed
partial reuse logits == full forward logits
full physical blocks share ids/refcounts
partial tail gets private storage
shared partial tail → decode COW
source release does not invalidate child
```

---

## 10　这一版没有证明什么

当前仍然不是 vLLM/SGLang 级 production prefix cache。

它仍有明确缺口：

```text
Token trie            → 还不是 radix/block-hash production index
materialize()         → attention 前仍 gather 成 contiguous K/V
single-request suffix → 还没有 mixed-length batched suffix execution
eviction              → 未实现
admission policy       → 未实现
cross-node cache       → 未实现
GPU benchmark          → 未实现
```

尤其重要：

> **物理 block storage 已经存在，不等于 attention 已经 page-aware。**

当前模型最终仍吃：

```text
[1, H_kv, T, D]
```

的 contiguous history。

真正下一阶段需要让 attention 直接根据：

```text
block_table
+ per-request sequence length
+ physical K/V slabs
```

读取不同请求的 cache，而不是先 `torch.cat`。

---

## 11　下一阶段

这条 inference 主线现在已经从：

```text
contiguous KV
→ logical pages
→ physical block allocator
→ physical tensor slabs
→ physical prefix sharing
```

走到了真正的分水岭。

接下来最关键的是：

```text
page-aware attention reference
→ heterogeneous decode batch
→ scheduler × block allocator × executor
→ continuous batching
→ TTFT / TPOT / throughput / fairness benchmark
```

完成这一层后，我们才真正开始拥有一个“小型现代 LLM serving engine”，而不是只有一组独立 cache primitives。
