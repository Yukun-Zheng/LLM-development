# 实现状态：v3 Reference System

> 本文件只记录**已经落地且有证据的实现**。机器可读事实源：[`../../能力清单-CAPABILITIES.json`](../../能力清单-CAPABILITIES.json)。任何“planned”都不能在 README 里被写成“已完成”。

---

# 1　当前总览

截至本次更新，Reference System 已经不再只是 `Transformer + Agent loop`，而形成五条真正可运行的纵向链：

```text
Model:
raw checkpoint → own Transformer → logits parity

Inference:
contiguous KV → reference paged KV → scheduler → real homogeneous batched forward

Post-training:
masked SFT → DPO → GRPO-style group-relative objective primitive

Agent Runtime:
submission → durable Thread → leased WorkItem → TurnExecutor → checkpoint / ACK

Evaluation:
toy case/grader → deterministic repository fixture → independent final-state verification
```

Fast CPU CI run 66：

```text
61 passed, 1 warning in 3.67s
Ruff correctness lint: All checks passed
```

这 61 个 tests 同时覆盖模型数学、cache parity、真实 checkpoint、工具、MCP、Codex harness、durability、security、evaluation、serving scheduler、batched execution、SFT/DPO 与 group-relative RL primitive。

---

# 2　Model Runtime

已实现：

- [x] UTF-8 byte tokenizer / 教学版 byte-level BPE
- [x] `ModelConfig` 与 tensor contract
- [x] Embedding / RMSNorm / RoPE / causal attention
- [x] half-split / interleaved RoPE
- [x] GQA
- [x] SwiGLU / residual block / LM head
- [x] raw safetensors loader / key mapping / shape audit
- [x] greedy / temperature / top-k / top-p / repetition penalty
- [x] public checkpoint adapter
- [x] `HuggingFaceTB/SmolLM2-135M` real checkpoint parity

受测 CPU float32 / HF eager reference：

```json
{"max_abs": 0.0, "mean_abs": 0.0, "argmax_agreement": 1.0}
```

边界：只证明当前 checkpoint / input / numerical setting；不能外推为“支持全部 Llama/Qwen/DeepSeek 架构”。

---

# 3　Inference / Serving

## 3.1　Contiguous KV

- [x] per-layer KV cache
- [x] prefill
- [x] one-token incremental decode
- [x] cached decode vs full-forward logits parity

## 3.2　Reference Paged KV

`paged_cache.py`：

```text
logical pages
→ per-layer K/V page table
→ append only new suffix
→ reconstruct reference past_key_values
→ decode
```

测试证明 paged path 的 prefill/decode logits 与 full recomputation 对齐。

**未声称** production benefit：当前仍会 `torch.cat` 回 contiguous K/V；还不是 vLLM-style physical block allocator / block-table kernel。

## 3.3　Reference Request Scheduler

`scheduler.py`：

```text
WAITING
→ PREFILL
→ DECODE
→ FINISHED / CANCELLED
```

已实现并测试：

```text
max_batch_size
prefill token budget
decode-first scheduling
cancellation
TTFT
TPOT
total latency
```

## 3.4　真实 Batched Model Forward

`batch_executor.py` 已经越过“只调度、不执行”的阶段。

当前 `HomogeneousBatchExecutor` 能真正执行：

```text
request A ─┐
request B ─┼→ ONE batched model forward
request C ─┘
```

并把 batched K/V 拆回每个 request 的独立 decode state。

自动测试验证：

```text
batched prefill logits
== individual prefill logits

batched decode logits
== individual full-recomputation logits
```

**当前限制**：同一 decode batch 中各 request 必须有相同 cached sequence length。真正 variable-length continuous batching 仍需要 per-request length / block table / specialized attention backend。

---

# 4　Post-training

源码：`posttraining.py`, `rlvr.py`。

## SFT

```text
logits [B,T,V]
+ labels [B,T]
→ causal shift
→ mask prompt/user target (-100)
→ CE
→ backward
→ optional grad clipping
→ optimizer step
```

测试：uniform logits 下 masked CE=`log(V)`；真实 AdamW step 更新 tiny Transformer 参数。

## DPO

```text
policy chosen/rejected log p
reference chosen/rejected log p
→ implicit reward margin
→ -log sigmoid(margin)
→ update policy
```

测试：

```text
policy == reference → loss = log(2)
policy weights      → changed
reference weights   → unchanged
```

## GRPO-style / RLVR primitive

`rlvr.py` 目前实现：

```text
verifier rewards [B,G]
→ group-relative advantage
→ policy ratio
→ clipped surrogate
→ optional reference KL penalty
```

测试验证：

- [x] advantage 每个 prompt group 内中心化；
- [x] 同组 reward 全相等时 advantage=0；
- [x] `new policy == old policy` 时 objective 数值可以为 0，但 high/low reward 样本仍产生方向正确的独立 gradient；
- [x] clipping 与 KL penalty 显式可观测。

**边界**：这是 objective primitive，不是完整 RLVR training pipeline。尚缺 rollout generation、old-policy snapshot、真正 verifier environment、token-level objective、optimizer loop 与 held-out evaluator。

---

# 5　Codex-style Turn Executor

`codex_harness.py` clean-room teaching slice：

```text
TURN_STARTED
→ model sampling
→ optional tool call
→ optional approval
→ tool execution
→ observation
→ follow-up sampling
→ TURN_COMPLETED / TURN_STOPPED
```

已验证 tool follow-up、approval deny/allow、max-step termination。

`SandboxPolicy` 仍只是 metadata contract，不是 OS isolation。

---

# 6　Durable Agent Kernel

## Thread / Event Store

`durable.py`：

- [x] persistent ThreadId / TurnId
- [x] append-only event log
- [x] process close/reopen replay
- [x] submission / turn / checkpoint
- [x] pause / resume
- [x] completed / failed / cancelled terminal states
- [x] thread fork
- [x] parent provenance (`parent_thread_id`, `parent_event_id`)
- [x] child cancellation 与 parent 独立

## Work Queue

`runtime_queue.py`：

```text
PENDING
→ LEASED(worker)
├─ ACK → COMPLETED
├─ FAIL → FAILED
├─ CANCEL → CANCELLED
└─ lease expires → another worker reclaim
```

## Integrated Runtime

`runtime.py` 已经把原本分散的 primitive 第一次串成：

```text
submit
→ Thread event
→ durable WorkItem
→ worker lease
→ TURN_STARTED
→ CodexHarness
→ checkpoint final result
→ TURN_COMPLETED
→ work ACK
```

测试还覆盖：

- [x] 多 submission 顺序执行；
- [x] terminal/cancelled thread 的 pending work 不执行；
- [x] model-step limit 同时落到 failed work + failed thread；
- [x] runtime 关闭后 Thread projection/checkpoint 仍能从 SQLite 恢复。

**关键未解问题**：worker 如果在 `TURN_STARTED` 后、side-effecting tool 执行期间突然死亡，当前 runtime 不会自动重跑该 turn。自动 at-least-once 重跑可能造成重复副作用；下一步必须引入 durable tool-execution records / idempotency semantics，而不是简单 retry。

---

# 7　Context / Memory

`memory.py`：persistent event history。  
`context.py`：model-visible context provenance。

Typed fragments：

```text
RAW_EVENT
NOTE
SUMMARY
ARTIFACT
RETRIEVAL
INSTRUCTION
```

Compaction：

```text
raw A ─┐
raw B ─┼→ summary S
raw C ─┘      │
              └→ parent lineage
```

测试保证 summary 不覆盖 raw evidence，能够沿 lineage 追回原观察。

下一步：persistent notes、semantic index、artifact provenance、token-budget context builder、historical-window retrieval。

---

# 8　Permission / Security

`security.py`：

```text
Tool Proposal
→ PermissionProfile
├─ ALLOW
├─ REQUIRE_APPROVAL
└─ DENY
→ optional approval
→ dispatch
```

DENY / approval reject 时，底层 tool body 不执行。

这只是 application-layer execution gate。真实 sandbox 仍缺：

```text
process isolation
filesystem namespace/policy
network isolation
credential scope
resource limits
audit / escape tests
```

---

# 9　Evaluation

## 通用 trajectory / case grader

`evaluation.py` + `benchmark.py` 已有：

```text
BenchmarkCase
→ Executor
→ trajectory
→ Grader
→ Grade
→ BenchmarkRecord
→ AggregateMetrics
```

## Level-1 Repository Fixture

新增 `repository_eval.py`。

一个可复现 coding case 现在可以固定：

```text
initial files
goal
verify argv
expected final files
```

执行后**独立于 Agent 自我陈述**运行 final-state grader。

自动测试专门构造两种情况：

```text
Agent says "I fixed it" but never edits
→ FAIL

Agent performs real edit + command verification
→ PASS
```

因此现在已经真正跨过：

```text
final answer
!=
final repository state
```

这一道评测门槛。

下一步：multi-file hidden-test fixtures → patch artifacts → SWE-bench adapter。

---

# 10　Protocols / Multi-Agent / Computer Use

已实现：minimal MCP JSON-RPC discover/list/call；task DAG、coordinator、worktree primitive。

仍未实现：完整 MCP transports/auth/tasks/extensions、A2A runtime、persistent AgentGraph/mailbox、parallel reviewer/merge、JS browser、DOM/A11y、screenshot/grounding、mouse/keyboard、Computer Use verifier。

---

# 11　最新硬证据

Fast CPU CI run 66：

```text
61 passed, 1 warning in 3.67s
Ruff correctness lint: All checks passed
```

这一次总回归已经同时包含：

```text
real checkpoint parity primitives
contiguous/paged KV
scheduler
real homogeneous batched forward
SFT/DPO
GRPO-style objective
Codex turn harness
durable replay/fork/cancel
leased work queue
integrated runtime
context provenance
permission enforcement
benchmark substrate
repository final-state grading
```

---

# 12　下一批最高优先级

## Inference

- [ ] heterogeneous / variable-length batching
- [ ] scheduler ↔ batch executor 完整 request-state integration
- [ ] real block allocator / free list / prefix sharing
- [ ] prefix cache
- [ ] chunked/disaggregated prefill
- [ ] speculative decoding
- [ ] actual throughput / memory / fairness benchmark

## Post-training

- [ ] conversation / pairwise dataset pipeline
- [ ] checkpointable SFT/DPO loops
- [ ] verifier-generated rollout dataset
- [ ] actual tiny-policy group-relative update
- [ ] train-reward vs held-out evaluator separation experiment

## Agent OS

- [ ] durable tool-execution record / idempotency key
- [ ] crash recovery after `TURN_STARTED`
- [ ] pending live steering
- [ ] worker heartbeat / lease renewal
- [ ] scoped `AGENTS.md` resolver + provenance
- [ ] App Server / external control plane
- [ ] artifact / rollout store

## Security / General Agent

- [ ] enforced OS/container sandbox + escape tests
- [ ] fs/network/secret scope
- [ ] JS browser / DOM / Accessibility Tree
- [ ] screenshot / grounding / mouse / keyboard

## Evaluation / Coding / Multi-Agent

- [x] deterministic repository fixture grader
- [ ] multi-file/hidden-test fixture suite
- [ ] SWE-bench adapter
- [ ] tree-sitter / LSP / semantic patch
- [ ] parallel workers / mailbox / reviewer / merge
- [ ] controlled 1/2/4/8-agent comparison
