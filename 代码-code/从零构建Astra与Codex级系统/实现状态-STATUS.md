# 实现状态：v3 Reference System

> 本文件只记录**已经落地并有证据的代码**。完整机器可读状态以 [`../../能力清单-CAPABILITIES.json`](../../能力清单-CAPABILITIES.json) 为准；README / 看板不能把计划冒充实现。

---

# 1　Model Runtime / Inference

当前已实现并测试：

- [x] UTF-8 byte tokenizer / 教学版 byte-level BPE
- [x] `ModelConfig` 与 tensor contract
- [x] Embedding / RMSNorm / RoPE / causal mask
- [x] half-split 与 interleaved RoPE
- [x] Grouped-Query Attention (GQA)
- [x] SwiGLU / residual blocks / LM head
- [x] contiguous per-layer KV Cache
- [x] prefill / one-token incremental decode
- [x] greedy / temperature / top-k / top-p / repetition penalty
- [x] raw safetensors loader / key remap / shape audit
- [x] full-forward vs cached-decode logits parity
- [x] `HuggingFaceTB/SmolLM2-135M` public checkpoint adapter
- [x] own runtime vs HF eager reference logits parity
- [x] **reference Paged KV Cache + page table + paged decode parity**

真实 checkpoint 受测 CPU float32 结果：

```json
{"max_abs": 0.0, "mean_abs": 0.0, "argmax_agreement": 1.0}
```

只对当前受测 checkpoint / input / numerical setting 成立。

## Reference Paged KV 的边界

`paged_cache.py` 现在已经建立：

```text
logical page size
→ per-layer K/V pages
→ incremental suffix ingest
→ page table
→ reconstruct past_key_values
→ one-token decode
```

自动测试证明 paged path 的 prefill / decode logits 与 full forward 对齐，并验证页面从：

```text
(2, 2, 1)
```

增长到：

```text
(2, 2, 2)
```

但当前模型 forward 仍会产生完整 K/V，`as_past_key_values()` 也会重建 contiguous tensor。因此这是**Paged KV 语义 reference implementation**，不是 vLLM 级 zero-copy block allocator，也不声称已经获得生产级显存/吞吐收益。

---

# 2　Agent / Coding Runtime

当前已实现：

- [x] explicit JSON tool-call protocol + schema subset
- [x] tool registry / error-as-observation
- [x] rooted filesystem / repository search
- [x] exact-match ambiguity-safe editing
- [x] shell + exit-code observation
- [x] Git status / diff / log / show
- [x] Python AST + Markdown repo map
- [x] minimal observe → act → observe loop
- [x] Coding Agent contract
- [x] typed task DAG / cycle detection / failure propagation
- [x] external file / command / composite verifier
- [x] SQLite persistent event memory
- [x] worktree primitive
- [x] deterministic coordinator primitive
- [x] minimal HTTP text browser

---

# 3　Codex Source-Anatomy / MiniCodex

官方源码课程：[`../../Codex源码解剖-codex-anatomy/README.md`](../../Codex源码解剖-codex-anatomy/README.md)

`codex_harness.py` 当前重建：

```text
TurnSettings
→ TURN_STARTED
→ model sampling
→ optional tool call
→ approval request / decision
→ tool execution
→ observation
→ follow-up sampling
→ TURN_COMPLETED / TURN_STOPPED
```

自动验证：

- [x] tool → observation → follow-up → final
- [x] approval deny → tool body 不执行
- [x] approval allow → tool 执行
- [x] max-model-step → explicit `TURN_STOPPED`

`SandboxPolicy` 仍只是 metadata contract；不等于 OS isolation。

---

# 4　Durable Thread / Event Store

`durable.py` 已实现：

- [x] persistent `ThreadId` / `TurnId`
- [x] append-only SQLite event log
- [x] `THREAD_CREATED / USER_SUBMISSION / TURN_STARTED / TURN_COMPLETED`
- [x] checkpoint event
- [x] pause / resume
- [x] completed / failed terminal state
- [x] `project(thread_id)` 通过 event replay 重建当前状态
- [x] 关闭数据库后重新打开，可恢复 running turn、submission 和 checkpoint
- [x] invalid state transition tests

这解决了“任务状态只活在一个 Python 进程内”的最小问题。

---

# 5　Durable Work Queue / Lease / Cancellation

新增 `runtime_queue.py`，把长期任务进一步从 thread state 推到 worker execution：

```text
enqueue
→ PENDING
→ claim(worker)
→ LEASED
├─ ack → COMPLETED
├─ fail → FAILED
├─ cancel → CANCELLED
└─ lease timeout → another worker reclaim
```

已经自动验证：

- [x] active lease 阻止第二 worker 重复领取；
- [x] lease 过期后另一 worker 可以 reclaim；
- [x] ACK 后持久化 result；
- [x] cancel 后该 work item 不再可 claim。

这仍不是分布式生产队列：还没有 heartbeat renewal、priority/fairness、跨节点 clock/transaction 设计和真正异步 worker pool。

---

# 6　Context Provenance / Compaction Ledger

新增 `context.py`。现在 memory 不再只是“把字符串塞 SQLite”，而开始区分：

```text
RAW_EVENT
NOTE
SUMMARY
ARTIFACT
RETRIEVAL
INSTRUCTION
```

每个 `ContextFragment` 保存：

```text
fragment_id
kind
content
source
parent_ids
metadata
created_at
```

Compaction 不覆盖原历史，而是：

```text
raw fragment A ─┐
raw fragment B ─┼→ summary fragment
raw fragment C ─┘        │
                         └→ parent provenance
```

自动测试已经证明：summary 建立后仍能沿 lineage 找回原始 tool observations，原始 fragment 本身也不会被删除。

下一步是 token-budget builder、persistent notes、semantic index、artifact provenance 和 historical-window retrieval。

---

# 7　Permission Enforcement

`security.py` 当前执行顺序：

```text
Tool Proposal
→ PermissionProfile
├─ ALLOW
├─ REQUIRE_APPROVAL
└─ DENY
→ optional approval callback
→ ToolRegistry dispatch
```

自动测试已经证明：

- [x] `DENY` 时底层 tool body 调用次数保持 0；
- [x] approval reject 时底层 tool body 不执行；
- [x] approval allow 后才进入 dispatch。

这是 **application-layer execution gate**，仍不是 process/filesystem/network/credential/syscall sandbox。真实 OS / container sandbox 继续是 P0。

---

# 8　Evaluation Metrics

`evaluation.py` 已把 unit correctness 与 capability evaluation 分开。

当前统一记录：

```text
success
verifier_passed
model_steps
tool_calls
tool_failures
approvals_requested
stopped_by_limit
wall_time_s
input_tokens
output_tokens
cost_usd
```

已有 aggregation tests；但 SWE-bench / Browser / OS 等真实 benchmark adapter 尚未实现。

---

# 9　Agent Protocol Runtime

当前：

- [x] JSON-RPC 2.0 educational envelope
- [x] minimal MCP `server/discover`
- [x] `tools/list`
- [x] `tools/call`
- [x] ToolRegistry → MCP schema mapping
- [x] in-process transport / client
- [x] protocol / tool failure tests

当前 `mcp.py` 仍是教学子集。MCP 2026-07-28 的完整 stateless semantics、stdio/HTTP、authorization、tasks/extensions，以及 A2A 1.0 都仍在后续。

---

# 10　最新 Fast CI

加入 reference Paged KV、Context provenance 与 durable work queue 后，Fast CI run 42 已真实通过：

```text
40 passed, 1 warning in 2.46s
Ruff correctness lint: All checks passed
```

当前新增测试明确覆盖：

```text
paged KV prefill/decode parity
context compaction lineage
lease exclusivity / expiry reclaim
work cancellation
```

---

# 11　当前明确未实现

## Inference / Serving

- [ ] SDPA / high-performance FlashAttention backend
- [x] reference paged KV semantics + parity
- [ ] production block allocator / free list / prefix sharing
- [ ] prefix cache
- [ ] chunked / disaggregated prefill
- [ ] continuous batching scheduler
- [ ] speculative decoding
- [ ] grammar-level constrained decoding
- [ ] multi-architecture checkpoint parity matrix

## Post-training

- [ ] tiny SFT pipeline
- [ ] pairwise preference / DPO
- [ ] reward / verifier training lab
- [ ] GRPO / RLVR-style small policy update
- [ ] Agent trajectory learning

## Durable Agent OS

- [x] event-sourced thread replay
- [x] durable leased work queue
- [ ] pending steering integrated into active turn
- [ ] thread fork
- [ ] thread-level cancellation semantics
- [ ] worker heartbeat / lease renewal
- [ ] scoped `AGENTS.md` resolver + instruction provenance
- [x] provenance-aware context fragments / compaction lineage
- [ ] persistent notes / semantic index / artifact provenance
- [ ] App Server / external runtime control plane
- [ ] rollout trace / artifact store

## Security

- [x] pre-dispatch permission / approval primitive
- [ ] enforced process/container sandbox
- [ ] filesystem / network policy
- [ ] credential / secret scope
- [ ] trajectory monitor / audit policy
- [ ] indirect prompt-injection defenses

## Coding / Multi-Agent

- [ ] tree-sitter / LSP semantic intelligence
- [ ] unified diff / semantic patch engine
- [ ] test selection / coverage-aware verifier
- [ ] stable AgentGraph / mailbox
- [ ] parallel worker scheduler
- [ ] reviewer / merge agent
- [ ] controlled 1/2/4/8-agent benchmark

## Browser / Computer / Multimodal

- [ ] JS-capable browser automation
- [ ] DOM / accessibility-tree observation
- [ ] screenshot / mouse / keyboard
- [ ] visual grounding
- [ ] typed multimodal observation object
- [ ] Computer Use verifier

## Evaluation

- [ ] benchmark case / grader framework
- [ ] SWE-bench adapter
- [ ] browser environment adapter
- [ ] OSWorld-style adapter
- [ ] long-horizon crash/recovery benchmark
- [ ] full token / latency / cost instrumentation
