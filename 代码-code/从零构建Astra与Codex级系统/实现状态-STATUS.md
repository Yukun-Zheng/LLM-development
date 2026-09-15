# 实现状态：v3 Reference System

> 本文件只记录**已经落地并有证据的代码**。机器可读事实源以 [`../../能力清单-CAPABILITIES.json`](../../能力清单-CAPABILITIES.json) 为准；README / 看板不能把计划冒充实现。

---

# 1　Model Runtime

已实现并测试：

- [x] UTF-8 byte tokenizer / 教学版 byte-level BPE
- [x] `ModelConfig` 与 tensor contract
- [x] Embedding / RMSNorm / RoPE / causal mask
- [x] half-split 与 interleaved RoPE
- [x] Grouped-Query Attention (GQA)
- [x] SwiGLU / residual blocks / LM head
- [x] raw safetensors loader / key remap / shape audit
- [x] greedy / temperature / top-k / top-p / repetition penalty
- [x] `HuggingFaceTB/SmolLM2-135M` public checkpoint adapter
- [x] own runtime vs HF eager reference logits parity

真实 checkpoint 受测 CPU float32：

```json
{"max_abs": 0.0, "mean_abs": 0.0, "argmax_agreement": 1.0}
```

只对当前受测 checkpoint / input / numerical setting 成立。

---

# 2　Inference / Serving Reference Layer

## Contiguous KV

- [x] per-layer KV Cache
- [x] prefill / one-token incremental decode
- [x] cached decode vs full-forward parity

## Reference Paged KV

`paged_cache.py` 已建立：

```text
logical page size
→ per-layer K/V pages
→ incremental suffix ingest
→ page table
→ reconstruct past_key_values
→ one-token decode
```

测试证明 paged prefill/decode logits 与 full forward 对齐。

**边界**：当前模型仍会产生完整 K/V，`as_past_key_values()` 仍会重建 contiguous tensor；因此这是 Paged KV **语义 reference**，不是 vLLM 级 zero-copy allocator。

## Reference Request Scheduler

`scheduler.py` 已实现：

```text
WAITING
→ PREFILL admission
→ DECODE
→ FINISHED / CANCELLED
```

当前调度策略是可检查的 deterministic decode-first reference，并支持：

- [x] max batch size
- [x] prefill token budget
- [x] request cancellation
- [x] TTFT
- [x] TPOT
- [x] total latency

Fast CI run 47 已验证 admission、decode batch lifecycle 与 serving metrics。

**仍未完成**：真正 batched model executor、continuous batching 下的实际 kernel execution、chunked/disaggregated prefill、throughput/fairness benchmark。

---

# 3　Post-training Reference Primitives

新增 `posttraining.py`，不再只停留在公式章节。

## SFT

已实现：

```text
logits [B,T,V]
+ labels [B,T]
→ causal shift
→ prompt/user token mask (-100)
→ next-token cross entropy
→ backward
→ optional grad clipping
→ optimizer.step()
```

测试包括：

- [x] uniform logits 下 masked CE 与手算 `log(V)` 一致；
- [x] supervised token count 正确；
- [x] tiny Transformer 上真实 AdamW step 会更新参数。

## DPO

已实现：

```text
policy chosen/rejected log p
reference chosen/rejected log p
→ preference log-ratio difference
→ beta-scaled reward margin
→ -log sigmoid(margin)
→ backward / optimizer step
```

测试包括：

- [x] policy == reference 时，DPO loss = `log(2)`；
- [x] policy 参数在 DPO step 后更新；
- [x] reference 参数保持冻结。

Fast CI run 51：

```text
49 passed, 1 warning in 2.11s
Ruff correctness lint: All checks passed
```

**边界**：当前是 objective + optimization primitive，还不是完整 dataset/dataloader/checkpoint/eval training pipeline；RLVR/GRPO-style small-policy update 仍未实现。

---

# 4　Agent / Coding Runtime

已实现：

- [x] explicit JSON tool-call protocol + schema subset
- [x] tool registry / error-as-observation
- [x] rooted filesystem / repository search
- [x] ambiguity-safe exact editing
- [x] shell + exit-code observation
- [x] Git status / diff / log / show
- [x] Python AST + Markdown repo map
- [x] minimal observe → act → observe loop
- [x] Coding Agent contract
- [x] typed task DAG / cycle detection / failure propagation
- [x] external file / command / composite verifier
- [x] worktree primitive
- [x] deterministic coordinator primitive
- [x] minimal HTTP text browser

---

# 5　Codex Source-Anatomy / MiniCodex

`codex_harness.py` 当前 clean-room teaching slice：

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

自动验证：tool follow-up、approval deny/allow、step-limit termination。

`SandboxPolicy` 仍只是 metadata contract；不等于 OS isolation。

---

# 6　Durable Agent Kernel

## Event-sourced Thread / Turn

`durable.py` 已实现：

- [x] persistent `ThreadId` / `TurnId`
- [x] append-only SQLite event log
- [x] submission / turn / checkpoint / pause / resume
- [x] completed / failed / cancelled terminal state
- [x] process close/reopen 后通过 replay 恢复状态
- [x] invalid transition rejection
- [x] **thread fork**：复制 source event prefix 到独立 child stream
- [x] fork provenance：`parent_thread_id / parent_event_id`
- [x] child cancellation 不改变 parent state

Fast CI run 49：

```text
45 passed, 1 warning in 1.95s
Ruff correctness lint: All checks passed
```

## Durable Work Queue

`runtime_queue.py`：

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

已验证 active lease exclusion、expiry reclaim、ACK result persistence、cancel unclaimable。

**仍未完成**：heartbeat renewal、priority/fairness、async worker pool，以及 ThreadStore ↔ WorkQueue ↔ TurnExecutor 的完整组合。

---

# 7　Context / Memory Provenance

`memory.py` 提供 persistent event history；`context.py` 进一步把 model-visible context 分成：

```text
RAW_EVENT
NOTE
SUMMARY
ARTIFACT
RETRIEVAL
INSTRUCTION
```

每个 fragment 保存 source / parent lineage / metadata。

Compaction 不是覆盖历史，而是：

```text
raw A ─┐
raw B ─┼→ summary
raw C ─┘      │
              └→ parent provenance
```

测试证明 summary 生成后仍可追回原始 tool observations。

下一步：token-budget builder、persistent notes、semantic index、artifact provenance、historical-window retrieval。

---

# 8　Permission / Security

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

自动测试证明 DENY / approval reject 时底层 tool body 不执行。

这是 **application-layer execution gate**；真实 OS/container/process/filesystem/network/credential sandbox 仍是 P0。

---

# 9　Evaluation Harness

`evaluation.py` 统一 trajectory metrics：success、verifier、steps、tool calls/failures、approvals、wall time、tokens、cost。

新增 `benchmark.py`：

```text
BenchmarkCase
→ Executor
→ CodexTurnResult / trajectory
→ Grader
→ Grade
→ BenchmarkRecord
→ AggregateMetrics
```

Fast CI run 47 验证 execution 与 grading 分离，toy 两案例得到 `success_rate = 0.5`。

**边界**：这只是 benchmark substrate，不等于 SWE-bench/WebArena/OSWorld。真实 environment setup、artifact/final-state graders 仍要单独实现。

---

# 10　Agent Protocol Runtime

当前已有 minimal MCP：JSON-RPC 2.0、discover、`tools/list`、`tools/call`、ToolRegistry adapter、in-process transport/client。

仍未完成 MCP 完整 transports/auth/tasks/extensions 与 A2A 1.0 runtime。

---

# 11　下一批最高优先级

## Runtime / Serving

- [ ] scheduler → **真实 batched model executor**
- [ ] production Paged KV block allocator / free list / prefix sharing
- [ ] Prefix Cache
- [ ] Chunked / Disaggregated Prefill
- [ ] Speculative Decoding
- [ ] actual throughput / fairness / memory benchmark

## Post-training

- [x] SFT objective / optimization step
- [x] DPO objective / optimization step
- [ ] conversation / pairwise dataset pipeline
- [ ] checkpointable training loops
- [ ] reward / verifier learning lab
- [ ] GRPO / RLVR-style policy update
- [ ] Agent trajectory learning

## Durable Agent OS

- [x] event replay
- [x] fork / cancellation
- [x] durable leased work queue
- [x] context provenance / compaction lineage
- [ ] pending steering integrated into active turn
- [ ] thread/work-queue/turn-executor integration
- [ ] worker heartbeat / lease renewal
- [ ] scoped `AGENTS.md` resolver + instruction provenance
- [ ] App Server / external control plane
- [ ] artifact store / rollout trace

## Security / General Agent

- [ ] enforced OS/container sandbox
- [ ] filesystem/network/secret policies + escape tests
- [ ] JS browser + DOM/A11y + screenshot
- [ ] visual grounding + mouse/keyboard
- [ ] Computer Use verifier

## Evaluation / Coding / Multi-Agent

- [x] benchmark case/grader substrate
- [ ] repository fixture benchmark
- [ ] SWE-bench adapter
- [ ] browser / OS adapters
- [ ] tree-sitter / LSP / semantic patch / test selection
- [ ] parallel workers / mailbox / reviewer / merge
- [ ] 1/2/4/8-agent controlled comparison
