# 实现状态：v3 Reference System

> 本文件只记录**已经落地且有证据的实现**。机器可读事实源：[`../../能力清单-CAPABILITIES.json`](../../能力清单-CAPABILITIES.json) 与 `能力清单-CAPABILITIES-*.json`。任何 `planned` 都不能在 README 中被写成“已完成”。

---

# 1　当前总览

Reference System 已经形成六条可运行纵向链：

```text
Model:
raw checkpoint → own Transformer → logits parity

Inference:
contiguous KV → reference paged KV → prefix reuse → scheduler
→ real homogeneous batched forward → speculative decoding reference path

Post-training:
masked SFT → DPO → GRPO-style group-relative objective primitive

Agent Runtime:
submission → durable Thread → leased WorkItem → TurnExecutor
→ journaled tools → checkpoint / recovery / ACK
→ live steering / lease heartbeat

Control Plane / Context:
Durable Event Feed → JSON-RPC App Server → real localhost HTTP
→ Bearer AuthN → Method/Thread AuthZ
→ scoped AGENTS.md instructions + provenance
→ content-addressed artifacts

Evaluation:
toy case/grader → deterministic repository fixture
→ independent final-state verification
```

最新已核验 Fast CPU CI（run 114）：

```text
100 passed, 1 warning in 5.67s
Ruff correctness lint: All checks passed
```

这 100 个 tests 覆盖模型数学、cache/parity、真实 checkpoint、推理 primitives、post-training、工具/MCP、Codex harness、durability、tool idempotency、live steering、worker lease、artifact integrity、replayable runtime events、HTTP App Server、control-plane AuthN/AuthZ、scoped AGENTS instructions、security、evaluation 与 repository final-state grading。

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

## 3.1 Contiguous KV

- [x] per-layer KV cache
- [x] prefill
- [x] one-token incremental decode
- [x] cached decode vs full-forward logits parity

## 3.2 Reference Paged KV

`paged_cache.py` 已实现 logical pages、page growth 与 reference materialization，并验证 paged prefill/decode 与 full recomputation 对齐。

**未声称 production benefit**：当前 reference path 仍可 materialize 为 contiguous K/V；它不是 vLLM-style physical block allocator + page-aware attention kernel。

## 3.3 Prefix Cache

`prefix_cache.py` 已支持 longest-prefix reuse：

```text
partial prefix reuse → logits / K/V parity
exact prefix hit     → no extra model forward
```

下一步是 radix/block-hash index、eviction、physical block sharing 与 scheduler integration。

## 3.4 Request Scheduler

`scheduler.py` 已测试 max batch size、prefill token budget、decode-first、cancellation、TTFT、TPOT 与 total latency。

## 3.5 真实 Batched Model Forward

`batch_executor.py` 能让多个等长 request 共用一次真实 batched forward，并把 K/V 拆回独立 request state。

当前限制：同一 decode batch 仍要求相同 cached sequence length。真正 heterogeneous continuous batching 需要 per-request lengths / block tables / specialized attention backend。

## 3.6 Speculative Decoding

`speculative.py` 已进入 reference implementation + tests。后续重点：draft/target distribution correctness、accept/reject parity、speedup crossover 与真实 latency benchmark。

---

# 4　Post-training

源码：`posttraining.py`, `rlvr.py`。

## SFT

```text
logits [B,T,V]
+ labels [B,T]
→ causal shift
→ prompt/user masking
→ CE
→ backward
→ optimizer step
```

已测试 uniform logits 下 masked CE=`log(V)`，且真实 AdamW step 更新 tiny Transformer 参数。

## DPO

已测试 `policy == reference → loss = log(2)`；policy 更新而 reference 保持冻结。

## GRPO-style / RLVR primitive

当前实现 verifier rewards → group-relative advantage → clipped surrogate → optional reference KL。

**边界**：仍是 objective primitive，不是完整 RLVR pipeline；尚缺 rollout generation、old-policy snapshot、真实 verifier environment、optimizer loop 与 held-out evaluator。

---

# 5　Codex-style Turn Executor

`codex_harness.py` 当前状态：

```text
TURN_STARTED
→ model sampling
→ optional tool call / approval
→ tool execution
→ observation
→ follow-up sampling
→ TURN_COMPLETED / TURN_STOPPED
```

新增 `HarnessEventSink` 后，事件不再只能等 `run_turn()` 结束后批量返回，而可以在发生时写入 durable runtime feed。

`SandboxPolicy` 仍只是 metadata contract，不是 OS isolation。

---

# 6　Durable Agent Kernel

## 6.1 Thread / Event Store

`durable.py` 已有 persistent ThreadId/TurnId、append-only event log、restart replay、submission/checkpoint、pause/resume、terminal states、fork、parent provenance 与独立 child cancellation。

## 6.2 Work Queue / Lease

`runtime_queue.py`：

```text
PENDING
→ LEASED(worker)
├─ ACK → COMPLETED
├─ FAIL → FAILED
├─ CANCEL → CANCELLED
└─ lease expires → another worker reclaim
```

`runtime_control.py` 的 `LeaseHeartbeat` 已验证 lease renewal 能阻止过早 reclaim。

**边界**：当前 integrated runtime 在 model sampling 边界续约。单个超长 tool/RPC 仍需要 background heartbeat 或 tool-owned lease。

## 6.3 Tool Journal / Crash Recovery

`tool_journal.py` 已验证：completed call replay 不重复副作用、restart 后 replay、key rebinding rejection、STARTED-only in-doubt protection、recovered TurnId 复用、`turn_finished` checkpoint 无 model resampling 完成 work。

仍不等于通用 exactly-once；远端服务需 tool-specific reconciliation / external idempotency key。

## 6.4 Live Steering

`steering.py` + `ControlledBackend`：

```text
heartbeat
→ atomically consume pending steering
→ append steering to transcript
→ model.generate(...)
```

已测试 steering 在 tool body 执行期间到达时，下一次 model sampling 可看到新要求。

---

# 7　Context / Instructions / Artifacts

## 7.1 Context provenance

`context.py` 支持：

```text
RAW_EVENT
NOTE
SUMMARY
ARTIFACT
RETRIEVAL
INSTRUCTION
```

Compaction summary 保留 parent lineage，raw evidence 不被覆盖。

## 7.2 Scoped AGENTS.md

新增 `instructions.py`，基于公开 Codex `agents_md.rs` 的已验证行为做 clean-room reference：

```text
nearest project root
→ root ... cwd
→ each directory selects one candidate
   AGENTS.override.md
   > AGENTS.md
   > configured fallbacks
→ one global byte budget
→ exact source/scope/truncation provenance
```

`coding.py` 已把 resolved instructions 注入 Coding Agent system prompt。

负对照测试证明：

```text
root instruction   → visible
cwd instruction    → visible
sibling instruction→ not visible
```

当前未复现 Codex 的 remote filesystem abstraction、multi-environment labeling、project trust gating 和完整 config layering。

## 7.3 Artifact Store

`artifacts.py` 使用 SHA-256 content-addressed immutable snapshots；相同 bytes 可复用 object，读取时重新校验 checksum，篡改会报错。

Checksum 证明 byte integrity，不证明 semantic correctness；后者仍需要 verifier/grader/reviewer。

---

# 8　Security

## 8.1 Tool permission gate

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

DENY / approval reject 时底层 tool body 不执行。

## 8.2 Control-plane AuthN/AuthZ

新增 `control_auth.py` 与 App Server integration：

```text
Bearer token
→ Authentication
→ Principal
├─ allowed_methods
└─ thread_ids
→ Authorization
→ App Server dispatch
```

已验证：

```text
missing / invalid token → -32001 Unauthenticated
wrong method            → -32003 Forbidden
wrong thread            → -32003 Forbidden
scoped event/poll(all)  → Forbidden
scoped runtime/runOne   → Forbidden
admin global worker     → allowed
HTTP bearer             → same policy as in-process
```

特别地，当前 `runtime/runOne` 是 global queue claim，没有 thread selector，因此 thread-scoped principal 被显式禁止调用；不能用 API 层授权假装 queue 层已经有 tenant isolation。

Bearer token 当前仅做 SHA-256 digest → in-memory Principal mapping。没有 expiry、rotation、OIDC、mTLS、distributed policy engine。

## 8.3 仍缺真正 sandbox

真实 OS/container enforcement 仍未实现：process isolation、filesystem namespace/policy、network isolation、credential scope、resource limits、escape tests。

**Auth ≠ Permission ≠ Sandbox。** 三层不能互相冒充。

---

# 9　Event Feed / App Server / HTTP

## 9.1 Durable Runtime Event Feed

`event_stream.py` 是独立于 authoritative Thread event log 的 integration feed：

```text
event_id
+ topic
+ payload
+ thread_id
+ turn_id
+ created_at
```

支持：

```text
afterEventId cursor replay
thread filter
topic filter
high watermark
restart-safe replay
```

`CodexHarness` 事件通过 `event_sink` 在发生时进入 feed，而不是等整个 Turn 完成。

## 9.2 App Server

`app_server.py` 当前方法：

```text
server/discover
thread/create / get / submit / steer
thread/pause / resume / cancel / fork
runtime/runOne
artifact/list
event/poll
```

`event/poll` 提供 cursor-based reconnect/replay。

## 9.3 Real localhost HTTP

`http_app_server.py` 已让 JSON 真正走：

```text
AgentAppClient
→ HTTPAppTransport
→ localhost TCP/HTTP
→ LocalHTTPAppServer
→ AgentAppServer
→ DurableAgentRuntime
```

HTTP server thread 为自己打开独立 runtime/SQLite handles，避免跨线程复用 SQLite connection。

现在还支持：

```text
Authorization: Bearer <token>
```

并在 HTTP / in-process 两种 transport 上使用同一 App Server policy。

**边界**：默认仅 loopback、serialized reference server；无 TLS、SSE/WebSocket、production concurrency、rate limit、OIDC 或公网 hardening。

配套课程：Lesson 09–12。

---

# 10　Evaluation

`evaluation.py` + `benchmark.py` 提供 trajectory metrics 与 case/grader abstraction。

`repository_eval.py` 已有 deterministic coding fixture，并验证：

```text
Agent says "I fixed it" but never edits → FAIL
real edit + command verification      → PASS
```

下一步：multi-file hidden-test fixtures → patch artifacts → SWE-bench adapter。

---

# 11　Protocols / Multi-Agent / Computer Use

已实现：minimal MCP JSON-RPC discover/list/call；task DAG、coordinator、worktree primitive；replayable App Server control plane；local HTTP bearer auth。

仍未实现：完整 MCP transports/auth/tasks/extensions、A2A runtime、persistent AgentGraph/mailbox、parallel reviewer/merge、JS browser、DOM/A11y、screenshot/grounding、mouse/keyboard、Computer Use verifier。

---

# 12　最新硬证据

Fast CPU CI run 114：

```text
100 passed, 1 warning in 5.67s
Ruff correctness lint: All checks passed
```

本轮新增回归覆盖：

```text
restart-safe runtime event cursor replay
thread/topic event filtering
live harness event publication
real localhost HTTP round-trip
Bearer authentication
method/thread-scoped authorization
scoped-principal negative controls
AGENTS root→cwd ordering
override/fallback priority
global byte budget/truncation
project-root boundary
sibling instruction negative control
Coding Agent instruction injection
```

---

# 13　下一批最高优先级

## Inference

- [ ] heterogeneous / variable-length batching
- [ ] scheduler ↔ batch executor 完整 request-state integration
- [ ] real physical block allocator / free list / prefix sharing
- [ ] chunked / disaggregated prefill
- [ ] speculative decoding real speed/distribution benchmark
- [ ] throughput / memory / fairness benchmark

## Post-training

- [ ] conversation / pairwise dataset pipeline
- [ ] checkpointable SFT/DPO loops
- [ ] verifier-generated rollout dataset
- [ ] actual tiny-policy group-relative update
- [ ] train-reward vs held-out evaluator separation experiment

## Agent OS / Control Plane

- [x] durable tool execution journal / idempotency semantics
- [x] crash recovery for completed side effects / finished checkpoints
- [x] live steering
- [x] worker lease renewal primitive
- [x] content-addressed artifact store
- [x] durable cursor-replay runtime event feed
- [x] App Server `event/poll`
- [x] real localhost HTTP JSON-RPC transport
- [x] Bearer AuthN + method/thread AuthZ
- [x] scoped `AGENTS.md` resolver + coding-prompt injection
- [ ] background heartbeat for one long tool/RPC
- [ ] steering/instruction → unified ContextStore provenance
- [ ] SSE / WebSocket push backed by durable cursor
- [ ] token expiry / rotation / session identity / queue-level tenant filtering
- [ ] rollout trace linked to artifacts/context/instructions

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
