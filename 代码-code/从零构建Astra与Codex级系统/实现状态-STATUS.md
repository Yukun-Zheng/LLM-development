# 实现状态：v3 Reference System

> 本文件只记录**已经落地且有证据的实现**。机器可读事实源：[`../../能力清单-CAPABILITIES.json`](../../能力清单-CAPABILITIES.json)。任何“planned”都不能在 README 里被写成“已完成”。

---

# 1　当前总览

Reference System 现在已经形成六条可运行纵向链：

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

Control Plane / Artifacts:
JSON-RPC App Server → thread control → runtime execution
→ content-addressed artifact snapshots

Evaluation:
toy case/grader → deterministic repository fixture
→ independent final-state verification
```

最新已核验 Fast CPU CI（run 91）：

```text
86 passed, 1 warning in 5.14s
Ruff correctness lint: All checks passed
```

这 86 个 tests 覆盖模型数学、cache/parity、真实 checkpoint、推理 primitives、post-training、工具/MCP、Codex harness、durability、tool idempotency、live steering、worker lease、artifact integrity、App Server、security、evaluation 与 repository final-state grading。

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

`paged_cache.py` 已实现 logical pages、page growth 与 reference materialization，并验证 paged prefill/decode 与 full recomputation 对齐。

**未声称 production benefit**：当前 reference path 仍可 materialize 为 contiguous K/V；它不是 vLLM-style physical block allocator + page-aware attention kernel。

## 3.3　Prefix Cache

`prefix_cache.py` 已支持 longest-prefix reuse。已有测试证明：

```text
partial prefix reuse → logits / K/V parity
exact prefix hit     → no extra model forward
```

下一步是 radix/block-hash index、eviction、physical block sharing 与 scheduler integration。

## 3.4　Reference Request Scheduler

`scheduler.py`：

```text
WAITING
→ PREFILL
→ DECODE
→ FINISHED / CANCELLED
```

已实现并测试 max batch size、prefill token budget、decode-first、cancellation、TTFT、TPOT 与 total latency。

## 3.5　真实 Batched Model Forward

`batch_executor.py` 能让多个等长 request 共用一次真实 batched forward，并把 K/V 拆回独立 request state。

当前限制：同一 decode batch 仍要求相同 cached sequence length。真正 heterogeneous continuous batching 需要 per-request lengths / block tables / specialized attention backend。

## 3.6　Speculative Decoding

`speculative.py` 已进入 reference implementation + tests 阶段。后续重点不再是“有这个函数”，而是：

```text
draft / target distribution correctness
accept/reject parity
speedup crossover
real latency benchmark
```

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

```text
policy chosen/rejected log p
reference chosen/rejected log p
→ implicit reward margin
→ -log sigmoid(margin)
→ policy update
```

已测试 `policy == reference → loss = log(2)`；policy 更新而 reference 保持冻结。

## GRPO-style / RLVR primitive

当前实现 verifier rewards → group-relative advantage → clipped surrogate → optional reference KL。

**边界**：仍是 objective primitive，不是完整 RLVR pipeline；尚缺 rollout generation、old-policy snapshot、真实 verifier environment、optimizer loop 与 held-out evaluator。

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

## 6.1 Thread / Event Store

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

`runtime_control.py` 新增 `LeaseHeartbeat`：worker 可以延长自己的 lease，测试证明续约期间另一个 worker 不能提前 reclaim。

**边界**：当前 integrated runtime 在 model sampling 边界续约。若单个外部 tool/RPC 本身运行时间超过 lease，还需要 background heartbeat 或 tool-owned lease，不能把当前实现宣传成完整 distributed liveness。

## 6.3 Durable Tool Journal / Crash Recovery

`tool_journal.py` 已实现稳定 scope + call index 的 durable execution record：

- [x] completed tool call 可 replay，不重复执行 side effect；
- [x] restart 后 replay 仍成立；
- [x] key rebinding 被拒绝；
- [x] STARTED-only 非幂等调用标为 in-doubt，不盲目 retry；
- [x] recovered turn 可复用 active TurnId；
- [x] `turn_finished` checkpoint 可直接完成 work，不重新 model sampling。

这仍不等于通用 exactly-once；外部服务如果“已提交但本地 COMPLETED 尚未落盘”，仍需要 tool-specific reconciliation / external idempotency key。

## 6.4 Live Steering

新增 `steering.py`：

```text
PENDING
→ CONSUMED
or
→ CANCELLED
```

新增 `ControlledBackend`：每次 model sampling 前：

```text
heartbeat
→ atomically consume pending steering
→ append steering to transcript
→ model.generate(...)
```

自动测试专门模拟 steering 在 tool body 执行期间到达，并验证**下一次** model sampling 能看到新要求，不必重启整个 turn。

当前 steering 有独立 durable inbox；下一步会把 consumption 同步进 Thread event stream，形成统一 replay provenance。

---

# 7　Context / Memory / Artifact

`memory.py`：persistent event history。  
`context.py`：model-visible context provenance。  
`artifacts.py`：immutable content-addressed work products。

Context fragments 支持 raw observation、note、summary、artifact、retrieval、instruction，并保证 compaction summary 可沿 lineage 找回原始证据。

Artifact Store 使用 SHA-256 对 bytes 做 content-addressed snapshot：

```text
artifact_id
thread_id
kind
sha256
size_bytes
metadata
created_at
```

相同 bytes 可复用同一 object；读取时重新校验 checksum，篡改会直接报错。

重要边界：checksum 只能证明 bytes 完整性，不能证明语义正确性；语义正确仍需要 verifier / grader / review。

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

真实 sandbox 仍缺：process isolation、filesystem namespace/policy、network isolation、credential scope、resource limits、audit/escape tests。

---

# 9　App Server / 外部控制面

新增 `app_server.py`，把 UI/CLI/IDE 与 Agent Kernel 解耦成最小 JSON-RPC control plane。

当前方法：

```text
server/discover
thread/create
thread/get
thread/submit
thread/steer
thread/pause
thread/resume
thread/cancel
thread/fork
runtime/runOne
artifact/list
```

测试验证 thread lifecycle、runtime execution、steering、artifact metadata、fork provenance 与 protocol errors。

**边界**：当前只有 in-process transport；没有 HTTP/WebSocket、streaming subscription、auth、multi-tenant ACL，因此不声称 API-compatible/production App Server。

配套课程：[`教程-lessons/09-实时Steering租约与AppServer-live-control-plane.md`](教程-lessons/09-实时Steering租约与AppServer-live-control-plane.md)。

---

# 10　Evaluation

`evaluation.py` + `benchmark.py` 已提供 trajectory metrics 和 case/grader abstraction。

`repository_eval.py` 提供 deterministic coding fixture：

```text
initial files
goal
verify argv
expected final files
```

独立 final-state grader 已测试：

```text
Agent says "I fixed it" but never edits → FAIL
real edit + command verification      → PASS
```

下一步：multi-file hidden-test fixtures → patch artifacts → SWE-bench adapter。

---

# 11　Protocols / Multi-Agent / Computer Use

已实现：minimal MCP JSON-RPC discover/list/call；task DAG、coordinator、worktree primitive；最小 App Server control plane。

仍未实现：完整 MCP transports/auth/tasks/extensions、A2A runtime、persistent AgentGraph/mailbox、parallel reviewer/merge、JS browser、DOM/A11y、screenshot/grounding、mouse/keyboard、Computer Use verifier。

---

# 12　最新硬证据

Fast CPU CI run 91：

```text
86 passed, 1 warning in 5.14s
Ruff correctness lint: All checks passed
```

本轮新增回归覆盖：

```text
worker lease heartbeat
mid-turn live steering
artifact checksum / dedup / tamper detection
JSON-RPC App Server thread lifecycle
App Server steering + artifact listing
thread fork provenance through control plane
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
- [x] pending live steering primitive
- [x] worker lease renewal primitive
- [x] content-addressed artifact store
- [x] minimal in-process App Server
- [ ] background heartbeat for long single tool/RPC execution
- [ ] steering → unified Thread event provenance
- [ ] scoped `AGENTS.md` resolver + provenance
- [ ] streaming Event subscription
- [ ] stdio / HTTP / WebSocket transport
- [ ] auth / session / multi-tenant permission model
- [ ] rollout trace linked to artifacts

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
