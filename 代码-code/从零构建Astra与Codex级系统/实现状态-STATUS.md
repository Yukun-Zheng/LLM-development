# 实现状态：v3 Reference System

> 本文件只记录**已经落地且有证据的实现**。机器可读事实源是根目录的 [`../../能力清单-CAPABILITIES.json`](../../能力清单-CAPABILITIES.json) 与 `能力清单-CAPABILITIES-*.json`。任何 `planned` 都不能在 README 中被写成“已完成”。

---

# 1　当前总览

Reference System 现在已经形成七条可运行纵向链：

```text
Model Runtime
raw checkpoint → own Transformer → logits parity

Inference Runtime
contiguous KV → paged KV reference → prefix reuse → scheduler
→ homogeneous batched forward → speculative-decoding reference path

Post-training
masked SFT → DPO → group-relative / RLVR objective primitive

Durable Agent Kernel
submission → durable Thread → WorkItem lease → TurnExecutor
→ journaled tools → checkpoint / crash recovery / ACK
→ live steering → foreground/background lease heartbeat

Control Plane
Durable Event Feed → JSON-RPC App Server → localhost HTTP
→ Bearer AuthN/AuthZ → queue-level thread fencing
→ SSE cursor replay / reconnect

Context / Artifacts
scoped AGENTS.md → prompt injection → ContextStore provenance
→ immutable content-addressed artifacts

Security / Evaluation
permission/approval gate → restricted subprocess execution
→ independent repository final-state grading
```

最新已核验 Fast CPU CI（run **152**）：

```text
119 passed, 1 warning in 8.48s
Ruff correctness lint: All checks passed
```

这 119 个 tests 已经同时覆盖模型数学、真实 checkpoint parity、cache/serving primitives、post-training objectives、MCP/Codex harness、durability、tool idempotency、live steering、lease/reclaim、background heartbeat、artifact integrity、runtime event replay、HTTP/SSE、control-plane AuthN/AuthZ、scoped AGENTS provenance、restricted-process security guards、evaluation 与 repository final-state grading。

---

# 2　Model Runtime

已经实现并验证：

- [x] UTF-8 byte tokenizer / 教学版 byte-level BPE
- [x] `ModelConfig` 与 tensor contract
- [x] Embedding / RMSNorm / RoPE / causal GQA
- [x] half-split / interleaved RoPE
- [x] SwiGLU / residual block / LM head
- [x] raw safetensors loader / key mapping / shape audit
- [x] greedy / temperature / top-k / top-p / repetition penalty
- [x] public checkpoint adapter
- [x] `HuggingFaceTB/SmolLM2-135M` tested real-checkpoint parity

当前受测 CPU float32 / HF eager reference：

```json
{"max_abs": 0.0, "mean_abs": 0.0, "argmax_agreement": 1.0}
```

这个结果只证明当前 checkpoint / input / numerical setting；不能外推为“支持全部 Llama/Qwen/DeepSeek 架构”。

下一步模型侧重点不再是继续复制普通 Transformer，而是 multi-architecture adapter、hybrid attention / SSM reference implementation 与 activation-level parity。

---

# 3　Inference / Serving

## 3.1 Contiguous KV

- [x] per-layer KV cache
- [x] prefill
- [x] one-token incremental decode
- [x] cached decode vs full-forward logits parity

## 3.2 Reference Paged KV

`paged_cache.py` 已实现 logical page growth、per-layer page table 和 reference materialization，并验证 paged prefill/decode 与 full recomputation 对齐。

**未声称 production benefit**：当前 reference path 仍会 materialize contiguous K/V；它不是 vLLM-style physical block allocator + page-aware attention kernel。

## 3.3 Prefix Cache

`prefix_cache.py` 已支持 longest-prefix reuse：

```text
partial prefix reuse → logits / K/V parity
exact prefix hit     → zero additional model forward
```

仍缺 radix/block-hash index、eviction、physical block sharing 和 scheduler integration。

## 3.4 Request Scheduler / Batched Executor

`scheduler.py` 已覆盖 max batch size、prefill token budget、decode-first、cancellation、TTFT、TPOT 和 total latency。

`batch_executor.py` 能让多个等长 request 共用一次真实 batched forward，并把 K/V 拆回独立 request state。

当前最大缺口是 heterogeneous / variable-length continuous batching：

```text
per-request sequence length
+ physical block table
+ page-aware attention backend
+ scheduler ↔ executor integration
```

## 3.5 Speculative Decoding

`speculative.py` 已进入 reference implementation + tests。下一层验收必须加入：

```text
draft/target distribution correctness
accept/reject parity
speedup crossover
real latency benchmark
```

---

# 4　Post-training

## 4.1 SFT

当前实现：

```text
logits [B,T,V]
+ labels [B,T]
→ causal shift
→ prompt/user masking
→ CE
→ backward
→ optimizer step
```

已测试 uniform logits 下 masked CE=`log(V)`，且真实 AdamW step 会更新 tiny Transformer 参数。

## 4.2 DPO

已测试：

```text
policy == reference → loss = log(2)
policy parameters   → update
reference model     → remains frozen
```

## 4.3 Group-relative / RLVR primitive

`rlvr.py` 当前实现：

```text
verifier rewards
→ group-relative advantage
→ policy ratio
→ clipped surrogate
→ optional reference KL
```

**边界**：仍是 objective primitive，不是完整 RLVR pipeline。尚缺 rollout generation、old-policy snapshot、真正 environment verifier、optimizer loop、checkpoint 和 held-out evaluator。

---

# 5　Codex-style Turn Executor

`codex_harness.py` clean-room teaching slice：

```text
TURN_STARTED
→ model sampling
→ optional tool call / approval
→ tool execution
→ observation
→ follow-up sampling
→ TURN_COMPLETED / TURN_STOPPED
```

`HarnessEventSink` 允许事件发生时直接写入 durable runtime feed，而不是必须等 `run_turn()` 返回后批量获取。

这里的 `codex_harness.SandboxPolicy` 只是**模型可见的 turn/session metadata contract**；真正的执行限制现在单独进入 `sandbox.py`，两者不能混为一谈。

---

# 6　Durable Agent Kernel

## 6.1 Thread / Event Store

`durable.py` 已有：persistent ThreadId / TurnId、append-only event log、restart replay、submission/checkpoint、pause/resume、terminal states、fork、parent provenance 与独立 child cancellation。

## 6.2 Work Queue / Lease / Thread-fenced Claim

`runtime_queue.py`：

```text
PENDING
→ LEASED(worker)
├─ ACK → COMPLETED
├─ FAIL → FAILED
├─ CANCEL → CANCELLED
└─ lease expires → another worker reclaim
```

`DurableWorkQueue.claim(...)` 支持 `thread_ids`，thread authorization 已下沉到 SQL claim transaction，而不是先拿到未授权 work 再拒绝。

## 6.3 Foreground + Background Heartbeat

当前有两层 lease protection：

```text
model-sampling boundary
→ LeaseHeartbeat

long blocking model/tool call
→ BackgroundLeaseHeartbeat
```

CI 曾经真实暴露一个 startup race：heartbeat 线程被创建并不代表第一轮续租已经成功，极短 lease 在 CI 调度压力下会先过期。

现在 `BackgroundLeaseHeartbeat.start()` 使用：

```text
synchronous renewal
→ spawn thread
→ immediate thread-side renewal
→ ready barrier
→ start() returns
```

只有后台续租真正 arm 成功后，调用方才继续进入受保护的长阻塞工作。测试还把 protected section 设置为**长于原始 lease**，并验证 competitor 仍不能 reclaim。

这仍不是分布式 fencing-token / consensus protocol；worker 若失去 lease，外部 side effect 仍需更强的 ownership fencing。

## 6.4 Tool Journal / Crash Recovery

`tool_journal.py` 已验证：

- [x] completed call replay 不重复副作用
- [x] restart 后 replay
- [x] idempotency-key rebinding rejection
- [x] STARTED-only non-idempotent call → in-doubt，不盲重试
- [x] recovered TurnId 复用
- [x] `turn_finished` checkpoint 可无 model resampling 完成 work

仍不等于通用 exactly-once；远端服务需 tool-specific reconciliation 或 external idempotency key。

## 6.5 Live Steering

`steering.py` + `ControlledBackend` 已验证：steering 即使在 tool body 执行期间到达，也会在下一次 model sampling 前被持久化消费并注入 transcript。

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

`ContextStore.add_resolved_instructions(...)` 会把真正进入 Coding Agent prompt 的仓库指令保存为 `INSTRUCTION` fragment，并记录 source path、scope、candidate name、truncation、byte budget、project root、cwd 和 order。

## 7.2 Scoped AGENTS.md

`instructions.py` 基于公开 Codex `agents_md.rs` 的已验证行为做 clean-room reference：

```text
nearest project root
→ root ... cwd
→ each directory chooses one candidate
   AGENTS.override.md
   > AGENTS.md
   > configured fallback
→ global byte budget
→ exact source/scope/truncation provenance
```

负对照测试证明 sibling instruction 不会污染当前 cwd。

## 7.3 Artifact Store

`artifacts.py` 使用 SHA-256 content-addressed immutable snapshots。相同 bytes 可复用 object，读取时重新校验 checksum，篡改会失败。

Checksum 只证明 byte integrity，不证明 semantic correctness；后者仍由 verifier/grader/reviewer 负责。

---

# 8　Security

安全栈现在明确拆成四层：

```text
Control-plane AuthN/AuthZ
        ↓
Tool Permission / Approval
        ↓
Restricted Process Execution
        ↓
[planned] OS / Container Isolation
```

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

`control_auth.py` 已形成：

```text
Bearer token
→ Authentication
→ Principal
├─ allowed_methods
└─ thread_ids
→ App Server Authorization
→ SQL-filtered WorkQueue claim
```

已覆盖 missing/invalid/expired/revoked token、method scope、thread scope、scoped `runtime/runOne`、HTTP bearer 与队列级 thread fencing。Authorizer 还已有显式 expiry、revoke、rotate 生命周期。

**边界**：credential metadata 仍是 process-local；没有 OIDC、mTLS、secure distributed persistence、distributed revocation propagation 或外部 policy engine。

## 8.3 Restricted Subprocess Sandbox Guards

新增 `sandbox.py`。当前真正 enforce 的是：

```text
✅ argv-only execution；不经过 shell parser
✅ executable basename allowlist
✅ cwd 不能逃出 workspace root
✅ child environment allowlist；默认不继承任意 secret
✅ wall-clock timeout
✅ timeout 时 kill process group
✅ POSIX CPU / file-size / open-file / process-count rlimits
✅ optional address-space limit
✅ core dump disabled
✅ Linux PR_SET_NO_NEW_PRIVS
✅ output clipping
```

测试会直接验证：cwd escape / disallowed executable / forbidden env key 被拒绝；父进程假 secret 不会出现在 child；长任务超时；Linux child 的 `/proc/self/status` 显示 `NoNewPrivs: 1`。

`coding.py` 现在支持 opt-in：

```text
legacy teaching mode → ShellTool
restricted mode      → SandboxExecTool
```

restricted mode 中 `shell` 会从工具表消失，Agent 只能提交 argv array。

## 8.4 尚未完成的真正 OS / Container Sandbox

当前**没有**声称可以安全运行任意恶意代码。仍未实现：

```text
mount namespace / isolated rootfs
filesystem read/write policy
network namespace / network-none
seccomp or equivalent syscall filtering
capability drop
container/VM boundary
credential broker
host-secret / fs-escape / network-escape adversarial tests
```

因此 machine-readable capability 将：

```text
security.restricted_subprocess_sandbox = validated
security.os_container_sandbox          = planned
```

**Auth ≠ Approval ≠ Restricted Process ≠ OS Sandbox。**

---

# 9　Event Feed / App Server / HTTP / SSE

## 9.1 Durable Runtime Event Feed

`event_stream.py` 是独立于 authoritative Thread event log 的 integration feed，支持 cursor replay、thread/topic filter、high watermark 与 restart-safe replay。Harness events 会在发生时进入 feed。

## 9.2 App Server + HTTP

`app_server.py` 当前暴露 thread lifecycle、runtime execution、artifact listing、event polling 等 JSON-RPC 方法；`http_app_server.py` 已让同一语义真正经过 localhost TCP/HTTP。

HTTP server thread 使用独立 runtime/SQLite handles，避免跨线程复用 SQLite connection。

## 9.3 SSE durable push

`sse_events.py` 已提供 durable-cursor-backed SSE reference path：

```text
catch-up from cursor / Last-Event-ID
→ live delivery
→ disconnect
→ reconnect from last durable event id
```

并复用 control-plane AuthN/AuthZ 做 thread-scoped event visibility。

**边界**：仍是 localhost/reference server；没有 TLS termination、WebSocket duplex channel、production backpressure、rate limiting、OIDC/session gateway 或多节点 fan-out。

配套课程：Lesson 09–14。

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

已经实现：minimal MCP JSON-RPC discover/list/call；task DAG、sequential coordinator、worktree primitive；replayable App Server；HTTP/SSE；scoped worker claim。

仍未实现：完整 MCP transports/auth/tasks/extensions、A2A runtime、persistent AgentGraph/mailbox、parallel reviewer/merge、JS browser、DOM/A11y、screenshot grounding、mouse/keyboard 与 Computer Use verifier。

---

# 12　最新硬证据

Fast CPU CI run **152**：

```text
119 passed, 1 warning in 8.48s
Ruff correctness lint: All checks passed
```

这一轮最重要的新增/强化回归是：

```text
restricted argv-only subprocess execution
workspace cwd escape negative control
executable allowlist negative control
environment secret scrubbing
wall-time timeout + process-group kill
Linux NoNewPrivs = 1 direct observation
SandboxExecTool metadata
Coding Agent shell → sandbox_exec replacement
background heartbeat synchronous arming
heartbeat ready barrier
blocking model call longer than original lease
competitor reclaim remains forbidden
```

CI 之前确实捕获了 heartbeat startup race；不是把 flaky test 删除，而是先补两阶段 arming，再把测试时间窗改成明显大于 scheduler quantum。现在它验证的是 lease/liveness 语义，而不是 CI 调度运气。

---

# 13　下一批最高优先级

## Inference

- [ ] heterogeneous / variable-length batching
- [ ] scheduler ↔ batch executor 完整 request-state integration
- [ ] physical KV block allocator / free list / prefix sharing
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

- [x] durable tool journal / idempotency semantics
- [x] crash recovery for completed side effects / finished checkpoints
- [x] live steering
- [x] foreground + background lease heartbeat
- [x] content-addressed artifact store
- [x] durable cursor-replay runtime event feed
- [x] localhost HTTP JSON-RPC transport
- [x] Bearer AuthN + method/thread AuthZ + credential lifecycle
- [x] queue-level authorized thread claim
- [x] scoped `AGENTS.md` + ContextStore provenance
- [x] SSE durable cursor push / reconnect
- [ ] steering / artifact / verifier → one unified rollout trace
- [ ] WebSocket duplex control channel
- [ ] production session gateway / distributed identity

## Security / General Agent

- [x] restricted subprocess execution guards
- [ ] real OS/container sandbox + escape tests
- [ ] filesystem/network/secret capability broker
- [ ] JS browser / DOM / Accessibility Tree
- [ ] screenshot grounding / mouse / keyboard

## Evaluation / Coding / Multi-Agent

- [x] deterministic repository fixture grader
- [ ] multi-file hidden-test fixture suite
- [ ] SWE-bench adapter
- [ ] tree-sitter / LSP / semantic patch
- [ ] persistent AgentGraph / mailbox
- [ ] parallel workers / reviewer / merge
- [ ] controlled 1/2/4/8-agent success/cost/time comparison
