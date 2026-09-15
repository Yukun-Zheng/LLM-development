# 实现状态：v3 Reference System

> 本文件只记录**已经落地且有自动化证据的实现**。机器可读事实源是根目录的 [`../../能力清单-CAPABILITIES.json`](../../能力清单-CAPABILITIES.json) 与 `能力清单-CAPABILITIES-*.json`。任何 `planned` 都不能在 README 中写成“已完成”。

---

# 1　当前总览

截至当前版本，Reference System 已经形成九条可以实际运行、测试和继续替换优化的纵向链：

```text
Model Runtime
raw public checkpoint
→ own Decoder-only Transformer
→ tested logits parity

Inference Runtime
contiguous KV
→ logical paged KV
→ physical KV block allocator
→ tensor slabs
→ physical prefix sharing
→ page-aware mixed-length decode
→ continuous-batching lifecycle

Post-training
masked SFT
→ DPO
→ group-relative / RLVR objective primitive

Durable Agent Kernel
submission
→ Thread / Turn / WorkItem
→ lease / heartbeat
→ journaled tools
→ checkpoint / crash recovery
→ steering

Control Plane
runtime event feed
→ JSON-RPC App Server
→ localhost HTTP
→ Bearer AuthN/AuthZ
→ SSE cursor replay / reconnect

Context / Artifact / Evidence
scoped AGENTS.md
→ ContextStore provenance
→ immutable artifacts
→ unified Rollout Trace
→ verifier parent edges
→ hash-chain integrity

Security
permission / approval
→ restricted subprocess
→ Docker / Bubblewrap reference isolation

Evaluation
trajectory metrics
→ deterministic repository fixture
→ independent final-state verifier

Multi-Agent / Coding Team
Persistent AgentGraph
→ durable mailbox
→ message lease / reclaim
→ real parallel workers
→ worker-specific Git worktrees
→ candidate patch artifact
→ independent verifier
→ reviewer selection
→ merge apply / rollback
```

当前最新完整 coding-team 硬证据来自 Fast CPU Capstone CI run **219**：

```text
165 passed, 14 skipped, 1 warning in 8.96s
Ruff correctness lint: All checks passed
```

其中 14 个 skip 主要来自依赖宿主环境能力的 Docker / Bubblewrap security runtime tests；这类隔离边界另有专门 security workflow。不能把“runner 不允许 namespace”记成“隔离成功”。

---

# 2　Model Runtime

已实现并验证：

- [x] UTF-8 byte tokenizer / 教学版 byte-level BPE
- [x] `ModelConfig` 与 tensor contract
- [x] Embedding / RMSNorm / RoPE / causal GQA
- [x] half-split / interleaved RoPE
- [x] SwiGLU / residual block / LM head
- [x] raw safetensors loader / key mapping / shape audit
- [x] greedy / temperature / top-k / top-p / repetition penalty
- [x] public checkpoint adapter
- [x] `HuggingFaceTB/SmolLM2-135M` tested real-checkpoint parity

受测 CPU float32 / Hugging Face eager reference：

```json
{"max_abs": 0.0, "mean_abs": 0.0, "argmax_agreement": 1.0}
```

这个结果只证明**当前 checkpoint、当前 input、当前 numerical setting**，不能外推成“支持全部 Llama/Qwen/DeepSeek 架构”。

下一阶段：更多公开 architecture adapter、hybrid attention/SSM reference、activation-level parity。

---

# 3　Inference / Serving

## 3.1 Contiguous KV

已验证：prefill、one-token decode、cached logits 与 full recomputation parity。

## 3.2 Logical Paged KV

`paged_cache.py` 用逻辑 page 建立 page-table 语义，并与 full recomputation 对齐。

它是教学 reference，不声称 production memory benefit，因为 reference path 仍可 materialize 回 contiguous K/V。

## 3.3 Physical KV Block Allocator

`kv_block_allocator.py` 已实现：

```text
finite free list
per-request block table
refcount
prefix sharing
partial-tail private copy
copy-on-write
truncate / release
fragmentation / utilization metrics
```

## 3.4 Physical K/V Tensor Slabs

`kv_tensor_pool.py` 把 block id 真正绑定到：

```text
[layer, physical_block, H_kv, block_size, D]
```

已测试：physical-cache decode logits 与 full recomputation 对齐，shared prefix COW 不污染 parent，释放后的 private block storage 可清零。

## 3.5 Physical Prefix Cache

`physical_prefix_cache.py` 已实现 longest live prefix → physical cache fork → suffix-only forward。

已验证：

```text
exact prefix hit → zero additional model forward
partial hit      → only unmatched suffix forward
partial logits   → full-forward parity
source release   → child remains valid
```

## 3.6 Page-Aware Mixed-Length Decode

`page_aware_decode.py` 不再把所有 historical K/V `materialize()` 成 contiguous cache，而是直接根据每个 request 的 block table 读取 physical slabs。

测试把 `materialize` monkeypatch 成直接报错；不同 cached length 的请求仍能通过同一 decode API，并与各自 full recomputation logits/KV 对齐。

边界：当前仍有 Python request/block loops，不是 fused Triton/CUDA page-attention kernel。

## 3.7 Continuous-Batching Lifecycle

`continuous_batching.py` 已把 scheduler 和 physical serving path 串成：

```text
WAITING
→ prefix-aware prefill
→ first token / TTFT
→ DECODE
→ page-aware decode
→ FINISHED / CANCELLED
→ physical blocks released
```

并验证：老请求持续 decode 时，新请求可以在同一 serving iteration 进入 prefill。

仍缺：chunked prefill、preemption/swap、GPU fused kernel、真实 arrival process、throughput/fairness benchmark。

## 3.8 Speculative Decoding

已有 reference implementation/tests。下一层需要真实 draft/target benchmark、acceptance-rate 与 speedup crossover，而不是只证明函数存在。

---

# 4　Post-training

## 4.1 SFT

```text
logits + labels
→ causal shift
→ prompt/user masking
→ cross entropy
→ backward
→ optimizer step
```

已测试 uniform logits 下 CE=`log(V)`，真实 AdamW step 会改变 tiny Transformer 参数。

## 4.2 DPO

已测试：

```text
policy == reference → loss = log(2)
policy updates
reference remains frozen
```

## 4.3 Group-relative / RLVR primitive

`rlvr.py` 已实现 verifier rewards → group-relative advantage → policy ratio → clipped surrogate → optional reference KL。

**尚不是完整 RLVR pipeline。** 仍缺 rollout generation、old-policy snapshot、真实 environment verifier、optimizer loop、checkpoint 与 held-out evaluator。

---

# 5　Codex-style Turn Executor

`codex_harness.py` 当前 clean-room teaching slice：

```text
TURN_STARTED
→ model sampling
→ optional tool call / approval
→ tool execution
→ observation
→ follow-up sampling
→ TURN_COMPLETED / TURN_STOPPED
```

Harness event 可以实时写入 durable runtime event feed。

`codex_harness.SandboxPolicy` 只是 model-visible metadata；真正 isolation 在独立 security/sandbox subsystem，二者不能混淆。

---

# 6　Durable Agent Kernel

## 6.1 Thread / Turn / Event Store

已经有 persistent ThreadId/TurnId、append-only events、restart replay、checkpoint、pause/resume、terminal states、fork 与 parent provenance。

## 6.2 Work Queue / Lease

```text
PENDING
→ LEASED(worker)
├─ ACK → COMPLETED
├─ FAIL → FAILED
├─ CANCEL → CANCELLED
└─ lease expires → reclaim
```

授权范围已经可以下沉到 claim transaction，而不是先取出不属于当前 principal 的 work 再拒绝。

## 6.3 Foreground + Background Heartbeat

```text
model-sampling boundary → foreground renewal
long blocking call      → background heartbeat
```

Background heartbeat 有 startup barrier，先确认 lease 真正续过再进入受保护长调用。

边界：仍不是 distributed fencing-token / consensus ownership protocol。

## 6.4 Tool Journal / Crash Recovery

已验证：completed side effect replay 不重复执行；STARTED-only 非幂等调用进入 in-doubt；recover turn 可复用 TurnId；`turn_finished` checkpoint 可不重采样模型直接完成 work。

这不提供任意远端 side effect 的 universal exactly-once。

## 6.5 Live Steering

运行中 Steering 被持久化，在下一次 model sampling 前消费并进入同一 transcript。`ControlledBackend` 还暴露 steering observer seam，便于继续统一 provenance。

---

# 7　Context / Instructions / Artifacts / Rollout Trace

## 7.1 Context Provenance

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

## 7.2 AGENTS.md Scoped Resolver

`instructions.py` 按 project root → cwd 解析 scoped `AGENTS.md`，支持 override/fallback、global byte budget 和 source/scope/truncation provenance。

## 7.3 Content-addressed Artifact Store

`artifacts.py` 使用 SHA-256 immutable snapshots。Checksum 只能证明 bytes integrity，不能证明 semantic correctness。

## 7.4 Unified Rollout Trace

新增：

```text
rollout_trace.py
rollout_trace_collector.py
```

针对一个 stable `work_item_id`，现在可以聚合：

```text
runtime events
model/tool events
steering submitted + consumed
artifact snapshots
independent verifier results
```

Trace event 同时保留：

```text
source_event_id
artifact_id
parent_event_ids
prev_hash
event_hash
```

已验证：restart persistence、work-item/thread identity binding、重复 sync 幂等、payload tampering 检测、verifier → artifact parent edge。

SHA-256 chain 是 **tamper-evident**，不是 tamper-proof；如果攻击者可以重写整个数据库并重算 hash，仍需要 external trusted digest anchor。

---

# 8　Security

安全栈明确分层：

```text
Control-plane AuthN/AuthZ
→ Tool Permission / Approval
→ Restricted Process Execution
→ Container / Namespace Reference Isolation
```

Bearer credential 已支持 method/thread scope、expiry/revoke/rotate；HTTP/SSE 共享 policy。

Restricted subprocess enforce argv-only、executable/env allowlist、workspace cwd boundary、timeout/process-group kill、rlimits、core-dump disable 与 Linux `PR_SET_NO_NEW_PRIVS`。

Docker/Bubblewrap 已有 reference negative tests，但 Docker 共享 host Linux kernel，不能宣传成 VM-grade hostile-code containment。

**Auth ≠ Approval ≠ Restricted Process ≠ Container ≠ VM。**

---

# 9　Control Plane / Event Feed

已实现：

```text
Durable Runtime Event Feed
→ cursor replay
→ JSON-RPC App Server
→ localhost HTTP
→ Bearer AuthN/AuthZ
→ SSE catch-up / live delivery / reconnect
```

HTTP server thread 使用自己创建的 runtime/SQLite handles，避免跨线程复用 SQLite connection。

仍缺：production TLS/session gateway、WebSocket duplex control、rate limiting/backpressure、多节点 event fan-out。

---

# 10　Evaluation

`evaluation.py` + `benchmark.py` 提供 trajectory metrics 与 generic case/grader abstraction。

`repository_eval.py` 已验证：

```text
Agent says "I fixed it" but never edits → FAIL
real edit + command verification      → PASS
```

也就是 final answer 已与 final environment state 分离。

Rollout Trace 现在可以继续作为 benchmark / agentic RL trajectory 的证据底座。

下一层：multi-file hidden tests、patch artifact grading、SWE-bench adapter、time/cost/reliability benchmark。

---

# 11　Multi-Agent / Coding Team

这是本轮变化最大的部分。

## 11.1 Persistent AgentGraph

`agent_graph.py` 持久化：

```text
agent_id
thread_id
role
parent_agent_id
status
metadata
```

状态：ACTIVE / BUSY / WAITING / COMPLETED / FAILED / CANCELLED。

已验证 hierarchy/lineage restart persistence、terminal transition boundary、cross-thread negative control 与 subtree cancellation。

## 11.2 Durable Mailbox

Agent 间 point-to-point message 使用：

```text
PENDING
→ LEASED(consumer)
├─ ACK → ACKED
├─ release → PENDING
└─ lease expires → reclaim
```

已验证 lease exclusion、expiry reclaim、stale-owner ACK rejection、kind filtering 与 restart-persistent result。

当前 reference runtime 禁止本地 mailbox 跨 Thread 通信；跨 runtime/Thread 未来应走 A2A 或显式 remote handoff。

## 11.3 Real Parallel Coordinator

`parallel_agents.py` 用 coordinator-thread 持久化状态、worker thread 执行 immutable snapshots。

自动测试通过 `threading.Barrier` 强迫两个 worker 同时进入，证明存在真实 wall-clock overlap；同时验证：

```text
one active task per agent
worker exception → only that agent FAILED
healthy worker → continues queued work
wrong task/agent result identity → explicit failure
```

这仍是 local ThreadPool reference，不是 remote actor runtime。

## 11.4 Worktree-isolated Coding Team

`coding_team.py` 已实现：

```text
same base commit
→ one Git worktree per worker
→ parallel modifications
→ git diff --binary
→ candidate patch Artifact
→ independent CommandVerifier
→ ReviewerDecision
→ git apply --check
→ apply selected patch
→ post-merge verifier
├─ PASS → keep
└─ FAIL → reverse patch / rollback
```

Fast CPU CI run 219 的测试使用真实 Git repository/worktree：good candidate 通过 verifier、bad candidate 被 Reviewer 排除；selected patch 真正 apply 到 coordinator tree；另一个测试故意让 candidate-stage verifier PASS、post-merge stricter verifier FAIL，并验证 patch 被反向回滚、主树重新 clean。

边界：当前 Reviewer 是 deterministic substrate；还没有 LLM reviewer、多 patch dependency/conflict graph、remote worker、A2A 或 controlled scaling evidence。

---

# 12　Protocols / Computer Use

已实现：minimal MCP JSON-RPC discover/list/call、App Server/HTTP/SSE、AgentGraph/Mailbox local runtime。

仍未完成：完整 MCP transports/auth/tasks/extensions、A2A runtime、JS browser、DOM/A11y、screenshot grounding、mouse/keyboard、Computer Use verifier。

---

# 13　当前最新硬证据

Fast CPU Capstone CI run **219**：

```text
165 passed, 14 skipped, 1 warning in 8.96s
Ruff correctness lint: All checks passed
```

近期 Agent / Multi-Agent 关键里程碑：

```text
run 207 → unified Rollout Trace / evidence chain
run 214 → persistent AgentGraph + durable mailbox + real parallel coordinator
run 219 → real Git worktree candidate / verifier / reviewer / merge rollback
```

这条线已经从：

```text
"multiple prompts"
```

推进到：

```text
persistent identities
→ durable communication
→ real concurrency
→ isolated filesystems
→ immutable candidate artifacts
→ independent verification
→ review / merge gate
```

但**还没有**证明多 Agent 在真实任务上比单 Agent 更好。那个结论必须来自 controlled evaluation，而不是架构图。

---

# 14　下一批最高优先级

## Inference

- [ ] chunked / variable-length batched prefill
- [ ] preemption / swap
- [ ] prefix admission / eviction
- [ ] fused GPU page-attention kernel
- [ ] real throughput / memory / fairness workload benchmark
- [ ] speculative decoding real speed/distribution benchmark

## Post-training

- [ ] conversation / preference dataset pipeline
- [ ] checkpointable SFT / DPO loops
- [ ] verifier-generated rollout dataset
- [ ] actual tiny-policy group-relative update
- [ ] train reward vs held-out evaluator separation experiment

## Agent OS / Evidence

- [x] durable event stream / HTTP / SSE
- [x] scoped AuthN/AuthZ
- [x] AGENTS.md provenance
- [x] artifact store
- [x] Rollout Trace evidence bundle
- [ ] runtime-native trace sink instead of post-run collection
- [ ] external signed trace-head anchor
- [ ] reviewer/merge/policy decisions as provenance edges
- [ ] stronger fencing tokens for work/mailbox ownership

## Multi-Agent / Coding

- [x] persistent AgentGraph
- [x] durable mailbox lease/reclaim
- [x] real local parallel workers
- [x] worker-specific Git worktrees
- [x] candidate patch artifacts
- [x] independent verifier / deterministic reviewer / merge rollback
- [ ] multi-patch dependency/conflict graph
- [ ] LLM Reviewer + hidden tests
- [ ] tree-sitter / LSP semantic patch analysis
- [ ] multi-file hidden-test fixture suite
- [ ] SWE-bench adapter
- [ ] controlled 1/2/4/8-agent success/cost/wall-time/conflict comparison
- [ ] process/remote worker execution
- [ ] A2A runtime

## Security / General Agent

- [ ] stronger process/container isolation on supported host
- [ ] filesystem/network/secret scoped policy composition
- [ ] browser DOM / Accessibility Tree
- [ ] screenshot / grounding / mouse / keyboard
- [ ] Computer Use verifier

---

# 15　当前定位

当前仓库已经不能准确描述成：

```text
LLM 教材 + 一些示例代码
```

更准确的是：

```text
Source-First Textbook
        +
Reference Model / Inference Runtime
        +
Durable Agent OS
        +
Security / Protocol Layer
        +
Persistent Multi-Agent Coding Runtime
        +
Evaluation / Evidence Infrastructure
```

最终毕业标准仍然不变：读者应该能够从空目录开始解释并亲手实现模型 runtime、推理服务、持久 Agent OS、tool/environment、安全边界、多 Agent coding team、evaluation harness，并知道每一层**为什么这样设计、怎样验证、哪里还不是 production/frontier 等价物**。
