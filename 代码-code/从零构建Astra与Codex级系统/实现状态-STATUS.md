# 实现状态：v3 Reference System

> 本文件只记录**已经落地且有自动化证据的实现**。机器可读事实源是根目录的 [`../../能力清单-CAPABILITIES.json`](../../能力清单-CAPABILITIES.json) 与 `能力清单-CAPABILITIES-*.json`。任何 `planned` 都不能在 README 中写成“已完成”。

---

# 1　当前总览

Reference System 现在形成十条真正可以运行、测试和继续替换优化的纵向链：

```text
Model Runtime
raw public checkpoint → own Transformer → logits parity

Inference Runtime
contiguous KV → paged KV → physical blocks/slabs
→ physical prefix reuse → mixed-length decode → continuous batching

Post-training
masked SFT → DPO → group-relative / RLVR objective primitive

Durable Agent Kernel
Thread / Turn / WorkItem → lease / heartbeat
→ journaled tools → checkpoint / recovery → steering

Control Plane
runtime event feed → JSON-RPC → localhost HTTP
→ Bearer AuthN/AuthZ → SSE replay/reconnect

Context / Evidence
scoped AGENTS.md → ContextStore provenance
→ immutable artifacts → unified Rollout Trace / hash chain

Security
permission/approval → restricted process
→ Docker / Bubblewrap reference isolation

Evaluation
trajectory metrics → repository fixture
→ independent final-state verification

Multi-Agent Coding Team
AgentGraph / mailbox → parallel workers
→ isolated Git worktrees → reviewer → merge / rollback

Agent Protocols
MCP teaching runtime
→ A2A v1 Task/Message/Artifact
→ HTTP+JSON + source-aligned ListTasks
→ A2A ↔ DurableAgentRuntime
→ remote HTTP gateway
→ SendStreamingMessage / SSE
→ durable SubscribeToTask / reconnect replay
```

当前最新已核验 Fast CPU Capstone CI：run **248**。

```text
188 passed, 14 skipped, 1 warning in 46.16s
Ruff correctness lint: All checks passed
```

14 个 skip 主要来自依赖宿主环境能力的 Docker/Bubblewrap security runtime tests；不能把 runner 不具备 namespace/container 条件写成“隔离成功”。

---

# 2　Model Runtime

已验证：UTF-8 byte tokenizer、教学 byte-level BPE、Embedding、RMSNorm、RoPE、causal GQA、SwiGLU、residual、LM head、sampling、raw safetensors mapping/loading，以及 `HuggingFaceTB/SmolLM2-135M` 的真实 checkpoint parity。

受测 CPU float32 / Hugging Face eager reference：

```json
{"max_abs": 0.0, "mean_abs": 0.0, "argmax_agreement": 1.0}
```

边界：只证明当前 checkpoint/input/numerical setting；不外推到全部 Llama/Qwen/DeepSeek/Hybrid 模型。

---

# 3　Inference / Serving

已落地并验证：

```text
contiguous KV
→ reference logical paged KV
→ finite physical KV block allocator
→ physical K/V tensor slabs
→ copy-on-write prefix sharing
→ longest-prefix physical cache reuse
→ page-aware heterogeneous decode reference
→ request scheduler
→ continuous-batching lifecycle
→ speculative decoding reference
```

关键 parity 已覆盖 cached/full logits、paged/full、prefix reuse/full、physical cache/full 和 mixed-length decode/full。

仍未声称 production 等价：当前 page-aware path 仍有 Python request/block loops，没有 fused Triton/CUDA PagedAttention kernel；continuous batching 也尚缺 chunked prefill、preemption/swap、真实 arrival/fairness/throughput benchmark。

---

# 4　Post-training

已实现：

```text
SFT masked causal-LM loss + AdamW step
DPO objective + tiny optimization step
GRPO-style group-relative verifier-RL objective primitive
```

已验证 `policy == reference → DPO loss = log(2)`、reference frozen、group-relative advantage 中心化、clipping/KL 等数学性质。

仍缺 end-to-end RLVR：rollout generation、old-policy snapshot、真实 verifier environment、optimizer loop、checkpoint 与 held-out evaluator。

---

# 5　Durable Agent OS

当前核心不是一个 `while model → tool`，而是：

```text
Submission
→ Durable Thread
→ WorkItem / Lease
→ Turn
→ Model / Tool Events
→ Durable Tool Journal
→ Checkpoint
→ ACK / Fail / Recover
```

已验证：restart replay、fork/cancel、lease reclaim、foreground/background heartbeat、completed side-effect replay、不确定 in-doubt side-effect protection、finished-checkpoint recovery、live steering。

Background heartbeat 已覆盖单个长 model/tool blocking call；仍不是 distributed fencing-token/consensus ownership protocol。

---

# 6　Control Plane / Context / Evidence

已实现：

```text
Durable runtime event feed
→ JSON-RPC App Server
→ real localhost HTTP
→ Bearer token AuthN
→ method/thread AuthZ
→ queue-level authorized claim
→ SSE cursor replay/reconnect
```

Context/evidence 已包括：scoped `AGENTS.md` resolver、INSTRUCTION provenance、compaction lineage、content-addressed Artifact Store、unified Rollout Trace、artifact/verifier parent edges 与 SHA-256 event hash chain。

Auth、approval、sandbox 被明确分层；HTTP reference 默认 loopback，不声称 TLS/OIDC/mTLS/multi-tenant production gateway。

---

# 7　Security

当前有三层真实实现：

```text
Tool Permission / Approval
→ RestrictedSubprocessSandbox
→ Docker / Bubblewrap reference isolation
```

Restricted subprocess enforce argv-only、workspace boundary、env/executable allowlist、timeout/process-group kill、rlimits、core-dump disable、Linux `NO_NEW_PRIVS`。

Docker/Bubblewrap 有 negative tests，但 Docker 与宿主共享 kernel，不能宣传为 VM-grade hostile-code isolation。

---

# 8　Evaluation / Multi-Agent Coding

Evaluation 已经明确区分：

```text
final answer ≠ final repository state
unit test ≠ agent capability benchmark
```

Repository fixture 可以独立验证真实 edit/command/final state。

Multi-Agent 已从“多个 prompt”推进到：

```text
Persistent AgentGraph
→ durable mailbox lease/reclaim
→ real parallel worker overlap
→ one Git worktree per worker
→ candidate patch artifacts
→ independent verifier
→ reviewer selection
→ merge apply
→ stricter post-merge verification
→ rollback on failure
```

仍缺 remote worker/A2A team、LLM reviewer、multi-patch dependency/conflict graph、1/2/4/8-agent controlled scaling evidence。

---

# 9　MCP 与 A2A

## 9.1 MCP

已有 inspectable minimal MCP teaching runtime：JSON-RPC、discover/list/call、ToolRegistry mapping 和 deterministic tests。完整 transports/auth/tasks/extensions 仍在后续层。

## 9.2 A2A v1 Core

A2A 固定官方规范快照：

```text
a2aproject/A2A
commit 6d6640c29b102f7a8d23784901351b5d2454fe71
specification/a2a.proto
```

已从零实现并测试：

```text
AgentCard / AgentInterface / AgentSkill
Message / Part
Task / TaskStatus / TaskState
Artifact
SendMessageConfiguration
SQLite TaskStore
SendMessage / GetTask / ListTasks / CancelTask
```

TaskState、Part oneof 和 protocolVersion 都按 pinned v1 wire semantics，不用社区旧资料猜。

## 9.3 A2A HTTP+JSON

真实 localhost HTTP reference 已支持：

```text
GET  /.well-known/agent-card.json
POST /message:send
GET  /tasks/{id}
GET  /tasks
POST /tasks/{id}:cancel
```

`ListTasks` 已按 v1 语义支持：

```text
contextId
status
pageSize / opaque pageToken
historyLength
statusTimestampAfter
includeArtifacts
nextPageToken / pageSize / totalSize
```

## 9.4 A2A → Durable Runtime Bridge

```text
A2A Task
→ metadata.runtimeThreadId
→ Durable Thread
→ WorkItem
→ queue-level thread fencing
→ Turn Executor
→ RuntimeExecutionRecord
→ A2A TaskStatus / Message / Artifact
```

`raw`/`url` Part 在没有明确 media-fetch security policy 前主动拒绝，而不是暗中下载。

## 9.5 Real HTTP → Agent OS Gateway

`a2a_runtime_http.py` 让 HTTP server thread 自己打开 A2ATaskStore 与 DurableAgentRuntime SQLite handles：

```text
POST /message:send
→ A2A Task
→ Durable Thread / WorkItem / Turn
→ A2A result
```

server restart 后 A2A Task 和 Agent-OS committed state 都可以恢复。

## 9.6 SendStreamingMessage / SSE

`a2a_streaming.py` + `a2a_sse.py` 显式建模 pinned v1 `StreamResponse`：

```text
task
message
statusUpdate
artifactUpdate
```

真实 `POST /message:stream` 测试故意阻塞 handler，验证客户端先收到 durable `SUBMITTED Task` 首帧，再收到 artifact delta / final status；不是把最终数组伪装成 stream。

## 9.7 SubscribeToTask / Durable Replay

新增：

```text
a2a_subscription.py
```

实现 append-only SQLite Task update journal 与：

```text
GET /tasks/{id}:subscribe
→ text/event-stream
```

测试覆盖：

```text
SUBMITTED snapshot durable
→ subscriber reads SUBMITTED
→ another DB connection completes Task
→ subscriber reads terminal snapshot
→ stream closes
```

连续相同 snapshot 会去重；fresh subscription 对 already-terminal Task 返回 409 reference unsupported-operation boundary。

另外 reference SSE transport 显式使用：

```text
Last-Event-ID
```

作为**非 normative 的 transport-level replay extension**。最新 regression 验证：

```text
observe update N
→ disconnect
→ Task reaches N+1 terminal update
→ reconnect Last-Event-ID=N
→ replay exactly N+1
```

Fast CI run 248：

```text
188 passed, 14 skipped, 1 warning in 46.16s
Ruff correctness lint: All checks passed
```

当前仍缺 executor-native WORKING/intermediate `TaskStatusUpdateEvent` / `TaskArtifactUpdateEvent`、push notification config、extended authenticated Agent Card、tenant/security binding、active local WorkItem cancellation propagation 和官方 conformance differential tests。

---

# 10　下一批最高优先级

## Protocol / Agent OS

- [x] A2A durable Task update journal
- [x] `GET /tasks/{id}:subscribe`
- [x] disconnect / Last-Event-ID replay reference semantics
- [ ] executor-native `TaskStatusUpdateEvent` / `TaskArtifactUpdateEvent`
- [ ] A2A cancellation propagation into local running WorkItem
- [ ] A2A INPUT_REQUIRED / AUTH_REQUIRED continuation mapping
- [ ] push notification configuration
- [ ] remote A2A auth / tenant scope
- [ ] official A2A SDK/conformance differential fixtures

## Inference

- [ ] chunked/disaggregated prefill
- [ ] fused page-aware attention kernel
- [ ] request preemption / swap
- [ ] speculative decoding real latency/speedup benchmark
- [ ] throughput/memory/fairness benchmark

## Post-training

- [ ] conversation/pairwise dataset pipeline
- [ ] checkpointable SFT/DPO loops
- [ ] verifier-generated rollouts
- [ ] actual tiny-policy group-relative update
- [ ] train reward vs held-out evaluator separation

## General Agent / Computer Use

- [ ] JS browser / DOM / Accessibility Tree
- [ ] screenshot grounding
- [ ] mouse/keyboard action runtime
- [ ] Computer Use verifier

## Evaluation / Coding / Multi-Agent

- [ ] multi-file hidden-test fixture suite
- [ ] SWE-bench adapter
- [ ] tree-sitter / LSP / semantic patch
- [ ] remote parallel workers + A2A handoff
- [ ] controlled 1/2/4/8-agent comparison

---

最终毕业标准不变：读者应能从空目录开始解释并亲手实现模型 runtime、推理服务、post-training primitives、持久 Agent OS、control plane、MCP/A2A、security boundary、多 Agent coding team、evaluation harness，并知道每一层**为什么这样设计、怎样验证、哪里还不是 production/frontier 等价物**。
