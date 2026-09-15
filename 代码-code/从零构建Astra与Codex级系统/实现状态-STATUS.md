# 实现状态：v3 Reference System

> 本文件只记录**已经落地并有证据的代码**。完整机器可读状态以 [`../../能力清单-CAPABILITIES.json`](../../能力清单-CAPABILITIES.json) 为准；README / 看板不能把计划冒充实现。

---

# 1　Model Runtime

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

受测 CPU float32 结果：

```json
{"max_abs": 0.0, "mean_abs": 0.0, "argmax_agreement": 1.0}
```

只对当前受测 checkpoint / input / numerical setting 成立。

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

# 4　v3 新增：Durable Thread / Event Store

新增 `durable.py`：

- [x] persistent `ThreadId` / `TurnId`
- [x] append-only SQLite event log
- [x] `THREAD_CREATED / USER_SUBMISSION / TURN_STARTED / TURN_COMPLETED`
- [x] checkpoint event
- [x] pause / resume
- [x] completed / failed terminal state
- [x] `project(thread_id)` 通过 event replay 重建当前状态
- [x] 关闭数据库后重新打开，可恢复 running turn、submission 和 checkpoint
- [x] invalid state transition tests

这已经解决了“状态只活在一个 Python 进程内”的最小问题，但还没有：async durable queue、lease、cancellation、fork、distributed worker ownership。

---

# 5　v3 新增：Permission Enforcement

新增 `security.py`：

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

- [x] `DENY` 时底层 tool body 调用次数保持 0
- [x] approval reject 时底层 tool body 不执行
- [x] approval allow 后才进入 dispatch

这是 **application-layer execution gate**，仍不是：

```text
process isolation
filesystem namespace
network isolation
credential isolation
syscall sandbox
```

真实 OS / container sandbox 继续是 P0。

---

# 6　v3 新增：Evaluation Metrics

新增 `evaluation.py`，把 unit correctness 与 capability evaluation 分开。

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

# 7　Agent Protocol Runtime

当前：

- [x] JSON-RPC 2.0 educational envelope
- [x] minimal MCP `server/discover`
- [x] `tools/list`
- [x] `tools/call`
- [x] ToolRegistry → MCP schema mapping
- [x] in-process transport / client
- [x] protocol / tool failure tests

注意：当前 `mcp.py` 是教学子集。MCP 2026-07-28 的完整 stateless semantics、stdio/HTTP、authorization、tasks/extensions，以及 A2A 1.0 都仍在后续。

---

# 8　最新 Fast CI

修复 permission metadata edge case 后，最新 Fast CI 已真实通过：

```text
36 passed, 1 warning in 2.14s
Ruff correctness lint: All checks passed
```

新增加的 6 个测试覆盖 durable replay、state transition、permission/approval enforcement 和 trajectory metrics。

---

# 9　当前明确未实现

## Inference / Serving

- [ ] SDPA / high-performance FlashAttention backend
- [ ] paged KV cache
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

- [ ] async Submission Queue / Event Queue
- [ ] pending steering / cancellation
- [ ] thread fork
- [ ] worker lease / ownership
- [ ] scoped `AGENTS.md` resolver + instruction provenance
- [ ] context budget / compaction / historical-window retrieval
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

---

# 10　成熟度规则

一个 capability 标为 `validated` 至少要求：

```text
Source / Theory
+ Code
+ Explicit Contract
+ Automated Test
+ Evidence
```

若声称与公开/工业实现等价，还必须有 numerical / behavioral / protocol parity 或 benchmark。真正状态统一进入 [`../../能力清单-CAPABILITIES.json`](../../能力清单-CAPABILITIES.json)。
