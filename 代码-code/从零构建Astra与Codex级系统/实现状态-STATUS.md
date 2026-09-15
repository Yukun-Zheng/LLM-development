# 实现状态：v3 Reference System

> 本文件只记录**已经落地且有证据的实现**。机器可读事实源是根目录的 [`../../能力清单-CAPABILITIES.json`](../../能力清单-CAPABILITIES.json) 与 `能力清单-CAPABILITIES-*.json`。任何 `planned` 都不能在 README 中被写成“已完成”。

---

# 1　当前总览

Reference System 已形成七条可运行纵向链：

```text
Model Runtime
raw checkpoint → own Transformer → logits parity

Inference Runtime
contiguous KV
→ logical paged KV
→ physical block allocator
→ physical K/V tensor slabs
→ physical prefix sharing
→ page-aware mixed-length decode
→ interleaved continuous-batching lifecycle

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
permission/approval gate
→ restricted subprocess / container reference isolation
→ independent repository final-state grading
```

最新已核验 Fast CPU Capstone CI（run **189**）：

```text
145 passed, 14 skipped, 1 warning in 8.74s
Ruff correctness lint: All checks passed
```

14 个 skip 主要来自依赖宿主环境能力的 Docker/Bubblewrap runtime tests；这类安全边界另有专门 security workflow，不会把“runner 不允许 namespace”误记为隔离成功。

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

受测 CPU float32 / HF eager reference：

```json
{"max_abs": 0.0, "mean_abs": 0.0, "argmax_agreement": 1.0}
```

边界：这个结果只证明当前 checkpoint / input / numerical setting，不能外推成“支持全部 Llama/Qwen/DeepSeek 架构”。

下一阶段模型侧重点：multi-architecture adapter、hybrid attention / SSM reference、activation-level parity。

---

# 3　Inference / Serving

## 3.1 Contiguous KV

- [x] per-layer KV cache
- [x] prefill
- [x] one-token incremental decode
- [x] cached decode vs full-forward logits parity

## 3.2 Logical Paged KV

`paged_cache.py` 已实现 logical page growth、per-layer page table 和 reference materialization，并验证 paged prefill/decode 与 full recomputation 对齐。

这层主要用于解释 page table 语义，不再承担“物理存储已经 production 化”的错误暗示。

## 3.3 Physical KV Block Allocator

`kv_block_allocator.py` 已从逻辑 page 推进到有限 physical block pool：

```text
free list
+ per-request block table
+ refcount
+ prefix sharing
+ partial-tail private copy
+ copy-on-write
+ release / truncate
+ fragmentation/utilization metrics
```

已测试 OOM append atomicity、shared block 生命周期与 COW。

## 3.4 Physical K/V Tensor Slabs

`kv_tensor_pool.py` 把 block id 真正绑定到：

```text
K/V storage:
[layer, physical_block, H_kv, block_size, D]
```

并验证：

```text
physical-cache decode logits == full recomputation
shared prefix COW 不污染 parent
released private block storage 会清零
```

## 3.5 Physical Prefix Cache

`physical_prefix_cache.py` 已把前缀发现、物理 block ownership 与实际计算复用接起来：

```text
TokenPrefixIndex
→ longest live prefix
→ fork physical cache
→ full blocks share
→ partial tail private copy
→ forward unmatched suffix only
```

已验证：

```text
exact prompt hit → 0 additional model forward
partial hit       → only suffix forward
partial logits    → full-forward parity
source release    → child cache remains valid
```

当前 prefix index 仍是教学 token trie；production 下一步应转 block hash / radix index，并增加 admission/eviction。

## 3.6 Page-Aware Heterogeneous Decode

`page_aware_decode.py` 是当前 inference 主线的重要分水岭。

它不再调用：

```text
pool.materialize(request)
→ contiguous historical K/V
```

而是直接根据每个 request 的 block table，从 physical K/V slabs 逐 block 读取历史状态。

当前数据流：

```text
batched embedding / norm / QKV projection
        ↓
per-request RoPE absolute position
        ↓
block-table attention over physical K/V
        ↓
score concatenation + global softmax
        ↓
batched residual / FFN / LM head
```

自动测试会把 `pool.materialize` monkeypatch 成直接报错；在这种条件下，cached length 不同的请求仍能通过同一个 decode API，并与各自 full recomputation logits 对齐。

还验证了 page-aware decode 后每层存储的 K/V 与 `model(full_sequence, use_cache=True)` 对齐。

**边界**：attention 目前仍是 Python request/block loop，不是 fused Triton/CUDA block-table kernel。

## 3.7 Reference Continuous Batching Lifecycle

`continuous_batching.py` 第一次把 scheduler 与真实 physical serving path 串成闭环：

```text
submit
→ WAITING
→ physical-prefix prefill
→ first token / TTFT
→ DECODE
→ heterogeneous page-aware decode
→ next token
→ FINISHED / CANCELLED
→ physical block release
```

一个 iteration 有两个 lane：

```text
1. decode currently active requests
2. admit/prefill waiting requests
```

因此已经验证：老请求仍处于 DECODE 时，新到请求可以在同一 serving iteration 进入 prefill，而不是等所有旧请求结束。

同时已测试：

- [x] mixed cached lengths in one decode API
- [x] newly arrived request admission while old request keeps decoding
- [x] terminal request immediately releases physical blocks
- [x] cancellation releases admitted cache
- [x] `run_until_idle` leaves no leaked blocks
- [x] TTFT / TPOT / total latency driven by actual engine lifecycle

**尚未声称 production continuous batching**。仍缺：

```text
chunked/variable-length batched prefill
unified token-budget scheduler
batch-wide block reservation before execution
preemption / swap
prefix-cache eviction/admission
fused page-attention GPU kernel
live-arrival throughput/fairness benchmark
```

## 3.8 Speculative Decoding

`speculative.py` 已有 reference implementation + tests。下一层必须从“有函数”升级成：distribution correctness、accept/reject parity、speedup crossover 和真实 latency benchmark。

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

已测试 uniform logits 下 masked CE=`log(V)`，真实 AdamW step 会更新 tiny Transformer 参数。

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

边界：仍是 objective primitive，不是完整 RLVR pipeline。尚缺 rollout generation、old-policy snapshot、真实 environment verifier、optimizer loop、checkpoint 与 held-out evaluator。

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

`HarnessEventSink` 允许事件在发生时直接进入 durable runtime event feed。

这里的 `codex_harness.SandboxPolicy` 只是模型可见 turn/session metadata；真正 execution isolation 在独立 sandbox 层，二者不能混为一谈。

---

# 6　Durable Agent Kernel

## 6.1 Thread / Event Store

已实现：persistent ThreadId/TurnId、append-only event log、restart replay、submission/checkpoint、pause/resume、terminal states、fork、parent provenance、独立 child cancellation。

## 6.2 Work Queue / Lease / Scoped Claim

`runtime_queue.py` 支持：

```text
PENDING
→ LEASED(worker)
├─ ACK → COMPLETED
├─ FAIL → FAILED
├─ CANCEL → CANCELLED
└─ expired lease → reclaim
```

`claim(...)` 支持 thread filter，授权边界已经下沉到 SQL claim transaction，而不是“拿到未授权 work 后再拒绝”。

## 6.3 Foreground + Background Heartbeat

两层 lease protection：

```text
model-sampling boundary → LeaseHeartbeat
long blocking call      → BackgroundLeaseHeartbeat
```

Background heartbeat 已有 startup barrier，避免“线程启动了但第一轮 renewal 还没发生”导致短 lease 被其他 worker reclaim。

边界：仍不是 distributed fencing-token / consensus ownership protocol。

## 6.4 Tool Journal / Crash Recovery

已验证：

- [x] completed side effect replay 不重复执行
- [x] restart 后 replay
- [x] idempotency-key rebinding rejection
- [x] STARTED-only non-idempotent call → in-doubt
- [x] recovered TurnId 复用
- [x] `turn_finished` checkpoint 可无 model resampling 完成 work

任意远端 side effect 仍不能靠本地 journal 自动获得 universal exactly-once；需要 external idempotency/reconciliation。

## 6.5 Live Steering

Steering 在 tool body 执行期间到达时，会被持久化，并在下一次 model sampling 前注入同一 transcript。

---

# 7　Context / Instructions / Artifacts

`context.py` 支持：

```text
RAW_EVENT / NOTE / SUMMARY / ARTIFACT / RETRIEVAL / INSTRUCTION
```

Compaction summary 保留 parent lineage，raw evidence 不被覆盖。

`instructions.py` 实现 scoped `AGENTS.md` clean-room reference：从 project root 到 cwd，按 override > AGENTS > fallback 选择，并带 global byte budget 与 source/scope/truncation provenance。

`coding.py` 会把 resolved instructions 注入模型，同时可写入 `ContextStore`，因此可以追踪“模型为什么看到了某条仓库规则”。

`artifacts.py` 使用 SHA-256 content-addressed immutable snapshots。Checksum 证明 byte integrity，不证明 semantic correctness；后者仍由 verifier/grader/reviewer 负责。

---

# 8　Security

安全栈明确分层：

```text
Control-plane AuthN/AuthZ
→ Tool Permission / Approval
→ Restricted Process Execution
→ Container / Namespace Reference Isolation
```

## 8.1 Control-plane

Bearer credentials 支持 method/thread scope，并已有 expiry、revoke、rotate lifecycle；HTTP 与 in-process transport 共用同一 policy。

边界：还没有 OIDC、mTLS、secure distributed credential persistence、distributed revocation propagation 或外部 policy engine。

## 8.2 Restricted process

`sandbox.py` enforce：argv-only、executable allowlist、workspace cwd boundary、env allowlist、timeout/process-group kill、POSIX rlimits、core-dump disable、Linux `PR_SET_NO_NEW_PRIVS`、output clipping。

## 8.3 Container / namespace reference

Docker path 已有 readonly workspace/rootfs、network-none、capability drop、NoNewPrivs 等负测试；Bubblewrap 在 runner 不允许完整 namespace 时显式 skip。

Docker 仍共享 host Linux kernel，不能宣传成 VM-grade hostile-code containment。

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

HTTP server thread 使用独立 runtime/SQLite handles，避免跨线程复用 SQLite connection。

边界：仍缺 production TLS/session gateway、WebSocket duplex control、backpressure/rate limit、多节点 event fan-out。

---

# 10　Evaluation

`evaluation.py` + `benchmark.py` 提供 trajectory metrics 和 case/grader abstraction。

`repository_eval.py` 的 deterministic coding fixture 已证明：

```text
Agent says "I fixed it" but never edits → FAIL
real edit + command verification      → PASS
```

也就是 final answer 已经和 final environment state 分离。

下一层：multi-file hidden-test fixtures、patch artifact grading、SWE-bench adapter、time/cost/reliability benchmark。

---

# 11　Protocols / Multi-Agent / Computer Use

已经实现：

- [x] minimal MCP JSON-RPC discover/list/call
- [x] task DAG / sequential coordinator
- [x] Git worktree primitive
- [x] replayable App Server / HTTP / SSE
- [x] scoped worker claim

仍未完成：完整 MCP transports/auth/tasks/extensions、A2A runtime、persistent AgentGraph/mailbox、parallel reviewer/merge、JS browser、DOM/A11y、screenshot grounding、mouse/keyboard、Computer Use verifier。

---

# 12　最新硬证据

Fast CPU Capstone CI run **189**：

```text
145 passed, 14 skipped, 1 warning in 8.74s
Ruff correctness lint: All checks passed
```

最近四个 inference 里程碑：

```text
run 172 → physical KV block allocator
run 176 → physical K/V tensor slabs
run 179 → physical prefix cache / compute reuse
run 184 → page-aware mixed-length decode
run 189 → scheduler-integrated continuous-batching reference lifecycle
```

这里的提升不是单纯 test 数增加，而是数据路径已经从：

```text
Python tuple cache
```

推进到：

```text
finite physical blocks
→ real tensor slabs
→ shared prefixes
→ direct page-table attention
→ live request lifecycle
```

---

# 13　下一批最高优先级

## Inference

- [ ] batch-wide physical-block reservation / rollback semantics
- [ ] chunked prefill + unified token-budget scheduler
- [ ] preemption / swap / eviction
- [ ] block-hash/radix prefix index
- [ ] fused Triton/CUDA page-attention kernel
- [ ] live-arrival TTFT / TPOT / throughput / fairness benchmark
- [ ] speculative decoding real speed/distribution benchmark

## Post-training

- [ ] conversation / pairwise dataset pipeline
- [ ] checkpointable SFT/DPO loops
- [ ] verifier-generated rollout dataset
- [ ] actual tiny-policy group-relative update
- [ ] train-reward vs held-out evaluator separation experiment

## Agent OS / Control Plane

- [ ] steering / artifacts / verifier → unified rollout trace provenance
- [ ] WebSocket duplex control channel
- [ ] production session identity gateway
- [ ] persistent distributed credential/policy state

## Coding / Evaluation / Multi-Agent

- [ ] multi-file hidden-test fixture suite
- [ ] SWE-bench adapter
- [ ] tree-sitter / LSP / semantic patch
- [ ] persistent AgentGraph / mailbox
- [ ] parallel workers / reviewer / merge
- [ ] controlled 1/2/4/8-agent success-cost-time comparison

## General Agent / Computer Use

- [ ] JS browser / DOM / Accessibility Tree
- [ ] screenshot grounding
- [ ] mouse / keyboard action runtime
- [ ] Computer Use verifier
- [ ] multimodal observation types
