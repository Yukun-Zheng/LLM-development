# 从零构建 Astra-class 与 Codex-class 系统

这是整本教材的**终极 Reference System**：从模型 forward、真实权重加载、推理状态、Serving，到 SFT/DPO/RL objective，再到 durable Agent kernel、工具、协议、安全、Coding Agent 与 Evaluation。

> **边界**：不声称复刻未公开的 frontier 权重、训练 recipe 或 OpenAI 云端全部生产基础设施。Open/tiny models 用于从零实现和 parity；frontier models 可以作为可插拔 Model Backend。**模型智能可以插拔，系统智能尽量自己实现。**

入口：

- [`实现状态-STATUS.md`](实现状态-STATUS.md)：当前真实代码状态；
- [`../../能力清单-CAPABILITIES.json`](../../能力清单-CAPABILITIES.json)：机器可读事实源；
- [`../../教材-book/99-全书架构蓝图-v3.md`](../../教材-book/99-全书架构蓝图-v3.md)：v3 总架构；
- [`../../评测-eval/README.md`](../../评测-eval/README.md)：独立实验事实层。

---

# 1　当前系统图

```text
Open/Tiny Model Runtime ─┐
Frontier Model Backend ──┤
                         ↓
                   Model Interface
                         │
       ┌─────────────────┼──────────────────┐
       ↓                 ↓                  ↓
Inference Runtime   Post-training      Agent Runtime
       │                 │                  │
KV / Paged KV       SFT / DPO          Thread/Event
Scheduler           Group-relative RL  Work Queue
Batched Executor                         Turn Executor
       │                                    │
       └────────────────┬───────────────────┘
                        ↓
                Context / Planner
                        ↓
             Permission / Approval
                        ↓
        Tool / MCP / Git / Browser / Agent
                        ↓
                   Environment
                        ↓
                    Verifier
                        ↓
           Checkpoint / Replan / Finish
                        ↓
          Benchmark / Artifact / Evidence
```

---

# 2　当前源码地图

```text
src/astra_codex/
│
├── Model
│   ├── config.py
│   ├── tokenizer.py
│   ├── model.py
│   ├── weights.py
│   └── public_checkpoint.py
│
├── Inference / Serving
│   ├── cache.py
│   ├── paged_cache.py
│   ├── engine.py
│   ├── scheduler.py
│   ├── batch_executor.py
│   └── sampling.py
│
├── Post-training
│   ├── posttraining.py       # SFT / sequence logp / DPO
│   └── rlvr.py               # group-relative verifier RL objective
│
├── Durable Agent Kernel
│   ├── agent.py
│   ├── codex_harness.py
│   ├── durable.py            # Thread/Turn/replay/fork/cancel
│   ├── runtime_queue.py      # lease/reclaim/ack/cancel
│   ├── runtime.py            # Thread→Queue→Turn integrated pipeline
│   ├── context.py            # typed context + provenance
│   ├── memory.py
│   ├── planning.py
│   └── verification.py
│
├── Environment / Protocol / Security
│   ├── structured.py
│   ├── tools.py
│   ├── editing.py
│   ├── repo_map.py
│   ├── general_tools.py
│   ├── mcp.py
│   └── security.py
│
├── Coding / Multi-Agent
│   ├── coding.py
│   ├── worktree.py
│   └── multi_agent.py
│
└── Evaluation
    ├── evaluation.py
    ├── benchmark.py
    └── repository_eval.py
```

暂时保持 flat package，避免为了目录美观破坏已经建立的 parity/tests。等 subsystem contract 稳定后再迁移子包。

---

# 3　Model Runtime：真实 checkpoint 已对齐

公开 checkpoint：

```text
HuggingFaceTB/SmolLM2-135M
```

```text
raw config + safetensors
→ our weight mapping
→ our RMSNorm / RoPE / GQA / SwiGLU / Transformer
→ our logits
        ↕
HF eager reference logits
```

固定受测 CPU float32：

```json
{"max_abs": 0.0, "mean_abs": 0.0, "argmax_agreement": 1.0}
```

这只是第一个真实模型族，不外推为通用兼容。

---

# 4　Inference：从 `generate()` 进入真正的 Serving 问题

## Contiguous KV

已验证 incremental cached decode 与 full recomputation logits parity。

## Reference Paged KV

`paged_cache.py` 已实现 logical page table 与 suffix ingest；paged prefill/decode 与 full forward 数值对齐。

当前 pages 仍会重新拼成 contiguous K/V，因此证明的是**分页语义**，不是生产级 block allocator 性能。

## Request Scheduler

`scheduler.py` 已实现：

```text
WAITING → PREFILL → DECODE → FINISHED/CANCELLED
```

并显式记录 TTFT / TPOT / total latency。

## Actual Batched Model Executor

`batch_executor.py` 已经执行真正的 batched model forward：多个 request 一次进入模型，并将 batched K/V 拆回各自 decode state。

测试证明：

```text
batched prefill == individual prefill
batched decode  == individual full recomputation
```

当前只支持同长度 cache cohort。真正 continuous batching 仍要解决 variable-length sequence、block table、allocator 和 scheduler/executor integration。

实验：[`教程-lessons/05-PagedKV与调度-serving-runtime.md`](教程-lessons/05-PagedKV与调度-serving-runtime.md)

---

# 5　Post-training：已经从公式进入真实 gradient

## SFT

`posttraining.py` 已有 prompt masking、causal shift、CE、backward、grad clip、optimizer step。

自动测试用 uniform logits 手算 `log(V)`，并验证 tiny model 参数真的更新。

## DPO

已经实现 policy/reference chosen/rejected sequence log-probs、implicit reward margin、DPO loss 与优化步骤。

自动不变量：

```text
policy == reference
→ DPO loss = log(2)
```

并验证 policy 更新、reference 冻结。

实验：[`教程-lessons/04-从SFT到DPO-post-training.md`](教程-lessons/04-从SFT到DPO-post-training.md)

## Group-relative verifier RL primitive

`rlvr.py` 已实现教学用 GRPO-style：

```text
verifier rewards [B,G]
→ group-relative advantages
→ policy ratios
→ clipped surrogate
→ optional reference KL
```

这还不是完整 RLVR pipeline；rollout generation、old-policy snapshot、真正 verifier reward、token-level update 与 held-out eval 仍待串联。

实验：[`教程-lessons/07-从Verifier到GroupRelativeRL-rlvr.md`](教程-lessons/07-从Verifier到GroupRelativeRL-rlvr.md)

---

# 6　Agent：从 while-loop 进入 Durable Runtime

## Turn Executor

`codex_harness.py`：sampling → tool/approval → observation → follow-up → complete/stop。

## Durable Thread

`durable.py`：Thread/Turn/event replay、checkpoint、pause/resume、fork、cancel、complete/fail。

进程关闭后重开 SQLite 可以重建 projection；fork 保存 parent provenance，并形成独立 child event stream。

## Durable Work Queue

`runtime_queue.py`：

```text
PENDING → LEASED(worker)
├─ ACK
├─ FAIL
├─ CANCEL
└─ lease timeout → reclaim
```

## Integrated Runtime

`runtime.py` 第一次把：

```text
Submission
→ Thread Event
→ Durable WorkItem
→ Worker Lease
→ Turn Executor
→ Checkpoint
→ Turn Complete
→ Work ACK
```

接成一个真正运行的 pipeline。

**重要未解问题**：如果 worker 在 side-effecting tool 执行期间死亡，不能简单自动重跑 turn，否则可能重复副作用。后续必须引入 durable tool-execution record / idempotency semantics。

实验：[`教程-lessons/06-从AgentLoop到DurableRuntime-durable-agent.md`](教程-lessons/06-从AgentLoop到DurableRuntime-durable-agent.md)

---

# 7　Context：Summary 不是 Evidence

`context.py` 区分：RAW_EVENT / NOTE / SUMMARY / ARTIFACT / RETRIEVAL / INSTRUCTION。

Compaction 生成新的 summary fragment，同时保存 parent lineage；raw observations 不会被 summary 覆盖或删除。

下一步：persistent notes、semantic index、artifact provenance、token-budget builder、historical-window retrieval。

---

# 8　Security：权限 gate 已有，OS sandbox 还没有

`security.py` 已实现：

```text
Action Proposal
→ ALLOW / REQUIRE_APPROVAL / DENY
→ optional approval
→ Tool Dispatch
```

测试证明 DENY / approval reject 时底层 tool body 不执行。

但 shell/browser 仍未处于真正 process/container sandbox 中。filesystem/network/secrets/resource isolation 与 escape tests 仍是 P0。

---

# 9　Evaluation：已经有真实 local repository final-state grader

通用层：

```text
BenchmarkCase
→ Executor
→ trajectory
→ Grader
→ Grade
→ AggregateMetrics
```

`repository_eval.py` 再往前一步，把 coding task 固定成：

```text
initial files
+ goal
+ verify argv
+ expected final files
```

然后 Agent 完成后，由独立 grader 检查 final state。

自动测试故意构造：

```text
Agent 只说 "I fixed it"
→ repository 仍错
→ FAIL

Agent 真改文件并验证
→ final-state grader
→ PASS
```

这意味着项目已经开始从“agent demo”进入可证伪的 coding-agent evaluation。

入口：[`../../评测-eval/README.md`](../../评测-eval/README.md)

---

# 10　当前硬证据

Fast CPU CI run 66：

```text
61 passed, 1 warning in 3.67s
Ruff correctness lint: All checks passed
```

测试覆盖范围现在已经横跨：

```text
Tokenizer / Transformer / real checkpoint
KV / Paged KV / Scheduler / Batched Executor
SFT / DPO / Group-relative RL primitive
Tool runtime / MCP / Coding tools
Codex-style harness
Thread replay / fork / cancellation
Lease / reclaim / integrated runtime
Context provenance
Permission enforcement
Trajectory / Benchmark / Repository final-state grading
```

---

# 11　下一阶段

```text
Inference:
heterogeneous continuous batching
→ real block allocator / prefix sharing
→ chunked/disaggregated prefill
→ speculative decoding

Post-training:
dataset/checkpoint loops
→ verifier rollouts
→ actual tiny-policy RLVR update
→ held-out evaluator

Agent OS:
durable tool execution / idempotency
→ crash recovery during active turn
→ live steering
→ heartbeat
→ App Server / artifact store

Security:
real process/container sandbox
→ fs/network/secret policies
→ escape tests

Evaluation:
multi-file hidden-test repository suite
→ SWE-bench adapter
→ Browser environment
→ Computer/OS environment

Coding / Multi-Agent:
tree-sitter / LSP / semantic patch
→ parallel workers
→ mailbox / reviewer / merge
→ 1/2/4/8-agent controlled comparison
```

**最终毕业标准不是“模块名都出现过”，而是每一层都能解释、实现、数值/行为验证、失败恢复、真实评测，并明确自己还没有证明什么。**
