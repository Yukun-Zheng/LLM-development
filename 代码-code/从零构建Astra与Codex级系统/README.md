# 从零构建 Astra-class 与 Codex-class 系统

这是整本教材的**终极 Reference System**：从模型 forward、真实权重加载、推理状态管理，一直写到 post-training objective、durable Agent kernel、协议、安全、评测、Coding Agent、Computer Use 与 Multi-Agent。

> **严格边界**：我们不声称复刻未公开的 frontier 权重、训练 recipe 或 OpenAI 云端全部生产基础设施。Open/tiny models 用于真正从零理解与 parity；frontier models 可以作为可插拔 Model Backend。**模型智能可以插拔，系统智能尽量由本项目自己实现。**

- 实时实现状态：[`实现状态-STATUS.md`](实现状态-STATUS.md)
- 机器事实源：[`../../能力清单-CAPABILITIES.json`](../../能力清单-CAPABILITIES.json)
- v3 蓝图：[`../../教材-book/99-全书架构蓝图-v3.md`](../../教材-book/99-全书架构蓝图-v3.md)
- 独立评测层：[`../../评测-eval/README.md`](../../评测-eval/README.md)

---

# 1　当前源码地图

```text
src/astra_codex/
│
├── Model / Inference
│   ├── config.py
│   ├── tokenizer.py
│   ├── model.py
│   ├── weights.py
│   ├── public_checkpoint.py
│   ├── cache.py
│   ├── paged_cache.py       # reference logical pages + parity
│   ├── engine.py
│   ├── scheduler.py         # request lifecycle / prefill / decode / TTFT / TPOT
│   └── sampling.py
│
├── Post-training
│   └── posttraining.py      # masked SFT + sequence logp + DPO + optimizer steps
│
├── Durable Agent Kernel
│   ├── agent.py
│   ├── codex_harness.py
│   ├── durable.py           # Thread / Turn / replay / fork / cancellation
│   ├── runtime_queue.py     # lease / reclaim / ack / cancel
│   ├── context.py           # typed context + compaction provenance
│   ├── memory.py
│   ├── planning.py
│   ├── verification.py
│   └── evaluation.py
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
├── Multi-Agent / Coding
│   ├── worktree.py
│   ├── multi_agent.py
│   └── coding.py
│
└── Evaluation
    └── benchmark.py         # BenchmarkCase / Grader / Record / aggregation
```

暂时保持 flat imports 以保护已经通过的 tests/parity；subsystem contract 稳定后再逐步迁移到 `model/ inference/ posttraining/ agent/ protocols/ security/ eval/` 子包。

---

# 2　模型侧：已经从 toy runtime 跨到真实 checkpoint

当前真实 checkpoint：

```text
HuggingFaceTB/SmolLM2-135M
```

```text
config.json + raw safetensors
        ↓
our key mapper
        ↓
our RMSNorm / RoPE / GQA / SwiGLU / Transformer
        ↓
our logits
        ↕
HF eager reference logits
```

固定受测 CPU float32：

```json
{"max_abs": 0.0, "mean_abs": 0.0, "argmax_agreement": 1.0}
```

这个证据只覆盖当前受测模型/输入/数值设置。下一阶段是 multi-architecture parity matrix。

---

# 3　Inference：从 KV Cache 进入 Serving Runtime

## 3.1 Contiguous KV

已经验证 cached incremental decode 与 full recomputation logits 对齐。

## 3.2 Reference Paged KV

`paged_cache.py` 已有逻辑 page、per-layer page table、suffix ingest 与 paged decode parity。

当前 reference path 会把 pages 重新拼成 contiguous `past_key_values`，所以它证明的是：

> **分页状态语义正确。**

它不证明已经拥有 vLLM 的 block allocator / zero-copy kernel / 显存收益。

## 3.3 Reference Request Scheduler

`scheduler.py` 把 serving 生命周期显式化：

```text
arrival
→ WAITING
→ PREFILL admission
→ DECODE
→ FINISHED / CANCELLED
```

已支持：

```text
max_batch_size
max_prefill_tokens
decode-first scheduling
cancellation
TTFT
TPOT
total latency
```

Fast CI 已验证调度和指标行为。下一步不是继续改接口，而是接入**真实 batched model executor**，再测 continuous batching 的 throughput / fairness / memory。

---

# 4　Post-training：数学已经第一次变成 optimizer step

此前 RLHF / DPO 主要在教材公式里。现在 `posttraining.py` 已经落到 tensor。

## SFT

```text
input ids
→ logits [B,T,V]
→ causal shift
→ mask prompt/user targets with -100
→ CE
→ backward
→ grad clip (optional)
→ optimizer.step
```

## DPO

```text
policy chosen/rejected sequence logp
reference chosen/rejected sequence logp
→ log-ratio preference margin
→ beta-scaled implicit rewards
→ -log sigmoid(margin)
→ backward
→ update policy only
```

自动测试包括：uniform logits 手算 CE、policy==reference 时 DPO loss=`log(2)`、SFT/DPO 真实更新 policy、reference 保持冻结。

Fast CI run 51：

```text
49 passed, 1 warning in 2.11s
Ruff correctness lint: All checks passed
```

这还不是完整训练平台：dataset/dataloader/checkpoint/eval、reward model、GRPO/RLVR 仍待实现。

---

# 5　Agent：从 Turn Loop 进入 Durable Kernel

现在不再把：

```text
model → tool → model
```

当作整个 Agent architecture。

## 5.1 Turn Executor

`codex_harness.py`：

```text
TURN_STARTED
→ sampling
→ optional approval
→ tool
→ observation
→ follow-up
→ TURN_COMPLETED / TURN_STOPPED
```

## 5.2 Event-sourced Thread

`durable.py`：

```text
Thread
→ Submission
→ Turn
→ Checkpoint
→ Replay
→ Pause / Resume
→ Fork
→ Cancel / Complete / Fail
```

进程关闭后重新打开数据库仍能恢复 running turn / submissions / checkpoint。

Fork 会复制指定 event prefix 到独立 child stream，并保存：

```text
parent_thread_id
parent_event_id
```

子 thread 的 cancellation 不改变 parent state。

## 5.3 Durable Work Queue

`runtime_queue.py`：

```text
PENDING
→ LEASED(worker)
├─ ACK → COMPLETED
├─ FAIL → FAILED
├─ CANCEL → CANCELLED
└─ lease expires → reclaim
```

这把 task state 与 worker execution ownership 分开。

---

# 6　Context：不再把 Summary 当成历史本身

`context.py` 已开始区分：

```text
RAW_EVENT
NOTE
SUMMARY
ARTIFACT
RETRIEVAL
INSTRUCTION
```

Compaction 产生新的 `SUMMARY` fragment，并记录 parent IDs；raw observations 不被覆盖。

因此以后模型看到一个摘要时，可以追问：

```text
这个 summary 来自哪些 raw events？
是谁/哪种 policy 压缩的？
原始证据还能不能取回？
```

下一步是 persistent notes、semantic index、artifact provenance 与 token-budget context builder。

---

# 7　Security：第一道真实 gate 已经进入执行路径

当前：

```text
Action Proposal
→ PermissionProfile
├─ ALLOW
├─ REQUIRE_APPROVAL
└─ DENY
→ optional approval
→ ToolRegistry dispatch
```

DENY / rejected approval 会在 tool body 执行前阻断。

**仍然没有冒充 OS sandbox。** process / filesystem / network / credentials / syscall isolation 仍是独立 P0。

---

# 8　Evaluation：测试与能力评测正式分层

`evaluation.py` 记录 trajectory cost/behavior；`benchmark.py` 新增：

```text
BenchmarkCase
→ Executor
→ Result / trajectory
→ Grader
→ Grade
→ BenchmarkRecord
→ AggregateMetrics
```

这允许以后同一个 task 在不同系统设置下做严格 A/B：

```text
memory vs no memory
planner vs reactive
1 agent vs N agents
reviewer vs no reviewer
paged vs contiguous KV
```

当前 grader 只包含 deterministic toy reference。真实 repository/browser/OS benchmark adapter 仍未完成。

---

# 9　当前系统总图

```text
                        Model Backend
                             │
          ┌──────────────────┴──────────────────┐
          ↓                                     ↓
   Open/Tiny Runtime                      Frontier API
          │                                     │
          └──────────────────┬──────────────────┘
                             ↓
                     Durable Thread
                             ↓
                       Work Queue
                             ↓
                       Turn Executor
                             ↓
                    Context Builder
                             ↓
                   Planner / Policy
                             ↓
              Permission / Approval
                             ↓
       Tool / MCP / Browser / Remote Agent
                             ↓
                       Environment
                             ↓
                        Verifier
                             ↓
              Checkpoint / Replan / Finish
                             ↓
                  Benchmark / Evidence
```

---

# 10　下一阶段最高优先级

### Inference

```text
Reference Scheduler
→ actual batched model executor
→ continuous batching
→ prefix cache
→ block allocator
→ chunked/disaggregated prefill
→ speculative decoding
```

### Post-training

```text
SFT/DPO objective primitives
→ real dataset pipeline
→ checkpointable loops
→ verifier reward
→ GRPO/RLVR-style toy policy update
```

### Agent OS

```text
Thread + Queue + Turn
→ integrated runtime
→ pending steering
→ worker heartbeat
→ App Server control plane
→ artifact/rollout store
```

### Security

```text
Permission Gate
→ enforced process/container sandbox
→ filesystem/network/secret scope
→ escape tests
```

### Evaluation

```text
Benchmark Harness
→ repository fixture suite
→ SWE-bench adapter
→ browser environment
→ OS/computer environment
→ long-horizon crash/recovery
```

**终点不是代码文件数量，而是：每一层都能解释、执行、测试、恢复、评测，并且知道自己还没有证明什么。**
