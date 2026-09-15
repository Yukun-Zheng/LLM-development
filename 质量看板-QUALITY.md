# 全书质量看板 / Quality Dashboard v3

> **目的**：把“还差什么”变成可审计问题，而不是靠 Markdown 数量制造完成感。机器可读 capability 状态以 [`能力清单-CAPABILITIES.json`](能力清单-CAPABILITIES.json) 为准。

成熟度：L0=边界/来源；L1=教材机制完整；L2=代码+自动测试/parity；L3=真实 benchmark、消融、反例和研究证据。

---

# 1　当前总体状态

| 技术层 | 当前状态 | 已有硬证据 | 最大缺口 |
|---|---|---|---|
| Model Science | **L1–L2，覆盖强** | own Transformer + SmolLM2 checkpoint parity | 更多 architecture parity；SSM/MoE/Diffusion 实验化 |
| Inference Systems | **L2 reference layer** | contiguous KV、reference Paged KV parity、request scheduler metrics | real batched executor、allocator、prefix、speculation、throughput |
| Post-training | **从 L1 进入 L2 primitive** | masked SFT + DPO objective/optimizer tests | dataset/checkpoint/eval pipeline；RLVR/GRPO update |
| Agent Kernel | **L2** | Codex turn、event replay、fork/cancel、leased work queue | steering、integrated runtime、App Server、worker pool |
| Context / Memory | **L2 reference layer** | persistent event memory + compaction provenance | notes、semantic index、artifact provenance、token budget |
| Protocols | **L2 subset** | minimal MCP tests | fuller MCP + A2A 1.0 |
| Security | **L1–L2** | pre-dispatch deny/approval enforcement | **真实 OS/container sandbox、network、credentials** |
| Multi-Agent | **L2 primitive** | task DAG + coordinator + worktree | parallel scheduler、mailbox、review/merge、controlled eval |
| Evaluation | **L2 substrate** | trajectory metrics + BenchmarkCase/Grader harness | real repository/browser/OS benchmark adapters |
| Browser / Computer | **L1** | HTTP text fetch only | DOM/A11y、screenshot、grounding、actions、verifier |
| Observatory | **L1** | machine-readable snapshot + freshness audit | version history、auto-generated frontier tables |

---

# 2　已经跨过的关键门槛

## 2.1　真实 checkpoint，而不是 toy-only model

`HuggingFaceTB/SmolLM2-135M` raw safetensors → 本项目自己的 Transformer runtime，在固定 CPU float32 / HF eager reference 条件：

```json
{"max_abs": 0.0, "mean_abs": 0.0, "argmax_agreement": 1.0}
```

## 2.2　KV Cache → Reference Paged KV

已有逻辑 page、per-layer page table、incremental suffix ingest，并验证 paged prefill/decode logits 与 full forward 一致。

边界：当前仍重建 contiguous K/V，因此是语义 reference，不是 production allocator。

## 2.3　Serving Scheduler 第一次进入代码

`scheduler.py` 已实现 WAITING→PREFILL→DECODE→FINISHED/CANCELLED 生命周期、prefill token admission、decode-first batch selection、TTFT/TPOT/latency。

Fast CI run 47：

```text
43 passed, 1 warning in 2.41s
Ruff correctness lint: All checks passed
```

尚未实现真正 batched model execution，所以不能把它宣传成完整 continuous batching engine。

## 2.4　Post-training 不再只有公式

`posttraining.py` 已实现：

```text
masked causal SFT objective
sequence log-probabilities
DPO implicit reward / preference margin
SFT optimizer step
DPO optimizer step with frozen reference
```

Fast CI run 51：

```text
49 passed, 1 warning in 2.11s
Ruff correctness lint: All checks passed
```

其中 policy==reference 时 DPO loss=`log(2)` 已被自动测试；SFT/DPO 真实参数更新也已验证。

## 2.5　Agent turn → Durable Agent Kernel

现在已经分成：

```text
Turn Executor       codex_harness.py
Thread/Event Store  durable.py
Work Ownership      runtime_queue.py
Context Provenance  context.py
```

Thread 支持 close/reopen replay、checkpoint、pause/resume、fork、cancel。Fork 记录 parent provenance，child cancellation 不改变 parent。

Fast CI run 49：

```text
45 passed, 1 warning in 1.95s
Ruff correctness lint: All checks passed
```

## 2.6　Memory → Provenance-aware Context

Compaction 生成新的 summary fragment，并保存 parent IDs；原 raw events 不被覆盖。

## 2.7　Permission 第一次成为执行 gate

DENY / rejected approval 会在 `ToolRegistry` dispatch 之前物理阻断应用层 tool body。仍不等于 OS sandbox。

## 2.8　Evaluation substrate 已经落地

`evaluation.py` 记录 trajectory metrics；`benchmark.py` 把 Case / Executor / Grader / Grade / Record 分开。模型说“完成”不再自动等于 benchmark success。

---

# 3　模型与系统主线

| 主题 | 等级 | 已有证据 | 下一关键动作 |
|---|---|---|---|
| Transformer / RoPE / RMSNorm / GQA / SwiGLU | **L2** | own runtime + tests + real checkpoint parity | 第二模型族；activation parity |
| Tokenization / Data | L2 | BPE code/tests | fertility / dedup / mixture experiments |
| GPT-3 / Scaling / ICL | L1 | primary sources + math | tiny scaling / controlled ICL |
| MoE | L1 | theory/source | router + load balance + expert-parallel toy |
| **SFT** | **L2 primitive** | mask + CE + optimizer step | dataset/batch/checkpoint/training curves |
| **DPO** | **L2 primitive** | analytic + optimizer tests | pairwise pipeline + preference evaluation |
| Reasoning / RLVR | L1 | theory + verifier | group-relative objective + actual policy update |
| Inference | **L2 reference** | contiguous/paged parity + request scheduler | batched executor → prefix → batching → speculation |
| Hardware / Kernels | L1 | theory/source | roofline / SDPA / bandwidth lab |
| Hybrid Sequence Architectures | L1 | theory/source | SSM/linear-attention reference code |
| Diffusion LM | L1 | objective/mechanism | tiny denoise/remask runtime |
| Interpretability | L1 | theory/source | activation patch / SAE |
| Multimodal Runtime | L0–L1 | historical/theory | typed multimodal tensors + vision path |

---

# 4　Agent / Runtime 主线

| Capability | 状态 | 当前证据 | 关键缺口 |
|---|---|---|---|
| Codex-style Turn Executor | validated | tool/approval/event tests | steering + external event stream |
| Durable Thread/Event Store | **validated** | replay + fork + cancel tests | runtime integration / protocol parity |
| Durable Work Queue | **validated** | lease / expiry reclaim / cancel | heartbeat / fairness / async workers |
| Planning DAG | validated | dependency/cycle/failure tests | dynamic replan integration |
| Verifier | validated | file/command/composite tests | artifact/benchmark graders |
| Persistent Memory | validated primitive | SQLite tests | semantic/artifact retrieval |
| Context Provenance | **validated** | summary lineage preserves raw evidence | token budget / notes / index |
| Coding Agent | L2 primitive | repo tools/edit/test/Git | tree-sitter/LSP/semantic patch/SWE |
| MCP | L2 subset | discover/list/call | transports/auth/tasks/extensions |
| A2A | planned | theory/spec | AgentCard/Task/Message/Artifact |
| Multi-Agent | L2 primitive | coordinator/worktree | parallel/mailbox/reviewer/merge |
| Browser | partial | HTTP text | JS/DOM/A11y/screenshot/actions |
| Computer Use | planned | theory | real GUI loop + verifier |

---

# 5　Security P0

已经有：

```text
Action Proposal
→ Permission Policy
→ Optional Approval
→ Dispatch / Deny
```

仍必须实现：

```text
Credential Scope
→ Process / Container Sandbox
→ Filesystem Policy
→ Network Policy
→ Execution
→ Audit / Monitor
```

因此在扩张 Browser/Computer Use 之前，**enforced sandbox 仍是最高风险工程缺口之一**。

---

# 6　Evaluation：从 substrate 到真实环境

现在已经有：

```text
BenchmarkCase
→ Executor
→ trajectory
→ Grader
→ Grade
→ AggregateMetrics
```

但真实能力还要进入：

```text
repository fixture
→ SWE-bench
→ browser environment
→ OS/computer environment
→ long-horizon crash/recovery
```

每个架构改动最终都应该能做 controlled comparison：memory/no-memory、planner/reactive、1/2/4/8 agents、reviewer/no-reviewer、paged/contiguous、speculative/normal。

---

# 7　v3 P0 当前进度

| P0 | 当前 | 验收 |
|---|---|---|
| Capability Manifest | ✅ | CI 校验 code/test/evidence/path |
| Durable Thread Replay | ✅ | process restart 后恢复 |
| Thread Fork / Cancellation | ✅ | child prefix/provenance/independent terminal state |
| Durable Work Queue / Lease | ✅ | exclusion + expiry reclaim + cancel |
| Context Provenance | ✅ | summary 可追回 raw evidence |
| Trajectory Metrics | ✅ | success/steps/tools/time/cost schema |
| Benchmark Case/Grader substrate | ✅ | execution ≠ grading；suite aggregate |
| Permission Gate | ✅ | deny/reject 时 tool body 不执行 |
| Reference Paged KV | ✅ | paged vs full-forward logits parity |
| Reference Request Scheduler | ✅ | admission/decode/cancel/TTFT/TPOT |
| SFT / DPO objective primitives | ✅ | hand-check invariants + optimizer update |
| **Actual Batched Model Executor** | ❌ | scheduler batch 真正进入 batched prefill/decode |
| **Enforced Sandbox** | ❌ | process/filesystem/network/secret isolation + escape tests |
| **Real Environment Benchmark** | ❌ | final state/artifact graders |
| **RLVR / GRPO-style policy update** | ❌ | verifier reward + actual optimization |

---

# 8　下一优先顺序

1. **把 Reference Scheduler 接到真实 batched model execution**，记录 TTFT/TPOT/throughput，而不是只调度状态机。
2. **做 repository fixture benchmark + artifact/final-state grader**，再接 SWE-bench。
3. **构建 OS/container sandbox reference**，并用 escape tests 验证，而不是只扩 permission enum。
4. **把 Thread + WorkQueue + TurnExecutor 接成一个 resumable runtime**，加入 pending steering/heartbeat。
5. **补 RLVR/GRPO toy update**，形成 SFT→DPO→verifier-RL 教学闭环。
6. Prefix Cache / Speculative Decoding、MCP fuller semantics/A2A、parallel agent reviewer、Browser/Computer Use。

---

# 9　决策原则

```text
真实 parity / final environment evidence
> recovery / safety invariants
> controlled benchmark / ablation
> automated tests
> Source-First interpretation
> Markdown count
```

项目最终质量看：**原始 claim 能否追到形式化；形式化能否追到代码；代码能否追到测试；系统能力能否追到真实评测；失败边界能否明确复现。**
