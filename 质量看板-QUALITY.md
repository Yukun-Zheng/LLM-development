# 全书质量看板 / Quality Dashboard v3

> **目的**：把“还差什么”变成可审计问题。  
> v3 不再主要统计 Markdown 数量；机器可读 capability 状态以 [`能力清单-CAPABILITIES.json`](能力清单-CAPABILITIES.json) 为准。

成熟度仍使用：

| 等级 | 标准 |
|---|---|
| **L0** | 有问题定义、边界、原始资料入口 |
| **L1** | 机制 / 数学 / 状态机 / 原始来源基本完整 |
| **L2** | 有从零代码、自动测试、parity 或可复现实验 |
| **L3** | 有独立复现、消融、反例、争议、真实 benchmark 与开放问题 |

---

# 1　v3 总体评价

| 技术层 | 当前状态 | 最强证据 | 最大缺口 |
|---|---|---|---|
| Model Science | **L1–L2，覆盖强** | Transformer + SmolLM2 real checkpoint parity | 更多 architecture parity；实验化 SSM/MoE/Diffusion |
| Inference Systems | **L1–L2，代码偏早** | contiguous KV + incremental parity | Paged KV、scheduler、batching、speculation、serving metrics |
| Post-training | **L1，明显短板** | 理论 / 数学 / verifier primitive | SFT → DPO → RLVR 可运行闭环 |
| Agent Kernel | **L2 起步** | Codex-style turn + durable event store | async queues、fork/cancel、context provenance、App Server |
| Protocols | **L2 subset** | minimal MCP tests | MCP 2026-07-28 fuller semantics + A2A 1.0 |
| Security | **L1–L2** | pre-dispatch deny/approval enforcement | **真实 OS/container sandbox、credentials、network** |
| Multi-Agent | **L2 primitive** | task DAG + coordinator + worktree | parallel scheduler、mailbox、review/merge、controlled eval |
| Evaluation | **L1–L2 起步** | trajectory metric schema + tests | SWE-bench/browser/OS real graders |
| Browser / Computer | **L1** | HTTP text fetch only | DOM/A11y、screenshot、grounding、actions、verifier |
| Observatory | **L1** | machine-readable frontier snapshot + drift audit | 自动生成前沿表、版本历史与 revalidation workflow |

---

# 2　当前已经真正跨过的门槛

## 2.1　真实公开 checkpoint parity

`HuggingFaceTB/SmolLM2-135M` raw safetensors → 本项目自己的 Transformer runtime，在固定受测 CPU float32 / HF eager setting 下：

```json
{"max_abs": 0.0, "mean_abs": 0.0, "argmax_agreement": 1.0}
```

这比“写了一个像 Llama 的 toy model”更强，但只证明当前受测模型与设置。

## 2.2　Agent turn 不再只是 ReAct while-loop 描述

`codex_harness.py` 已有显式 turn / model / approval / tool / completion events，并有 tool follow-up、approval allow/deny、step-limit 回归测试。

## 2.3　Long-running state 开始脱离单进程内存

`durable.py` 已实现 SQLite append-only event log 与 replay projection。测试真实关闭数据库、重新打开，再恢复 running turn、submission 与 checkpoint。

## 2.4　Permission 开始成为执行边界

`security.py` 在 dispatch 前执行 ALLOW / REQUIRE_APPROVAL / DENY。测试验证 deny / approval rejection 时底层 tool body 没有执行。

这仍不等于 sandbox。

## 2.5　Evaluation 与 pytest 开始分层

`evaluation.py` 统一记录 success、verifier、steps、tool calls、failures、approvals、wall time、tokens、cost。下一阶段才进入真实 benchmark adapters。

## 2.6　最新代码证据

Fast CPU CI：

```text
36 passed, 1 warning in 2.14s
Ruff correctness lint: All checks passed
```

---

# 3　模型科学与系统主线

| 主题 | 当前等级 | 现有证据 | 下一步进入 L2/L3 的关键动作 |
|---|---|---|---|
| Transformer / RoPE / RMSNorm / GQA / SwiGLU | **L2** | own runtime + tests + real checkpoint parity | activation-level parity；第二模型族 |
| Tokenization / Data | L2 | BPE code + tests + Source-First chapter | fertility / dedup / mixture experiments |
| GPT-3 / Scaling / ICL | L1 | 原始资料 + 公式 | tiny scaling curves；ICL controlled toy experiments |
| MoE | L1 | theory/source | router + load-balance + expert-parallel toy |
| Optimization Dynamics | L1 | theory/source | optimizer trajectories / loss spike lab |
| RLHF / DPO | **L1** | math/source | tiny pairwise pipeline；DPO loss parity |
| Reasoning / RLVR | **L1** | theory + verifier primitive | trajectory/reward schema + actual policy update |
| Training Systems | L1 | theory/source | communication/memory simulator |
| Inference Systems | L2 reference layer | contiguous KV / engine | paged KV → scheduler → batching → speculation |
| Hardware / Kernels | L1 | theory/source | roofline / bandwidth / SDPA kernel lab |
| Hybrid Sequence Architectures | L1 | S4/Mamba/RWKV/xLSTM/Titans theory | reference SSM / linear-attention code + parity |
| Diffusion LM | L1 | objective/mechanism | tiny corruption-denoise-remask runtime |
| Interpretability | L1 | source/mechanism | activation patch / SAE experiments |
| Multimodal Model Runtime | L0–L1 | historical/theory | vision projector / typed multimodal inputs |

**判断**：模型侧现在不缺“新章节名”，缺把已有核心主题做成 L2/L3。

---

# 4　Agent / Runtime 主线

| Capability | 状态 | 当前证据 | 关键缺口 |
|---|---|---|---|
| Minimal Agent Loop | validated | scripted tests | 被 durable kernel 吸收，不再作为终点 |
| Codex-style Turn Executor | validated | behavioral tests | async input/event queues |
| Durable Thread/Event Store | **validated** | replay-after-reopen tests | fork / cancellation / leases / distributed workers |
| Planning DAG | validated | cycle/failure propagation tests | dynamic replan + scheduler integration |
| Verifier | validated | command/file/composite tests | artifact/benchmark graders |
| Persistent Memory | validated primitive | SQLite tests | provenance / notes / semantic index / compaction |
| Coding Agent | L2 primitive | repo tools/edit/test/Git | tree-sitter/LSP/semantic patch/SWE-bench |
| MCP | L2 subset | discover/list/call tests | 2026 stateless semantics / transports / auth / tasks |
| A2A | planned | theory/spec | AgentCard / Message / Task / Artifact runtime |
| Multi-Agent | L2 primitive | coordinator/worktree | parallelism + mailbox + reviewer + merge |
| Browser | partial | HTTP text only | JS/DOM/A11y/screenshot/actions |
| Computer Use | planned | theory | real GUI loop + state verifier |

---

# 5　Security：从“安全章节”转成 Runtime P0

当前已经完成的只是第一层：

```text
Action Proposal
→ Permission Policy
→ Optional Approval
→ Dispatch / Deny
```

真正安全边界仍必须实现：

```text
Credential Scope
→ Process / Container Sandbox
→ Filesystem Policy
→ Network Policy
→ Execution
→ Audit Event
→ Monitor / Verifier
```

在 shell / browser / computer use 大规模扩展前，**enforced sandbox 是 P0，不是 P2**。

---

# 6　Evaluation：当前最重要的科研缺口之一

Unit tests 只回答代码正确性。能力结论最终必须进入独立 [`评测-eval/`](评测-eval/)：

```text
trajectory
+ final environment state
+ artifact
+ grader/verifier
+ cost/latency accounting
```

未来必须能做这些 controlled comparisons：

```text
memory vs no-memory
planner vs reactive
compaction A vs B
1 vs 2 vs 4 vs 8 agents
reviewer vs no-reviewer
DOM vs screenshot observation
paged vs contiguous KV
speculative vs normal decode
```

任何“更强”结论没有 baseline 就不升 L3。

---

# 7　v3 P0

| P0 | 当前 | 验收标准 |
|---|---|---|
| Machine-readable Capability Manifest | **已落地** | CI 校验 theory/code/test/evidence 路径与状态 |
| Durable Thread / Event Store / Replay | **已落地并测试** | 进程关闭后 replay 恢复状态 |
| Trajectory Metrics | **已落地并测试** | 统一 success/steps/tools/time/cost schema |
| Permission Gate | **已落地并测试** | deny 时底层 tool body 不执行 |
| **Enforced Sandbox** | 未完成 | process/filesystem/network/secret isolation + escape tests |
| **Context Provenance System** | 未完成 | fragment → raw event/artifact 可追溯 |
| **Paged KV Reference** | 未完成 | contiguous-vs-paged logits parity |
| **Request Scheduler** | 未完成 | continuous batching + TTFT/TPOT/throughput metrics |

---

# 8　v3 P1

- [ ] Prefix Cache / Chunked Prefill / Speculative Decoding
- [ ] SFT → DPO → verifier/RL tiny post-training pipeline
- [ ] MCP 2026-07-28 fuller semantics
- [ ] minimal A2A 1.0
- [ ] scoped AGENTS / context provenance / compaction / notes / index
- [ ] tree-sitter / LSP / semantic patch / test-selection
- [ ] Browser DOM/A11y + screenshot + Computer Use action/verifier
- [ ] parallel worktree agents + reviewer/merge
- [ ] SWE-bench / browser / OS environment benchmark adapters
- [ ] Hybrid sequence model reference labs

---

# 9　教材与证据基础设施

已经有：

- [x] Source-First 方法论
- [x] paper/source card + claim ledger 起步
- [x] strict internal-link audit
- [x] source coverage audit
- [x] math normalization
- [x] Theory ↔ Code ↔ Source ↔ Test 学习地图
- [x] **machine-readable Capability Manifest + CI validator**
- [x] **Frontier Observatory + machine-readable snapshot + freshness audit**
- [x] **独立 Evaluation Lab**

仍需：

- [ ] 从 manifest 自动生成 README/QUALITY/MAP 的部分表格
- [ ] 外部来源健康检查（非硬阻塞）
- [ ] 扩大 Source Card / Claim Ledger 覆盖
- [ ] 网站 / PDF / print build
- [ ] 双语术语表 / Index
- [ ] 章节习题与答案分离版

---

# 10　决策原则

遇到“继续加章节”和“把核心系统做实”的冲突时，顺序固定为：

```text
真实 parity / real environment evidence
> recovery / safety invariants
> benchmark / controlled ablation
> automated tests
> Source-First interpretation
> new Markdown count
```

项目的质量最终看：

> **原始 claim 能否追到形式化，形式化能否追到代码，代码能否追到测试，系统能力能否追到真实评测，失败边界能否被明确观测。**
