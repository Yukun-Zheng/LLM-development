# 全书质量看板 / Quality Dashboard v3

> **原则**：全书不再以 Markdown 文件数衡量成熟度。机器事实源：[`能力清单-CAPABILITIES.json`](能力清单-CAPABILITIES.json)。  
> L0=边界/来源；L1=教材机制；L2=代码+自动测试/parity；L3=真实 benchmark、消融、反例和研究证据。

---

# 1　总体状态

| 技术层 | 当前 | 已有硬证据 | 最大缺口 |
|---|---|---|---|
| Model Science | **L1–L2，覆盖强** | own Transformer + SmolLM2 real checkpoint parity | 更多模型族、SSM/MoE/Diffusion 实验化 |
| Inference Systems | **L2 reference** | contiguous/paged parity + scheduler + real homogeneous batched forward | variable-length continuous batching、allocator、prefix/speculation、系统 benchmark |
| Post-training | **L2 primitive 已形成** | masked SFT、DPO、group-relative clipped RL objective tests | dataset/checkpoint loop、rollout/verifier、真正 tiny-policy RL update |
| Agent Kernel | **L2** | Codex Turn + replay/fork/cancel + leased queue + integrated runtime | crash recovery/idempotency、steering、heartbeat、App Server |
| Context / Memory | **L2 reference** | persistent history + typed context + compaction lineage | notes、semantic index、artifact provenance、token budget |
| Protocols | **L2 subset** | minimal MCP | fuller MCP + A2A |
| Security | **L1–L2** | pre-dispatch permission/approval enforcement | **OS/container sandbox、network、credentials、escape tests** |
| Multi-Agent | **L2 primitive** | DAG/coordinator/worktree | parallel workers、mailbox、reviewer/merge、controlled eval |
| Evaluation | **L2 local substrate** | trajectory metrics + case/grader + repository final-state grader | hidden tests/SWE/browser/OS/long-horizon benchmarks |
| Browser / Computer | **L1** | HTTP text fetch only | DOM/A11y、screenshot、grounding、actions、verifier |
| Observatory | **L1** | machine snapshot + freshness audit | version history、auto-generated frontier tables |

---

# 2　最新硬证据

Fast CPU CI run 66：

```text
61 passed, 1 warning in 3.67s
Ruff correctness lint: All checks passed
```

这一轮回归已经同时覆盖：

```text
real model checkpoint parity primitives
contiguous / paged KV
request scheduler
actual homogeneous batched model forward
SFT / DPO
group-relative verifier-RL objective
Codex-style turn harness
Thread replay / fork / cancellation
leased work queue
integrated Thread→Queue→Turn runtime
context provenance
permission enforcement
benchmark substrate
repository final-state grading
```

---

# 3　已跨过的关键门槛

## Model：toy → public checkpoint

`HuggingFaceTB/SmolLM2-135M` raw weights 进入本项目自己的 Transformer runtime；固定 CPU float32 / HF eager reference：

```json
{"max_abs": 0.0, "mean_abs": 0.0, "argmax_agreement": 1.0}
```

## Inference：generation loop → serving primitives

已经形成：

```text
Contiguous KV
→ Reference Paged KV
→ Request Scheduler
→ Homogeneous Batched Executor
```

`HomogeneousBatchExecutor` 一次真实 model forward 同时处理多 request，并通过 individual full recomputation 对照验证 logits。当前只支持 equal-length cache cohort；因此仍不能声称完成 production continuous batching。

## Post-training：公式 → gradient

现在已经能从代码里直接检查：

```text
SFT mask / shift / CE
DPO four sequence log-probs / implicit margin
Group-relative reward normalization / clipped policy ratio / optional KL
```

并已经有 analytic invariants 与真实 optimizer/gradient tests。

## Agent：loop → durable kernel

```text
Thread/Event Store
+ Work Queue/Lease
+ Turn Executor
+ Checkpoint
→ Integrated Durable Runtime
```

正常路径、多个 submission、terminal-thread work cancellation、step-limit failure 都已有自动验证。

## Evaluation：assistant claim → final-state evidence

`repository_eval.py` 已证明：Agent 只说“修好了”但 repository state 未变时，独立 grader 必须判 FAIL；真正 edit + final-state verifier 才 PASS。

---

# 4　模型 / Post-training / 系统成熟度

| 主题 | 等级 | 当前证据 | 下一动作 |
|---|---|---|---|
| Transformer / RoPE / RMSNorm / GQA / SwiGLU | L2 | own runtime + checkpoint parity | 第二架构/model family + activation parity |
| Tokenization / Data | L2 | BPE code/tests | fertility / dedup / mixture controlled labs |
| Scaling / ICL | L1 | primary sources + math | tiny scaling / ICL experiments |
| MoE | L1 | theory/source | router/load balance/expert parallel toy |
| SFT | **L2 primitive** | analytic mask/CE + optimizer update | dataset/packing/checkpoint/eval |
| DPO | **L2 primitive** | `loss=log(2)` invariant + frozen ref | pairwise pipeline + held-out preference eval |
| Group-relative verifier RL | **L2 objective primitive** | advantages/gradient/clipping/KL tests | rollouts + old policy + verifier + actual model update |
| Full RLVR pipeline | L1 | theory | end-to-end tiny policy experiment |
| Paged KV | L2 reference | paged/full numerical parity | physical block allocator / fragmentation |
| Scheduler | L2 reference | lifecycle + TTFT/TPOT | scheduler/executor integration |
| Batched Executor | **L2** | true batched prefill/decode parity | variable-length state / block tables |
| Speculative Decode | L1 | theory | draft-target reference + distribution parity |
| Hybrid Sequence Architectures | L1 | theory/source | SSM/linear-attention reference parity |
| Diffusion LM | L1 | theory | tiny denoise/remask code |
| Interpretability | L1 | theory/source | activation patch / SAE experiments |

---

# 5　Agent Runtime 成熟度

| Capability | 状态 | 证据 | 最大缺口 |
|---|---|---|---|
| Codex-style Turn Executor | validated | tool/approval/follow-up/stop tests | external event stream / live steering |
| Durable Thread | validated | restart replay / fork / cancel | active-turn crash recovery |
| Durable Work Queue | validated | lease exclusion/reclaim/cancel | heartbeat / fairness / async pool |
| **Integrated Durable Runtime** | **validated** | Thread→Queue→Turn→checkpoint→ACK | durable side-effect record / idempotency |
| Planning DAG | validated | cycle/failure propagation | dynamic replan integration |
| Verifier | validated | file/command/composite | artifact/final-state graders expansion |
| Persistent Memory | validated primitive | SQLite | semantic/artifact retrieval |
| Context Provenance | validated | compaction lineage retains raw evidence | notes/index/token-budget builder |
| Coding Agent | L2 primitive | repo tools/edit/test/Git | semantic code intelligence / SWE |
| MCP | L2 subset | discover/list/call | transports/auth/tasks/extensions |
| A2A | planned | theory/spec | protocol runtime/conformance |
| Multi-Agent | L2 primitive | coordinator/worktree | parallel/mailbox/reviewer/merge |
| Browser | partial | HTTP text | JS/DOM/A11y/screenshot/actions |
| Computer Use | planned | theory | GUI loop + verifier |

---

# 6　Security P0

当前做到：

```text
proposal
→ permission decision
→ optional approval
→ dispatch / deny
```

下一层必须是真正执行边界：

```text
credential scope
→ process/container isolation
→ filesystem policy
→ network policy
→ resource limits
→ execution
→ audit / monitor
→ escape tests
```

在 Browser/Computer Use 扩张之前，**真实 sandbox 仍是最高优先级风险项之一**。

---

# 7　Evaluation Roadmap

已完成：

```text
Level 0: deterministic toy Case/Grader            ✅
Level 1a: deterministic local repository fixture  ✅
```

下一步：

```text
Level 1b: multi-file + hidden-test fixture suite
Level 2 : SWE-bench adapter
Level 3 : browser environment
Level 4 : OS/computer environment
Level 5 : long-horizon crash/recovery + cross-app
```

每次必须记录 trajectory、final state、artifact、grader、wall time、tool calls、tokens/cost（可得时）。

---

# 8　当前 P0 / P1

| 任务 | 当前 | 验收标准 |
|---|---|---|
| Capability Manifest | ✅ | CI 强校验 code/test/evidence/path |
| Real checkpoint parity | ✅ 第一模型族 | 扩为 multi-architecture matrix |
| Reference Paged KV | ✅ | 下一步 allocator/memory metrics |
| Request Scheduler | ✅ reference | 下一步 variable-length executor integration |
| Real batched model forward | ✅ homogeneous | heterogeneous continuous batching |
| SFT/DPO objective + update | ✅ | dataset/checkpoint/eval loop |
| Group-relative RL objective | ✅ primitive | verifier rollout + real tiny-policy update |
| Thread replay/fork/cancel | ✅ | active-turn crash recovery |
| Work lease/reclaim | ✅ | heartbeat / worker pool |
| Integrated Runtime | ✅ normal path | idempotent side-effect recovery |
| Context Provenance | ✅ | notes/index/retrieval provenance |
| Benchmark Harness | ✅ | benchmark metadata/versioning |
| Repository Final-state Grader | ✅ | hidden/multi-file fixtures → SWE |
| Permission Gate | ✅ | **OS/container sandbox** |
| Browser/Computer | ❌ | real environment + verifier |

---

# 9　接下来优先级

1. **Inference**：scheduler + batch executor 真正支持 variable-length request / block-table state；随后 prefix cache、chunked prefill、speculative decoding。
2. **Agent reliability**：为 side-effecting tool 加 durable execution record / idempotency key，再做 active-turn crash/recovery，而不是盲目 retry。
3. **Evaluation**：扩 repository fixture 到 multi-file hidden tests，形成第一个可重复 coding benchmark suite，再接 SWE-bench。
4. **Security**：实现可测试的 process/container sandbox，明确 filesystem/network/secret boundary 与 escape tests。
5. **Post-training**：把 group-relative objective 接到真实 tiny-model rollouts + verifier，形成 SFT→DPO→RLVR held-out 闭环。
6. 再推进 tree-sitter/LSP、parallel agents、A2A、Browser/Computer Use。

---

# 10　决策规则

```text
final-state / numerical / protocol evidence
> recovery & security invariants
> controlled benchmark / ablation
> unit tests
> Source-First interpretation
> Markdown count
```

这本教材真正达到 L3 的标志不是“解释得很全”，而是：**关键 claim 可以被运行、测量、反驳，并且失败边界是显式的。**
