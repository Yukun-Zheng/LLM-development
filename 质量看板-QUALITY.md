# 全书质量看板 / Quality Dashboard v3

> **目的**：把“还差什么”变成可审计问题。  
> v3 不再主要统计 Markdown 数量；机器可读 capability 状态以 [`能力清单-CAPABILITIES.json`](能力清单-CAPABILITIES.json) 为准。

成熟度：L0=问题/边界/来源；L1=教材机制完整；L2=代码+测试/parity；L3=真实 benchmark、消融、反例与研究证据。

---

# 1　v3 总体评价

| 技术层 | 当前状态 | 最强证据 | 最大缺口 |
|---|---|---|---|
| Model Science | **L1–L2，覆盖强** | Transformer + SmolLM2 real checkpoint parity | 更多 architecture parity；实验化 SSM/MoE/Diffusion |
| Inference Systems | **L2 reference layer 起步** | contiguous KV + reference paged KV parity | production allocator、scheduler、batching、speculation、serving metrics |
| Post-training | **L1，明显短板** | 理论 / 数学 / verifier primitive | SFT → DPO → RLVR 可运行闭环 |
| Agent Kernel | **L2 持续推进** | Codex turn + durable event store + leased work queue | fork/cancel/steering、App Server、worker pool |
| Context / Memory | **L2 reference layer** | persistent event memory + provenance/compaction lineage | notes、semantic index、artifact provenance、token-budget builder |
| Protocols | **L2 subset** | minimal MCP tests | MCP fuller semantics + A2A 1.0 |
| Security | **L1–L2** | pre-dispatch deny/approval enforcement | **真实 OS/container sandbox、credentials、network** |
| Multi-Agent | **L2 primitive** | task DAG + coordinator + worktree | parallel scheduler、mailbox、review/merge、controlled eval |
| Evaluation | **L1–L2 起步** | trajectory metric schema + tests | benchmark cases + SWE/browser/OS graders |
| Browser / Computer | **L1** | HTTP text fetch only | DOM/A11y、screenshot、grounding、actions、verifier |
| Observatory | **L1** | machine-readable frontier snapshot + drift audit | 自动生成前沿表、版本历史与 revalidation workflow |

---

# 2　当前已经真正跨过的门槛

## 2.1　真实公开 checkpoint parity

`HuggingFaceTB/SmolLM2-135M` raw safetensors → 本项目自己的 Transformer runtime，在固定受测 CPU float32 / HF eager setting 下：

```json
{"max_abs": 0.0, "mean_abs": 0.0, "argmax_agreement": 1.0}
```

## 2.2　Contiguous KV → Reference Paged KV

现在不只解释 PagedAttention，而是已有 `paged_cache.py`：固定容量逻辑页、per-layer page table、incremental suffix ingest，并通过 full-forward 对照验证 prefill / decode logits parity。

当前实现仍会重建 contiguous `past_key_values`，因此它是**语义 reference**，不是生产级 zero-copy allocator。

## 2.3　Agent turn → Durable runtime primitives

`codex_harness.py` 负责 Turn Executor；`durable.py` 负责 event-sourced Thread/Turn replay；`runtime_queue.py` 新增 lease/reclaim/cancel work semantics。进程重启恢复与 worker ownership 开始分层，而不是继续塞进一个 while-loop。

## 2.4　Memory → Context Provenance

`context.py` 把 RAW_EVENT / NOTE / SUMMARY / ARTIFACT / RETRIEVAL / INSTRUCTION 变成 typed fragment。Compaction 创建新 summary 并保留 parent lineage，不覆盖 raw evidence。

## 2.5　Permission 开始成为执行边界

`security.py` 在 dispatch 前执行 ALLOW / REQUIRE_APPROVAL / DENY；deny/reject 时底层 tool body 不执行。仍不等于 OS/container sandbox。

## 2.6　Evaluation 与 pytest 开始分层

`evaluation.py` 统一 success、verifier、steps、tool calls、failures、approvals、wall time、tokens、cost；真实 benchmark adapters 仍待实现。

## 2.7　最新代码证据

Fast CPU CI run 42：

```text
40 passed, 1 warning in 2.46s
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
| RLHF / DPO | **L1** | math/source | tiny SFT/preference pipeline；DPO loss parity |
| Reasoning / RLVR | **L1** | theory + verifier primitive | trajectory/reward schema + actual policy update |
| Inference Systems | **L2 reference** | contiguous KV + paged KV semantics/parity | allocator → prefix → scheduler → batching → speculation |
| Hardware / Kernels | L1 | theory/source | roofline / bandwidth / SDPA kernel lab |
| Hybrid Sequence Architectures | L1 | S4/Mamba/RWKV/xLSTM/Titans theory | reference SSM / linear-attention code + parity |
| Diffusion LM | L1 | objective/mechanism | tiny corruption-denoise-remask runtime |
| Interpretability | L1 | source/mechanism | activation patch / SAE experiments |
| Multimodal Model Runtime | L0–L1 | historical/theory | vision projector / typed multimodal inputs |

---

# 4　Agent / Runtime 主线

| Capability | 状态 | 当前证据 | 关键缺口 |
|---|---|---|---|
| Codex-style Turn Executor | validated | behavioral tests | external event stream / steering integration |
| Durable Thread/Event Store | **validated** | replay-after-reopen tests | fork / thread cancellation / protocol parity |
| Durable Work Queue | **validated** | lease exclusion / expiry reclaim / cancel tests | heartbeat、fairness、async worker pool |
| Planning DAG | validated | cycle/failure propagation tests | dynamic replan + scheduler integration |
| Verifier | validated | command/file/composite tests | artifact/benchmark graders |
| Persistent Event Memory | validated primitive | SQLite tests | semantic index / artifact retrieval |
| Context Provenance | **validated** | summary lineage keeps raw evidence | token budget / notes / retrieval provenance |
| Coding Agent | L2 primitive | repo tools/edit/test/Git | tree-sitter/LSP/semantic patch/SWE-bench |
| MCP | L2 subset | discover/list/call tests | transports / auth / tasks/extensions |
| A2A | planned | theory/spec | AgentCard / Message / Task / Artifact runtime |
| Multi-Agent | L2 primitive | coordinator/worktree | parallelism + mailbox + reviewer + merge |
| Browser | partial | HTTP text only | JS/DOM/A11y/screenshot/actions |
| Computer Use | planned | theory | real GUI loop + state verifier |

---

# 5　Security：从“安全章节”转成 Runtime P0

当前已完成第一层：

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

在 shell / browser / computer use 扩展前，**enforced sandbox 是 P0**。

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

未来必须能做：memory vs no-memory、planner vs reactive、compaction A vs B、1/2/4/8 agents、reviewer vs no-reviewer、DOM vs screenshot、paged vs contiguous KV、speculative vs normal decode。

---

# 7　v3 P0

| P0 | 当前 | 验收标准 |
|---|---|---|
| Capability Manifest | **已落地** | CI 校验 theory/code/test/evidence |
| Durable Thread / Event Replay | **已落地并测试** | 进程关闭后 replay 恢复状态 |
| Durable Work Queue / Lease | **已落地并测试** | exclusive lease + expiry reclaim + cancellation |
| Context Provenance | **已落地并测试** | summary/compaction 可追回 raw evidence |
| Trajectory Metrics | **已落地并测试** | success/steps/tools/time/cost schema |
| Permission Gate | **已落地并测试** | deny 时底层 tool body 不执行 |
| Reference Paged KV | **已落地并 parity** | paged path 与 full forward logits 对齐 |
| **Enforced Sandbox** | 未完成 | process/filesystem/network/secret isolation + escape tests |
| **Request Scheduler** | 未完成 | continuous batching + TTFT/TPOT/throughput metrics |
| **Benchmark Case/Grader Harness** | 未完成 | deterministic case + artifacts + verifier + metrics |

---

# 8　接下来的优先顺序

1. **Request Scheduler + Continuous Batching reference**：先正确性，再 TTFT/TPOT/throughput。
2. **Benchmark Case / Grader Harness**：让 Agent 能力变化进入统一实验。
3. **Thread Fork / Steering / Cancellation integration**：把 durable primitives 连成真正 runtime。
4. **OS/container sandbox**：在 Browser/Computer Use 之前建立强边界。
5. **Tiny SFT → DPO → RLVR**：补当前最明显的 Post-training 实现缺口。
6. Prefix Cache / Speculative Decoding；MCP fuller semantics / A2A；parallel workers；Browser/Computer Use。

---

# 9　决策原则

```text
真实 parity / real environment evidence
> recovery / safety invariants
> benchmark / controlled ablation
> automated tests
> Source-First interpretation
> new Markdown count
```

项目质量最终看：**原始 claim 能否追到形式化，形式化能否追到代码，代码能否追到测试，系统能力能否追到真实评测，失败边界能否被明确观测。**
