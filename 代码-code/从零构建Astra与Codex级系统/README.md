# 从零构建 Astra-class 与 Codex-class 系统

这是整本教材的**终极 Reference System**：不是调用高层 Agent framework 拼一个 demo，而是从模型 forward、权重加载、推理缓存一路写到 durable thread、工具、协议、权限、评测、Coding Agent、Computer Use 与 Multi-Agent。

> **边界**：不要求自己训练 frontier 权重；可以用 tiny/open models 学机制、做 parity，也可以把 GPT/Claude/Gemini 等作为可插拔 Model Backend。我们构建的是 **Astra-class / Codex-class system runtime**，不声称复刻未公开模型权重或云端全部生产基础设施。

实时实现状态：[`实现状态-STATUS.md`](实现状态-STATUS.md)  
机器事实源：[`../../能力清单-CAPABILITIES.json`](../../能力清单-CAPABILITIES.json)  
v3 蓝图：[`../../教材-book/99-全书架构蓝图-v3.md`](../../教材-book/99-全书架构蓝图-v3.md)  
独立评测层：[`../../评测-eval/README.md`](../../评测-eval/README.md)

---

# 1　当前源码：仍然扁平，但 subsystem 已经开始成形

```text
src/astra_codex/
├── config.py             # 模型结构约束
├── tokenizer.py          # UTF-8 byte tokenizer + 从零 BPE
├── model.py              # RMSNorm / RoPE / GQA / SwiGLU / Transformer
├── cache.py              # per-layer contiguous KV Cache
├── sampling.py           # greedy / temperature / top-k / top-p
├── engine.py             # prefill / incremental decode / streaming
├── weights.py            # raw safetensors / key remap / shape audit
├── public_checkpoint.py  # 公开 Llama-family checkpoint adapter
│
├── agent.py              # 最小 observe → act → observe loop
├── codex_harness.py      # Codex-style Turn / Event / Approval teaching harness
├── durable.py            # [v3] event-sourced Thread / Turn / replay / resume
├── memory.py             # SQLite persistent event memory
├── planning.py           # typed task DAG
├── verification.py       # external verifier
├── evaluation.py         # [v3] trajectory capability metrics
├── security.py           # [v3] pre-dispatch permission / approval enforcement
│
├── structured.py         # JSON tool call / schema validation
├── tools.py              # filesystem / shell / Git
├── editing.py            # ambiguity-safe exact edit
├── repo_map.py           # Python AST + Markdown structure map
├── general_tools.py      # minimal HTTP text browser
├── mcp.py                # minimal MCP teaching subset
│
├── worktree.py           # Git worktree primitive
├── multi_agent.py        # coordinator primitive
└── coding.py             # repository coding-agent assembly
```

短期保持这些 import 稳定；等 subsystem contract 充分成熟后，再逐步迁移到 `model/`、`inference/`、`agent/`、`protocols/`、`security/`、`eval/` 等子包，避免为了目录漂亮而破坏已经通过的 parity/tests。

---

# 2　真实公开模型 parity：当前最重要的模型证据

真实 checkpoint：

```text
HuggingFaceTB/SmolLM2-135M
```

数据流：

```text
config.json + raw safetensors
        ↓
ModelConfig + key mapping
        ↓
our RMSNorm / RoPE / GQA / SwiGLU / Transformer
        ↓
our logits

same input
        ↓
HF reference eager implementation
        ↓
reference logits
```

当前受测 CPU float32 设置：

```json
{"max_abs": 0.0, "mean_abs": 0.0, "argmax_agreement": 1.0}
```

这只证明当前模型/输入/数值设置，不外推为“兼容所有模型族”。后续要做 architecture parity matrix。

---

# 3　v3 新增：Durable Thread / Event Store / Replay

`durable.py` 把 Agent 生命周期从“一个 Python 函数跑到底”升级为显式持久状态：

```text
THREAD_CREATED
→ USER_SUBMISSION
→ TURN_STARTED
→ CHECKPOINT_CREATED
→ TURN_COMPLETED
→ THREAD_PAUSED / THREAD_RESUMED
→ THREAD_COMPLETED / THREAD_FAILED
```

SQLite event log 是 source of truth，`project(thread_id)` 通过 replay 得到当前状态。因此：

```text
process A
→ running turn + checkpoint
→ close process

process B
→ reopen same DB
→ replay events
→ recover thread / active turn / checkpoint
```

这是 long-running Agent OS 的最小地基。它还不是 distributed scheduler、lease manager 或 durable queue。

---

# 4　v3 新增：Permission Enforcement

`security.py` 把：

```text
"模型被提示不要做危险动作"
```

和：

```text
"runtime 在执行前物理阻断动作"
```

分开。

当前数据流：

```text
Tool Proposal
    ↓
PermissionProfile
├─ ALLOW
├─ REQUIRE_APPROVAL
└─ DENY
    ↓
optional approval callback
    ↓
ToolRegistry dispatch
```

测试验证：DENY / approval reject 时底层 tool body 的计数保持 0，说明动作没有被 dispatch。

**但这仍不是 OS sandbox。** Bash 进程、filesystem namespace、network、secret、syscall isolation 仍是独立 P0。

---

# 5　v3 新增：Agent Trajectory Evaluation

`evaluation.py` 开始把 unit tests 与 capability evaluation 分开。

统一记录：

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

这只是 metrics layer；后续 `评测-eval/` 会继续加入 benchmark case format、graders 与 SWE-bench / Browser / OS adapters。

---

# 6　模型运行时数据流

```text
Text
 ↓
Tokenizer / Token IDs
 ↓
Embedding
 ↓
RMSNorm
 ↓
Q / K / V
 ↓
RoPE
 ↓
Grouped-Query Causal Attention
 ↓
Residual
 ↓
RMSNorm
 ↓
SwiGLU
 ↓
... N blocks
 ↓
Final RMSNorm
 ↓
LM Head
 ↓
Logits
```

自回归推理：

```text
Prompt
→ Prefill
→ contiguous KV Cache
→ one-token Decode
→ Sampling
→ repeat
```

当前下一道系统门槛不是“再加一种 sampling”，而是：

```text
contiguous KV
→ paged KV
→ prefix cache
→ scheduler
→ continuous batching
→ chunked/disaggregated prefill
→ speculative decoding
```

所有优化必须保留 reference implementation 和 parity test。

---

# 7　Agent Runtime v3 目标

最终核心对象：

```text
Submission Queue
      ↓
Thread
      ↓
Turn Executor
      ↓
Event Stream
      ↓
Event Store / Replay
      ↓
Context Builder
├─ raw history
├─ notes
├─ semantic index
├─ artifacts
└─ compaction provenance
      ↓
Planner / Policy
      ↓
Permission / Approval / Sandbox
      ↓
Tool / MCP / A2A / Remote Agent
      ↓
Observation
      ↓
Verifier
      ↓
Checkpoint / Continue / Finish
```

`codex_harness.py` 只负责其中的 **TurnExecutor teaching slice**，不再被当成整个 Agent OS。

---

# 8　Protocol 分层

不要把所有外部交互都叫“Agent API”。最终至少分：

```text
Tool Protocol
└─ MCP: Agent/Host ↔ Tool / Context Server

Agent Protocol
└─ A2A: Agent ↔ Agent

Application Runtime Protocol
└─ App-Server-like: UI / Client ↔ Agent Runtime
```

当前 `mcp.py` 是教学子集。下一步优先按 2026-07-28 语义审计 stateless core、discovery、cache/auth/tasks extensions，再实现 minimal A2A 1.0。

---

# 9　未来源码目录目标

不立即搬文件，但最终目标明确为：

```text
astra_codex/
├── model/
├── inference/
├── posttraining/
├── agent/
├── protocols/
├── environments/
├── security/
├── multi_agent/
└── eval/
```

迁移必须以稳定 API、全量 tests、compatibility update 为前提。

---

# 10　P0 / P1

## P0

```text
[x] Capability Manifest + CI validator
[x] Durable Thread / Event Store / Replay primitive
[x] Trajectory Metrics primitive
[x] Pre-dispatch Permission primitive
[ ] Enforced OS/container sandbox
[ ] Context provenance / notes / semantic index / compaction record
[ ] Paged KV reference + parity
[ ] request scheduler + serving metrics
```

## P1

```text
[ ] prefix cache / continuous batching / speculative decode
[ ] tiny SFT → DPO → verifier/RL pipeline
[ ] MCP 2026-07-28 fuller semantics + A2A 1.0 minimal runtime
[ ] tree-sitter / LSP / semantic patch
[ ] browser DOM/A11y + screenshot + actions + verifier
[ ] parallel worktree workers + reviewer/merge
[ ] real benchmark adapters
```

真实状态以 [`../../能力清单-CAPABILITIES.json`](../../能力清单-CAPABILITIES.json) 为准，而不是 README 里的手工勾选。

---

# 11　运行

基础测试：

```bash
cd '代码-code/从零构建Astra与Codex级系统'
pip install -e '.[dev]'
pytest
```

真实 checkpoint parity：

```bash
pip install -e '.[parity]'
python examples/02_smollm2_parity.py
```

教材内容 / manifest / frontier 审计由根仓库 GitHub Actions 执行。

---

# 12　最终毕业验收

## Model / Inference

公开模型 parity matrix + paged/state cache + scheduler + batching + speculation + profiling。

## Post-training

至少在 tiny/open model 上真正跑通 SFT、preference/DPO、verifier/reward 与小规模 RL update。

## Codex-class

真实 repository + issue：

```text
resume durable thread
→ read scoped instructions
→ inspect / localize
→ edit
→ targeted tests
→ diagnose / repair
→ broader verification
→ review diff
→ produce artifact + evidence
```

## Astra-class

跨 browser/computer/files/code 的长程任务，支持 crash/restart、context compaction、permission/sandbox、安全审计和 external verifier。

## Multi-Agent

不仅能 spawn，而能用 controlled benchmark 回答：

> 2/4/8 agents 相比 1 agent 到底提高了什么，又增加了多少成本、冲突与 merge failure？

**终点不是“代码很多”，而是每一层都能解释、运行、测试、对齐、恢复、评测并暴露失败。**
