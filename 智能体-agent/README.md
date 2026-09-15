# 智能体系统主线 / Agentic Systems Track v3

> **定位**：Agent 不是“LLM 后面接几个工具”。模型主线研究 `model(x) -> tokens`；智能体主线研究一个**可持久、可恢复、可验证、受权限约束的环境闭环系统**。

第一性闭环仍可写为：

$$
a_t\sim \pi_\theta(a_t\mid h_t,g),
$$

$$
o_{t+1}\sim P(o_{t+1}\mid s_t,a_t),
$$

$$
h_{t+1}=F(h_t,a_t,o_{t+1}).
$$

但工程上，$h_t$ 不能只是一串聊天消息。v3 明确把 Thread / Turn / Event、Context / Memory、Permission / Sandbox、Verifier / Evaluation 都变成一等对象。

---

# 1　课程 A1–A10

| 模块 | 主题 | 文件 |
|---|---|---|
| A1 | 智能体第一性原理 | [`01-智能体基础-agent-foundations.md`](01-智能体基础-agent-foundations.md) |
| A2 | 工具、动作空间与环境接口 | [`02-工具与环境-tool-use-environments.md`](02-工具与环境-tool-use-environments.md) |
| A3 | 规划、反思、验证与恢复 | [`03-规划反思与验证-planning-reflection-verification.md`](03-规划反思与验证-planning-reflection-verification.md) |
| A4 | 记忆、上下文与长期任务状态 | [`04-记忆与上下文-memory-context.md`](04-记忆与上下文-memory-context.md) |
| A5 | Coding Agent 与真实仓库闭环 | [`05-编程智能体-coding-agents.md`](05-编程智能体-coding-agents.md) |
| A6 | Multi-Agent、并行与协调 | [`06-多智能体与协调-multi-agent.md`](06-多智能体与协调-multi-agent.md) |
| A7 | Evaluation、Safety、Verification | [`07-评测安全与开放问题-agent-evaluation-safety.md`](07-评测安全与开放问题-agent-evaluation-safety.md) |
| A8 | Environment Learning / Agentic RL | [`08-环境学习与AgenticRL-agentic-rl.md`](08-环境学习与AgenticRL-agentic-rl.md) |
| A9 | MCP、A2A 与 Agent Protocols | [`09-Agent协议与互操作-agent-protocols.md`](09-Agent协议与互操作-agent-protocols.md) |
| A10 | Browser / Computer Use | [`10-浏览器与计算机使用-browser-computer-use.md`](10-浏览器与计算机使用-browser-computer-use.md) |

一手资料：[`../参考文献-references/01-智能体原始资料-agent-sources.md`](../参考文献-references/01-智能体原始资料-agent-sources.md)  
工业源码支线：[`../Codex源码解剖-codex-anatomy/README.md`](../Codex源码解剖-codex-anatomy/README.md)  
Reference System：[`../代码-code/从零构建Astra与Codex级系统/`](../代码-code/从零构建Astra与Codex级系统/)  
独立评测层：[`../评测-eval/README.md`](../评测-eval/README.md)

---

# 2　v3 Agent Kernel

最小 Agent：

```text
Model
→ Tool Call
→ Tool Result
→ Model
```

只是教学 baseline。目标 runtime 是：

```text
Submission
      ↓
Durable Thread
      ↓
Turn Executor
      ↓
Event Stream
      ↓
Event Store / Replay
      ↓
Context Builder
├─ raw history
├─ persistent notes
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
Environment Observation
      ↓
Verifier / Grader
      ↓
Checkpoint / Replan / Continue / Finish
```

这就是为什么 Agent architecture 不能被简化成 prompt template。

---

# 3　当前真正已经写出来的 Runtime

```text
agent.py          最小 observe → act → observe baseline
codex_harness.py  Codex-style Turn/Tool/Approval/Event executor
durable.py        event-sourced Thread/Turn/Event + replay/resume
planning.py       typed task DAG
verification.py   external verifier
memory.py         persistent event memory
security.py       pre-dispatch permission/approval gate
evaluation.py     trajectory metrics
structured.py     JSON action/schema
tools.py          filesystem/shell/Git
editing.py        ambiguity-safe edit
repo_map.py       AST/Markdown structure map
coding.py         coding-agent assembly
worktree.py       Git worktree primitive
multi_agent.py    coordinator primitive
mcp.py            minimal MCP teaching runtime
general_tools.py  HTTP text fetch only
```

最新 Fast CI：

```text
36 passed, 1 warning in 2.14s
Ruff correctness lint: All checks passed
```

其中 v3 新测试已经覆盖：进程式 reopen 后 durable state replay、invalid thread transitions、permission deny/reject 不 dispatch 工具、trajectory metrics / aggregation。

---

# 4　Long Context 不等于 Memory

最终长期任务状态至少分成：

```text
Raw Event Log
Episodic Store
Semantic Index
Persistent Notes
Working Context
Artifact Store
Compaction Record
Retrieval Provenance
```

当前 `memory.py` 与 `durable.py` 只覆盖前两类基础设施的一部分。后续任何 summary/context fragment 都要能回答：

> 它来自哪个原始 event / file / tool observation？是否经过压缩？压缩版本是什么？原始证据还能不能追回？

---

# 5　Security 是 Agent Runtime，不是“安全章节”

当前已经落地第一层：

```text
Action Proposal
→ PermissionProfile
├─ ALLOW
├─ REQUIRE_APPROVAL
└─ DENY
→ optional approval
→ dispatch
```

测试证明 deny / approval reject 时底层 tool body 不执行。

但真实边界还需要：

```text
Credential Scope
→ OS / Container Sandbox
→ Filesystem Policy
→ Network Policy
→ Execution
→ Audit Event
→ Monitor / Verifier
```

所以 `SandboxPolicy` enum、system prompt 或“请勿执行危险动作”都不能被写成已经实现安全隔离。

---

# 6　Protocol 必须分三层

```text
Tool Protocol
└─ MCP: Host/Agent ↔ Tool / Context Server

Agent Protocol
└─ A2A: Agent ↔ Agent

Application Runtime Protocol
└─ UI / Client ↔ Durable Agent Runtime
```

当前 `mcp.py` 是 minimal teaching subset；下一步按 2026-07-28 MCP 语义继续补 stateless lifecycle、transport/auth/tasks/extensions，并实现 minimal A2A 1.0。

---

# 7　Multi-Agent 必须可证伪

“能 spawn 4 个 Agent”不是科研结论。最终固定做：

```text
1 agent
2 agents
4 agents
8 agents
```

并统一比较：

```text
success
wall time
model steps
tool calls
tokens / cost
duplicate work
worktree conflicts
merge failures
human approvals
```

只有在任务、budget、model、grader 相同的条件下，多 Agent 的收益才有意义。

---

# 8　Computer Use 的真正闭环

当前只有 HTTP text fetch，不能称为 Computer Use。

真正目标：

```text
Observation
├─ DOM / Accessibility Tree
├─ Screenshot
├─ Text / File / Structured Data
└─ Environment State
        ↓
Grounding
        ↓
Mouse / Keyboard / Browser Action
        ↓
Environment Transition
        ↓
State Verifier
```

进一步再扩展到 image/audio/video 等 typed multimodal observations。

---

# 9　Codex Source Anatomy 的正确位置

[`../Codex源码解剖-codex-anatomy/`](../Codex源码解剖-codex-anatomy/) 现在被定义成：

> **Industrial Reference Implementation #1：Coding Agent Runtime**

它给出极高价值的官方实现证据，但不是 Agent 世界唯一架构。以后 inference runtime、model runtime、protocol runtime 也会建立自己的工业/官方 Reference Anatomy。

---

# 10　Agent 毕业标准

不依赖 LangChain 等高层 Agent framework，能够从零实现并解释：

```text
Durable Thread / Turn / Event / Replay
→ Context / Memory / Provenance
→ Planning / Verification / Recovery
→ Permission / Approval / Enforced Sandbox
→ MCP / A2A / Runtime Protocol
→ Coding Agent
→ Browser / Computer Agent
→ AgentGraph / Parallel Workers / Worktree / Reviewer
→ Evaluation / Grader / Cost Accounting
→ Agentic RL Trajectory Pipeline
```

而且系统崩溃、tool 失败、权限拒绝、context compaction、subagent 冲突时，都能通过 event/evidence 明确解释发生了什么。
