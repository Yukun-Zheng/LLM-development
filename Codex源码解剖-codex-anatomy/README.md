# OpenAI Codex 源码解剖 / Codex Source Anatomy

> **研究对象**：OpenAI 官方开源仓库 [`openai/codex`](https://github.com/openai/codex)。  
> **许可**：官方仓库 README 标注 Apache-2.0。  
> **本轮核验快照**：`d77ebc72237a639b6d877f2edc3b20b54631f25e`（2026-09-14）。  
> **边界**：这里研究的是 **Codex CLI / agent harness / runtime 的公开源码**，不是未公开的 frontier 模型权重、训练 recipe 或 Codex 云端全部基础设施。

---

# 1　为什么单独建立这条主线

此前教材把终点表述为“从零构建 Codex-class 编程智能体”。现在官方 harness 已经开源，学习方法应升级成：

```text
OpenAI 官方 Codex 源码
        ↓
Source-First 逐模块解剖
        ↓
提取可验证的软件架构契约
        ↓
在本仓库 clean-room 重写最小版本
        ↓
行为 / 协议 / 状态机 parity
        ↓
再扩展到自己的 Astra-class general agent
```

这比根据产品表面行为猜一个 Agent 架构严格得多。

我们不会照抄整个工业仓库，也不会把 Apache-2.0 “允许阅读/复用”误解成“复制越多越好”。教学目标是理解：

- 哪些状态必须显式存在；
- 模型、工具、环境与 UI 怎么解耦；
- approval 和 sandbox 为什么不是 prompt；
- tool output 怎样重新进入下一次 sampling；
- context compaction 怎样成为 runtime concern；
- multi-agent 为什么首先是 thread / message / concurrency / filesystem semantics 问题；
- App Server / protocol 怎样让 CLI、IDE、GUI 共享核心引擎。

---

# 2　官方源码给出的第一性架构

官方 [`codex-rs/docs/protocol_v1.md`](https://github.com/openai/codex/blob/main/codex-rs/docs/protocol_v1.md) 明确区分：

```text
Model
  ↑↓
Turn
  ↑↓
Task
  ↑↓
Session
  ↑↓
Codex Core
  ↑↓
Submission Queue / Event Queue
  ↑↓
CLI / TUI / IDE / arbitrary UI
```

一个 Task 由多次 Turn 构成；一次 Turn 大体是：

```text
model sampling
    ↓
assistant message ? ───────────────→ complete
    │
    └─ function/tool call
             ↓
        approval gate
             ↓
        tool execution
             ↓
         observation
             ↓
      next model sampling
```

当前官方核心实现可直接从：

[`codex-rs/core/src/session/turn.rs`](https://github.com/openai/codex/blob/main/codex-rs/core/src/session/turn.rs)

阅读。其 `run_turn` 注释直接说明：模型如果请求 function call，就执行并把结果送入下一次 sampling；若只返回 assistant message，则记录历史并结束 turn。

---

# 3　官方仓库不是“一个 agent.py”

当前 Rust workspace 已经拆成大量 crate。对教材最重要的族包括：

```text
codex-rs/
├─ core/                     agent/session/turn/tool orchestration
├─ protocol/                 core protocol types
├─ app-server/               UI-neutral app server
├─ app-server-protocol/      generated / stable-ish client protocol surface
├─ cli/ + tui/               local user interfaces
├─ exec/ + shell-command/    non-interactive execution
├─ apply-patch/              patch application
├─ sandboxing/               common sandbox abstractions
├─ linux-sandbox/            Linux enforcement
├─ windows-sandbox-service/  Windows enforcement
├─ codex-mcp/                MCP integration
├─ rmcp-client/              MCP client infrastructure
├─ worktree/                 Git worktree support
├─ agent-graph-store/        agent graph persistence
├─ agent-identity/           agent identity
├─ agent-roles/              agent roles
├─ thread-store/             thread persistence
├─ rollout/                  rollout/session artifacts
├─ rollout-trace/            execution/interaction trace
├─ state/                    state services
├─ context-fragments/        context assembly
├─ memories/                 memory read/write
├─ ext/mcp/                  MCP extension layer
├─ ext/memories/             memory extension
├─ ext/guardian-*            review / guardian extensions
├─ ext/web-search/           web-search extension
├─ code-mode*                code-mode runtime/protocol/host
└─ models-manager/           model catalog/messages/config
```

因此“重写 Codex”不能被简化成一个 ReAct while-loop。

---

# 4　课程目录

| 编号 | 主题 | 文件 |
|---|---|---|
| C00 | **官方仓库地图与源码阅读方法** | [`00-官方仓库地图-repository-map.md`](00-官方仓库地图-repository-map.md) |
| C01 | **Agent Loop：Task / Turn / Sampling / Tool Follow-up** | [`01-核心AgentLoop-agent-loop.md`](01-核心AgentLoop-agent-loop.md) |
| C02 | **Session、Thread、Event 与 App Server 协议** | [`02-会话线程与事件协议-session-thread-events.md`](02-会话线程与事件协议-session-thread-events.md) |
| C03 | **Tools、Exec、Patch、Approval 与 Sandbox** | [`03-工具执行审批与沙箱-tools-approval-sandbox.md`](03-工具执行审批与沙箱-tools-approval-sandbox.md) |
| C04 | **Prompt、AGENTS.md、Context、Compaction 与 Memory** | [`04-指令上下文压缩与记忆-context-memory.md`](04-指令上下文压缩与记忆-context-memory.md) |
| C05 | **MCP、App Server 与外部能力互操作** | [`05-MCP与AppServer-mcp-app-server.md`](05-MCP与AppServer-mcp-app-server.md) |
| C06 | **Multi-Agent、Agent Graph、Worktree 与协作语义** | [`06-多智能体与工作树-multi-agent-worktree.md`](06-多智能体与工作树-multi-agent-worktree.md) |
| C07 | **Clean-room MiniCodex 与 parity 路线** | [`07-CleanRoom重建与Parity-clean-room-parity.md`](07-CleanRoom重建与Parity-clean-room-parity.md) |

后续继续扩展：

```text
C08  Code Mode / V8 Runtime
C09  Hooks / Skills / Plugins / Connectors
C10  Guardian / Review / Approval Reviewer
C11  Thread Store / Rollout / Rollout Trace
C12  TUI / IDE / App integration
C13  Models Manager / Prompt Catalog
C14  Testing architecture
C15  性能、并发与故障恢复
C16  从 Codex harness 扩展到 Astra-class general agent
```

---

# 5　本仓库已经开始 clean-room 重建

代码入口：

[`../代码-code/从零构建Astra与Codex级系统/src/astra_codex/codex_harness.py`](../代码-code/从零构建Astra与Codex级系统/src/astra_codex/codex_harness.py)

当前第一版只复现最核心的软件契约：

```text
TurnSettings
→ TURN_STARTED event
→ model sampling
→ tool call
→ optional approval
→ TOOL_STARTED / TOOL_COMPLETED
→ observation enters conversation
→ follow-up sampling
→ TURN_COMPLETED / TURN_STOPPED
```

它**不是** OpenAI Codex API-compatible clone，也没有假装已经拥有真实 OS sandbox。真实 security boundary 会在后续独立模块实现。

测试入口：

[`../代码-code/从零构建Astra与Codex级系统/tests/test_codex_harness.py`](../代码-code/从零构建Astra与Codex级系统/tests/test_codex_harness.py)

---

# 6　源码解剖的证据等级

每个结论标注四种来源：

```text
O1  OpenAI repo source code
O2  OpenAI repo docs / generated protocol schema
O3  OpenAI official product / developer documentation
R1  我们自己的 clean-room reproduction / parity test
```

不要把：

```text
“Codex 当前公开源码这样实现”
```

偷换成：

```text
“所有 OpenAI 内部 Codex / Astra 系统一定都这样实现”。
```

公开仓库是极高价值的一手资料，但仍然只是可观察边界的一部分。

---

# 7　最终毕业标准

学完这条线，应能从空目录写出：

```text
Thread / Session State
→ Turn State Machine
→ Model Client
→ Event Stream
→ Tool Registry / Router
→ Exec / Patch
→ Approval Policy
→ Enforced Sandbox
→ Context Assembly
→ Compaction
→ MCP
→ Persistent Thread Store
→ Multi-Agent Graph
→ Worktree / Review / Merge
→ CLI / IDE-neutral App Server
```

然后能够把我们自己的实现与官方 Codex 在**公开可验证行为**上逐项比较，而不是只说“长得像”。
