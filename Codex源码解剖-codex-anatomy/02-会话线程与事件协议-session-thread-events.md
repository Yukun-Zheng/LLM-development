# C02　Session、Thread、Event 与 App Server：把 Agent Core 与 UI 解耦

> **Primary sources**  
> - Protocol mental model: https://github.com/openai/codex/blob/main/codex-rs/docs/protocol_v1.md  
> - App Server protocol: https://github.com/openai/codex/tree/main/codex-rs/app-server-protocol  
> - Thread settings schema: https://github.com/openai/codex/blob/main/codex-rs/app-server-protocol/schema/typescript/v2/ThreadSettings.ts  
> - Turn start schema: https://github.com/openai/codex/blob/main/codex-rs/app-server-protocol/schema/typescript/v2/TurnStartParams.ts

---

# 1　为什么工业 Agent 必须把 Core 和 UI 分开

一个 coding agent 可能被：

```text
CLI
TUI
IDE extension
Desktop app
remote client
CI / non-interactive exec
```

共同驱动。

如果 agent loop 和某个 terminal UI 写死在一起，就无法形成稳定系统边界。

Codex 官方协议文档明确把 UI 视为 Codex core 外部实体；core 通过 Submission / Event 交换信息。

因此我们的目标不是：

```text
print("thinking...")
input("approve?")
```

而是：

```text
UI / Client
   ↓ Submission
Core State Machine
   ↓ Event
UI / Client
```

---

# 2　Submission / Event 是两个方向

官方 mental model：

```text
SQ: UI → Codex
EQ: Codex → UI
```

典型 Submission：

```text
UserTurn
Interrupt
ExecApproval
UserInputAnswer
```

典型 Event：

```text
TurnStarted
AgentMessage
AgentMessageContentDelta
PlanDelta
ExecApprovalRequest
RequestUserInput
Warning
Error
TurnComplete
```

这比“函数返回一个字符串”强得多，因为真实任务是 streaming、可中断、可审批的。

---

# 3　Thread Settings 暴露了哪些 runtime concern

当前 App Server protocol 的 `ThreadSettings` schema 直接包含：

```text
cwd
approvalPolicy
approvalsReviewer
sandboxPolicy
activePermissionProfile
model
modelProvider
serviceTier
effort
summary
collaborationMode
personality
```

这说明这些都是**runtime configuration**，不应该被藏进一句 system prompt。

尤其：

```text
approvalPolicy != sandboxPolicy
```

前者决定“什么时候要问”；后者决定“即使不问，OS/runtime 实际允许做什么”。

---

# 4　Per-thread 与 Per-turn 设置

`TurnStartParams` 允许一些 turn-level override，例如 approval / sandbox / model 等。

因此更准确的状态结构是：

```text
Thread defaults
      ↓
Turn overrides
      ↓
Resolved Turn Settings
      ↓
Step Context
```

而不是全局只有一个 mutable config。

这对多任务和多 agent 特别重要。

---

# 5　为什么需要 response bookmark / resume / fork

官方协议文档说明每个完成的 turn 会保存 model `response_id`；它既可用于之后继续，也能支持从较早节点 fork。

抽象来看：

```text
Thread history
 ├─ response A
 ├─ response B   ← bookmark
 └─ response C
```

可以形成：

```text
original branch: A → B → C
fork branch:     A → B → D
```

因此 conversation history 不只是 append-only 聊天文本，还会成为**可恢复执行图**的一部分。

后续我们自己的 runtime 应显式加入：

```text
ThreadId
TurnId
ResponseBookmark
ParentTurnId
ForkOrigin
```

---

# 6　Streaming event 的价值

一个长 coding turn 可能经历：

```text
reasoning delta
assistant text delta
tool call start
approval request
exec output
patch result
plan delta
warning
final answer
```

如果只等一个最终 JSON：

```json
{"answer":"done"}
```

就失去了：

- 用户中途 cancel 的能力；
- UI progress；
- approval handshake；
- tracing；
- live tool output；
- long-running task observability。

所以 event stream 是 Agent runtime 的一级接口。

---

# 7　App Server 的正确定位

App Server 不应该重新实现 Agent intelligence。

合理层次：

```text
Client UI
   ↓
App Server protocol
   ↓
Thread Manager
   ↓
Codex Core
   ↓
Model / Tools / Environment
```

也就是说 App Server 是：

> **稳定的外部控制面，而不是第二套 agent loop。**

这也是我们最终让 Web UI、桌面 UI、IDE 插件共享一个 runtime 的基础。

---

# 8　我们自己的 clean-room 目标

当前 `codex_harness.py` 已有 `HarnessEvent`，但仍然只是内存数组。

后续演化：

```text
v1  list[HarnessEvent]
v2  async event iterator
v3  submission queue + event queue
v4  persistent ThreadStore
v5  JSON-RPC / app-server adapter
v6  resume / fork
```

关键要求：core 不依赖 TUI。

---

# 9　最低验收实验

### E1 Event ordering

验证一次 tool turn：

```text
TurnStarted
→ ModelOutput
→ ToolStarted
→ ToolCompleted
→ ModelOutput
→ TurnCompleted
```

### E2 Interrupt

在 tool execution 或 sampling 时发送 interrupt，验证 active task 进入明确终止状态，并能启动下一 turn。

### E3 Resume

把 thread state 持久化后重启进程，再从相同 bookmark 继续。

### E4 Fork

同一个 bookmark 启动两个后续输入，验证状态不串线。

---

# 10　一个重要工程结论

当 Agent 进入长期任务后，“聊天记录”这个词已经太弱。

真正需要的是：

```text
Thread state
+ Turn state
+ Event log
+ Execution evidence
+ Context snapshot
+ Permission snapshot
+ Resume/Fork metadata
```

这才是 Codex-class runtime 的持久化对象。
