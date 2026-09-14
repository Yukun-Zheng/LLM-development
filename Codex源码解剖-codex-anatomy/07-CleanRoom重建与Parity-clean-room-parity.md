# C07　Clean-room 重建与 Parity：怎样真正“从零搭一个 Codex”

> **目标**：不是复制 `openai/codex`，而是从公开源码提取行为契约，独立写出教学实现，并逐层做可复现 parity。  
> **Primary source**：OpenAI 官方仓库 https://github.com/openai/codex ，重点对照 https://github.com/openai/codex/blob/main/codex-rs/core/src/session/turn.rs 与 https://github.com/openai/codex/blob/main/codex-rs/docs/protocol_v1.md 。

---

# 1　为什么要 Clean-room

Apache-2.0 允许合法阅读、修改和再分发官方代码，但本教材的教育目标不是最大化代码复用，而是最大化理解。

因此固定流程：

```text
Official source
    ↓
Behavior contract / state diagram
    ↓
关闭官方实现细节，独立设计最小接口
    ↓
Clean-room implementation
    ↓
Unit test
    ↓
Behavior / protocol parity test
    ↓
Gap report
```

如果只是复制文件再改变量名，就无法证明自己理解了 architecture。

---

# 2　Parity 分七层

## P1　State-machine parity

比较：

```text
user input
→ model call
→ tool call
→ tool observation
→ follow-up
→ completion
```

不同分支是否一致。

## P2　Event parity

比较关键事件序列：

```text
turn started
approval request
exec start
exec end
message delta
turn complete
error
```

## P3　Protocol parity

对 App Server / MCP 一类协议，比较：

```text
request schema
response schema
stream ordering
error mapping
cancellation
```

## P4　Environment parity

比较：

```text
cwd
filesystem view
sandbox
network
process environment
worktree
```

## P5　Persistence parity

比较：

```text
resume
fork
thread state
compact window
agent graph
```

## P6　Coding behavior parity

同一个 deterministic backend / fixture 下比较：

```text
files read
commands executed
patches applied
tests run
verification evidence
```

## P7　End-to-end task parity

最后才比较真实 issue solving / SWE-bench 一类任务。

---

# 3　不要一上来比较模型自然语言

如果官方 Codex 和我们的 harness 用不同模型，那么：

```text
final prose difference
```

无法告诉你 harness 是否错。

早期 parity 应固定 policy：

```text
Scripted Model
or
Replay Model
```

让两个 runtime 收到完全相同的 model outputs，然后比较：

```text
state transition
execution
observation
event stream
```

这样才能隔离 harness 差异。

---

# 4　Replay Harness

后续要建立一种 fixture：

```json
{
  "turn": 1,
  "model_output": {
    "tool": "shell",
    "arguments": {"command": "pytest -q"}
  }
}
```

runtime 执行后记录：

```json
{
  "tool_result": {
    "ok": true,
    "returncode": 0
  },
  "next_event": "tool_completed"
}
```

然后继续 replay 下一步。

它能让 Agent runtime 像 compiler/runtime 一样做 regression test。

---

# 5　当前 MiniCodex v1

入口：

`代码-code/从零构建Astra与Codex级系统/src/astra_codex/codex_harness.py`

已经有：

```text
TurnSettings
SandboxPolicy metadata
ApprovalDecision
HarnessEvent
Turn started/completed/stopped
Model output
Approval requested/decided
Tool started/completed
Tool observation follow-up
Explicit step limit
```

自动测试：

`tests/test_codex_harness.py`

当前测试覆盖：

```text
tool → observation → follow-up → final
approval deny → tool not executed
approval allow → tool executed
step-limit termination
```

---

# 6　MiniCodex v2–v8 路线

## v2 Event-driven thread

```text
async SubmissionQueue
async EventQueue
interrupt
pending input
```

## v3 Durable thread

```text
ThreadId
TurnId
SQLite ThreadStore
resume
fork
```

## v4 Context system

```text
scoped AGENTS.md
ContextFragment provenance
budget
compaction
checkpoint
```

## v5 Security boundary

```text
PermissionProfile
async approval broker
enforced process sandbox
secret isolation
network policy
```

## v6 Protocol layer

```text
MCP stdio / HTTP
App Server JSON-RPC
streaming events
cancellation
```

## v7 Coding intelligence

```text
tree-sitter
LSP
semantic patch
test selection
Git evidence
```

## v8 Multi-Agent

```text
AgentGraph
mailbox
spawn/followup/message/wait/interrupt
worktree isolation
reviewer / merge
```

---

# 7　Source-to-Implementation Matrix

每个模块都维护：

| Official source | Extracted contract | Our module | Test | Status |
|---|---|---|---|---|
| `core/src/session/turn.rs` | tool output feeds next sampling | `codex_harness.py` | `test_codex_harness.py` | v1 ✓ |
| `docs/protocol_v1.md` | Task/Turn/Event model | `codex_harness.py` | event ordering | partial |
| `app-server-protocol` | external thread/turn API | — | — | todo |
| `tools/approvals.rs` | approval separated from action | `codex_harness.py` | denial/allow | partial |
| `tools/sandboxing.rs` | execution policy boundary | enum only | — | not enforced |
| `compact*` | compact/resume within long task | — | — | todo |
| `multi_agents.rs` | spawn/message/followup semantics | `multi_agent.py` | coordinator tests | partial |
| `worktree/` | isolated repo working copies | `worktree.py` | primitive tests | partial |

“partial” 必须保留，不能因为存在同名 class 就标成完成。

---

# 8　最终的 Codex Graduation Test

给一个新 Git repository + issue，系统必须能够：

```text
read scoped instructions
→ inspect repo
→ build repo map
→ create plan
→ choose isolated worktree when needed
→ edit code
→ run targeted tests
→ observe failure
→ repair
→ run broader verification
→ produce diff + evidence
→ request approval only where policy requires
→ survive context compaction
→ resume after process restart
→ hand off to reviewer
→ merge only after verifier pass
```

而且所有关键动作都能从 event log 重建。

这才是“搭出来 Codex”在系统工程意义上的验收，而不是做一个会输出 shell JSON 的聊天机器人。

---

# 9　再向 Astra-class 扩展

Codex source anatomy 给我们的是非常强的软件工程 agent reference。

更一般的 Astra-class 系统还要进一步把 environment 扩成：

```text
Code Repository
+ Browser
+ GUI Computer
+ Documents
+ External Apps
+ Long-running Cloud Tasks
+ Multimodal Observation
```

但 Agent OS 的大量基础设施：

```text
thread
turn
event
approval
sandbox
context
memory
tool protocol
multi-agent
verification
```

可以共享。

因此正确路线是：

```text
Codex source anatomy
→ MiniCodex parity
→ robust Agent Runtime
→ add general environments
→ Astra-class system
```
