# C00　OpenAI Codex 官方仓库地图：先看系统边界，再看单个函数

> **Primary source snapshot**：[`openai/codex@d77ebc72237a639b6d877f2edc3b20b54631f25e`](https://github.com/openai/codex/tree/d77ebc72237a639b6d877f2edc3b20b54631f25e)  
> **官方 README**：https://github.com/openai/codex/blob/main/README.md  
> **Rust workspace**：https://github.com/openai/codex/blob/main/codex-rs/Cargo.toml

---

# 1　先纠正一个常见误解

`openai/codex` 开源的是一个真实的**本地 coding-agent harness / CLI / runtime 工程**，并不是“Codex 模型权重仓库”。

因此源码阅读应分层：

```text
Model capability          ← 仍主要来自远端 / 本地模型 provider
Agent harness             ← openai/codex 可直接研究
Execution environment     ← openai/codex 可直接研究大量实现
CLI / TUI / App protocol  ← openai/codex 可直接研究
Cloud-only infrastructure ← 公开仓库不能穷尽
Training recipe / weights ← 公开仓库没有给出
```

官方 README 明确写 Codex CLI 是运行在本机的 coding agent，并把 Codex Web 单独称为 cloud-based agent；仓库许可证为 Apache-2.0。

---

# 2　从 workspace 看 Codex 已经是什么规模的软件

`codex-rs/Cargo.toml` 的 workspace members 已经包含大量独立 crate。这个事实本身非常重要：成熟 Agent 不会长期停留在一个 `agent.py`。

为了学习，把它们压缩成 12 个 subsystem：

| Subsystem | 代表目录 | 核心问题 |
|---|---|---|
| Core orchestration | `core/`, `core-api/` | session、turn、sampling、tool continuation |
| Protocol | `protocol/`, `app-server-protocol/` | UI 与 core 如何交换 typed events |
| User interfaces | `cli/`, `tui/`, `app-server/` | CLI/TUI/IDE 怎样复用同一核心 |
| Execution | `exec/`, `shell-command/`, `apply-patch/` | 命令、补丁怎样成为真实 action |
| Sandbox / permission | `sandboxing/`, `linux-sandbox/`, `windows-sandbox-service/`, `execpolicy/` | 谁能执行什么，边界怎样强制 |
| MCP / connectors | `codex-mcp/`, `rmcp-client/`, `connectors/`, `ext/mcp/` | 外部 tool/context server 怎样接入 |
| Context | `context-fragments/`, `prompts/`, `skills/`, `plugins/` | 什么信息最终进入模型可见上下文 |
| Memory / history | `history/`, `memories/`, `thread-store/`, `state/` | 长期状态怎样保存、读取和迁移 |
| Multi-agent | `agent-graph-store/`, `agent-identity/`, `agent-roles/` | subagent identity、graph、协作 |
| Git/workspace | `worktree/`, `git-utils/`, `file-system/` | repo/worktree 怎样成为任务空间 |
| Trace / observability | `rollout/`, `rollout-trace/`, `analytics/`, `otel/` | Agent 到底做了什么，怎样审计 |
| Extended capabilities | `code-mode*`, `ext/web-search/`, `ext/image-generation/`, `realtime-webrtc/` | Agent runtime 怎样扩展新能力 |

这是我们自己 capstone 后续模块化时的重要参照。

---

# 3　推荐源码阅读顺序

不要从 `main.rs` 开始机械逐行读整个仓库。推荐按数据流逆向定位：

```text
官方协议 mental model
codex-rs/docs/protocol_v1.md
        ↓
核心 turn loop
core/src/session/turn.rs
        ↓
工具路由
core/src/tools/
        ↓
执行 / patch / sandbox / approval
exec/ + apply-patch/ + sandboxing/
        ↓
context / AGENTS / compaction
core/src/context* + compact* + protocol prompts
        ↓
app-server protocol
app-server/ + app-server-protocol/
        ↓
MCP
codex-mcp/ + rmcp-client/
        ↓
multi-agent
core/src/session/multi_agents.rs
core/src/tools/handlers/multi_agents*
agent-graph-store/
        ↓
worktree / persistence / trace
worktree/ + thread-store/ + rollout-trace/
```

这个顺序对应的是“Agent 的数据流”，而不是仓库字母序。

---

# 4　三个最关键的入口文件

## 4.1 `codex-rs/docs/protocol_v1.md`

官方自己给出 `Model → Codex → Session → Task → Turn` 的 vocabulary。它非常适合建立 mental model。

其中明确定义：

```text
Task = 对一次用户输入执行工作的生命周期
Task = 多个 Turn
Turn = model request + streamed response + command/patch execution + approval
Turn output = next Turn input
```

并说明 UI 与 Codex core 通过 Submission Queue / Event Queue 通信。

## 4.2 `codex-rs/core/src/session/turn.rs`

这是目前最值得反复读的核心文件。

源码顶部直接写明 `run_turn` 的核心循环：

```text
model returns function call
        ↓
execute function
        ↓
send output back in next sampling request

model returns assistant-only message
        ↓
record conversation history
        ↓
turn complete
```

实际实现还加入：

- pre-sampling compaction；
- MCP startup requirements；
- skills/plugins injection；
- pending user input；
- context-window accounting；
- mid-turn compaction；
- hooks；
- guardian/reviewer；
- tool parallelism；
- error lifecycle。

所以“Agent loop”在生产实现里不是一个只有四行的 `while True`。

## 4.3 `codex-rs/Cargo.toml`

这个文件告诉你系统边界如何被拆成 crate。阅读成熟软件时，workspace graph 本身就是架构文档。

---

# 5　源码阅读时固定记录四张表

每读一个 subsystem，都记录：

## A. State Table

```text
state name
owner
lifetime
mutable by whom
persisted or ephemeral
model-visible or runtime-only
```

## B. Event Table

```text
event
producer
consumer
blocking?
can retry?
observable by UI?
```

## C. Boundary Table

```text
boundary
input
output
security trust level
failure semantics
```

## D. Parity Table

```text
official behavior
our clean-room behavior
unit test
integration test
gap
```

只有这样，源码解剖才会反哺我们自己的实现。

---

# 6　官方仓库中值得专门追踪的“不是模型”的能力

下面这些能力很容易被产品体验掩盖，但源码显示它们是独立系统组件：

```text
approval policy
sandbox policy
instruction source hierarchy
skills / plugins / connectors
MCP server lifecycle
context compaction
pending user input / steering
thread resume / fork
multi-agent spawning
agent identity / graph
worktree management
rollout trace
hooks
review / guardian
code mode runtime
```

它们都不是“底层模型自己天然拥有”的。

---

# 7　我们自己的对应路径

目前：

```text
官方 Codex                         我们的 clean-room capstone
──────────────────────────────────────────────────────────────
turn.rs                            codex_harness.py
core/tools                         tools.py / structured.py
approval metadata                  codex_harness.TurnSettings
sandbox metadata                   codex_harness.SandboxPolicy
protocol events                    HarnessEvent
MCP                                mcp.py
persistent history                 memory.py
planning                           planning.py
verification                       verification.py
multi-agent coordination           multi_agent.py
worktree                           worktree.py
```

这张表以后会不断细化到 class/function 级。

---

# 8　当前最重要的研究纪律

我们把官方代码当作一手资料，但仍然坚持：

```text
read source
→ extract behavior contract
→ independently implement
→ test
→ compare observable behavior
```

而不是：

```text
copy source
→ 改变量名
→ 宣称“从零实现”
```

真正有学习价值的是**重建背后的软件机制**。
