# A9　Agent Protocols：MCP、A2A 与智能体互操作

> **本章主线**：当 Agent 从“一个进程里调用几个 Python 函数”发展到跨进程、跨主机、跨组织协作时，工具和 Agent 本身都会成为协议端点。我们必须分别理解 **Tool Protocol、Agent Protocol 与 Application Control Protocol**，而不能把它们统称为“Agent API”。

---

# 1　协议层到底解决什么

最小 Tool Agent：

```text
LLM
↓
JSON tool call
↓
Python function
```

真实系统则可能是：

```text
User / IDE / App
       ↕
Application Control Plane
       ↕
Agent Runtime
├─ MCP → filesystem / database / browser / SaaS
├─ A2A → coding agent
├─ A2A → research agent
└─ local tools / sandbox / verifier
```

这里存在三类不同协议问题：

```text
Tool Protocol
Agent ↔ Tool / Context Server

Agent Protocol
Agent ↔ Agent

Application Runtime Protocol
UI / IDE / Client ↔ Agent Runtime
```

本项目分别用 MCP、A2A 和自己的 App Server reference implementation 研究这三层。

---

# 2　JSON-RPC 只是 RPC envelope，不是 Agent

JSON-RPC 2.0 的核心形状是：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "search",
    "arguments": {"q": "attention"}
  }
}
```

RPC envelope 解决“请求如何编码、如何关联响应”，并不自动提供 planning、memory、permission、sandbox、durability 或 task lifecycle。

---

# 3　MCP：Model Context Protocol

当前教材以 MCP 官方规范为一手来源：

- https://modelcontextprotocol.io/specification/
- 2026-07-28 版本说明：https://blog.modelcontextprotocol.io/posts/2026-07-28/

截至 2026-07-28，MCP 已朝 stateless protocol core 演进。教材必须把**协议版本**当成事实的一部分，不能把旧版生命周期描述直接套到新版本。

MCP 主要回答：

```text
Host / Client
      ↕
MCP
      ↕
Server
├─ tools
├─ resources
├─ prompts
└─ other capabilities
```

它标准化 capability/context access，而不是替 Agent 做 planning。

本项目的 `mcp.py` 已从零实现一个 inspectable teaching subset：JSON-RPC envelope、discover、tools/list、tools/call、ToolRegistry 映射、client 和 deterministic tests。完整 transports/auth/tasks/extensions 继续分层推进。

---

# 4　为什么“用了 MCP”仍然不等于“有 Agent”

完整执行系统至少还需要：

```text
Policy / Planner
       ↓
Durable Agent Runtime
       ↓
Permission / Sandbox
       ↓
MCP Client
       ↓
MCP Server
       ↓
Environment
       ↓
Observation / Artifact
       ↓
Verifier
```

MCP 不替你解决任务分解、重试、恢复、长期记忆和完成验证。

---

# 5　A2A v1：Agent2Agent

A2A 部分固定一份可重复核验的官方规范快照：

```text
a2aproject/A2A
commit: 6d6640c29b102f7a8d23784901351b5d2454fe71
```

原始规范：

https://github.com/a2aproject/A2A/blob/6d6640c29b102f7a8d23784901351b5d2454fe71/specification/a2a.proto

官方规范文档：

https://github.com/a2aproject/A2A/blob/6d6640c29b102f7a8d23784901351b5d2454fe71/docs/specification.md

A2A v1 的核心不是“远程调用另一个 LLM”，而是**跨边界任务协作**：

```text
Agent A
  ↓ Message
Agent B
  ↓
Task
├─ status
├─ history
├─ artifacts
└─ metadata
```

---

# 6　A2A v1 原始数据模型

pinned `a2a.proto` 定义 AgentCard、AgentInterface、AgentSkill、Message、Part、Task、TaskStatus、Artifact 等对象。

TaskState 的 wire enum 包括：

```text
TASK_STATE_UNSPECIFIED
TASK_STATE_SUBMITTED
TASK_STATE_WORKING
TASK_STATE_COMPLETED
TASK_STATE_FAILED
TASK_STATE_CANCELED
TASK_STATE_INPUT_REQUIRED
TASK_STATE_REJECTED
TASK_STATE_AUTH_REQUIRED
```

协议 enum 拼写本身就是 contract，不能为了“更 Pythonic”自行改成 `done`、`waiting`、`cancelled`。

## 6.1　Part 是多模态容器

原始 proto 的 `Part` oneof：

```text
text
raw
url
data
```

并带 filename、media_type、metadata。

因此 Agent-to-Agent 通信可以承载文本之外的数据，但“协议允许”不等于 runtime 已安全支持。项目当前 bridge 支持 `text`/`data`，对 `raw`/`url` 在没有 fetch/security policy 前主动拒绝。

---

# 7　AgentCard 与发现

官方 v1 discovery 文档使用标准公开入口：

```text
GET /.well-known/agent-card.json
```

AgentCard 描述 identity、supportedInterfaces、protocolVersion、capabilities、I/O modes、skills 和 security requirements。

其中 `protocolVersion` 位于每个 AgentInterface 上，是互操作 contract，而不是展示字段。

---

# 8　A2A v1 HTTP+JSON：URL 不能凭记忆写

pinned v1 `a2a.proto` 给出的核心 HTTP binding：

```text
POST /message:send
POST /message:stream
GET  /tasks/{id}
GET  /tasks
POST /tasks/{id}:cancel
GET  /tasks/{id}:subscribe
```

所以不能把旧资料中的 `/v1/message:send` 或自定义 `tasks/get` 当成 2026 v1 事实。

项目当前已经用真实 localhost HTTP 验证：

```text
GET  /.well-known/agent-card.json
POST /message:send
POST /message:stream
GET  /tasks/{id}
GET  /tasks
POST /tasks/{id}:cancel
GET  /tasks/{id}:subscribe
```

但这仍只是 reference subset，不代表完整 A2A conformance。tenant-prefixed binding、push notification、extended authenticated card、正式 transport security 和官方 conformance suite 仍未完成。

---

# 9　ListTasks：一个小接口足以暴露协议是否严谨

原始 v1 `ListTasksRequest` 定义：

```text
tenant
context_id
status
page_size
page_token
history_length
status_timestamp_after
include_artifacts
```

`ListTasksResponse`：

```text
tasks
next_page_token
page_size
total_size
```

HTTP+JSON 使用 camelCase query：

```text
contextId
status
pageSize
pageToken
historyLength
statusTimestampAfter
includeArtifacts
```

项目 reference 实现已经按这些字段工作，并使用 opaque `pageToken`。内部当前以 offset 编码，但客户端不依赖 token 内部格式。

---

# 10　`returnImmediately` 是 lifecycle，不只是 bool

原始 `SendMessageConfiguration` 的语义：

```text
returnImmediately = false
→ 等到 terminal / interrupted state 后返回

returnImmediately = true
→ 创建 Task 后即可返回
→ 后续通过 GetTask / streaming / subscription 观察状态
```

项目测试真正验证了 `SUBMITTED` Task 的 SQLite persistence、later processing 和 restart retrieval。

---

# 11　A2A Task 与 Durable Thread 必须分开

A2A 的 Task 是远程协议对象；项目的 Thread 是本地 Agent OS 对象。

```text
A2A Task
├─ id / contextId
├─ TaskStatus
├─ Message history
└─ Artifact

Durable Thread
├─ Event Log
├─ WorkItem / Lease
├─ Turn
├─ Tool Journal
├─ Checkpoint
├─ Context provenance
└─ Artifact Store
```

所以：

```text
A2A Task ≠ Runtime Thread
```

项目通过 `a2a_runtime_bridge.py` 显式映射：

```text
A2A Task
→ metadata.runtimeThreadId
→ Durable Thread
→ WorkItem
→ queue-level allowed_thread_ids fencing
→ Turn Executor
→ RuntimeExecutionRecord
→ A2A TaskStatus / Message / Artifact
```

---

# 12　真实 HTTP 已经进入本地 Agent OS

`a2a_runtime_http.py` 让 HTTP server thread 自己打开 A2ATaskStore 和 DurableAgentRuntime handles：

```text
remote HTTP client
→ POST /message:send
→ A2A Task
→ server-thread-owned DurableAgentRuntime
→ Thread / WorkItem / Turn
→ A2A result
```

这样避免跨线程复用 SQLite connection。测试关闭 server 后还能从原 runtime replay committed state，并验证 A2A Task 在 server restart 后仍存在。

---

# 13　SendStreamingMessage：真实 SSE，而不是最终数组伪装成流

pinned v1 `StreamResponse` oneof：

```text
task
message
status_update
artifact_update
```

项目 `a2a_streaming.py` + `a2a_sse.py` 实现：

```text
POST /message:stream
→ create durable SUBMITTED Task
→ flush SSE {task: ...}
→ run handler
→ artifactUpdate(s)
→ final statusUpdate
```

自动测试故意让 handler 阻塞，证明客户端先收到 SUBMITTED 首帧，再等 handler 完成；不是任务结束后一次性把数组叫作 streaming。

---

# 14　SubscribeToTask：durable update journal，而不是内存 Queue

新增 `a2a_subscription.py`：

```text
A2A Task snapshot
→ append-only SQLite update journal
→ GET /tasks/{id}:subscribe
→ text/event-stream
```

核心设计：

```text
producer/executor connection A
→ durable Task update journal

subscriber HTTP connection B
→ read updates after cursor
→ SSE delivery
```

subscriber 是否在线不会决定 update 是否存在。

项目还显式实现一个 SSE transport-level replay extension：

```text
Last-Event-ID
```

它**不是声称的 A2A normative request field**。它只作为当前 reference transport 的 durable reconnect cursor。

测试覆盖：

```text
SUBMITTED snapshot persisted
→ subscriber receives SUBMITTED
→ another DB connection completes Task
→ subscriber receives terminal Task
→ stream closes
```

以及：

```text
observed update N
→ disconnect
→ terminal update N+1 written
→ reconnect Last-Event-ID=N
→ only N+1 replayed
```

新的 fresh subscription 如果 Task 已经 terminal，则 reference server 返回 409，保持 pinned v1 对 terminal SubscribeToTask 的 unsupported-operation 边界。

---

# 15　MCP、A2A、App Server 的分工

```text
MCP
Agent ↔ Tool / Context Server

A2A
Agent ↔ Remote Agent / Long-running Task

App Server
UI / IDE / User Client ↔ Local Agent Runtime
```

真实系统可以同时存在三层：

```text
IDE
 ↓ App Server
Coordinator Agent
 ├─ MCP → Git / Browser / Database
 ├─ A2A → Coding Worker
 └─ A2A → Research Worker
```

---

# 16　协议安全必须落到 capability acquisition point

远程协议至少涉及：identity、AuthN、AuthZ、least privilege、schema validation、secret scope、injection、audit trail 与 sandbox boundary。

项目 App Server 已有 Bearer AuthN、method/thread AuthZ 和 queue-level thread fencing；A2A transport security 仍是后续独立层。

必须保持：

```text
valid schema
≠ authorized operation
≠ sandboxed execution
≠ semantically safe result
```

---

# 17　当前源码与实验入口

```text
src/astra_codex/mcp.py
src/astra_codex/a2a.py
src/astra_codex/a2a_http.py
src/astra_codex/a2a_runtime_bridge.py
src/astra_codex/a2a_runtime_http.py
src/astra_codex/a2a_streaming.py
src/astra_codex/a2a_sse.py
src/astra_codex/a2a_subscription.py
src/astra_codex/app_server.py

tests/test_mcp.py
tests/test_a2a.py
tests/test_a2a_http.py
tests/test_a2a_runtime_bridge.py
tests/test_a2a_runtime_http.py
tests/test_a2a_streaming.py
tests/test_a2a_subscription.py
```

配套逐代码课程：

```text
Lesson 23：A2A v1 远程任务、HTTP 与 Streaming
Lesson 24：SubscribeToTask、持久更新日志与断线重放
```

Fast CPU CI run 248 当前硬证据：

```text
188 passed, 14 skipped, 1 warning in 46.16s
Ruff correctness lint: All checks passed
```

---

# 18　下一层协议工程

还需要继续实现和验证：

```text
executor-native WORKING/intermediate updates
TaskStatusUpdateEvent journal
TaskArtifactUpdateEvent journal
A2A cancellation → local running WorkItem propagation
INPUT_REQUIRED / AUTH_REQUIRED continuation mapping
push notification config
extended authenticated Agent Card
tenant routing
A2A security schemes
MCP 完整 transports/auth/tasks/extensions
official SDK differential tests
protocol conformance fixtures
```

最终目标不是“支持很多协议名词”，而是读者能从原始 spec 推出 wire format、持久状态机、reconnect semantics 和安全边界，再亲手写出可以互操作、可以失败恢复的 reference runtime。
