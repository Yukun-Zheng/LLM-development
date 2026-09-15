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

这里其实存在三类不同协议问题：

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

# 2　JSON-RPC 只是 transport-friendly RPC envelope，不是 Agent

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

成功响应：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {...}
}
```

错误响应：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32601,
    "message": "Method not found"
  }
}
```

RPC envelope 解决“请求如何编码、如何关联响应”，并不自动提供 planning、memory、permission、sandbox、durability 或 task lifecycle。

---

# 3　MCP：Model Context Protocol

当前教材以 MCP 官方规范为一手来源：

- https://modelcontextprotocol.io/specification/
- 2026-07-28 版本说明：https://blog.modelcontextprotocol.io/posts/2026-07-28/

截至 2026-07-28，MCP 已朝 stateless protocol core 演进，并引入新的能力发现和请求元数据语义。教材必须把**协议版本**当成事实的一部分，不能把 2025 年的生命周期描述直接套到 2026 版本。

## 3.1　MCP 的职责

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

它主要回答：

> Agent/模型怎样用统一协议发现并访问外部能力与上下文？

而不是：

> Agent 下一步应该做什么？

## 3.2　本项目的 Minimal MCP

源码：

```text
代码-code/从零构建Astra与Codex级系统/src/astra_codex/mcp.py
```

当前 reference subset 已从零实现 JSON-RPC envelope、`server/discover`、`tools/list`、`tools/call`、ToolRegistry 映射、client 与 deterministic tests。

它仍然不是完整生产 SDK；transport/auth/tasks/extensions 等继续分层实现。

---

# 4　为什么“用了 MCP”仍然不等于“有 Agent”

一个完整执行系统至少仍需：

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

MCP 能让 capability access 标准化，但不会替你解决任务分解、重试、恢复、长期记忆和完成验证。

---

# 5　A2A v1：Agent2Agent

本章 A2A 部分不再只链接产品文档，而固定一份可重复核验的官方规范快照：

```text
a2aproject/A2A
commit: 6d6640c29b102f7a8d23784901351b5d2454fe71
```

规范源文件：

https://github.com/a2aproject/A2A/blob/6d6640c29b102f7a8d23784901351b5d2454fe71/specification/a2a.proto

官方规范文档：

https://github.com/a2aproject/A2A/blob/6d6640c29b102f7a8d23784901351b5d2454fe71/docs/specification.md

A2A v1 的核心不是“远程调用另一个 LLM”，而是**跨边界任务协作**。

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

pinned `a2a.proto` 定义了 AgentCard、AgentInterface、AgentSkill、Message、Part、Task、TaskStatus、Artifact 等对象。

TaskState 的 JSON 枚举语义包括：

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

这类拼写属于 wire contract；不能为了“更 Pythonic”就自行改成 `done`、`waiting`、`cancelled`。

## 6.1　Part 是真正的多模态容器

原始 proto 的 `Part` oneof 是：

```text
text
raw
url
data
```

并额外带 filename、media_type、metadata。

所以 Agent-to-Agent 通信天然可以承载文本之外的数据。真正困难的地方随之而来：URL 谁来下载？raw bytes 多大？MIME 是否可信？凭证能否转发？内容进入哪个 sandbox？

协议字段存在不等于 runtime 已经安全支持该内容类型。

---

# 7　AgentCard 与发现

官方 v1 discovery 文档规定常见公开发现入口：

```text
GET /.well-known/agent-card.json
```

AgentCard 描述：

```text
identity
supportedInterfaces
protocolVersion
capabilities
input/output modes
skills
security requirements
```

其中 `protocolVersion` 位于每个 AgentInterface 上，而不是一个可忽略的装饰字段。

本项目 `a2a.py` 和 `a2a_http.py` 都对这一结构做显式建模。

---

# 8　A2A v1 HTTP+JSON：不要凭记忆写 URL

pinned v1 `a2a.proto` 对 HTTP binding 给出的核心路由是：

```text
POST /message:send
POST /message:stream
GET  /tasks/{id}
GET  /tasks
POST /tasks/{id}:cancel
GET  /tasks/{id}:subscribe
```

这也是为什么旧资料中常见的 `/v1/message:send` 或自己编的 `tasks/get` 不能直接写进 2026 教材；v1 已移除固定 `/v1` path prefix，版本可以由 interface/base URL 管理。

当前源码：

```text
src/astra_codex/a2a_http.py
```

已经实现真实 localhost TCP/HTTP 的 reference subset：

```text
GET  /.well-known/agent-card.json
POST /message:send
GET  /tasks/{id}
GET  /tasks
POST /tasks/{id}:cancel
```

尚未实现 `/message:stream`、`:subscribe`、push notification、tenant-prefixed binding、extended authenticated card 和正式 conformance suite，因此项目不会把它标成“完整 A2A server”。

---

# 9　ListTasks 为什么是一个很好的协议严谨性测试

原始 v1 `ListTasksRequest` 不是随便一个 `GET /tasks`。

它定义：

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

`ListTasksResponse` 还必须表达：

```text
tasks
next_page_token
page_size
total_size
```

HTTP+JSON 映射使用 camelCase query parameters，例如：

```text
contextId
status
pageSize
pageToken
historyLength
statusTimestampAfter
includeArtifacts
```

因此本项目不再保留早期简化实现里的自定义 `state=` query，而是直接按照 pinned proto 的字段语义实现 reference pagination。

`pageToken` 对客户端是 opaque token；当前教学实现内部用 offset 编码，但客户端不能依赖其内部格式。

---

# 10　`returnImmediately` 是 lifecycle，不只是一个 bool

原始 `SendMessageConfiguration` 规定：

```text
returnImmediately = false
→ 等到 terminal / interrupted state 后返回

returnImmediately = true
→ Task 创建后即可返回
→ 后续通过 GetTask / streaming 等机制观察状态
```

项目测试真正验证：

```text
POST /message:send
→ SUBMITTED
→ SQLite durable Task
→ process / restart
→ GET /tasks/{id}
```

所以协议字段最终必须落到持久化状态机，而不是只出现在 dataclass 里。

---

# 11　A2A Task 与我们的 Durable Thread 必须分开

A2A 的 Task 是远程协议对象；本项目的 Thread 是本地 Agent OS 对象。

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

因此正确做法不是把两个类名改成一样，而是建立显式 bridge。

源码：

```text
src/astra_codex/a2a_runtime_bridge.py
```

当前映射：

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

这样协议世界与本地执行世界各自保持自己的不变量。

---

# 12　MCP、A2A、App Server 的分工

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

这也是项目为什么不把所有东西塞进一个 `agent.py`。

---

# 13　协议安全必须落到 capability acquisition point

远程协议至少涉及：

```text
identity
AuthN
AuthZ
least privilege
capability discovery
schema validation
secret scope
prompt/tool-output injection
audit trail
sandbox boundary
```

项目当前 App Server 已经做到 Bearer AuthN、method/thread AuthZ 和 queue-level thread fencing；MCP/A2A 的 transport security 则继续独立推进。

关键原则：

```text
valid schema
≠ authorized operation
≠ sandboxed execution
≠ semantically safe result
```

---

# 14　当前代码与实验入口

```text
src/astra_codex/mcp.py
src/astra_codex/a2a.py
src/astra_codex/a2a_http.py
src/astra_codex/a2a_runtime_bridge.py
src/astra_codex/app_server.py

tests/test_mcp.py
tests/test_a2a.py
tests/test_a2a_http.py
tests/test_a2a_runtime_bridge.py
```

配套逐代码课程：

```text
教程-lessons/23-A2Av1远程任务与HTTP绑定-a2a-runtime.md
```

---

# 15　下一层协议工程

后续必须继续实现和验证：

```text
A2A SendStreamingMessage / SSE
A2A SubscribeToTask
TaskStatusUpdateEvent
TaskArtifactUpdateEvent
push notification config
extended authenticated Agent Card
tenant routing
A2A security schemes
MCP 完整 transports/auth/tasks/extensions
official SDK differential tests
protocol conformance fixtures
```

最终目标不是“项目支持很多协议名词”，而是读者能从原始 spec 推出 wire format、状态机和安全边界，再自己写出一个可以互操作的 reference runtime。
