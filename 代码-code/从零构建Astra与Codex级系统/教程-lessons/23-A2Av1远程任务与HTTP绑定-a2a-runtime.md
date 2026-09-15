# Lesson 23　A2A v1：从远程任务协议到 Durable Agent Runtime

> 本课不是“介绍一下 A2A”。目标是把 **A2A v1 原始协议对象、HTTP+JSON wire binding、SSE streaming 和我们自己的 Durable Agent Runtime** 一层层对应起来，并明确哪些已经实现、哪些仍然没有实现。

---

## 1　原始资料先行

本课固定核验 A2A 官方仓库 commit：

```text
6d6640c29b102f7a8d23784901351b5d2454fe71
```

规范源文件：

https://github.com/a2aproject/A2A/blob/6d6640c29b102f7a8d23784901351b5d2454fe71/specification/a2a.proto

官方 v1 文档：

https://github.com/a2aproject/A2A/blob/6d6640c29b102f7a8d23784901351b5d2454fe71/docs/specification.md

因此本项目不会继续沿用旧版博客中常见的 `message/send`、`tasks/get` 等近似写法，而直接对齐 pinned v1 proto。

---

## 2　MCP 与 A2A 解决的是不同边界

MCP 主要回答：

```text
Agent / Host
    ↕
Tool / Resource / Context Server
```

A2A 主要回答：

```text
Agent A
   ↕
long-running Task / Message / Artifact
   ↕
Agent B
```

Tool 调用通常近似：

```text
arguments → result
```

Agent 协作则可能经历：

```text
goal
→ Task SUBMITTED
→ WORKING
→ INPUT_REQUIRED / AUTH_REQUIRED
→ more messages
→ artifacts
→ COMPLETED / FAILED / CANCELED / REJECTED
```

所以 A2A 的核心不是“远程函数调用”，而是**跨边界任务生命周期**。

---

## 3　v1 核心对象

当前 `a2a.py` 从原始 proto clean-room 实现：

```text
AgentCard
AgentInterface
AgentSkill
Message
Part
Task
TaskStatus
Artifact
SendMessageConfiguration
TaskStore
A2AService
```

Task state 严格保留 pinned v1 wire enum：

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

协议 enum 的拼写本身就是 wire contract，不能为了“更 Pythonic”自行改成 `done`、`waiting`、`cancelled`。

---

## 4　Part 为什么不是只有文本

原始 proto 中 `Part` 的 oneof 是：

```text
text
raw
url
data
```

此外还有：

```text
filename
media_type
metadata
```

因此 Agent-to-Agent 通信天然可以承载文本、结构化 JSON、文件引用和多模态内容。

但“协议允许”不等于“runtime 可以安全处理”。当前 `a2a_runtime_bridge.py`：

- 明确支持 `text`；
- 将 `data` 序列化为结构化文本进入当前模型上下文；
- 对 `raw` 与 `url` 主动拒绝。

这是因为真正支持 URL/raw 还需要下载策略、MIME 校验、大小限制、credential scope、malware/content inspection 与 sandbox。项目不会偷偷帮 Agent 下载外部内容后再宣称“多模态支持”。

---

## 5　HTTP+JSON v1 路由必须看原始 proto

pinned v1 `a2a.proto` 明确给出：

```text
POST /message:send
POST /message:stream
GET  /tasks/{id}
GET  /tasks
POST /tasks/{id}:cancel
GET  /tasks/{id}:subscribe
```

Agent Card 的公开发现入口：

```text
GET /.well-known/agent-card.json
```

当前 `a2a_http.py` 已实现并用真实 localhost TCP/HTTP 测试：

```text
GET  /.well-known/agent-card.json
POST /message:send
GET  /tasks/{id}
GET  /tasks
POST /tasks/{id}:cancel
```

当前还没有完成：

```text
/tasks/{id}:subscribe           ❌
push notification configs      ❌
extended authenticated card    ❌
tenant-prefixed routes          ❌
A2A transport security          ❌
official conformance suite      ❌
```

`POST /message:stream` 已进入独立 SSE reference path，见第 12 节。

---

## 6　ListTasks：用一个小接口检查协议严谨性

原始 `ListTasksRequest` 不是随便的 `GET /tasks`，而包含：

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

HTTP+JSON 使用 camelCase：

```text
contextId
status
pageSize
pageToken
historyLength
statusTimestampAfter
includeArtifacts
```

响应还需要：

```text
tasks
nextPageToken
pageSize
totalSize
```

当前 reference implementation 已完成：

- `status` 单状态过滤，而不是自己发明 `states=` wire 字段；
- `pageSize` 默认 50，约束 1–100；
- opaque `pageToken`；
- `historyLength` 投影；
- `statusTimestampAfter`；
- `includeArtifacts`；
- `nextPageToken/pageSize/totalSize`。

教学实现内部用 offset 编码 token，但客户端只把它看作 opaque token，不能依赖内部格式。

---

## 7　`returnImmediately` 是 lifecycle，不只是 bool

原始 `SendMessageConfiguration` 的语义是：

```text
returnImmediately = false
→ SendMessage 等到 terminal / interrupted state 再返回

returnImmediately = true
→ Task 创建后即可返回
→ 后续通过 GetTask / streaming 等机制观察状态
```

项目测试真正验证：

```text
POST /message:send
returnImmediately=true
        ↓
Task = SUBMITTED
        ↓
SQLite durable Task
        ↓
server restart / later processing
        ↓
GET /tasks/{id}
```

所以协议字段最终必须落到持久化状态机，而不是只出现在 dataclass。

---

## 8　A2A Task 与本地 Runtime Thread 不是一个东西

A2A：

```text
Task
├── id
├── contextId
├── status
├── history
└── artifacts
```

我们的 Agent OS：

```text
Thread
├── Event Log
├── WorkItem
├── Turn
├── Tool Journal
├── Checkpoint
├── Context provenance
└── Artifact Store
```

因此：

```text
A2A Task ≠ Runtime Thread
```

正确做法是显式 bridge：

```text
A2A Task
   │
   ├── metadata.runtimeThreadId
   ▼
Durable Thread
   ↓
WorkItem
   ↓
queue-level thread fencing
   ↓
Turn Executor
   ↓
result / artifacts
   ↓
A2A TaskStatus + Message + Artifact
```

源码：

```text
src/astra_codex/a2a_runtime_bridge.py
```

---

## 9　本地 runtime bridge 当前怎样执行

`DurableRuntimeA2AHandler` 第一次处理 Task 时建立稳定映射：

```text
A2A task id = task_abc
        ↓
runtime thread id = a2a_task_abc
```

然后：

```text
A2A Message
   ↓
extract text/data Parts
   ↓
runtime.submit(...)
   ↓
Durable WorkQueue
   ↓
runtime.run_one(
    allowed_thread_ids={runtime_thread_id}
)
   ↓
Codex-style Turn Executor
   ↓
RuntimeExecutionRecord
   ↓
A2AExecutionResult
```

这里 `allowed_thread_ids` 是真实 queue-level fencing：A2A worker 不能误 claim 其他 Thread 的工作。

---

## 10　HTTP → A2A → Agent OS 已经真正接通

如果 HTTP server 直接调用一个在主线程构造的 `DurableAgentRuntime`，Python SQLite 会触发 thread-affinity 错误。

因此新增：

```text
src/astra_codex/a2a_runtime_http.py
```

其结构是：

```text
remote HTTP client
      ↓
POST /message:send
      ↓
HTTP server thread
├── A2ATaskStore(thread-owned connection)
└── DurableAgentRuntime(thread-owned connections)
          ↓
    DurableRuntimeA2AHandler
          ↓
Thread → WorkItem → Turn
          ↓
A2A Task response
```

测试还会关闭 HTTP server，再用原始 runtime 重新读取 committed Thread/WorkItem，证明跨线程共享的是**durable state files**，不是 connection object。

Fast CI run 235 对这一链路给出的硬证据：

```text
182 passed, 14 skipped
Ruff: All checks passed
```

---

## 11　Artifact 不应该退化成聊天文字

Runtime artifact 记录：

```text
artifact_id
kind
sha256
size_bytes
metadata
```

bridge 会映射成 A2A `Artifact` + 结构化 `data` Part，例如：

```json
{
  "artifactId": "...",
  "kind": "patch",
  "sha256": "...",
  "sizeBytes": 1234,
  "metadata": {}
}
```

不会把服务端本机绝对路径伪装成远端可访问 URL。

原则是：

> **跨 Agent 交付协议 artifact，而不是泄漏服务端机器内部路径。**

---

## 12　SendStreamingMessage：真正的 SSE 首帧

pinned proto 的 `StreamResponse` oneof 可以携带：

```text
task
message
status_update
artifact_update
```

当前新增：

```text
src/astra_codex/a2a_streaming.py
src/astra_codex/a2a_sse.py
```

reference streaming 流程：

```text
POST /message:stream
        ↓
validate request
        ↓
create durable SUBMITTED Task
        ↓
SSE flush: {task: ...}          ← 首帧先到客户端
        ↓
process handler
        ↓
new artifacts?
        ├─ yes → artifactUpdate
        ↓
statusUpdate(final)
        ↓
close stream
```

关键测试故意让 handler 阻塞：客户端必须先读到 `SUBMITTED Task`，然后 handler 才完成。这证明它不是“任务全部跑完后一次性返回一个伪装成 stream 的数组”。

另一个测试验证新 artifact 在 final status 之前以 `artifactUpdate` 发送。

Fast CI run 241：

```text
184 passed, 14 skipped, 1 warning
Ruff correctness lint: All checks passed
```

### 当前 streaming 边界

底层 handler 目前仍是同步函数，因此 reference stream 不会伪造它没有暴露的 token-level 或 WORKING 中间增量。真正的 incremental runtime 需要 executor 主动发布：

```text
TaskStatusUpdateEvent
TaskArtifactUpdateEvent
model/tool progress events
```

这将与未来 `SubscribeToTask` 共享同一个 durable update source。

---

## 13　当前自动测试已经证明什么

```text
A2A v1 exact TaskState spelling
Part oneof wire shape
AgentInterface protocolVersion
returnImmediately durability
SendMessage / GetTask / ListTasks / CancelTask
well-known Agent Card over real localhost HTTP
source-aligned ListTasks fields + opaque pagination
real HTTP cancel
A2A Task → real Durable Thread → WorkItem → Turn
HTTP → A2A → thread-owned DurableAgentRuntime
server restart persistence
runtime result → A2A COMPLETED
raw/url media without policy → REJECTED
/message:stream real SSE
SUBMITTED first frame before slow handler completion
artifactUpdate → final statusUpdate ordering
```

项目已经从：

```text
几个 A2A dataclass
```

推进到：

```text
original v1 spec
→ wire objects
→ durable Task store
→ HTTP binding
→ source-aligned ListTasks
→ local Agent OS bridge
→ remote HTTP Agent-OS gateway
→ SSE SendStreamingMessage
```

---

## 14　下一层

后续优先级：

```text
SubscribeToTask
→ durable task-update journal
→ reconnect / replay cursor

TaskStatusUpdateEvent
TaskArtifactUpdateEvent
→ executor-native intermediate events

push notification configuration
extended authenticated Agent Card
A2A security schemes
HTTP tenant routing
official SDK differential tests
protocol conformance fixtures
```

最重要的下一步不是增加更多协议名词，而是让：

```text
Task lifecycle
+ streaming
+ reconnect
+ authorization
+ local Agent OS state
```

在崩溃、重连和并发情况下仍保持可解释的一致性。

---

## 15　毕业标准

读完本课后，应该能从空目录解释并实现：

```text
Agent Card discovery
→ Message / Part
→ Task creation
→ durable Task state
→ returnImmediately
→ Get/List pagination
→ cancellation
→ Artifact delivery
→ HTTP+JSON binding
→ A2A ↔ local Runtime bridge
→ real remote runtime gateway
→ SSE SendStreamingMessage
```

并能明确指出：

```text
Tool protocol ≠ Agent protocol
A2A Task ≠ local Thread
HTTP transport ≠ task lifecycle
streaming ≠ async executor
wire compatibility ≠ full conformance
authentication ≠ authorization ≠ sandbox
```

这才是协议层进入 Astra-class / Codex-class 系统后的正确学习方式。
