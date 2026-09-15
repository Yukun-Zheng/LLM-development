# Lesson 23　A2A v1：从远程任务协议到 Durable Agent Runtime

> 本课不是“介绍一下 A2A”。目标是把 **A2A v1 原始协议对象、HTTP+JSON wire binding 和我们自己的 Durable Agent Runtime** 一层层对应起来，并明确哪些已经实现、哪些仍然没有实现。

---

## 1　原始资料先行

本课固定核验 A2A 官方仓库 commit：

```text
6d6640c29b102f7a8d23784901351b5d2454fe71
```

规范源文件：

https://github.com/a2aproject/A2A/blob/6d6640c29b102f7a8d23784901351b5d2454fe71/specification/a2a.proto

官方 v1 文档同时说明标准 binding 包括 JSON-RPC、gRPC 和 HTTP+JSON：

https://github.com/a2aproject/A2A/blob/6d6640c29b102f7a8d23784901351b5d2454fe71/docs/specification.md

因此本项目不会继续沿用旧版博客中常见的 `message/send`、`tasks/get` 等模糊伪路径，而直接对齐 pinned v1 proto。

---

## 2　先分清 MCP 与 A2A

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

Tool 调用通常更像：

```text
arguments → result
```

Agent 协作则可能持续很久：

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

## 3　v1 的几个核心对象

当前 `a2a.py` 从原始 proto clean-room 实现了：

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

Task state 不是自己发明的字符串，而是 pinned v1 中的：

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

这里一个重要教学原则是：**协议 enum 拼写本身就是 wire contract。**

---

## 4　Part 为什么不是只有文本

A2A `Part` 是通信内容容器。原始 proto 的 oneof 包含：

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

这意味着未来 Astra-class Agent 间传递的不只是聊天文字，还可能是：

- JSON 数据；
- 图片/音频/视频；
- 文件 URL；
- 二进制内容；
- 结构化 artifact metadata。

我们当前 runtime bridge 对 `text` 和 `data` 有明确处理；对 `raw` 和 `url` **主动拒绝**，因为如果没有下载策略、MIME 校验、大小限制、凭证策略和 sandbox，就不应该偷偷替 Agent 拉取外部内容。

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

Agent Card 的标准公开发现入口是：

```text
GET /.well-known/agent-card.json
```

我们的 `a2a_http.py` 当前只实现一个**可验证子集**：

```text
GET  /.well-known/agent-card.json
POST /message:send
GET  /tasks/{id}
GET  /tasks
POST /tasks/{id}:cancel
```

没有实现的东西不会写成已完成：

```text
/message:stream                 ❌
/tasks/{id}:subscribe           ❌
push notification configs      ❌
extended authenticated card    ❌
tenant-prefixed routes          ❌
full pagination                 ❌
TLS / protocol auth             ❌
official conformance suite      ❌
```

---

## 6　`returnImmediately` 的语义

原始 `SendMessageConfiguration` 规定：

```text
returnImmediately = false
→ SendMessage 等到 terminal / interrupted state 再返回

returnImmediately = true
→ 创建 Task 后立即返回
→ Task 可以仍处于 SUBMITTED / processing lifecycle
```

我们的测试不只是检查一个布尔字段，而是真正验证：

```text
POST /message:send
returnImmediately=true
        ↓
Task = SUBMITTED
        ↓
SQLite persistence
        ↓
GET /tasks/{id}
        ↓
仍然能恢复同一个 Task
```

这已经把“协议字段”落到了 durable semantics。

---

## 7　为什么 HTTP server 不能直接复用 SQLite connection

Python SQLite 默认 connection 是 thread-affine 的。

错误做法：

```text
main thread 创建 A2ATaskStore
          ↓
HTTP server thread 直接拿同一个 connection 用
```

正确 reference design：

```text
main thread
└── prototype A2AService

HTTP server thread
└── A2ATaskStore(same_db_path)   ← 新 connection
    └── A2AService
```

因此跨线程共享的是**durable state file**，不是 SQLite connection object。

这与我们之前 App Server HTTP transport 的设计原则一致。

---

## 8　协议 Task 与本地 Runtime Thread 不是一个东西

这是本课最关键的结构区别。

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
└── Artifact Store
```

因此不能直接写：

```text
A2A Task == Runtime Thread
```

而应该做显式映射：

```text
A2A Task
   │
   ├── metadata.runtimeThreadId
   │
   ▼
Durable Thread
   │
   ├── submission
   ▼
WorkItem
   ▼
Turn Executor
   ▼
result / artifacts
   │
   ▼
A2A TaskStatus + Message + Artifact
```

源码：

```text
src/astra_codex/a2a_runtime_bridge.py
```

---

## 9　真实 bridge 当前怎样执行

`DurableRuntimeA2AHandler` 第一次处理 Task 时，会生成稳定映射：

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

这里 `allowed_thread_ids` 很重要：A2A worker 不能随手 claim 另一个 Thread 的 work。

---

## 10　Artifact 不应该退化成聊天文字

Runtime 中的 artifact 有：

```text
artifact_id
kind
sha256
size_bytes
metadata
```

bridge 会把新的 runtime artifact 映射成 A2A `Artifact`，并使用结构化 `data` Part 暴露：

```json
{
  "artifactId": "...",
  "kind": "patch",
  "sha256": "...",
  "sizeBytes": 1234,
  "metadata": {}
}
```

注意没有把本机绝对路径假装成远端可访问 URL。

这体现一个重要原则：

> **跨 Agent 交付的是协议 artifact，不是服务端机器内部路径。**

---

## 11　当前已经用测试证明什么

当前自动测试覆盖：

```text
A2A v1 exact TaskState spelling
Part oneof wire shape
AgentInterface protocolVersion
returnImmediately durability
SendMessage / GetTask / ListTasks / CancelTask
well-known Agent Card over real localhost HTTP
real POST /message:send round trip
GET task + historyLength
ListTasks context/state filtering
HTTP cancel
explicit pagination-not-implemented error
A2A Task → real Durable Thread → WorkItem → Turn
runtime result → A2A COMPLETED
raw media without policy → REJECTED
```

所以当前已经跨过：

```text
“我们定义了几个 A2A dataclass”
```

进入：

```text
original v1 spec
→ wire objects
→ durable Task store
→ real HTTP binding
→ local Agent OS execution
```

---

## 12　还缺什么

下一层不是继续堆字段，而是补协议里真正困难的部分：

```text
SendStreamingMessage / SSE
SubscribeToTask
TaskStatusUpdateEvent
TaskArtifactUpdateEvent
opaque pagination token
push notification configuration
extended authenticated Agent Card
A2A security schemes
HTTP tenant routing
official SDK / conformance differential tests
remote A2A → runtime bridge with server-thread-owned runtime handles
```

尤其最后一点很重要：当前 `a2a_runtime_bridge.py` 是本地同线程 reference adapter。若 HTTP server 直接调用一个在主线程构造的 bridge，会重新遇到 SQLite thread affinity。因此真正 remote runtime gateway 要像 App Server 一样，在 server thread / worker process 中拥有自己的 durable runtime handles。

---

## 13　毕业标准

读完这一课后，不应该只会说：

> A2A 是 Agent-to-Agent 协议。

而应该能从空文件开始解释并实现：

```text
Agent Card discovery
→ Message
→ Task creation
→ durable Task state
→ returnImmediately
→ polling
→ cancellation
→ Artifact delivery
→ HTTP+JSON route
→ local Agent Runtime bridge
```

并且能明确指出：

```text
Tool protocol ≠ Agent protocol
A2A Task ≠ local Thread
HTTP transport ≠ task lifecycle
authentication ≠ authorization
wire compatibility ≠ full conformance
```

这才是协议层进入 Astra-class / Codex-class 系统后的正确学习方式。
