# Lesson 24　A2A v1 SubscribeToTask：持久更新日志、SSE 与断线重放

> **本课目标**：把 A2A v1 的 `SubscribeToTask` 从一个 URL 变成真正可解释的 durable semantics。订阅者断线后，任务仍继续；重新连接时，客户端能从自己最后确认的更新位置继续读取，而不是只能“重新问一次当前状态”。

---

## 1　原始资料

固定规范快照：

```text
a2aproject/A2A
commit 6d6640c29b102f7a8d23784901351b5d2454fe71
```

原始 proto：

https://github.com/a2aproject/A2A/blob/6d6640c29b102f7a8d23784901351b5d2454fe71/specification/a2a.proto

其中 `SubscribeToTask` 的 HTTP+JSON route 是：

```text
GET /tasks/{id}:subscribe
```

RPC 返回 `stream StreamResponse`。

规范还明确说明：如果 Task 已经处于 terminal state，新的 SubscribeToTask 请求应返回 unsupported-operation 类错误，而不是假装开启一个永远不会再更新的流。

---

## 2　为什么 GetTask 不够

`GetTask` 解决：

```text
现在是什么状态？
```

Subscription 解决：

```text
从我开始观察以后，Task 怎样变化？
```

二者的语义完全不同。

如果只不断轮询：

```text
GET task
sleep
GET task
sleep
GET task
```

客户端可能只看到：

```text
SUBMITTED
→ COMPLETED
```

而错过中间所有可解释事件。

更重要的是，长任务客户端会断线。真正 durable 的系统必须把：

```text
Task update
```

和：

```text
当前有没有 subscriber 在线
```

彻底解耦。

---

## 3　错误设计：把 subscriber queue 当事实源

错误实现很容易写成：

```python
queue = asyncio.Queue()

async def update(task):
    await queue.put(task)
```

问题是：

```text
process dies
→ queue gone

subscriber disconnects
→ missed update gone

new subscriber
→ cannot replay
```

这只是 live notification，不是 durable task lifecycle。

---

## 4　本项目的 durable update journal

源码：

```text
src/astra_codex/a2a_subscription.py
```

核心表：

```text
a2a_task_updates
├─ update_id      INTEGER AUTOINCREMENT
├─ task_id
├─ task_json
└─ created_at
```

每一次外部可见 Task snapshot 都作为 append-only update 保存。

于是：

```text
Task state
      ↓
Journal append
      ↓
Subscriber online? ── no ──→ update remains durable
      │
     yes
      ↓
SSE delivery
```

---

## 5　为什么 snapshot journal 而不是只记“state changed”

教学 reference 选择保存完整 Task snapshot：

```text
update #17
Task {
  id
  contextId
  status
  history
  artifacts
  metadata
}
```

优点：

```text
replay 简单
reader 不必重放复杂 patch
schema 容易检查
```

缺点：

```text
数据重复
大 Task 成本更高
artifact/history 增长时 snapshot 变大
```

生产系统以后可以升级成：

```text
snapshot + delta log
```

或：

```text
status_update / artifact_update native journal
```

但必须先把 correctness 做清楚。

---

## 6　连续相同 snapshot 为什么要去重

如果：

```text
SUBMITTED
→ journal.append(SUBMITTED)
→ 某一层又 append 同一个 SUBMITTED
```

订阅者不应该收到两个语义完全相同的 update。

所以 `A2ATaskUpdateJournal.append()` 会比较该 Task 最新序列化 snapshot：

```text
same bytes/semantics
→ return previous update_id
→ no duplicate row
```

测试明确验证这一点。

---

## 7　JournaledA2AService

`JournaledA2AService` 在普通 `A2AService` 上增加 durable mirror：

```text
SendMessage
→ Task snapshot
→ journal

process_task
→ before snapshot
→ handler
→ final snapshot
→ journal

CancelTask
→ canceled snapshot
→ journal
```

注意：当前 handler 本身仍然是同步函数，因此服务层不会伪造不存在的 `WORKING` 中间事件。

这和项目一贯原则一致：

> **没有真实证据的 intermediate state，不制造。**

---

## 8　SubscribeToTask SSE 数据流

当前 reference server：

```text
GET /tasks/{id}:subscribe
        ↓
verify Task exists / not terminal
        ↓
HTTP 200 text/event-stream
        ↓
read journal after cursor
        ↓
SSE event
        ↓
wait for later durable rows
        ↓
terminal Task snapshot
        ↓
close stream
```

每个 SSE event 带：

```text
id: <update_id>
data: <A2A StreamResponse JSON>
```

当前 snapshot 通过：

```json
{
  "task": {...}
}
```

进入 `StreamResponse`。

---

## 9　断线之后如何恢复

项目引入 SSE 常见的：

```text
Last-Event-ID
```

作为**transport-level teaching extension**。

这个 header 不是我们声称的 A2A v1 normative request 字段；它只是 reference SSE transport 的 replay cursor。

例如客户端已经收到：

```text
id: 41
Task = SUBMITTED
```

随后断线。

任务在后台完成：

```text
id: 42
Task = COMPLETED
```

重连：

```http
GET /tasks/task_123:subscribe
Last-Event-ID: 41
```

服务端：

```text
read journal WHERE update_id > 41
→ deliver #42 only
→ terminal
→ close
```

最新回归专门覆盖这个路径。

---

## 10　为什么“已 terminal Task 的新订阅”和“断线重连”不冲突

规范要求对**新的 terminal subscription**报 unsupported operation。

所以：

```text
Task already COMPLETED
client has never subscribed
GET ...:subscribe
→ reference HTTP 409
```

但是一个已经看过 `SUBMITTED` 的 subscriber 断线后，需要补读 terminal update。

reference transport 用：

```text
Last-Event-ID > 0
```

区分这两个情况：

```text
fresh terminal subscription
≠
reconnect to existing durable stream
```

这属于本项目 transport extension 的 replay policy，不能误写成 A2A 原规范本身的字段。

---

## 11　SQLite 与跨线程

Subscriber HTTP server 和更新 Task 的 service 使用不同 SQLite connection：

```text
main/executor thread
└── JournaledA2AService
    ├── TaskStore connection A
    └── UpdateJournal connection A

subscription HTTP thread
├── TaskStore connection B
└── UpdateJournal connection B
```

共享的是数据库文件，不是 connection object。

因此真正验证了：

```text
writer and subscriber are independent runtime participants
```

而不是同一个 Python object 内部 callback。

---

## 12　当前自动化证据

Fast CPU CI run 246：

```text
187 passed, 14 skipped, 1 warning in 27.57s
Ruff correctness lint: All checks passed
```

这一轮验证：

```text
SUBMITTED snapshot durable
subscriber replays SUBMITTED
another connection completes Task
subscriber receives terminal snapshot
identical snapshots deduplicate
fresh terminal subscription → 409
```

随后新增的 reconnect regression 继续验证：

```text
subscriber observed update N
→ disconnect
→ Task reaches terminal at N+1
→ reconnect with Last-Event-ID=N
→ receive only N+1
```

---

## 13　现在还不是什么

当前实现不是 production message broker，也不是完整 A2A streaming runtime。

仍缺：

```text
executor-native WORKING events
TaskStatusUpdateEvent journal
TaskArtifactUpdateEvent journal
multi-subscriber backpressure
retention / compaction
subscriber auth / tenant isolation
push notifications
cross-process wake-up primitive
Kafka/NATS/Postgres notify equivalent
```

当前 subscriber 使用短周期 durable polling 检查 SQLite journal，目的是先验证 lifecycle/replay correctness。

---

## 14　下一步为什么是 executor-native update

目前：

```text
A2AService
→ before Task snapshot
→ synchronous handler
→ final Task snapshot
```

下一层必须让执行器自己产生：

```text
WORKING
progress message
artifact chunk
INPUT_REQUIRED
auth request
```

然后：

```text
executor event
→ durable A2A update journal
→ SendStreamingMessage
→ SubscribeToTask
→ reconnect replay
```

这样 `/message:stream` 与 `:subscribe` 才会真正共享同一个底层事实源，而不是两套相似代码。

---

## 15　毕业验收

读者应该能解释：

```text
GetTask
≠ SubscribeToTask

live queue
≠ durable journal

SSE connection
≠ task lifecycle

Last-Event-ID
= transport replay cursor in this reference
≠ claimed normative A2A field
```

并能亲手实现：

```text
append-only Task updates
→ subscription SSE
→ independent writer/subscriber connections
→ terminal close
→ disconnect
→ cursor replay
→ exactly the missing updates
```

这才算真正理解长程 Agent 的“订阅”，而不是只会打开一个 SSE socket。
