# Lesson 13　SSE 实时事件推送与断线重连：把“实时性”建立在 Durable Cursor 上

上一课已经有：

```text
DurableEventStream
→ event/poll(afterEventId)
```

这解决了**可靠重放**，但客户端仍需要主动轮询。对于 IDE、桌面端、Web UI 或观察长程 Agent 的控制台，更自然的需求是：

> **一旦 runtime 产生新事件，客户端尽快收到；断线后又不能丢事件。**

这两个目标不能混为一谈：

```text
低延迟推送
≠
可靠重放
```

本课采用 Server-Sent Events（SSE）作为第一种 push transport，但仍坚持：

> **SSE 只是 delivery layer，DurableEventStream 才是事实源。**

源码：

```text
src/astra_codex/sse_events.py
src/astra_codex/event_stream.py
src/astra_codex/control_auth.py
```

测试：

```text
tests/test_sse_events.py
```

---

## 1　为什么“直接把事件写进 socket”是不够的

如果 runtime 只做：

```text
new event
→ websocket.send(event)
```

客户端断线期间：

```text
event 101
event 102
event 103
```

可能全部丢失。

重新连接后，如果服务器没有 durable history，客户端只能知道：

```text
现在在线
```

却不知道：

```text
刚才漏了什么？
```

因此正确结构应是：

```text
Runtime Event
    ↓
DurableEventStream
    ↓
monotonic event_id
    ↓
SSE projection
```

SSE 连接消失时，数据库中的 event 仍然存在。

---

## 2　Cursor 是 reconnect contract

每个 runtime event 都有：

```text
event_id = 1, 2, 3, ...
```

客户端保存：

```text
last_seen = 42
```

重新连接可以发送：

```text
Last-Event-ID: 42
```

或者：

```text
GET /events?afterEventId=42
```

服务器只投递：

```text
43, 44, 45, ...
```

于是：

```text
push latency
+
replay durability
```

可以同时成立。

---

## 3　SSE wire format

当前 reference server 按标准 SSE 文本帧输出：

```text
id: 43
event: harness.tool_completed
data: {"eventId":43,"topic":"harness.tool_completed",...}

```

三个字段分别承担：

```text
id     → reconnect cursor
event  → runtime topic
data   → typed JSON payload
```

客户端不需要从 JSON 内部重新推导 topic/order，但 JSON 中仍保留完整 event metadata，方便日志落盘或跨 transport 转换。

---

## 4　为什么使用 `Last-Event-ID` 仍不能取代 durable store

`Last-Event-ID` 只是：

```text
客户端告诉服务器：我看到哪里了
```

它本身并不保存历史。

所以：

```text
Last-Event-ID
      ↓
DurableEventStream.read(after_id=...)
```

才构成完整 reconnect semantics。

如果服务器重启后 event DB 消失，即使客户端带着 `Last-Event-ID` 也无济于事。

---

## 5　Thread/topic filter 在 delivery 前执行

SSE endpoint 支持：

```text
threadId=thr_a
topic=harness.tool_started
topic=harness.tool_completed
```

真正查询发生在 SQLite：

```text
WHERE event_id > cursor
  AND thread_id = ...
  AND topic IN (...)
```

而不是先取全部事件再在客户端过滤。

这有两个意义：

1. 减少无关数据传输；
2. 对 thread-scoped auth 来说，避免越权事件先进入 transport 再丢弃。

---

## 6　SSE 复用同一套 Principal / AuthZ

当前 SSE server 不重新发明权限模型。

它继续使用：

```text
Authorization: Bearer <token>
        ↓
BearerTokenAuthorizer
        ↓
Principal
        ↓
method = event/poll
threadId = ...
        ↓
Authorization
```

因此一个只能访问：

```text
thr_a
```

的 principal：

```text
/events?threadId=thr_a    ✓
/events?threadId=thr_b    ✗
/events                    ✗
```

其中最后一种“不给 thread filter 看全局 feed”同样被拒绝。

这保证：

```text
poll transport
SSE transport
```

共享一套 resource policy，而不是两套逐渐漂移的安全逻辑。

---

## 7　为什么每个 SSE handler 自己打开 SQLite handle

当前 server 使用：

```text
ThreadingHTTPServer
```

每个连接可能在不同线程执行。

SQLite connection 默认有 thread affinity，所以错误做法是：

```text
main thread opens connection
→ hand same connection to all HTTP threads
```

reference implementation 的做法是：

```text
SSE request thread
→ DurableEventStream(event_db_path)
→ own SQLite connection
→ read cursor
→ close connection
```

这样避免通过 `check_same_thread=False` 掩盖并发边界。

---

## 8　Live delivery 的测试比“读取已有事件”更重要

仅测试：

```text
先写 event
再 GET /events
```

只能证明 catch-up replay。

我们还需要证明：

```text
client opens SSE
→ currently no event
→ handler waits/polls durable store
→ runtime later appends new event
→ open connection receives it
```

这才说明它确实具有 push-like live behavior。

因此测试用两个线程构造：

```text
reader thread: SSE wait
main thread:   append event later
```

并要求 reader 在 timeout 前拿到新 event。

---

## 9　为什么当前 reference stream 是有限的

真实 SSE 通常可以长时间保持：

```text
while connected:
    keep sending events
```

教学/CI 环境如果这样实现，很容易出现：

```text
测试永久阻塞
cleanup 不确定
GitHub Actions hang
```

所以当前 endpoint 明确接受：

```text
maxEvents
timeoutSeconds
```

达到条件后以 SSE comment：

```text
: stream-end
```

干净关闭连接。

这不是 production policy，只是可验证的 reference transport contract。

---

## 10　SSE 与 WebSocket 的职责差异

当前 Agent 事件流主要是：

```text
server → client
```

因此 SSE 已经很适合：

```text
runtime progress
model output event
tool start/end
artifact event
work finish
```

而双向低延迟交互，例如：

```text
live computer control
bidirectional voice
high-frequency interactive channel
```

更适合 WebSocket 或专用 realtime transport。

项目路线因此不是：

```text
SSE vs WebSocket 二选一
```

而是：

```text
Durable Event Store
      ↓
SSE for one-way event subscriptions
WebSocket/realtime for bidirectional control
      ↓
all reconnect through durable cursors where applicable
```

---

## 11　当前明确没有解决的事情

这一实现仍然不是 production streaming service。未解决：

```text
TLS
proxy buffering configuration
slow consumer backpressure
per-client retention cursor
server-side subscription registry
event retention / compaction
horizontal fan-out
multi-process notification
WebSocket
rate limiting
session expiry / rotation
```

尤其当前 SQLite poll loop 是 reference implementation，不应该被宣传成高吞吐 pub/sub broker。

---

## 12　验收标准

这一层只有在以下条件都被测试后才成立：

```text
cursor catch-up replay                 ✓
Last-Event-ID reconnect                ✓
no duplicate after reconnect           ✓
thread filter                          ✓
topic filter                           ✓
live event arriving after connect      ✓
missing bearer → 401                   ✓
wrong thread → 403                     ✓
unscoped feed for scoped principal →403✓
```

这使 Agent OS 的外部可观察性从：

```text
request → poll result
```

推进为：

```text
request
  ↓
durable execution
  ↓
replayable real-time event stream
  ↓
IDE / Web UI / Desktop Client
```

下一步不应该继续堆 transport 方法，而应继续强化两个更底层边界：**session/token lifecycle** 与 **真正 OS/container sandbox**。
