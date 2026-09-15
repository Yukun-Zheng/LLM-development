# Lesson 10　可重放事件流与 HTTP 控制面：让 Agent Runtime 真正能被外部系统驱动

这一课继续解决一个很容易被“Agent demo”忽略的问题：**一个长程 Agent 不应该要求 UI、CLI、IDE 与执行内核运行在同一个 Python 进程里。**

如果执行系统已经拥有 Thread、Turn、Work Queue、checkpoint、steering 与 artifact，那么下一个自然问题是：

> 外部客户端如何可靠地观察正在发生什么，并且在断线重连后继续从正确位置读取事件？

本课对应源码：

```text
src/astra_codex/event_stream.py
src/astra_codex/app_server.py
src/astra_codex/http_app_server.py
src/astra_codex/runtime.py
src/astra_codex/codex_harness.py
```

自动测试：

```text
tests/test_event_stream_http.py
```

---

## 1　为什么不能把“打印日志”当 Event Stream

最简单的 Agent demo 会这样做：

```text
print("tool started")
print("tool finished")
```

这只能服务当前进程的当前观察者。只要出现：

```text
IDE reconnect
browser refresh
client crash
worker restart
network interruption
```

客户端就不知道自己漏掉了哪些事件。

因此需要给每条 runtime event 一个稳定的单调 cursor：

```text
1  thread.created
2  thread.submitted
3  work.claimed
4  turn.opened
5  harness.turn_started
6  work.heartbeat
7  harness.model_output
8  harness.tool_started
9  harness.tool_completed
10 harness.model_output
11 harness.turn_completed
12 work.finished
```

客户端只需要记住：

```text
last_seen_event_id = 8
```

重新连接以后请求：

```text
afterEventId = 8
```

就能得到 9 之后的全部事件。

---

## 2　Thread Event Log 和 Runtime Event Feed 不是同一个东西

这是本课最重要的系统边界之一。

`DurableThreadStore` 的事件日志用于：

```text
replay
→ reconstruct authoritative Thread state
```

它是状态机事实源。

`DurableEventStream` 则用于：

```text
runtime activity
→ UI / IDE / observability / remote control client
```

它是**控制面 feed**。

二者虽然都 append-only，但不能混为一谈：

```text
Thread Event Log
= state reconstruction contract

Runtime Event Stream
= integration / observation contract
```

否则未来为了 UI 增加一个 event，很可能意外改变 Thread projection 的状态语义。

---

## 3　Cursor replay

当前 `event_stream.py` 使用 SQLite：

```text
runtime_events(
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic,
    payload_json,
    thread_id,
    turn_id,
    created_at
)
```

查询形式为：

```text
read(
    after_id,
    limit,
    optional thread_id,
    optional topics
)
```

其中 `event_id` 同时承担：

```text
ordering key
reconnect cursor
pagination cursor
high-watermark reference
```

因此一个分页客户端可以执行：

```text
cursor = 0

loop:
    page = event/poll(afterEventId=cursor)
    process(page.events)
    cursor = page.nextAfterEventId
```

这与简单的 `OFFSET 100` 不同：当新事件不断加入时，基于稳定 event id 的增量读取不会因为前面集合增长而发生分页漂移。

---

## 4　为什么先做 cursor-poll，再做 WebSocket

WebSocket 解决的是：

```text
push latency
```

但并不自动解决：

```text
replay
resume
ordering
missed events
cursor durability
```

所以正确的分层顺序应更接近：

```text
Durable Event Store
        ↓
Cursor Replay Contract
        ↓
Polling API
        ↓
SSE / WebSocket live push
        ↓
Reconnect with cursor
```

即使以后有 WebSocket，断线恢复仍应该回到 durable cursor。

本项目因此先实现：

```text
event/poll
```

而不是一上来就把 socket 当作系统事实源。

---

## 5　Turn Executor 需要实时向外发事件

以前 `CodexHarness` 是：

```text
run_turn()
    ↓
结束后返回 events[]
```

这意味着一个执行 30 分钟的 Turn，外部系统可能 30 分钟都只能等待最终结果。

现在增加：

```text
HarnessEventSink
```

内部每产生一个：

```text
TURN_STARTED
MODEL_OUTPUT
APPROVAL_REQUESTED
TOOL_STARTED
TOOL_COMPLETED
TURN_COMPLETED
TURN_STOPPED
```

都会立即调用 sink。

`DurableAgentRuntime` 再把它转换成：

```text
harness.turn_started
harness.model_output
harness.tool_started
...
```

写入 durable event stream。

这一步把：

```text
Turn return value
```

扩展成：

```text
Turn live event stream
```

但 Turn 的算法本身仍不知道 UI 是否存在。

---

## 6　App Server 的角色

现在控制面是：

```text
Client
  ↓ JSON-RPC
AgentAppServer
  ↓
DurableAgentRuntime
```

App Server 目前暴露：

```text
thread/create
thread/get
thread/submit
thread/steer
thread/pause
thread/resume
thread/cancel
thread/fork
runtime/runOne
artifact/list
event/poll
```

因此 GUI 不需要直接 import：

```python
DurableThreadStore
DurableWorkQueue
DurableToolJournal
```

它只依赖一个控制面协议。

这也是为什么 App Server 不应被写进 `agent.py`：它属于**外部控制平面**，不是模型 policy 本身。

---

## 7　真正的 HTTP transport

`http_app_server.py` 使用 Python 标准库提供最小 loopback HTTP：

```text
POST /rpc
Content-Type: application/json
```

客户端：

```text
AgentAppClient
    ↓
HTTPAppTransport
    ↓ HTTP
LocalHTTPAppServer
    ↓
AgentAppServer
```

测试已经不再只做：

```text
Python object → Python object
```

而是让 JSON 真正经过：

```text
serialize
→ localhost TCP/HTTP
→ parse
→ runtime
→ serialize response
→ HTTP client
```

这意味着以后 IDE extension、桌面端或其他进程已经有明确的接入边界。

---

## 8　为什么现在还不能叫 Production App Server

当前 HTTP 层故意只绑定：

```text
127.0.0.1
```

而且明确没有：

```text
TLS
Authentication
Session Authorization
Multi-tenant ACL
CSRF / Origin policy
rate limiting
WebSocket / SSE
backpressure
request cancellation transport
```

所以当前语义是：

> **Local educational external control plane**

而不是：

> production internet service。

如果没有 auth / ACL 就直接监听 `0.0.0.0`，Agent 的 shell/filesystem 能力会把普通 Web API 风险放大很多倍。

---

## 9　接下来为什么是 Auth 与 Streaming，而不是更多 RPC 方法

此时继续增加：

```text
thread/foo
thread/bar
```

边际价值已经变低。

下一层应该是：

```text
HTTP Control Plane
        ↓
Authentication
        ↓
Principal / Session
        ↓
Thread / Tool Authorization
        ↓
SSE / WebSocket
        ↓
Cursor reconnect
        ↓
backpressure / cancellation
```

同时安全执行层还必须继续向下：

```text
Approval
≠ Sandbox

Auth
≠ Sandbox

App Server permission
≠ OS isolation
```

它们分别解决不同边界。

---

## 10　验收标准

这一层只有同时满足以下条件才算完成第一版：

```text
1. events append-only 持久化
2. process restart 后 cursor replay 仍成立
3. thread/topic filtering 可工作
4. Harness event 在执行时写入 feed
5. App Server 可通过 event/poll 分页读取
6. HTTP round-trip 真正经过本地网络栈
7. HTTP client 能 create → submit → run → poll → inspect
8. 明确声明当前无 auth / TLS / WebSocket
```

这进一步把系统从：

```text
可恢复 Agent 内核
```

推进到：

```text
可被外部客户端可靠观察与控制的 Agent Runtime
```

下一课应进入 **Authentication / Session Authorization / SSE-WebSocket reconnect**，然后再把权限边界继续压到真正的 OS/container sandbox。
