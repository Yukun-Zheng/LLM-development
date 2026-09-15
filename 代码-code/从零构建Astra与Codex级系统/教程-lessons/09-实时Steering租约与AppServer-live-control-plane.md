# Lesson 09：实时 Steering、Worker Lease 与 App Server——长程 Agent 为什么需要控制面

> 本节不是教“再写一个 prompt”。目标是把长程 Agent 从一个阻塞函数，推进成**可被外部用户持续控制、可恢复、可观察的软件运行时**。

源码对应：

```text
src/astra_codex/steering.py
src/astra_codex/runtime_control.py
src/astra_codex/artifacts.py
src/astra_codex/app_server.py
src/astra_codex/runtime.py
```

自动测试：

```text
tests/test_runtime_control_artifacts.py
tests/test_app_server.py
```

---

# 1　为什么 `run(goal)` 不足以构成长期 Agent

最小 Agent 常写成：

```text
run(goal)
  ↓
model
  ↓
tool
  ↓
model
  ↓
finish
```

只要任务持续几十秒，这种结构就开始暴露问题：

1. 用户中途改变要求怎么办？
2. worker 长时间执行时，其他 worker 如何知道它还活着？
3. 进程崩溃后任务属于谁？
4. GUI / CLI / IDE 怎样在不侵入核心 runtime 的情况下控制线程？
5. Agent 产生的文件、patch、报告怎样形成可独立验证的证据？

因此要引入**数据面（data plane）与控制面（control plane）分离**。

```text
                       Control Plane
              UI / CLI / IDE / Remote Client
                           │
                  JSON-RPC App Server
                           │
       create / submit / steer / cancel / fork / inspect
                           │
                           ▼
                     Agent Runtime
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
         Data Plane                 Durable State
 model ↔ tools ↔ environment    thread / queue / journal
```

---

# 2　Lease：worker 不是“拿到任务就永远拥有任务”

分布式系统里，worker 可能：

- 崩溃；
- 机器断电；
- 网络分区；
- 卡在外部 RPC；
- 被调度系统杀死。

所以 Work Queue 不应该记录：

```text
owner = worker_A forever
```

而应该记录：

```text
lease_owner = worker_A
lease_until = t_expire
```

若：

$$
t > t_{expire},
$$

其他 worker 才可以 reclaim。

我们的 `LeaseHeartbeat` 做的是：

$$
t_{expire}\leftarrow t_{now}+\Delta_{lease}.
$$

因此正常 worker 要周期性续约，才能证明它仍然拥有这个 work item。

## 当前边界

当前 reference implementation 在**每次 model sampling 前**续约。

这能覆盖：

```text
sampling 1
→ heartbeat
→ tool
→ sampling 2
→ heartbeat
```

但如果一个单独 tool call 持续时间已经超过 lease，本实现仍可能过期。因此 production 版本还需要：

```text
background heartbeat task
or
long-running tool owns its own lease/heartbeat
```

不能因为“有 `LeaseHeartbeat` class”就声称已经解决全部 worker liveness。

---

# 3　Live Steering：用户为什么必须能在 Turn 中途改变方向

长期 Agent 的交互不是：

```text
user prompt
→ several hours
→ final answer
```

现实更像：

```text
User: 修这个 bug
Agent: inspect...
Agent: run tests...
User: 先不要改核心模块，保持 API 兼容
Agent: ← 下一决策必须看到这个新约束
```

所以 steering 必须拥有自己的 durable inbox：

```text
PENDING
→ CONSUMED
or
→ CANCELLED
```

当前 `DurableSteeringQueue` 使用 SQLite 保存消息。

关键时序：

```text
model step 1
    ↓
tool call begins
    ↓
user submits steering
    ↓
tool returns
    ↓
ControlledBackend before step 2
    ├─ heartbeat
    ├─ consume pending steering
    └─ append steering to transcript
    ↓
model step 2 sees new requirement
```

自动测试专门模拟 steering **在 tool body 执行期间到达**，并验证第二次 model sampling 确实看到它。

---

# 4　为什么 Steering 不能只是改内存里的 `messages`

如果只是：

```python
messages.append(new_user_message)
```

进程崩溃后会发生：

```text
steering 是否收到？
是否已经消费？
是否重复注入？
```

都无法回答。

所以消息先进入 durable queue，再由 executor 原子地：

```text
PENDING → CONSUMED
```

之后才加入模型 transcript。

这就是一个典型的软件系统原则：

> **先持久化事实，再驱动不可逆状态变化。**

下一阶段还要把 steering consumption 镜像进 Thread event stream，使完整历史可以从单一 event ledger 重建。

---

# 5　Artifact：Agent 的最终工作产品不是聊天文本

Coding / Research / Office Agent 可能产生：

```text
patch
diff
source file
PDF
spreadsheet
report
image
test log
benchmark result
```

因此：

```text
assistant: "done"
```

不能作为任务完成证据。

`ArtifactStore` 采用 content-addressed snapshot：

$$
h=\operatorname{SHA256}(bytes).
$$

对象路径由 hash 决定，相同字节可以共享底层 object，但每次产物仍有独立 `artifact_id` 与 metadata。

```text
ArtifactRecord
├─ artifact_id
├─ thread_id
├─ kind
├─ sha256
├─ size_bytes
├─ metadata
└─ created_at
```

读取时重新计算 SHA-256；如果磁盘内容被修改，校验直接失败。

这使 verifier/reviewer 能检查：

> **Agent 到底生成了哪些 bytes？**

而不是只相信它的自然语言描述。

---

# 6　App Server：为什么 Agent Kernel 不应该直接等于 CLI

若把 CLI 输入、终端渲染、Agent 状态机、工具执行全写在一起，那么：

```text
TUI
IDE
Web
mobile
remote orchestration
```

每增加一个客户端都要重写 runtime。

正确的边界是：

```text
Client
  ↓
Protocol
  ↓
Agent Runtime
```

当前 `AgentAppServer` 提供一个最小 JSON-RPC 控制面：

```text
server/discover
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
```

这不是 OpenAI Codex App Server 的 API-compatible 复刻；它是根据公开 runtime 分层思想写出的教学 reference control plane。

当前 transport 只有：

```text
InProcessAppTransport
```

后续才进入：

```text
stdio
HTTP / WebSocket
streaming events
authentication
multi-tenant permissions
subscriptions
```

---

# 7　从这里开始，Agent 更像一个操作系统进程模型

此时状态已经不再只有 prompt：

```text
Thread
├─ submissions
├─ turns
├─ checkpoints
├─ steering inbox
├─ work leases
├─ tool journal
├─ context provenance
├─ artifacts
└─ final state
```

控制路径：

```text
External Client
      ↓
App Server
      ↓
Thread / Work Queue
      ↓
Worker lease + heartbeat
      ↓
Turn Executor
      ↓
Tools / Environment
      ↓
Artifact / Observation
      ↓
Verifier
```

到这里才开始接近“Agent Runtime / Agent OS”，而不是“大模型套一个 while-loop”。

---

# 8　必须记住的三个边界

第一，**heartbeat 不是分布式共识**。SQLite 单机 reference runtime 与多机 production runtime 仍有巨大差距。

第二，**permission gate 不是 sandbox**。即使 App Server 控制谁能请求某个 tool，真正 shell / process / network / secret isolation 仍需 OS/container boundary。

第三，**artifact checksum 不是 semantic correctness**。SHA-256 只能证明 bytes 未被篡改；它不能证明报告结论正确。语义正确性仍由 verifier / grader / human review 判断。

---

# 9　验收实验

本节完成后，应能自己构造并解释四个实验：

```text
Experiment A
worker_A lease=5s
→ t=4 heartbeat renew to t=14
→ worker_B at t=6 cannot reclaim
→ worker_B at t=15 can reclaim

Experiment B
model step 1 calls tool
→ steering arrives while tool runs
→ model step 2 receives steering

Experiment C
same bytes saved twice
→ two artifact ids
→ same content-addressed object
→ tamper object
→ checksum verification fails

Experiment D
external client
→ thread/create
→ thread/submit
→ runtime/runOne
→ thread/get
→ result reconstructed from durable runtime
```

如果这些状态变化都能解释清楚，那么你掌握的已经不是 Agent prompt engineering，而是**长程智能体的软件系统基础**。
