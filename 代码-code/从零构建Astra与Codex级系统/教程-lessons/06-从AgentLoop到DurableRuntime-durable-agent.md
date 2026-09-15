# Lab 06：从 Agent Loop 到 Durable Runtime

> **目标**：理解为什么 `while model -> tool` 不是长程智能体系统，以及 Thread、Turn、Event、Checkpoint、Lease、Fork、Context Provenance 分别解决什么问题。

对应源码：

- [`../src/astra_codex/codex_harness.py`](../src/astra_codex/codex_harness.py)
- [`../src/astra_codex/durable.py`](../src/astra_codex/durable.py)
- [`../src/astra_codex/runtime_queue.py`](../src/astra_codex/runtime_queue.py)
- [`../src/astra_codex/context.py`](../src/astra_codex/context.py)
- [`../src/astra_codex/security.py`](../src/astra_codex/security.py)
- [`../src/astra_codex/benchmark.py`](../src/astra_codex/benchmark.py)

理论：

- [`../../../智能体-agent/04-记忆与上下文-memory-context.md`](../../../智能体-agent/04-记忆与上下文-memory-context.md)
- [`../../../Codex源码解剖-codex-anatomy/02-会话线程与事件协议-session-thread-events.md`](../../../Codex源码解剖-codex-anatomy/02-会话线程与事件协议-session-thread-events.md)
- [`../../../Codex源码解剖-codex-anatomy/04-指令上下文压缩与记忆-context-memory.md`](../../../Codex源码解剖-codex-anatomy/04-指令上下文压缩与记忆-context-memory.md)

官方源码入口： https://github.com/openai/codex

---

# 1　最小 Agent Loop 为什么不够

最小 loop：

```text
messages
   ↓
model
   ↓
tool call?
├─ no  → final
└─ yes → execute → observation → messages → repeat
```

这个 loop 足够解释 tool use，却隐含了一个非常强的假设：

> **整个任务期间，当前 Python 进程、内存对象、上下文历史和 tool session 都一直活着。**

真实长任务会遇到：

```text
process crash
machine restart
worker preemption
user pause
user steering
context overflow
timeout
parallel subtask
retry
cancel
```

所以 Agent 必须从函数升级成 runtime。

---

# 2　先区分 Thread 与 Turn

本项目采用：

```text
Thread
├─ Submission 1
│    └─ Turn 1
├─ Submission 2
│    └─ Turn 2
└─ ...
```

**Thread** 是长期工作对象。  
**Turn** 是一次 active execution slice。

一个 thread 可以跨越多个模型上下文窗口、多个进程甚至多个 worker。

因此：

```text
thread lifetime != model call lifetime
thread lifetime != process lifetime
turn lifetime   != token generation lifetime
```

---

# 3　Event Sourcing：状态来自历史，而不是 Python object

`durable.py` 不把：

```python
thread.status = "running"
```

当作唯一真相。

真正 source of truth 是 append-only event log：

```text
THREAD_CREATED
USER_SUBMISSION
TURN_STARTED
CHECKPOINT_CREATED
TURN_COMPLETED
THREAD_PAUSED
THREAD_RESUMED
THREAD_CANCELLED
...
```

当前状态是一个 projection：

$$
S_n=F(e_1,e_2,\ldots,e_n).
$$

其中 $e_i$ 是事件。

所以进程重启以后，只需要：

```text
open database
→ read events in order
→ replay
→ reconstruct projection
```

而不是依赖旧进程内存。

---

# 4　Checkpoint 与 Event Log 为什么都要有

Event log：

```text
发生过什么？
```

Checkpoint：

```text
执行器在某个位置需要恢复的显式工作状态是什么？
```

例如：

```json
{
  "phase": "tests-running",
  "attempt": 3,
  "target": "parser.py"
}
```

当前 reference 实现仍会 replay event log；未来长 trajectory 可以：

```text
checkpoint snapshot
+
events after checkpoint
→ recovery
```

减少恢复成本。

---

# 5　Fork：长程 Agent 为什么需要分支

假设 Agent 已经完成：

```text
inspect repo
→ identify two plausible fixes
```

我们不一定希望：

```text
在同一个 thread 中把 A 方案覆盖成 B 方案
```

而可以：

```text
parent thread
     │
     ├─ fork A → strategy A
     └─ fork B → strategy B
```

当前 `fork_thread()`：

1. 选择 source event prefix；
2. 建立新的 child thread；
3. 保存 `parent_thread_id` / `parent_event_id`；
4. 把该 prefix 复制成独立 child event stream；
5. 后续 parent / child 独立演化。

自动测试明确验证：取消 child 不会取消 parent。

这只是事件历史 fork；真实 coding agent 后续还要把它和 Git worktree / artifact store 结合。

---

# 6　Cancellation 必须是一等状态

长任务中：

```text
用户说 stop
```

不能只在下一次 prompt 里追加一句自然语言。

当前：

```text
THREAD_CANCELLED
```

进入 terminal state：

```text
submit      rejected
checkpoint  rejected
```

后续 worker / tool execution 还要继续把 cancellation signal 接到进程和远程任务层。

---

# 7　为什么还需要 Work Queue

Thread 表示：

> “任务是什么状态？”

Queue 表示：

> “哪一个 worker 现在拥有执行权？”

这两个问题不能混在一起。

当前：

```text
work item
→ PENDING
→ worker A claim
→ LEASED
```

lease 在时间：

$$
t_{lease}
$$

之后到期。

如果 worker A crash：

```text
lease expires
→ worker B reclaim
```

而不是永久卡死。

测试固定验证：

```text
active lease   → worker B cannot claim
expired lease  → worker B can reclaim
```

这才开始接近可靠 long-running runtime。

---

# 8　为什么 Lease 比简单 `status=running` 强

如果数据库只存：

```text
status = RUNNING
```

worker crash 后没有任何机制判断：

> 它是在运行，还是死了？

Lease 增加：

```text
lease_owner
lease_until
```

从而允许 runtime 判断 ownership 是否过期。

未来还需要 heartbeat：

```text
worker alive
→ renew lease
```

以及 idempotent execution，防止 lease race 导致副作用重复发生。

---

# 9　Context 不能等同于 Event Log

Event log 可以非常长，而且包含：

```text
terminal dumps
raw HTML
stack traces
failed attempts
old plans
obsolete observations
```

全部放入模型 context 会造成：

```text
cost ↑
noise ↑
relevant signal density ↓
```

因此我们显式区分：

```text
Durable Event Store
      ↓
Context Builder
      ↓
Model-visible Context
```

`context.py` 的 fragment kinds：

```text
RAW_EVENT
NOTE
SUMMARY
ARTIFACT
RETRIEVAL
INSTRUCTION
```

---

# 10　Compaction 最危险的错误：摘要取代证据

错误设计：

```text
raw history
→ summary
→ delete raw history
```

因为摘要可能遗漏、扭曲或过时。

当前设计：

```text
raw A ─┐
raw B ─┼→ summary S
raw C ─┘      │
              └─ parent_ids = [A,B,C]
```

所以：

$$
\text{summary}\neq\text{evidence}.
$$

Summary 是新的派生 artifact，而不是原历史。

自动测试会沿 lineage 找回原始 raw observations。

---

# 11　Permission 与 Sandbox 是不同层

当前 `security.py` 可以做到：

```text
proposal
→ DENY
→ tool body never executes
```

这是应用层 enforcement。

但如果允许 shell：

```text
ALLOW shell
```

当前 shell 仍然运行在宿主执行环境中。

所以：

```text
Permission Gate
!=
OS Sandbox
```

未来必须在实际执行边界继续加入：

```text
process isolation
filesystem namespace
network policy
credential scope
resource limits
audit log
```

---

# 12　Durability 必须被 benchmark，而不只是 unit test

未来一个真正的 resilience case 应该是：

```text
start task
→ execute 20 steps
→ kill worker
→ restart runtime
→ replay/checkpoint recovery
→ continue task
→ verify final artifact
```

然后与没有 durability 的 baseline 比：

```text
completion rate
recovery latency
duplicate side effects
lost work
human intervention
```

这才是“长程可靠性”证据。

---

# 13　运行当前测试

```bash
cd '代码-code/从零构建Astra与Codex级系统'
pip install -e '.[dev]'
pytest tests/test_durable_security_evaluation.py \
       tests/test_v3_p0_runtime.py -q
```

当前这些 tests 覆盖 replay、fork、cancel、lease/reclaim、context provenance、permission gate。

下一阶段把这些 primitive 真正连接成：

```text
Submission
→ Thread
→ Work Queue
→ Worker
→ Turn Executor
→ Events
→ Context Builder
→ Action
→ Checkpoint
```

届时项目才从“多个 Agent 模块”进一步跨到一个完整的 **Agent Runtime**。
