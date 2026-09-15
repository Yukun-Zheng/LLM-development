# Lesson 21：持久 AgentGraph、Mailbox 与真正并行的 Worker Runtime

> 多智能体系统最容易掉进的陷阱，是把“给三个 LLM 起三个角色名”误认为实现了 Multi-Agent。本课从系统第一性原理出发：如果 Agent 会并行工作数小时、会崩溃、会互相发任务、会产生独立 artifact，那么**拓扑、身份、生命周期、通信所有权与任务并发都必须成为持久系统状态**。

---

## 1　Multi-Agent 不是角色扮演

最弱的“多智能体”通常长这样：

```text
Planner prompt
Coder prompt
Reviewer prompt
```

然后在一个 Python list 里轮流调用模型。

它可以用于教学 prompt pattern，但回答不了真正运行时问题：

```text
谁创建了谁？
worker 进程死后，它正在处理的消息归谁？
coordinator 取消一个子树时，孙 Agent 怎么办？
一个 worker 会不会同时被分配两个互相冲突的任务？
worker 的结果来自哪个 task / mailbox message？
进程重启后 Agent topology 是否还在？
```

所以我们把 Multi-Agent 拆成三个层次：

```text
AgentGraph
    = 谁存在、谁属于谁、当前生命周期是什么

Mailbox
    = Agent 之间可靠地传什么消息、谁拥有消费权

Parallel Scheduler
    = 哪些任务此刻可以真正并发执行
```

---

## 2　AgentGraph 是显式数据结构

源码：

```text
src/astra_codex/agent_graph.py
```

一个 Agent 节点现在至少包含：

```text
AgentNode
├─ agent_id
├─ thread_id
├─ role
├─ parent_agent_id
├─ status
├─ metadata
├─ created_at
└─ updated_at
```

`role` 可以是：

```text
coordinator
worker
reviewer
merge-agent
researcher
browser-worker
```

但 role 只是一段语义 metadata；真正决定系统行为的是生命周期与权限，而不是名字。

---

## 3　生命周期为什么不能只放内存

当前状态：

```text
ACTIVE
BUSY
WAITING
COMPLETED
FAILED
CANCELLED
```

允许的非终态转换例如：

```text
ACTIVE → BUSY
BUSY   → ACTIVE
BUSY   → WAITING
WAITING→ BUSY
```

以及：

```text
ACTIVE / BUSY / WAITING
        ↓
COMPLETED / FAILED / CANCELLED
```

而终态不能重新“复活”：

```text
COMPLETED → ACTIVE   ✗
FAILED    → BUSY     ✗
CANCELLED → ACTIVE   ✗
```

如果未来真的需要 retry，应创建新的 attempt / child execution identity，而不是偷偷篡改旧 Agent 的历史语义。

所有状态都写入 SQLite，并配套 append-only `agent_events`。因此：

```text
process dies
   ↓
reopen agents.sqlite
   ↓
agent identity / parent / status / mailbox still exist
```

---

## 4　父子关系为什么重要

当 coordinator spawn worker：

```text
coordinator
├─ worker-A
├─ worker-B
└─ reviewer
```

我们保存：

```text
parent_agent_id
```

于是系统可以回答：

```text
children(agent)
descendants(agent)
lineage(agent)
```

例如：

```text
root
└─ coding-worker
   └─ test-debugger
```

`lineage(test-debugger)` 给出：

```text
root → coding-worker → test-debugger
```

这以后会直接进入 Rollout Trace：一个 artifact 不只知道来自哪个 Thread，还应知道**是谁的哪条 Agent lineage 产生的**。

---

## 5　为什么当前禁止 Cross-Thread Mail

Reference runtime 当前要求：

```text
sender.thread_id == recipient.thread_id
```

否则消息被拒绝。

这不是说未来 Agent 永远不能跨任务通信，而是为了先建立最小清晰边界：

```text
Thread = 一个 durable long-running task/session namespace
AgentGraph = Thread 内部的执行拓扑
```

真正跨 Thread / 跨服务通信应走更显式的协议：

```text
A2A
remote task handoff
external message bus
```

而不是让本地 mailbox 偷偷打破隔离。

---

## 6　Mailbox 为什么也需要 Lease

消息不是普通 Python queue item。

假设：

```text
coordinator
    ↓ task.assign
worker-A process
    ↓ 已经取走消息
process crashes
```

如果“取走”就等于“删除”，任务永久丢失。

所以 mailbox 使用：

```text
PENDING
   ↓ claim
LEASED(worker-process)
   ├─ ACK → ACKED
   ├─ release → PENDING
   └─ lease expires → another process reclaim
```

与主 `DurableWorkQueue` 一样，这是典型的 at-least-once substrate。

它解决的是：

> **所有权失效后任务能够重新变得可领取。**

它没有自动解决：

> **外部副作用 exactly-once。**

后者仍然要依赖 `DurableToolJournal`、idempotency key、reconciliation 等机制。

---

## 7　ACK 必须检查 Lease Owner

如果：

```text
process-A lease expires
        ↓
process-B reclaims message
        ↓
process-A 晚到一步说“我完成了”
```

不能允许 A 覆盖 B 的当前所有权。

所以：

```python
ack_message(message_id, worker_id)
```

会检查：

```text
status == LEASED
lease_owner == worker_id
```

否则：

```text
PermissionError
```

这是分布式执行中非常基础的 stale owner 问题。

当前 mailbox 还没有完整 fencing token；未来跨机器执行时，应进一步使用 monotonically increasing lease generation / execution epoch，而不能只依赖字符串 worker ID。

---

## 8　Subtree Cancellation

如果 coordinator 判断一个研究分支不再需要：

```text
worker-A
└─ debugger-A1
```

只 cancel `worker-A` 而让 A1 继续跑会形成 orphan work。

所以当前：

```python
cancel_subtree(worker_A)
```

按：

```text
leaf descendants
      ↓
parent
```

顺序标记取消。

这样至少不会出现“父 Agent 已终止，但子 Agent 仍被系统认为 ACTIVE”的短暂逻辑状态。

注意：状态取消不等价于已经终止任意外部 OS process。真正 process/container cancellation 仍需环境执行层配合。

---

# 9　从持久 Graph 到真正并行

源码：

```text
src/astra_codex/parallel_agents.py
```

核心对象：

```text
ParallelTask
ParallelTaskResult
ParallelAgentCoordinator
```

最小调度器执行：

```text
Pending Tasks
   │
   ├─────────────┐
   ▼             ▼
worker-A       worker-B
   │             │
 BUSY           BUSY
   │             │
 task-a          task-b
   │             │
   └──────┬──────┘
          ▼
      Results
```

而不是：

```text
worker-A task-a 完成
        ↓
worker-B task-b 才开始
```

自动测试使用 `threading.Barrier`：两个 worker 必须同时进入 barrier 才能继续。如果调度器是假并行、其实按顺序调用，第一个 worker 会直接超时。因此这个测试验证的不是“代码里写了 ThreadPoolExecutor”，而是**两个 worker callback 确实发生 wall-clock overlap**。

---

## 10　为什么 SQLite 仍然只在 Coordinator Thread 使用

Python 的 thread 并发很容易制造一个误区：

> “既然 worker callback 在不同 thread，那就把同一个 SQLite connection 一起传进去。”

这是错误的系统边界。

我们的 reference scheduler 刻意设计为：

```text
Coordinator Thread
├─ AgentGraph DB
├─ mailbox send
├─ mailbox claim
├─ status updates
└─ ACK

Worker Thread
└─ receives immutable AgentNode + AgentMessage snapshot
```

这样并发工作与持久状态协调分离。

如果以后进入多进程 / 多机器：

```text
SQLite coordinator
```

会进一步换成：

```text
transactional DB / queue / actor runtime
```

但状态机语义可以保持不变。

---

## 11　一 Agent 同时最多一个 Task

对于 coding worker，最危险的情况之一是：

```text
同一个 Agent identity
├─ task-A：修改 parser.py
└─ task-B：同时修改 parser.py
```

这会让“Agent 的上下文、worktree、artifact、mailbox result”全部失去单一归属。

当前调度器因此固定：

```text
one active task per worker agent
```

测试记录每个 Agent 的并发计数：

```text
peak_concurrency_per_agent == 1
```

注意这不限制整个系统并行：

```text
worker-A concurrency = 1
worker-B concurrency = 1
worker-C concurrency = 1

system concurrency = 3
```

---

## 12　Worker Crash 不应该让整个 Pool 自动报成功

如果一个 worker callback 抛异常：

```text
worker-A
→ RuntimeError
```

调度器把该 task 转换为显式：

```text
ok = false
summary = worker raised RuntimeError: ...
```

同时：

```text
worker-A → FAILED
```

而其他 healthy worker 仍可以继续接剩余任务。

这使得：

```text
N agents
```

不再等于一个不可分割的“大模型调用”。每个 agent 都有独立健康状态。

---

## 13　Result Identity 也必须验证

Worker 返回：

```text
ParallelTaskResult(
    task_id=...,
    agent_id=...,
)
```

Coordinator 检查它是否与 dispatch record 一致。

否则：

```text
wrong task id
or
wrong agent id
       ↓
failed result
```

这是为了防止将来 remote worker / A2A 接入后，结果错配到错误任务。

---

# 14　真正的 Coding Multi-Agent 还差什么

当前这一课只完成：

```text
identity
hierarchy
lifecycle
mailbox
lease/reclaim
parallel execution
```

要变成 Codex-class parallel coding，还要继续接：

```text
Task DAG
   ↓
worker-specific Git worktree
   ↓
Coding Agent runtime
   ↓
patch artifact
   ↓
independent verifier
   ↓
Reviewer Agent
   ↓
merge decision
   ↓
conflict handling
   ↓
Rollout Trace provenance
```

其中每个 worker 必须有独立 worktree，不能让几个 Agent 直接并发写同一个 working tree。

---

# 15　为什么必须做 1 / 2 / 4 / 8 Agent 对照

“多 Agent”不是天然更强。

随着 Agent 数量增加：

```text
parallelism ↑
```

可能带来：

```text
wall time ↓
```

但也可能带来：

```text
tokens ↑
tool calls ↑
duplicate work ↑
merge conflict ↑
coordination failures ↑
```

所以最终 benchmark 必须统一记录：

| Workers | Success | Wall Time | Model Steps | Tool Calls | Cost | Duplicate Work | Conflicts | Human Interventions |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | | | | | | | | |
| 2 | | | | | | | | |
| 4 | | | | | | | | |
| 8 | | | | | | | | |

只有这样才能回答：

> Multi-Agent 到底在哪些任务上值得？

而不是展示一个“8 个 Agent 同时聊天”的动画。

---

# 16　本课的验证层次

当前测试已经覆盖：

```text
Agent topology persistence
parent/child lineage
terminal lifecycle boundary
cross-thread negative controls
mailbox lease exclusion
lease expiry reclaim
stale-owner ACK rejection
release → pending
subtree cancellation
real two-worker overlap
one-task-per-agent invariant
worker crash isolation
wrong result identity rejection
```

这些都属于 **runtime correctness**。

仍未证明：

```text
parallel agents improve real coding success
parallel worktree patches can merge safely
reviewer improves final correctness
4/8 agents beat 1/2 agents after cost normalization
```

这些必须留给下一阶段 controlled evaluation。

---

# 17　下一步

从这里开始，多智能体主线最合理的工程顺序是：

```text
Persistent AgentGraph + Mailbox       ← 本课
        ↓
Parallel Agent Coordinator            ← 本课
        ↓
Worker-specific Git Worktrees
        ↓
Patch Artifact + Rollout Trace
        ↓
Independent Reviewer
        ↓
Merge / Conflict Resolution
        ↓
1/2/4/8-Agent Controlled Benchmark
        ↓
Remote Agent / A2A
```

这条路线坚持一个原则：

> **先把并发、身份、状态与证据做成可靠软件系统，再谈“群体智能”。**
