# C01　核心 Agent Loop：Task、Turn、Sampling 与 Tool Follow-up

> **Primary sources**  
> - `codex-rs/docs/protocol_v1.md`: https://github.com/openai/codex/blob/main/codex-rs/docs/protocol_v1.md  
> - `codex-rs/core/src/session/turn.rs`: https://github.com/openai/codex/blob/main/codex-rs/core/src/session/turn.rs  
> **核验快照**：`d77ebc72237a639b6d877f2edc3b20b54631f25e`

---

# 1　Agent Loop 不是“Reason → Act”四个字

官方 protocol 文档把系统拆成：

```text
Session
  └─ Task
      ├─ Turn 1
      ├─ Turn 2
      └─ ...
```

其中 Task 在一次用户输入后启动，直到以下条件之一：

- 模型不再产生需要继续执行的输出；
- 新用户输入中断旧 task；
- UI interrupt；
- fatal error；
- approval 等待导致阻塞。

一次 Turn 则是模型与环境的一次闭环迭代。

因此最小形式化更接近：

$$
H_{k+1}=F(H_k,R_k,E_k),
$$

其中：

- $H_k$：当前 conversation / runtime state；
- $R_k$：第 $k$ 次 model response；
- $E_k$：工具执行、审批、错误、用户 steering 等环境事件。

如果 response 包含 tool call：

$$
R_k\rightarrow a_k\rightarrow o_k\rightarrow H_{k+1}\rightarrow R_{k+1}.
$$

如果 response 只有最终 assistant message：

$$
R_k\rightarrow \mathrm{Complete}.
$$

---

# 2　官方 `run_turn` 的核心契约

`core/src/session/turn.rs` 在 `run_turn` 上方的源码注释已经给出最重要的语义：

```text
model replies with function call
→ execute it
→ send output back in next sampling request

model replies only with assistant message
→ record it
→ consider turn complete
```

这条逻辑就是 Codex harness 的心脏。

但真实实现外围还有大量 runtime concern：

```text
pre-turn compaction
→ hooks
→ MCP requirements
→ skills / plugins
→ step context snapshot
→ world-state diff
→ sampling request
→ tool execution
→ pending input
→ token accounting
→ mid-turn compaction
→ stop hooks
→ completion / error event
```

所以生产 Agent 的复杂度主要来自**状态和边界**，而不是来自 `while` 关键字本身。

---

# 3　为什么要有 Step Context

官方实现并不是每次 tool call 都随手从全局 mutable config 中抓数据。

`run_turn` 会为 sampling step 捕获一个一致的 context view，使：

```text
prompt context
advertised tools
permissions
MCP requirements
environment selection
model settings
```

尽量共享同一个 request-time snapshot。

这是一个非常重要的工程原则：

> **模型看到的世界和 runtime 实际允许执行的世界，必须来自同一份或可追踪的一致状态。**

否则会出现典型 race：

```text
prompt: tool A 可用
runtime: tool A 已被撤权
```

或者反过来。

我们的 clean-room 版本后续也要把 `TurnSettings` 继续扩展成 immutable step snapshot。

---

# 4　Follow-up 是 Agent 的真正循环条件

在官方实现里，sampling 后会得到一个类似：

```text
needs_follow_up
last_agent_message
```

的结果。

同时 runtime 还会检查是否出现 pending user input。

因此循环继续条件不是单纯：

```python
while tool_call:
```

而更接近：

```text
model needs follow-up
OR
new pending input exists
OR
hook requests continuation
OR
context rollover requires compact + resume
```

这说明长任务 agent loop 是一个**事件驱动状态机**。

---

# 5　用户 steering 不是新开一个完全无关的 chat

官方实现允许模型运行过程中积累 pending input。下一轮 sampling 前，runtime 可以把这些输入写入 history，再构造新的 step context。

因此：

```text
User prompt
   ↓
Agent working
   ↓
User sends additional direction
   ↓
Pending input queue
   ↓
next model step observes steering
```

这和“取消整个 agent 再重新 prompt”不是一个交互模型。

对于 coding agent，这非常重要，因为真实用户常会在任务运行时追加：

- “不要动这个目录”；
- “顺便跑一下 integration tests”；
- “这个方向不对，换方案”。

---

# 6　Context rollover 也是 loop 的一部分

官方 `run_turn` 会在模型继续需要 follow-up、但 token budget 达到阈值时执行 compaction，然后继续同一个逻辑任务。

所以：

```text
context window exhausted
```

不必等价于：

```text
agent task terminated
```

更合理的是：

```text
active state
→ summarize / compact / preserve critical world state
→ start next context window
→ continue task
```

这就是为什么长程 agent runtime 必须把“任务生命周期”与“单个 context window 生命周期”分开。

---

# 7　错误不是只有 exception

Agent loop 中至少要区分：

```text
model transport error
context window exceeded
tool execution failure
approval denial
invalid image/input
policy violation
user interrupt
hook stop/block
step limit
```

不同错误的恢复语义不同。

例如 tool failure 一般应该成为 observation：

```text
command failed
→ stderr + exit code
→ feed back to model
→ repair
```

而不是整个进程崩溃。

这也是我们现有 `ToolRegistry.execute()` 采用 error-as-observation 的原因。

---

# 8　Clean-room v1 对应实现

我们新增：

[`../代码-code/从零构建Astra与Codex级系统/src/astra_codex/codex_harness.py`](../代码-code/从零构建Astra与Codex级系统/src/astra_codex/codex_harness.py)

当前状态机：

```text
TURN_STARTED
    ↓
MODEL_OUTPUT
    ↓
final? ─────────────→ TURN_COMPLETED
    │
    └─ tool call
         ↓
approval needed?
   ├─ deny → observation → next MODEL_OUTPUT
   └─ allow
         ↓
TOOL_STARTED
         ↓
TOOL_COMPLETED
         ↓
observation
         ↓
next MODEL_OUTPUT
```

这只是 C01 的最小正确实现。

还没有复现：

```text
pending input
context compaction
response bookmark
hooks
MCP lifecycle snapshot
parallel tools
turn diff
retry state
thread persistence
```

这些会逐章加入，而不是把它们藏进一个巨型 `Agent.run()`。

---

# 9　Parity 应该测什么

不能只测“最终回答看起来差不多”。

Agent harness parity 至少分成：

| 层 | 可测对象 |
|---|---|
| State-machine parity | tool call 后是否一定产生 follow-up sampling |
| Event parity | started / approval / exec / complete 的事件顺序 |
| Failure parity | denial、tool error、interrupt 后怎样继续 |
| Context parity | observation 进入下一步的格式和位置 |
| Persistence parity | resume / fork 是否保留正确状态 |
| Permission parity | policy 改变后 model-visible + execution boundary 是否一致 |

最终目标是写出 behavioral test harness，而不是肉眼点两次 CLI。

---

# 10　本章实验

### E1　最小 tool follow-up

让 scripted backend 第一步请求 `echo`，第二步根据 observation 返回 final answer，验证模型调用次数为 2。

### E2　approval denial

工具被 gate，approval 返回 deny；验证真实 tool 没被执行，但 denial 作为 observation 进入下一步。

### E3　step limit

模型永远请求工具，验证 runtime 最终以显式 `TURN_STOPPED` 退出，而不是死循环。

当前这些实验已写入：

[`../代码-code/从零构建Astra与Codex级系统/tests/test_codex_harness.py`](../代码-code/从零构建Astra与Codex级系统/tests/test_codex_harness.py)
