# Lab 08：副作用、幂等键、In-Doubt 与崩溃恢复

> **目标**：理解为什么“任务可以 resume”仍不等于“工具副作用可以安全重试”，以及 durable journal、idempotency key、lease 与 checkpoint 怎样一起缩小重复执行风险。

对应源码：

- [`../src/astra_codex/tool_journal.py`](../src/astra_codex/tool_journal.py)
- [`../src/astra_codex/runtime.py`](../src/astra_codex/runtime.py)
- [`../src/astra_codex/runtime_queue.py`](../src/astra_codex/runtime_queue.py)
- [`../src/astra_codex/durable.py`](../src/astra_codex/durable.py)

对应测试：

- [`../tests/test_tool_journal.py`](../tests/test_tool_journal.py)
- [`../tests/test_integrated_runtime.py`](../tests/test_integrated_runtime.py)

理论入口：

- [`../../../智能体-agent/07-评测安全与开放问题-agent-evaluation-safety.md`](../../../智能体-agent/07-评测安全与开放问题-agent-evaluation-safety.md)
- [`../../../Codex源码解剖-codex-anatomy/02-会话线程与事件协议-session-thread-events.md`](../../../Codex源码解剖-codex-anatomy/02-会话线程与事件协议-session-thread-events.md)
- [`../../../Codex源码解剖-codex-anatomy/03-工具执行审批与沙箱-tools-approval-sandbox.md`](../../../Codex源码解剖-codex-anatomy/03-工具执行审批与沙箱-tools-approval-sandbox.md)

---

# 1　最危险的崩溃窗口

假设 Agent 要调用：

```text
send_email(...)
```

执行时发生：

```text
1. runtime 决定调用 send_email
2. 邮件服务已经发送成功
3. 本地进程突然崩溃
4. runtime 没来得及记录“成功”
```

重启后系统只能看到：

```text
这个 tool call 开始过
但不知道外部副作用到底提交成功没有
```

如果直接 retry：

```text
send_email(...)
```

用户可能收到两封邮件。

这个状态叫：

> **In-Doubt Execution（结果不确定执行）**。

---

# 2　为什么 Lease 还不够

Work Queue 的 lease 解决：

```text
哪个 worker 现在拥有执行权？
```

它不能解决：

```text
上一个 worker 在死之前是否已经完成外部副作用？
```

所以：

```text
Lease
!=
Idempotency
!=
Transaction
```

三者解决不同层的问题。

---

# 3　Durable Tool Journal

当前 `tool_journal.py` 为每个 side-effect proposal 建立持久记录：

```text
idempotency_key
tool_name
arguments_hash
status
result
```

状态只有：

```text
STARTED
COMPLETED
```

执行顺序：

```text
create STARTED row
        ↓
execute tool
        ↓
persist COMPLETED + result
```

因此已经完成的调用可以被 replay，而不需要再次执行 tool body。

---

# 4　稳定 Idempotency Key 从哪里来

`TurnScopedJournaledTools` 使用：

```text
{work_item_id}:tool:{call_index}
```

例如：

```text
work_abc:tool:0
work_abc:tool:1
work_abc:tool:2
```

为什么使用 durable `work_item_id` 而不是随机 UUID？

因为 worker crash 后重新领取的是**同一个 work item**。

如果重新执行同一个 turn 并产生相同 tool-call sequence：

```text
first run:   work_abc:tool:0
retry run:   work_abc:tool:0
```

journal 才能知道：

> “这是同一逻辑调用的恢复，而不是新的动作。”

---

# 5　COMPLETED：可以安全 replay

第一次：

```text
key = work_abc:tool:0
STARTED
→ tool body executes
→ COMPLETED(result)
```

进程重启后：

```text
same key
same tool
same arguments
→ return stored result
→ tool body DOES NOT execute
```

自动测试使用一个 `CountingTool`：

```text
first execution   calls = 1
replay execution  calls = 1
```

这证明 replay 没有重复副作用。

---

# 6　为什么还要存 arguments hash

如果同一个 key 被错误地用于：

```text
first:
filesystem.write(path="A", content="x")

retry:
filesystem.write(path="B", content="y")
```

系统不能说：

> “key 一样，所以这是同一个动作。”

当前使用 canonical JSON：

```text
sort keys
→ UTF-8 bytes
→ SHA-256
```

得到：

$$
h=\mathrm{SHA256}(\mathrm{canonicalJSON}(args)).
$$

同一个 idempotency key 如果绑定不同 tool 或不同 arguments，直接拒绝。

这是一种保守的 divergence detection。

---

# 7　STARTED：真正困难的 In-Doubt 状态

如果数据库只有：

```text
STARTED
```

可能发生两种现实：

### 情况 A

```text
STARTED persisted
→ process crashes before tool executes
```

此时 retry 本来是安全的。

### 情况 B

```text
STARTED persisted
→ external side effect succeeds
→ process crashes before COMPLETED persisted
```

此时 retry 可能重复副作用。

但本地 journal 无法区分 A/B。

所以默认策略是：

```text
STARTED + no result
→ mark in_doubt
→ DO NOT blindly retry
```

这不是系统“不够智能”，而是**信息论上确实缺少事实**。

---

# 8　什么情况下可以 retry In-Doubt

只有在额外满足某种条件时。

例如：

### 工具本身是幂等的

```text
set_config(key="x", value=1)
```

重复执行得到同样最终状态。

### 外部 API 支持 idempotency key

```text
POST /payments
Idempotency-Key: work_abc:tool:0
```

服务端保证相同 key 不会创建两笔支付。

### 可以先 reconcile

```text
query external system
→ 查找该 operation_id 是否已存在
→ 已存在则 reconstruct result
→ 不存在才执行
```

否则：

> **不要把 at-least-once retry 伪装成 exactly-once。**

---

# 9　Runtime Crash Recovery 现在怎样工作

当前 `runtime.py` 已经把：

```text
ThreadStore
WorkQueue
ToolJournal
TurnExecutor
```

连起来。

正常：

```text
submit
→ checkpoint: claimed
→ TURN_STARTED
→ checkpoint: turn_started
→ journaled tool calls
→ checkpoint: turn_finished
→ TURN_COMPLETED
→ work ACK
```

worker crash 后 lease 到期，新 worker 可以 reclaim。

如果 Thread 仍是 RUNNING，而且 last checkpoint 属于同一个 work item：

```text
reclaim same WorkItem
→ reuse same active TurnId
→ rerun turn with same tool-call scope
```

已完成的工具会从 journal replay。

---

# 10　更细的恢复点：`turn_finished`

另一个重要 crash window：

```text
model/tool execution 已全部结束
→ final checkpoint 已持久化
→ process crashes
→ 还没 TURN_COMPLETED / ACK
```

这里根本不应该重新调用模型。

当前 runtime 检测：

```text
checkpoint.phase == "turn_finished"
```

则直接：

```text
recover final_answer/model_steps
→ complete turn
→ ACK work
```

测试用一个**没有任何输出的 `ScriptedBackend([])`**恢复成功，证明恢复路径没有重新 sampling。

---

# 11　仍然没有解决的 exactly-once 难题

即使有 ToolJournal，仍然存在：

```text
external effect commits
↓
process dies
↓
COMPLETED row not persisted
```

本地状态只能看到 `STARTED`。

所以当前项目只能准确声称：

```text
completed call replay duplication   → solved
in-doubt detection                  → solved
blind retry of non-idempotent call  → blocked
universal exactly-once side effects → NOT solved
```

真正 exactly-once 往往需要：

```text
transactional outbox
external idempotency key
2PC-like coordination
operation reconciliation
idempotent state transition
```

具体取决于环境。

---

# 12　这对 Coding Agent 有什么意义

某些 coding tool 本身相对容易 reconcile：

```text
write file
apply patch
create git commit
```

因为可以检查 final filesystem/Git state。

但有些动作更危险：

```text
publish package
merge PR
push branch
delete remote resource
send message
create cloud resource
```

这些动作必须拥有更严格的 operation identity 与 recovery protocol。

因此未来 tool spec 最终还应标记：

```text
read-only
idempotent
reconcilable
non-idempotent
requires-approval
requires-external-idempotency-key
```

安全与可靠性会在这里汇合。

---

# 13　运行测试

```bash
cd '代码-code/从零构建Astra与Codex级系统'
pip install -e '.[dev]'
pytest tests/test_tool_journal.py \
       tests/test_integrated_runtime.py -q
```

你应该重点读两个不是 happy-path 的测试：

```text
STARTED-only record
→ automatic retry blocked

crash after completed tool
→ lease expires
→ new worker reclaims
→ journal replay
→ new tool body calls == 0
```

这两个场景比再写一个普通 Agent loop 更接近真实长程智能体系统的难点。
