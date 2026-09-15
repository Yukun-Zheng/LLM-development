# Lesson 20：统一 Rollout Trace 与证据链

> 本课解决一个长程智能体里非常容易被忽略的问题：**任务“发生过什么”不能只存在聊天记录里，更不能靠 Agent 最后的自然语言自述。** 一个可评测、可恢复、可审计的智能体系统，需要把模型输出、工具动作、Steering、产物、Verifier 结果组织成一条能追溯来源的执行证据链。

---

## 1　为什么 Event Log 还不够

到这一阶段，我们已经有多种持久化状态：

```text
DurableThreadStore
    └─ 负责重建 Thread / Turn 状态

DurableEventStream
    └─ 负责 UI / IDE / SSE 的控制面 replay

DurableToolJournal
    └─ 负责 side-effect idempotency / in-doubt 状态

DurableSteeringQueue
    └─ 负责运行中用户改向

ArtifactStore
    └─ 负责不可变工作产物
```

这些 store 的职责都不同，因此不能简单把其中一个宣布成“完整 trajectory”。

例如控制面事件里可能有：

```text
thread.submitted
work.claimed
harness.model_output
harness.tool_started
harness.tool_completed
work.finished
```

但真正进行评测时，我们还想知道：

```text
哪一个 artifact 是这个 work item 产生的？
哪条 steering 最终被消费并进入模型上下文？
Verifier 是基于哪些 artifact 作出 PASS/FAIL？
这份证据是否在导出后被悄悄改过？
```

因此需要一个新的概念：

> **Rollout Trace = 面向一个可执行任务的、跨 store 的证据投影。**

它不是新的业务状态真相，而是把已有状态源里的相关证据组织成一个可导出、可验证的 bundle。

---

## 2　三类日志必须区分

### 2.1 State Log

目标是回答：

> “系统现在是什么状态？”

例如 `DurableThreadStore` 可以 replay 得到：

```text
READY
RUNNING
PAUSED
FAILED
COMPLETED
```

### 2.2 Integration Event Feed

目标是回答：

> “外部客户端从 cursor X 之后错过了哪些事件？”

所以 `DurableEventStream` 采用单调 `event_id`：

```text
client cursor = 152
        ↓
read(after_id=152)
        ↓
153, 154, 155, ...
```

这非常适合 SSE reconnect，但它不是专门为实验证据设计的。

### 2.3 Rollout Trace

目标是回答：

> “work_123 从目标进入系统，到模型、工具、Steering、artifact、Verifier，完整证据是什么？”

因此 trace 的主键不是 UI session，而是稳定的：

```text
work_item_id
    ↓
trace_work_item_id
```

---

## 3　当前 Reference Implementation

源码：

```text
src/astra_codex/rollout_trace.py
src/astra_codex/rollout_trace_collector.py
```

核心数据对象：

```text
RolloutTrace
├─ trace_id
├─ work_item_id
├─ thread_id
└─ created_at

RolloutTraceEvent
├─ event_id
├─ ordinal
├─ kind
├─ turn_id
├─ source_event_id
├─ artifact_id
├─ parent_event_ids
├─ payload
├─ prev_hash
├─ event_hash
└─ created_at
```

注意三个不同的 ID：

```text
source_event_id
    = 原 DurableEventStream 的事件编号

artifact_id
    = ArtifactStore 中的不可变工作产物

parent_event_ids
    = trace 内部的证据依赖边
```

因此它同时可以表达“来自哪里”和“依赖什么”。

---

## 4　为什么需要 parent_event_ids

假设 coding agent 产生 patch：

```text
artifact A
```

随后 hidden-test verifier 得到：

```text
PASS
```

最弱的记录方式是：

```text
artifact exists
verification PASS
```

但二者没有结构关系。

我们现在允许：

```text
artifact.snapshot [event 17]
          │
          ▼
verification.completed [event 18]
parent_event_ids = [17]
```

于是我们可以明确声明：

> 这个验证结论是在检查哪一个工作产物。

以后 Reviewer、merge decision、benchmark grade 都可以继续沿用这个结构。

---

## 5　Steering 为什么必须进入 Trace

长程 Agent 常见情况：

```text
用户最初：修改代码
        ↓
Agent 开始执行
        ↓
用户中途：先别改，先看文档
        ↓
Agent 改变策略
```

如果最后只保存初始 prompt 和 final answer，我们会错误地解释 trajectory。

所以 collector 会同时记录：

```text
thread.steering_submitted
        ↓
DurableSteeringQueue status == CONSUMED
        ↓
steering.consumed
```

这使得一次实验可以回答：

```text
任务失败是模型能力不足？
还是用户中途改变了目标？
Steering 是否真的被执行器消费？
```

这里的 `ControlledBackend` 还提供 `steering_observer` seam，为后续把 consumption 直接实时写入 unified trace/event feed 留出接口。

---

## 6　Artifact 不等于自然语言答案

对于 coding agent：

```text
"I fixed the bug."
```

没有多少证据价值。

真正有意义的是：

```text
patch.diff
modified tree
pytest output
coverage report
build artifact
review report
```

所以 `ArtifactStore` 保存 bytes，Rollout Trace 保存 artifact reference：

```text
artifactId
kind
sha256
sizeBytes
metadata
```

推荐 metadata 至少带：

```json
{
  "work_item_id": "work_123",
  "producer": "worker_A",
  "purpose": "candidate_patch"
}
```

这样 collector 可以把 thread 上众多 artifact 中真正属于当前 work item 的部分投影进 trace。

---

## 7　Hash Chain：我们证明了什么，没证明什么

每条 trace event 保存：

```text
prev_hash
      ↓
canonical current event bytes
      ↓
SHA-256
      ↓
event_hash
```

于是：

```text
E0 ─hash→ E1 ─hash→ E2 ─hash→ E3
```

测试会直接修改 SQLite 中某个 `payload_json`，然后：

```python
assert not store.verify_integrity(trace_id)
```

这证明：

> 在 hash 没有同步重算的情况下，历史篡改可以被发现。

但**不能**证明：

> 一个拥有数据库完全写权限的恶意进程无法伪造整条链。

因为它可以把 payload 和后续 hash 全部重新计算。

因此 production audit 还需要：

```text
local hash chain
      ↓
periodic head digest
      ↓
external trusted anchor
```

例如外部 append-only log、签名服务、remote attestation 或不可由 Agent 自己覆盖的审计系统。

所以准确术语是：

> **tamper-evident（篡改可察觉）**，不是 **tamper-proof（不可篡改）**。

---

## 8　Collector 的数据流

当前 `RolloutTraceCollector.sync(work_item_id)` 做四件事：

```text
1. WorkQueue
   work_item_id → thread_id

2. DurableEventStream
   筛选 workItemId / turnId 相关事件

3. DurableSteeringQueue
   核验 steering 是否真的 CONSUMED

4. ArtifactStore
   metadata.work_item_id == target
```

然后统一写成：

```text
RolloutTraceStore
```

而且 sync 是幂等的：重复同步不会因为同一个 `source_event_id` 再写一遍同一事件。

---

## 9　Verifier 如何进入证据链

Verifier 必须独立于 Agent 的自我陈述。

例如：

```python
result = VerificationResult(
    Verdict.PASS,
    "hidden tests passed",
    {"tests": 12},
)
```

collector 可以写：

```text
verification.completed
├─ verdict = pass
├─ summary
├─ evidence
├─ artifactIds
└─ parent_event_ids
```

于是一个 benchmark case 最终可以输出：

```text
Goal
  ↓
Trajectory
  ↓
Artifacts
  ↓
Independent Verification
  ↓
Grade
```

而不是：

```text
Agent final answer
  ↓
LLM judge asks "looks good?"
```

---

## 10　当前自动测试

`tests/test_rollout_trace.py` 当前验证三组性质。

第一组是完整性：

```text
append event
→ hash chain valid
→ mutate stored payload
→ integrity verification fails
```

第二组是 durability / identity：

```text
create trace
→ close SQLite
→ reopen
→ trace persists

same work_item_id + another thread_id
→ rejected
```

第三组是真实跨子系统证据链：

```text
create Thread
→ submit WorkItem
→ submit Steering
→ execute Turn
→ Steering consumed
→ snapshot patch Artifact
→ sync Rollout Trace
→ attach independent Verification
→ export evidence bundle
```

并检查第二次 `sync()` 为零新增，保证 collector 的 reference semantics 是幂等的。

---

## 11　为什么这对 RL / Agentic RL 也重要

RLVR / Agentic RL 最终训练的不是一句 answer，而是 trajectory。

未来 trajectory dataset 至少需要：

```text
observation
model action
model/tool event
external action
artifact
verifier reward
environment transition
termination
```

Rollout Trace 正好提供一个“训练之前”的可靠数据层：

```text
Agent Runtime
      ↓
Rollout Trace
      ↓
Verifier / reward attribution
      ↓
Trajectory Dataset
      ↓
RL / offline analysis
```

因此它不仅服务 debugging，也会成为后续 full RLVR pipeline 和 Agent evaluation 的共同接口。

---

## 12　下一步

当前实现仍然是 reference layer，下一批升级应包括：

```text
runtime-native trace sink
→ 不依赖事后 collector 扫描

artifact / verifier / reviewer / merge
→ 全部显式 parent edges

trace export
→ benchmark harness / SWE-bench adapter

trace head hash
→ external signed anchor

multi-agent
→ parent/child trace IDs + agent identity

training
→ trajectory dataset materializer
```

最终我们希望能够对任何 Codex/Astra-class 长程任务导出一份这样的对象：

```text
Task Evidence Bundle
├─ immutable task identity
├─ model/tool trajectory
├─ steering history
├─ artifacts
├─ independent verifier results
├─ reviewer decisions
├─ provenance graph
└─ integrity chain
```

到那一步，“Agent 完成了任务”才不再是一句自然语言声明，而是一个**可以复盘、可以评测、可以训练、可以审计的工程事实**。
