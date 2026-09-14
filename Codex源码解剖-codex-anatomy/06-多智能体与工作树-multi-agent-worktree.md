# C06　Multi-Agent、Agent Graph、Worktree 与协作语义

> **Primary sources**  
> - `core/src/session/multi_agents.rs`: https://github.com/openai/codex/blob/main/codex-rs/core/src/session/multi_agents.rs  
> - `core/src/tools/handlers/multi_agents/`: https://github.com/openai/codex/tree/main/codex-rs/core/src/tools/handlers/multi_agents  
> - `core/src/tools/handlers/multi_agents_v2/`: https://github.com/openai/codex/tree/main/codex-rs/core/src/tools/handlers/multi_agents_v2  
> - `agent-graph-store/`: https://github.com/openai/codex/tree/main/codex-rs/agent-graph-store  
> - `agent-identity/`: https://github.com/openai/codex/tree/main/codex-rs/agent-identity  
> - `agent-roles/`: https://github.com/openai/codex/tree/main/codex-rs/agent-roles  
> - `worktree/`: https://github.com/openai/codex/tree/main/codex-rs/worktree

---

# 1　官方源码已经把 Multi-Agent 当成真实 runtime feature

当前 `multi_agents.rs` 的公开提示与 runtime 配置明确出现：

```text
spawn_agent
followup_task
send_message
wait_agent
interrupt_agent
list_agents
```

而且 child agent 可以继续 spawn 自己的 sub-agent。

这与“让三个 role 在一个 prompt 里轮流说话”完全不同。

真实 multi-agent 要有：

```text
identity
thread
parent/child relation
message delivery
concurrency slot
lifecycle
state visibility
filesystem semantics
result handoff
```

---

# 2　Agent Graph 是比“角色列表”更准确的抽象

可以定义：

$$
G=(V,E),
$$

其中 $V$ 是 agent/thread，$E$ 是 spawn / message / dependency / artifact handoff 等关系。

例如：

```text
/root
├─ /root/tester
│   └─ /root/tester/reproducer
├─ /root/refactor
└─ /root/reviewer
```

需要回答：

- 谁创建谁？
- 什么时候 active？
- 结果送给谁？
- 是否能发送消息但不启动 turn？
- 父 agent 停止时 child 怎么办？
- resume 后 identity 是否稳定？

这就是 `agent-graph-store` / `agent-identity` 一类 subsystem 的意义。

---

# 3　Fork Context 是成本与隔离的核心选择

官方 multi-agent instructions 公开提到 `fork_turns`：parent 可以决定传播多少上下文给 child。

抽象：

```text
fork all history
  优点：child 上手快
  缺点：token 贵、污染多、耦合强

fork none
  优点：干净、便宜、独立
  缺点：需要显式 task package

fork last k turns
  折中
```

因此多 Agent 调度的成本函数至少包含：

$$
J=
\alpha T_{wall}
+\beta C_{token}
+\gamma C_{tool}
+\delta C_{merge}
+\epsilon R_{conflict}.
$$

“agent 越多越强”没有理论保证。

---

# 4　共享 filesystem 带来的并发语义

官方 multi-agent usage hint 明确说明多个 agent 可以共享 container/filesystem/cwd，某个 agent 的 edits 会立即对其他 agent 可见。

这带来经典并发问题：

```text
Agent A reads file v1
Agent B edits file → v2
Agent A writes based on stale v1
→ lost update
```

所以共享工作区模式必须处理：

- stale read；
- file conflict；
- build/test interference；
- process/port collision；
- destructive command race。

这也是为什么 coding workflow 中 worktree 隔离非常重要。

---

# 5　Shared Workspace 与 Worktree Isolation 是两种模式

## Mode A：共享目录

```text
Agent A ─┐
Agent B ─┼→ same repo checkout
Agent C ─┘
```

适合：

- 只读调研；
- 不重叠文件修改；
- 极低 overhead 协作。

## Mode B：独立 Worktree

```text
main repo
├─ worktree-A / branch-A
├─ worktree-B / branch-B
└─ worktree-C / branch-C
```

适合：

- 并行 patch；
- 独立 test；
- reviewer / merge；
- 避免工作区污染。

Codex workspace 中独立 `worktree` crate 说明 worktree 已经不是外围脚本，而是 runtime concern。

---

# 6　Multi-Agent 的消息语义比聊天更细

官方源码区分概念上不同的协作动作：

```text
spawn_agent
  创建 child + initial task

send_message
  给 running agent 信息，不一定触发新 turn

followup_task
  给现有 agent 新 task，并触发执行

wait_agent
  等待 agent 状态变化
```

这说明：

$$
\text{message delivery} \neq \text{task scheduling}.
$$

把二者绑死会导致：

- 每条消息都浪费一次 model call；
- 无法给正在执行的 worker 注入信息；
- scheduler 无法区分 mailbox 和 new task。

---

# 7　我们的当前实现

现有：

```text
planning.py       typed task DAG
multi_agent.py    deterministic coordinator primitive
worktree.py       Git worktree primitive
memory.py         persistent event memory
verification.py   evidence/verifier
```

它们仍然缺：

```text
persistent AgentGraph
stable AgentId
mailbox
async worker lifecycle
concurrency semaphore
subagent spawn
wait / interrupt
artifact handoff
reviewer
merge policy
```

---

# 8　Clean-room v2 设计目标

```text
AgentRuntime
├─ AgentGraphStore
├─ ThreadStore
├─ Scheduler
├─ WorkerRuntime
├─ Mailbox
├─ WorktreeManager
├─ ArtifactStore
├─ Verifier
└─ MergeCoordinator
```

关键 state：

```text
AgentId
ParentAgentId
ThreadId
TaskId
Status
WorktreePath
Branch
Mailbox
Artifacts
Evidence
```

---

# 9　必须做的并发实验

### MA1 read-only parallelism

4 个 worker 并发搜索不同子目录，验证 wall time 降低、结果可合并。

### MA2 same-file conflict

两个 agent 修改同一行，验证 runtime 能检测冲突，而不是静默覆盖。

### MA3 isolated tests

两个 worktree 分别运行可能修改缓存/生成文件的测试，确认相互不污染。

### MA4 message without turn

向正在工作的 child 发送补充信息，不额外启动新的独立 worker。

### MA5 nested spawn

child 再 spawn grandchild，验证 graph、权限和 concurrency cap 正确。

### MA6 reviewer/merge

worker 提交 patch → reviewer 验证 → merge coordinator 合并；失败时退回具体 evidence。

---

# 10　评价 Multi-Agent 的正确指标

不要只报：

> “用了 8 个 agents。”

至少报告：

```text
task success
wall-clock time
total model tokens
tool calls
parallel utilization
conflict count
merge failure rate
review retries
cost
```

只有当并行收益超过通信、重复 context 与 merge 成本，多 Agent 才真正有价值。
