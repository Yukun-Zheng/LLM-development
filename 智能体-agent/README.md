# 智能体系统主线 / Agentic Systems Track

> **定位**：智能体不是“LLM 后面再接几个工具”的附录，而是本教材的第二条主线。模型主线研究 `model(x) -> tokens`；智能体主线研究 `goal + state + environment -> actions -> observations -> updated state`。

这条主线与 [`教材-book/`](../教材-book/) 并行：前者回答**模型怎样获得与表达能力**，这里回答**模型怎样进入真实环境、调用外部能力、长期规划、验证、恢复、记忆与协作**。

---

# 1　第一性形式化

单次语言模型推理近似：

$$
y\sim p_\theta(y\mid x).
$$

智能体则是闭环：

$$
a_t\sim \pi_\theta(a_t\mid h_t,g),
$$

$$
o_{t+1}\sim P(o_{t+1}\mid s_t,a_t),
$$

$$
h_{t+1}=F(h_t,a_t,o_{t+1}).
$$

其中：

- $g$：目标；
- $h_t$：可见历史、memory 与任务状态；
- $a_t$：动作；
- $o_{t+1}$：环境 observation；
- $P$：环境动力学；
- $\pi_\theta$：由 LLM / reasoning model 驱动的 policy。

Agent 的核心对象不再只是“下一个 token”，而是**长程决策与真实反馈**。

---

# 2　完整学习路线

| 模块 | 主题 | 文件 |
|---|---|---|
| A1 | **智能体第一性原理** | [`01-智能体基础-agent-foundations.md`](01-智能体基础-agent-foundations.md) |
| A2 | **工具调用、动作空间与环境接口** | [`02-工具与环境-tool-use-environments.md`](02-工具与环境-tool-use-environments.md) |
| A3 | **规划、反思、验证与失败恢复** | [`03-规划反思与验证-planning-reflection-verification.md`](03-规划反思与验证-planning-reflection-verification.md) |
| A4 | **记忆、上下文管理与长期任务状态** | [`04-记忆与上下文-memory-context.md`](04-记忆与上下文-memory-context.md) |
| A5 | **编程智能体：从代码生成到仓库闭环** | [`05-编程智能体-coding-agents.md`](05-编程智能体-coding-agents.md) |
| A6 | **多智能体、角色分工与并行工作树** | [`06-多智能体与协调-multi-agent.md`](06-多智能体与协调-multi-agent.md) |
| A7 | **智能体评测、安全与可验证性** | [`07-评测安全与开放问题-agent-evaluation-safety.md`](07-评测安全与开放问题-agent-evaluation-safety.md) |
| A8 | **环境学习、Agentic RL 与持续适应** | [`08-环境学习与AgenticRL-agentic-rl.md`](08-环境学习与AgenticRL-agentic-rl.md) |
| A9 | **MCP、A2A 与 Agent Protocols** | [`09-Agent协议与互操作-agent-protocols.md`](09-Agent协议与互操作-agent-protocols.md) |
| A10 | **Browser / Computer Use 与 GUI Agent** | [`10-浏览器与计算机使用-browser-computer-use.md`](10-浏览器与计算机使用-browser-computer-use.md) |

一手资料地图：[`../参考文献-references/01-智能体原始资料-agent-sources.md`](../参考文献-references/01-智能体原始资料-agent-sources.md)

源码工程：[`../代码-code/从零构建Astra与Codex级系统/`](../代码-code/从零构建Astra与Codex级系统/)

---

# 3　Agent Stack

```text
Model / Reasoner
      ↓
Policy / Planner
      ↓
Structured Action
      ↓
Protocol Layer
├─ direct tool schema
├─ MCP
└─ A2A
      ↓
Environment
├─ Filesystem
├─ Shell
├─ Git
├─ Browser
├─ Computer
└─ Remote Agent
      ↓
Observation
      ↓
Context / Memory
      ↓
Verifier
├─ pass
├─ fail → retry / replan
└─ risky → approval gate
      ↓
Next Decision / Finish
```

模型能力只是其中一层；Agent 还会受：

- observation quality；
- action schema；
- tool latency；
- memory policy；
- planning search；
- verifier quality；
- permission boundary；
- scheduler；
- environment nondeterminism；

独立影响。

---

# 4　把 Agent 看成“软件运行时”，而不是 Prompt

最终我们希望拥有一个接近 Agent OS 的结构：

```text
Agent Runtime
├─ Scheduler
├─ Task DAG
├─ Context Manager
├─ Memory Store
├─ Tool Router
├─ MCP Client
├─ Remote Agent Client
├─ Permission Manager
├─ Sandbox
├─ Artifact Store
├─ Verifier
├─ Worker Pool
├─ Event Log
└─ Checkpoint / Recovery
```

因此本教材不会把“换一个 system prompt”当作 Agent architecture innovation。

---

# 5　当前代码映射

终极工程已经有：

```text
agent.py          最小 observe → act → observe loop
structured.py     structured action / schema
planning.py       typed task DAG
verification.py   external verifier
memory.py         SQLite event memory
coding.py         coding-agent contract
tools.py          filesystem / shell / Git
editing.py        safe exact edit
repo_map.py       code structure map
worktree.py       Git worktree primitive
multi_agent.py    coordinator primitive
general_tools.py  minimal HTTP browser
```

下一批将加入：

```text
mcp.py
computer/
scheduler.py
checkpoint.py
permissions.py
sandbox.py
artifact.py
review.py
```

---

# 6　学习标准

学完一个 Agent 概念，必须回答：

1. **形式化**：state / observation / action / policy / verifier 分别是什么？
2. **原始工作**：论文或规范真正定义了什么？
3. **代码**：机制在 runtime 哪一层实现？
4. **证据**：怎样证明 task completion？
5. **失败模式**：在哪些环境下会失效？
6. **系统代价**：step、token、latency、tool cost、retries、memory 分别多少？

所以：

```text
“I fixed the bug and tests pass.”
```

不是证据。

只有：

```text
shell observation
→ exit code = 0
→ verifier pass
```

才能确认测试真的通过。

---

# 7　毕业标准

不依赖 LangChain 等高层 Agent framework，从零搭出：

```text
Tool Calling
→ Environment Loop
→ Planning / Replanning
→ Verification / Recovery
→ Persistent Memory
→ Coding Agent
→ Browser / Computer Agent
→ MCP / A2A Interoperability
→ Multi-Agent Scheduler
→ Reviewer / Merge
→ Agent Evaluation
→ Agentic RL Trajectory Pipeline
```

最终与模型主线、系统主线汇合到 Astra-class / Codex-class 总工程。