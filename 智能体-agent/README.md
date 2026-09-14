# 智能体系统主线 / Agentic Systems Track

> **定位**：智能体不是“LLM 后面再接几个工具”的附录，而是本教材的第二条主线。模型主线研究 `model(x) -> tokens`；智能体主线研究 `goal + state + environment -> actions -> observations -> updated state`。

这条主线与 `教材-book/` 并行：前者回答**模型怎样获得与表达能力**，这里回答**模型怎样在真实环境中持续行动、验证、恢复、记忆与协作**。

---

## 1　为什么智能体必须单独成体系

单次语言模型推理通常近似：

$$
y\sim p_\theta(y\mid x).
$$

智能体问题则是一个闭环：

$$
a_t\sim \pi_\theta(a_t\mid h_t,g),
$$

$$
o_{t+1}\sim P(o_{t+1}\mid s_t,a_t),
$$

$$
h_{t+1}=h_t\oplus(a_t,o_{t+1}).
$$

其中：

- $g$：目标（goal）；
- $h_t$：历史与当前可见状态；
- $a_t$：动作，可能是文本、工具调用、代码编辑、浏览器操作等；
- $o_{t+1}$：环境 observation；
- $P$：环境动力学；
- $\pi_\theta$：决策策略，通常由 LLM / reasoning model 驱动。

所以 Agent 的核心对象不再只是“下一个 token”，而是**长程决策、状态管理和真实反馈**。

---

## 2　完整学习路线

| 模块 | 主题 | 文件 |
|---|---|---|
| A1 | **智能体第一性原理：从语言模型到闭环决策系统** | [`01-智能体基础-agent-foundations.md`](01-智能体基础-agent-foundations.md) |
| A2 | **工具调用、动作空间与环境接口** | [`02-工具与环境-tool-use-environments.md`](02-工具与环境-tool-use-environments.md) |
| A3 | **规划、反思、验证与失败恢复** | [`03-规划反思与验证-planning-reflection-verification.md`](03-规划反思与验证-planning-reflection-verification.md) |
| A4 | **记忆、上下文管理与长期任务状态** | [`04-记忆与上下文-memory-context.md`](04-记忆与上下文-memory-context.md) |
| A5 | **编程智能体：从代码生成到仓库闭环** | [`05-编程智能体-coding-agents.md`](05-编程智能体-coding-agents.md) |
| A6 | **多智能体、角色分工与并行工作树** | [`06-多智能体与协调-multi-agent.md`](06-多智能体与协调-multi-agent.md) |
| A7 | **智能体评测、安全与可验证性** | [`07-评测安全与开放问题-agent-evaluation-safety.md`](07-评测安全与开放问题-agent-evaluation-safety.md) |
| A8 | **环境学习、Agentic RL 与持续适应** | [`08-环境学习与AgenticRL-agentic-rl.md`](08-环境学习与AgenticRL-agentic-rl.md) |

一手资料地图：[`../参考文献-references/01-智能体原始资料-agent-sources.md`](../参考文献-references/01-智能体原始资料-agent-sources.md)

源码工程：[`../代码-code/从零构建Astra与Codex级系统/`](../代码-code/从零构建Astra与Codex级系统/)

---

## 3　这条主线与 LLM 主线怎样连接

```text
Tokenizer / Model / Reasoning
          │
          ▼
     Policy / Planner
          │
          ▼
 Structured Action / Tool Call
          │
          ▼
 Environment / Files / Shell / Browser / Computer
          │
          ▼
      Observation
          │
          ├───────────────┐
          ▼               ▼
      Context           Memory
          │               │
          └───────┬───────┘
                  ▼
             Next Decision
                  │
          ┌───────┴────────┐
          ▼                ▼
       Verifier          Finish
```

模型越强，Agent 往往越强；但 Agent 失败不能全部归因于模型：工具协议、环境可观测性、状态压缩、搜索策略、验证器、重试预算和安全边界都能独立成为瓶颈。

---

## 4　学习标准

学完一个 Agent 概念，必须至少能回答四层问题：

1. **形式化**：state / observation / action / policy / reward 分别是什么？
2. **原始工作**：论文真正提出了什么，而不是后来社区如何包装？
3. **代码**：这一机制在 `agent.py / tools.py / memory.py / coding.py / worktree.py` 中落到哪里？
4. **验证**：怎样证明一次任务真的完成，而不是模型“说完成了”？

因此教材禁止把下面这种输出当作 Agent 成功：

```text
“I fixed the bug and all tests pass.”
```

除非系统真的执行过测试并获得可核验 observation。

---

## 5　最终目标

这条线的毕业标准不是“会用 LangChain / 某个 Agent SDK”，而是能从零搭出：

```text
Goal
→ inspect environment
→ form/update plan
→ choose tool
→ execute action
→ parse observation
→ persist useful state
→ verify progress
→ recover from failure
→ continue
→ produce artifact
→ verify final state
```

然后把它扩展到两类系统：

- **Codex-class**：真实代码仓库、编辑、测试、Git、worktree、review、merge；
- **Astra-class**：浏览器、计算机使用、文件、终端、多模态 observation、长程任务与跨工具协作。

---

## 6　证据原则

本目录完全继承教材的 Source-First 规范：

- ReAct 就回到 ReAct 原论文；
- Toolformer 就回到 Toolformer 原论文；
- Coding Agent 就看 SWE-bench / SWE-agent 等原始工作和源码；
- Browser / Computer Agent 就看 WebArena / OSWorld 等原始 benchmark；
- Multi-Agent 就区分真正的交互机制与只是多次调用模型的脚手架；
- 闭源 Agent 产品只写官方公开资料能支持的结论。

**智能体是 LLM 的重要延伸，但从研究对象上看，它已经是“模型 + 系统 + 环境”的交叉学科。**
