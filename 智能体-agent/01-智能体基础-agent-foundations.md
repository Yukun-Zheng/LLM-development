# A1　智能体第一性原理：从语言模型到闭环决策系统

> **核心问题**：LLM、Tool-Using LLM、Agent 三者到底有什么结构性区别？为什么“多轮调用模型”还不一定构成一个好的智能体？

---

# 1　从静态映射到动态过程

普通语言模型推理可以写成：

$$
y\sim p_\theta(y\mid x).
$$

而一个最小智能体需要至少包含：目标、状态、动作、环境、观察与停止条件。

令：

- $g$：任务目标；
- $s_t$：环境真实状态；
- $o_t$：智能体能看到的 observation；
- $h_t$：智能体累计历史；
- $a_t$：动作；
- $\pi_\theta$：决策策略。

则：

$$
a_t\sim \pi_\theta(a_t\mid h_t,g),
$$

$$
s_{t+1}\sim P(s_{t+1}\mid s_t,a_t),
$$

$$
o_{t+1}\sim O(o_{t+1}\mid s_{t+1}),
$$

$$
h_{t+1}=h_t\oplus(a_t,o_{t+1}).
$$

这已经接近一个部分可观测决策过程（POMDP）的表达。LLM 在这里通常只实现 policy / planner 的一部分，而不是整个 Agent。

---

# 2　Agent 的最小闭环

```text
Goal
 ↓
Policy / LLM
 ↓
Action
 ↓
Environment
 ↓
Observation
 ↓
Update state/history
 ↓
Policy / LLM
```

关键不是“循环”两个字，而是**动作真实改变环境，并且新环境反馈会改变下一步决策**。

例如代码任务：

```text
read file
→ edit
→ pytest
→ test failure
→ inspect traceback
→ edit again
→ pytest
→ pass
```

失败 observation 是新的信息，因此下一步 policy 与第一次不同。

---

# 3　Tool-Using LLM 与 Agent 的边界

可以按能力层级理解：

| 层级 | 结构 | 例子 |
|---|---|---|
| L0 | 单次生成 | 问答、摘要 |
| L1 | Structured Output | JSON、schema output |
| L2 | Tool Use | 一次函数调用 |
| L3 | Agent Loop | action → observation → next action |
| L4 | Stateful Agent | memory、checkpoint、recovery |
| L5 | Long-Horizon Agent | plan、verifier、budget、resume |
| L6 | Multi-Agent System | coordinator、workers、reviewer |

因此“模型支持 function calling”不等于“系统已经解决长期自主任务”。

---

# 4　ReAct 的历史位置

ReAct 把 reasoning traces 与 environment actions 交错组织：

```text
Reason
→ Act
→ Observe
→ Reason
→ Act
→ Observe
```

原始资料：Yao et al., **ReAct: Synergizing Reasoning and Acting in Language Models**  
https://arxiv.org/abs/2210.03629

ReAct 的重要性不是给出一个固定提示词模板，而是明确展示：**语言推理与环境交互可以在同一轨迹中互相校正**。

后续 Agent 系统即使不显式输出“Thought”，仍然保留了 action / observation 闭环。

---

# 5　Agent 的状态不等于聊天记录

聊天记录只是 $h_t$ 的一种表示。真实系统还需要：

- current plan；
- files changed；
- test state；
- tool permissions；
- user constraints；
- task budget；
- persistent notes；
- checkpoints；
- unresolved failures；
- artifact locations。

因此更一般地：

$$
z_t=f(h_t,m_t,e_t),
$$

其中 $m_t$ 是长期 memory，$e_t$ 是外部环境摘要；真正送进模型的是压缩后的 decision context $z_t$。

---

# 6　Agent 的成功标准必须外部化

语言模型自身可以生成：

> “任务已经完成。”

但这不是证据。

应定义 verifier：

$$
V(s_T,g)\rightarrow \{\text{pass},\text{fail},\text{unknown}\}.
$$

不同任务的 verifier 不同：

- 数学题：最终答案；
- 代码：tests / typecheck / lint；
- 浏览器：网页目标状态；
- 文件任务：artifact 是否存在且满足 schema；
- 数据分析：数值约束与输出文件；
- 机器人：环境 reward / success condition。

这也是为什么 Agent evaluation 比普通 benchmark 更接近系统工程。

---

# 7　成本与预算也是 Agent 状态

长程系统必须考虑：

$$
B=(B_{tokens},B_{steps},B_{time},B_{money},B_{tools}).
$$

一个“能做成但需要无限 retry”的系统并不等价于高质量 Agent。

因此真实评价至少要报告：

- success rate；
- step count；
- tool calls；
- latency；
- token / compute cost；
- recovery rate；
- destructive-action rate。

---

# 8　源码映射

当前工程：

- `agent.py`：最小 observe → act → observe 状态机；
- `structured.py`：模型文本到 action 的协议边界；
- `tools.py`：环境 action 与 observation；
- `memory.py`：persistent event state；
- `coding.py`：代码仓库任务的 specialized policy scaffold。

工程入口：  
[`../代码-code/从零构建Astra与Codex级系统/`](../代码-code/从零构建Astra与Codex级系统/)

---

# 9　原始资料

- WebGPT: Nakano et al., 2021, **WebGPT: Browser-assisted question-answering with human feedback**  
  https://arxiv.org/abs/2112.09332
- ReAct: Yao et al., 2022/2023  
  https://arxiv.org/abs/2210.03629
- Toolformer: Schick et al., 2023  
  https://arxiv.org/abs/2302.04761
- MRKL Systems: Karpas et al., 2022  
  https://arxiv.org/abs/2205.00445

下一章把“动作”进一步拆成可执行工具、环境协议和安全边界。
