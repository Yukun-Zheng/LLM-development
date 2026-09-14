# A3　规划、反思、验证与失败恢复

> **核心问题**：为什么“先想好完整计划再执行”经常失败？Planner、Executor、Verifier、Reviewer 应该怎样分工？Reflection 到底是重新生成文本，还是基于新证据更新状态？

---

# 1　规划不是把任务切成几条 bullet

将目标 $g$ 分解为子任务集合：

$$
\mathcal T=\{\tau_1,\tau_2,\ldots,\tau_n\}.
$$

若存在依赖关系，则更合理的表示是 DAG：

$$
G=(V,E),\qquad V=\mathcal T.
$$

边：

$$
\tau_i\rightarrow\tau_j
$$

表示完成 $\tau_i$ 是执行 $\tau_j$ 的前置条件。

这种表示比一个固定文本计划更接近真实软件工程、研究和浏览器任务。

---

# 2　Open-loop plan 与 closed-loop plan

Open-loop：

```text
plan once
→ execute step 1
→ execute step 2
→ execute step 3
```

Closed-loop：

```text
plan
→ act
→ observe
→ update plan
→ act
→ observe
→ replan
```

真实环境会出现：

- 文件与预期不同；
- API 失败；
- test 暴露隐藏约束；
- 网站结构变化；
- 子任务已被其他 worker 修改。

所以高质量 Agent 的计划应被视为**可变状态**，而不是一次性自然语言输出。

---

# 3　Verifier 与 policy 分离

让同一个模型同时负责“做”和“判断自己做对没有”，会产生自我确认偏差。

可将系统拆为：

```text
Planner / Policy
      ↓
Action / Artifact
      ↓
Environment
      ↓
Verifier
      ↓
pass / fail / unknown + evidence
```

形式化：

$$
V(g,s_t,a_t,o_{t+1})\rightarrow (v,e),
$$

其中 $v$ 是 verdict，$e$ 是 evidence。

代码任务中的 verifier 可以直接使用 tests、compiler、type checker；这类外部 verifier 通常比让模型“自评”更可靠。

---

# 4　Self-Consistency、Tree Search 与 Agent Planning 的关系

推理阶段可以采样多个候选：

$$
y^{(1)},\ldots,y^{(K)}\sim p_\theta(y\mid x),
$$

然后通过 voting 或 verifier 选择。

相关原始工作：

- Self-Consistency: https://arxiv.org/abs/2203.11171
- Tree of Thoughts: https://arxiv.org/abs/2305.10601

但 Agent planning 与纯文本 reasoning search 的区别在于：**分支可以调用环境，获得真实 observation。**

因此搜索树节点不必只是 thought，也可以是：

```text
state + tool result + artifact state
```

---

# 5　Reflection：关键是新信息，不是“再想一遍”

Reflexion 将反馈转化为语言形式的经验，供后续尝试使用。

原始资料：Shinn et al., **Reflexion: Language Agents with Verbal Reinforcement Learning**  
https://arxiv.org/abs/2303.11366

更一般地，反思可以表示：

$$
m_{t+1}=U(m_t,o_{t+1},v_t),
$$

其中 memory update $U$ 应保留真正影响未来决策的信息。

低价值 reflection：

> “下次应该更仔细。”

高价值 reflection：

> “`cache_position` 必须使用 past length，而不能从 0 重新开始；失败测试为 `test_cached_logits_match_full_forward`。”

---

# 6　Failure Recovery 是一等公民

Agent 不应把错误当作异常终止：

```text
ToolError
Timeout
TestFailure
PermissionDenied
InvalidSchema
AmbiguousEdit
```

都应被转换为结构化 observation。

策略可以选择：

1. retry same action；
2. change arguments；
3. inspect more context；
4. replan；
5. use another tool；
6. request user intervention；
7. stop safely。

---

# 7　停止条件

长程 Agent 必须显式定义 stop policy：

$$
\text{stop}=f(goal\_verified,budget,irrecoverable\_failure,user\_cancel).
$$

不应该仅靠模型输出一句“done”。

建议状态至少包括：

```text
RUNNING
WAITING_FOR_TOOL
WAITING_FOR_USER
BLOCKED
VERIFIED_SUCCESS
FAILED
BUDGET_EXHAUSTED
```

---

# 8　源码扩展目标

当前 `agent.py` 已经实现最小 observe-act loop。下一阶段要加入：

- typed run state；
- mutable task plan；
- dependency graph；
- verifier protocol；
- evidence records；
- retry / recovery policy；
- explicit stop reason；
- checkpoint / resume。

这会把 Agent 从“循环调用工具”升级为**可观察、可调试、可恢复的任务执行系统**。

---

# 9　原始资料

- ReAct: https://arxiv.org/abs/2210.03629
- Self-Consistency: https://arxiv.org/abs/2203.11171
- Tree of Thoughts: https://arxiv.org/abs/2305.10601
- Reflexion: https://arxiv.org/abs/2303.11366
- Self-Refine: https://arxiv.org/abs/2303.17651

下一章讨论 Agent 最容易被混淆的另一个概念：Memory 与 Context Window 到底有什么区别。
