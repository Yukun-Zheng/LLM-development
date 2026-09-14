# A8　环境学习、Agentic RL 与持续适应

> **核心问题**：一个 Agent 能不能不仅在环境中执行，还通过环境反馈持续变强？Agentic RL 与传统 RL、RLHF、RLVR 有什么关系？

---

# 1　从“推理时 Agent”到“学习型 Agent”

很多 Agent 系统只是：

```text
frozen model
+ tools
+ memory
+ planner
+ environment loop
```

模型参数在任务过程中并不更新。

学习型 Agent 则进一步考虑：

$$
\pi_{\theta_{k+1}} = \mathrm{Update}(\pi_{\theta_k},\tau_k,r_k),
$$

其中轨迹：

$$
\tau_k=(o_0,a_0,o_1,a_1,\ldots,o_T).
$$

也就是说，Agent 不只“使用环境”，还把环境反馈转成后续 policy 的学习信号。

---

# 2　RLHF、RLVR 与 Agentic RL 的区别

## RLHF

reward 来自人类偏好模型或直接人工反馈。

## RLVR

reward 来自可验证任务，例如：

- 数学答案；
- 单元测试；
- 编译结果。

## Agentic RL

reward / trajectory 来自**长期环境交互**，动作不再只是一段最终文本，而可能包括：

```text
search
browse
edit
execute
click
wait
retry
ask-user
```

因此 credit assignment 横跨多个 action。

---

# 3　轨迹回报

标准 discounted return：

$$
G_t=\sum_{k=t}^{T}\gamma^{k-t}r_k.
$$

但真实 Agent reward 常常稀疏：

```text
step 1..37: 0
step 38: task success = 1
```

问题变成：哪些中间决策真正贡献了最终成功？

这使 long-horizon agent learning 比“数学题最后答案对不对”困难很多。

---

# 4　环境反馈不一定需要立即更新参数

广义的 adaptation 有多层：

1. **in-context adaptation**：只改变当前上下文；
2. **memory update**：持久化经验但不改权重；
3. **retrieval update**：更新外部知识库；
4. **policy prompt / scaffold update**；
5. **offline fine-tuning**；
6. **online RL / continual learning**。

因此“Agent 会学习”必须明确到底是哪一层在更新。

---

# 5　Verifier 是 Agentic RL 的关键基础设施

如果环境能够返回可靠 success signal：

$$
r=V(trajectory,final\ state),
$$

就可以自动生成大规模交互训练数据。

软件工程就是典型例子：

```text
patch
→ tests
→ pass/fail
```

浏览器任务则可以检查：

```text
cart state
form state
account state
URL / DOM condition
```

但 verifier 不完美时，Agent 会学会 exploit reward，而不是完成真实目标。

---

# 6　探索与安全的冲突

RL 需要 exploration；真实工具环境却不能任意探索：

```text
rm -rf
send email
purchase
push production config
```

因此真实 Agentic RL 需要把 exploration 限定在 sandbox / simulator / reversible environment。

形式上可以使用 constrained MDP：

$$
\max_\pi \mathbb E[R]\quad
\mathrm{s.t.}\quad \mathbb E[C_i]\le d_i.
$$

其中 $C_i$ 表示安全成本。

---

# 7　Agent trajectory 数据本身是一种新数据类型

普通预训练数据：

```text
text tokens
```

Agent 数据：

```text
goal
observation
action
observation
action
...
reward
artifact
verifier evidence
```

这意味着未来的数据工程不仅要管理 corpus，还要管理：

- environment version；
- tool schema；
- action validity；
- trajectory provenance；
- reward provenance；
- replay / simulation；
- off-policy contamination。

---

# 8　持续学习问题

如果 Agent 长期在线更新，还要面对：

- catastrophic forgetting；
- policy drift；
- reward hacking；
- stale memories；
- security poisoning；
- distribution shift；
- rollback / versioning。

所以“永远在线学习”不是简单把 optimizer 常开，而是系统级研究问题。

---

# 9　与本教材的连接

这条线连接：

```text
第三篇：RLHF / preference
第八篇：Reasoning RL / RLVR
智能体 A1-A7：环境执行系统
        ↓
A8：Agentic RL / environment learning
        ↓
第十八篇：Astra-class / Codex-class capstone
```

也就是说，智能体不是一个纯 inference-time 话题；它最终会反过来影响训练数据、reward design、后训练和持续学习。

---

# 10　当前工程边界

我们的源码目前实现的是**冻结 policy 下的 Agent runtime**，尚未实现在线参数更新。

当前可做的是：

- 记录完整 trajectory；
- 持久化 observations；
- 保存 verifier evidence；
- 为未来 offline / online Agentic RL 构造数据接口。

只有真正加入 learning update 并有实验后，才会把“Agentic RL 已实现”打勾。

---

# 11　原始资料连接

这一方向与以下工作直接相连：

- WebGPT（浏览器交互 + 人类反馈）: https://arxiv.org/abs/2112.09332
- DeepSeekMath / GRPO（可验证推理奖励）: https://arxiv.org/abs/2402.03300
- DeepSeek-R1（reasoning RL）: https://arxiv.org/abs/2501.12948
- ReAct（推理与环境动作交错）: https://arxiv.org/abs/2210.03629

Agentic RL 仍是快速演化的领域，教材会严格区分公开证据、系统脚手架与真正参数学习。
