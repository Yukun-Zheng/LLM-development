# 附录 C　RLHF、DPO、GRPO 与 RLVR：后训练数学速查

> 本附录不是“算法名词表”，而是统一回答：**语言模型经过预训练以后，怎样再利用人类偏好、奖励模型和可验证反馈改变行为？**

---

## C.1　从预训练到后训练：优化目标发生了什么

预训练主要优化 next-token negative log-likelihood：

$$
\mathcal L_{\mathrm{pretrain}}
=-\mathbb E_{x\sim\mathcal D}
\sum_t\log\pi_\theta(x_t\mid x_{<t}).
$$

它学习“互联网/语料中什么文本更可能出现”。但用户真正希望的是：

- 按要求回答；
- 遵守格式；
- 更有帮助；
- 少胡说；
- 在推理任务上找到正确答案；
- 必要时调用工具；
- 在长程环境中完成任务。

这意味着训练信号要从“观测文本”扩展到“期望行为”。后训练就是围绕这个差异展开。

---

# C.2　SFT：最简单也最重要的后训练

给定 prompt $x$ 和高质量回答 $y=(y_1,\ldots,y_T)$，SFT 仍然是交叉熵：

$$
\mathcal L_{\mathrm{SFT}}
=-\sum_{t=1}^{T}\log \pi_\theta(y_t\mid x,y_{<t}).
$$

与预训练的区别主要不在数学形式，而在**数据分布和意图**：

```text
预训练：互联网中“接下来通常是什么”
SFT：   给定指令后“我们希望模型怎样回答”
```

SFT 是 imitation learning。它可以教行为格式、风格、任务模式和一部分能力，但它无法直接表达：

> 回答 A 和回答 B 都可行，但人更喜欢 A。

这引出 preference learning。

InstructGPT 的经典训练流程见：Ouyang et al., 2022, https://arxiv.org/abs/2203.02155

---

# C.3　Preference Data：监督信号从“答案”变成“比较”

典型偏好样本：

$$
(x,y_w,y_l),
$$

其中：

- $x$：prompt；
- $y_w$：winner / preferred response；
- $y_l$：loser / dispreferred response。

标注者不必写出完美答案，只需要判断哪个更好。这通常比从零撰写专家答案更容易扩展。

---

# C.4　Reward Model：把偏好变成标量

经典 RLHF 先训练 reward model：

$$
r_\phi(x,y)\in\mathbb R.
$$

常见 Bradley–Terry 形式：

$$
P(y_w\succ y_l\mid x)
=
\sigma\left(
 r_\phi(x,y_w)-r_\phi(x,y_l)
\right).
$$

因此 reward-model loss：

$$
\mathcal L_{RM}
=-\log\sigma\left(
 r_\phi(x,y_w)-r_\phi(x,y_l)
\right).
$$

直觉：只要求 preferred response 的 reward 比 rejected response 高，而不是要求绝对分数有某种物理意义。

来源：

- Christiano et al., 2017, *Deep Reinforcement Learning from Human Preferences*: https://arxiv.org/abs/1706.03741
- Ouyang et al., 2022: https://arxiv.org/abs/2203.02155

---

# C.5　RLHF 的核心目标：既追求奖励，又别偏离太远

一个常见抽象形式：

$$
\max_\pi
\mathbb E_{x\sim\mathcal D,\,y\sim\pi(\cdot|x)}
[r_\phi(x,y)]
-
\beta D_{KL}
\left[
\pi(\cdot|x)\|\pi_{ref}(\cdot|x)
\right].
$$

第一项：提高 reward。

第二项：限制新 policy 不要离 reference model 太远。

原因很实际：如果只最大化 learned reward，policy 可能利用 reward model 的缺陷，即 reward hacking / overoptimization。

`\beta` 控制“追求偏好”与“保持原模型分布”的权衡。

---

# C.6　PPO：为什么早期 RLHF 常用它

PPO 的经典 clipped surrogate objective：

$$
L^{CLIP}(\theta)
=
\mathbb E_t
\left[
\min\left(
 r_t(\theta)\hat A_t,
 \operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
\right)
\right],
$$

其中 probability ratio：

$$
r_t(\theta)
=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{old}}(a_t\mid s_t)}.
$$

在 LLM 中可以把：

- state 理解为 prompt + 已生成 token；
- action 理解为下一个 token；
- rollout 是整条回答；
- reward 通常主要在 sequence 级得到。

PPO 的优势是成熟、能做在线采样；缺点是系统复杂，需要同时维护/调用 policy、reference、reward model，往往还有 value model，并进行 rollout。

来源：Schulman et al., 2017, https://arxiv.org/abs/1707.06347

---

# C.7　DPO：为什么“不要显式 RL”也能学偏好

DPO 从 KL-regularized RLHF 的最优 policy 关系出发，把隐式 reward 写成 policy 与 reference policy 的 log-ratio，从而直接在 preference pair 上训练。

经典 DPO loss：

$$
\mathcal L_{DPO}(\theta)
= -\mathbb E_{(x,y_w,y_l)}
\log\sigma\Bigg(
\beta
\Big[
\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}
-
\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}
\Big]
\Bigg).
$$

定义：

$$
\Delta_\theta
=
\log\pi_\theta(y_w|x)-\log\pi_\theta(y_l|x),
$$

$$
\Delta_{ref}
=
\log\pi_{ref}(y_w|x)-\log\pi_{ref}(y_l|x),
$$

则：

$$
\mathcal L_{DPO}
=-\log\sigma[\beta(\Delta_\theta-\Delta_{ref})].
$$

直观上：

> 新模型要比 reference model **更强烈地偏好 winner，而不是简单地把 winner 概率拉高**。

DPO 不需要显式训练 reward model，也不需要 PPO rollout，因此工程明显简化。

来源：Rafailov et al., 2023, https://arxiv.org/abs/2305.18290

---

# C.8　DPO 不等于“RLHF 已经没用了”

这是常见误读。

DPO 主要适合已有离线 preference pair 的场景。在线 RL 的优势在于 policy 可以持续产生新行为、从环境或 verifier 获得新反馈，并探索训练数据之外的轨迹。

所以应该区分：

```text
离线 preference optimization
    DPO / IPO / KTO / ORPO / ...

在线 reinforcement learning
    PPO / GRPO / variants
```

它们解决的计算与数据条件并不完全相同。

---

# C.9　GRPO：用组内相对奖励替代独立 value model

GRPO（Group Relative Policy Optimization）在 DeepSeekMath 中被系统提出并用于数学推理训练。核心思想之一是：针对同一个 prompt 采样一组输出，使用组内 reward 的相对位置估计 advantage，从而避免 PPO 中额外训练与 policy 同规模的 critic/value model。

对同一问题 $q$ 采样：

$$
\{o_1,o_2,\ldots,o_G\}\sim\pi_{old}(\cdot|q).
$$

获得 reward：

$$
r_1,r_2,\ldots,r_G.
$$

一个最直观的 group-relative 标准化 advantage：

$$
\hat A_i
=
\frac{r_i-\operatorname{mean}(r_1,\ldots,r_G)}
{\operatorname{std}(r_1,\ldots,r_G)+\epsilon}.
$$

然后仍使用 importance ratio / clipping 约束更新幅度，并通常加入 KL regularization。

重要的是理解它的**相对比较逻辑**：

```text
同一个 prompt
  ├─ response 1 → reward 0
  ├─ response 2 → reward 1
  ├─ response 3 → reward 0
  ├─ response 4 → reward 1
  └─ response 5 → reward 0

不是问“1 分绝对有多好”
而是问“同组里哪些轨迹更值得强化”
```

来源：DeepSeekMath, Shao et al., 2024, https://arxiv.org/abs/2402.03300

---

# C.10　RLVR：奖励从“人觉得好”变成“机器能验证对错”

Reasoning 模型时代的关键变化，是大量任务存在便宜、可靠、自动化的 verifier：

- 数学最终答案；
- 代码 unit tests；
- theorem prover；
- constraint satisfaction；
- tool execution result；
- environment success signal。

这类训练常被概括为 reinforcement learning with verifiable rewards（RLVR）。

例如数学任务：

$$
r(y)=
\begin{cases}
1,&\operatorname{answer}(y)=a^*\\
0,&\text{otherwise}
\end{cases}
$$

相比 learned reward model，这种 reward 的优势是“标准更硬”，不那么容易因为 reward model 主观偏差而漂移；但它也有明显边界：大量真实任务并没有廉价的自动 verifier。

DeepSeek-R1 是 reasoning RL 路线中的关键公开案例之一。R1-Zero 展示了在没有先进行传统 SFT 冷启动的条件下，直接进行大规模 RL 可以出现长推理、反思等行为；正式 R1 则加入 cold-start data 和多阶段训练来改善可读性与综合能力。

来源：DeepSeek-AI, 2025, https://arxiv.org/abs/2501.12948

---

# C.11　Outcome Reward 与 Process Reward

## Outcome reward

只看最终结果：

$$
r=r(y_{final}).
$$

优点：标注简单、目标明确。

缺点：credit assignment 困难。一条 5000-token reasoning trajectory 最后错了，并不知道错误发生在第 300 token 还是第 4300 token。

## Process reward

对中间步骤提供反馈：

$$
r_t=r(s_t,a_t).
$$

理论上可改善 credit assignment，但需要可靠地判断“中间一步是否真的正确”，成本和偏差都可能很高。

相关代表工作：Lightman et al., 2023, *Let’s Verify Step by Step*, https://arxiv.org/abs/2305.20050

---

# C.12　为什么 reasoning RL 会改变输出长度

假设更多 test-time computation 有机会提升成功率，那么 policy 可能学会：

- 写更长的中间推导；
- 回溯；
- 验证前一步；
- 尝试多个策略；
- 在不确定处延长搜索。

所以 reasoning token 数本身会成为一种隐式 compute allocation。

但“更长”绝不自动等于“更聪明”：模型也可能产生冗余循环、伪反思和无效 token。评测必须同时看：

$$
\text{accuracy},\quad
\text{tokens},\quad
\text{latency},\quad
\text{cost}.
$$

只比较最终 benchmark 而不比较 test-time compute，是 reasoning 模型时代很容易产生误导的做法。

---

# C.13　KL 项的真正作用

KL regularization 常写：

$$
D_{KL}(\pi_\theta\|\pi_{ref}).
$$

它不是“让模型永远不变化”，而是在说：

> 在 reward 信号可能有噪声、不完整或可被利用时，不要让 policy 为追逐 reward 而无约束地跑到极端分布。

若 $\beta$ 太大：

- policy 太保守；
- 学不到新行为。

若 $\beta$ 太小：

- policy 可能过度优化 reward；
- 语言质量、泛化或安全能力下降。

因此 post-training 本质上也是一个 regularized optimization problem。

---

# C.14　Reward Hacking：为什么“分数上升”未必是能力上升

如果训练目标是：

$$
\max_\pi \mathbb E[r(y)],
$$

那么模型会优化 **reward function 实际写下来的东西**，而不是设计者脑中真正想要的东西。

例子：

- verifier 只检查最终字符串，模型可能利用格式漏洞；
- reward model 喜欢长答案，模型可能无限扩写；
- judge model 存在固定偏好，policy 可能学会讨好 judge；
- coding benchmark tests 不完整，程序可以“过测试但不正确”。

因此可靠 RL 系统需要：

```text
任务设计
 + verifier 设计
 + adversarial evaluation
 + held-out tests
 + reward audits
 + distribution-shift tests
```

不能只看训练 reward 曲线。

---

# C.15　SFT、DPO、PPO、GRPO、RLVR 一张表

| 方法 | 数据/反馈 | 是否在线采样 | 显式 Reward Model | Critic / Value Model | 最适合回答的问题 |
|---|---|---:|---:|---:|---|
| SFT | 标准答案/示范 | 否 | 否 | 否 | “应该怎么回答？” |
| DPO | winner/loser preference | 否 | 否 | 否 | “两种回答更喜欢哪种？” |
| PPO-RLHF | learned reward | 是 | 通常是 | 通常是 | “怎样最大化人类偏好 reward？” |
| GRPO | 一组 rollout 的相对 reward | 是 | 可选 | 通常不需要独立 critic | “组内哪些策略更成功？” |
| RLVR | 自动 verifier | 是 | 通常不需 learned RM | 取决于算法 | “答案/轨迹是否可验证地成功？” |

注意：**RLVR 是 reward 来源/训练范式描述，不是单一 optimizer 的名字。** RLVR 可以结合不同 policy-optimization 算法。

---

# C.16　从数学上看后训练到底在干什么

把整个过程抽象成一句话：

$$
\text{pretraining prior}
+
\text{behavioral supervision}
+
\text{preference / reward signal}
+
\text{regularization}
\rightarrow
\text{new policy}.
$$

预训练提供巨大行为先验；后训练不是从零“创造智能”，而是在已有模型分布上重塑哪些轨迹更容易出现。

这也是为什么 base model 质量仍然重要：后训练可以重新分配已有能力、强化部分能力，并通过在线探索获得新行为，但它不是无限免费的能力生成器。

---

# C.17　一个最小 preference-learning 手算例子

假设对同一个 prompt：

```text
winner: y_w
loser : y_l
```

reference model：

$$
\log\pi_{ref}(y_w|x)=-4.0,
\qquad
\log\pi_{ref}(y_l|x)=-3.5.
$$

它原本反而更偏好 loser：

$$
\Delta_{ref}=-4.0-(-3.5)=-0.5.
$$

当前 policy：

$$
\log\pi_\theta(y_w|x)=-3.0,
\qquad
\log\pi_\theta(y_l|x)=-3.2.
$$

于是：

$$
\Delta_\theta=-3.0-(-3.2)=0.2.
$$

相对 improvement：

$$
\Delta_\theta-\Delta_{ref}=0.7.
$$

若 $\beta=1$，DPO 目标中的 sigmoid 输入为 0.7：

$$
\sigma(0.7)\approx0.668.
$$

loss：

$$
-\log(0.668)\approx0.403.
$$

如果 policy 进一步提高 winner 相对 loser 的优势，loss 继续下降。

这个例子说明：DPO 关心的是**相对 reference 的 preference margin 改变**。

---

# C.18　Reasoning RL 最小实验设计

如果要真正掌握，而不是只读 R1 论文，建议做一个小实验：

### 任务

小学/中学算术表达式或 GSM8K 子集。

### Policy

一个 0.5B–3B 开源模型或更小教学模型。

### Reward

只解析最终 `Answer:` 字段：

$$
r=\mathbb{1}[\hat a=a^*].
$$

### 对照

1. base model；
2. 只 SFT；
3. SFT + GRPO/RLVR；
4. 改变 rollout group size；
5. 改变最大 reasoning tokens。

### 至少记录

- pass@1；
- pass@k；
- 平均输出 token 数；
- reward；
- entropy；
- KL to reference；
- wall-clock time；
- 每道正确题的平均 inference FLOPs/近似成本。

只有把“准确率提升”与“额外计算”一起记录，才能判断是否真的学到了更好的 reasoning policy。

---

# C.19　核心来源

1. Christiano et al., 2017, *Deep Reinforcement Learning from Human Preferences*: https://arxiv.org/abs/1706.03741
2. Schulman et al., 2017, *Proximal Policy Optimization Algorithms*: https://arxiv.org/abs/1707.06347
3. Ouyang et al., 2022, *Training language models to follow instructions with human feedback*: https://arxiv.org/abs/2203.02155
4. Rafailov et al., 2023, *Direct Preference Optimization*: https://arxiv.org/abs/2305.18290
5. Lightman et al., 2023, *Let’s Verify Step by Step*: https://arxiv.org/abs/2305.20050
6. Shao et al., 2024, *DeepSeekMath*: https://arxiv.org/abs/2402.03300
7. DeepSeek-AI, 2025, *DeepSeek-R1*: https://arxiv.org/abs/2501.12948

---

## C.20　读完后的最低验收标准

你应该能准确回答：

- SFT 和预训练为什么数学形式类似、语义却不同；
- reward model 如何从 pairwise preference 学习；
- PPO 的 ratio 与 clipping 在约束什么；
- DPO 为什么能绕过显式 reward model；
- GRPO 为什么可以不用独立 critic；
- RLVR 为什么特别适合数学和代码；
- outcome reward 与 process reward 的 credit assignment 差异；
- 为什么 reasoning benchmark 必须同时报告 test-time compute；
- reward hacking 为什么是优化问题而不是“模型道德问题”。

掌握这些以后，你再看到“某模型通过 RL 获得 reasoning 能力”时，就能继续追问：**reward 是谁给的、数据怎么采、policy 怎么更新、reference 是谁、KL 怎么控、verifier 能否被 hack、额外消耗了多少 test-time compute。**