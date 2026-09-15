# Lab 07：从 Verifier 到 Group-Relative RL

> **目标**：把“可验证奖励（verifiable reward）”从概念变成一个可手算、可求导、可自动测试的 policy objective。当前实现是**教学用 GRPO-style clipped objective primitive**，不声称复刻任意厂商完整 RL recipe。

对应源码：[`../src/astra_codex/rlvr.py`](../src/astra_codex/rlvr.py)  
对应测试：[`../tests/test_rlvr.py`](../tests/test_rlvr.py)  
理论入口：[`../../../教材-book/08-推理模型时代-reasoning-era.md`](../../../教材-book/08-推理模型时代-reasoning-era.md)

原始资料：

- Shao et al., **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models**, 2024: https://arxiv.org/abs/2402.03300
- DeepSeek-AI et al., **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning**, 2025: https://arxiv.org/abs/2501.12948

---

# 1　RLVR 的关键变化：reward 可以由环境判断

普通 preference 数据依赖：

```text
human / preference model
→ chosen vs rejected
```

而很多 reasoning / coding 任务存在程序化 verifier：

```text
math answer exact match
test suite exit code
compiler result
proof checker
structured constraint
```

因此 trajectory $y$ 对 prompt $x$ 的 reward 可以直接写：

$$
r(x,y)=V(x,y),
$$

其中 $V$ 是可执行 verifier。

这并不意味着 reward 一定“真实无偏”。Verifier 仍可能有漏洞、覆盖不足、reward hacking 或 train/eval leakage。

---

# 2　为什么一个 prompt 要采多个 rollout

对于同一个 prompt $x_i$，采样 $G$ 个输出：

$$
y_{i,1},y_{i,2},\ldots,y_{i,G}.
$$

得到 reward：

$$
r_{i,1},r_{i,2},\ldots,r_{i,G}.
$$

Current teaching code expects：

```text
rewards: [B, G]
```

其中：

- $B$：prompt batch；
- $G$：每个 prompt 的 rollout group size。

这样可以比较：

> 对同一道题，哪些 sampled trajectories 比同组其它 trajectories 更好？

---

# 3　Group-relative advantage

当前 reference 定义：

$$
A_{i,j}
=
\frac{r_{i,j}-\mu_i}{\max(\sigma_i,\epsilon)},
$$

其中：

$$
\mu_i=\frac{1}{G}\sum_j r_{i,j}.
$$

这意味着不同 prompt 的 absolute reward scale 不直接互相比较。

例如：

```text
reward = [1, 0, -1]
```

归一化后仍满足：

```text
A_high > A_mid > A_low
mean(A) = 0
```

如果整组 reward 全相同：

```text
[5, 5, 5]
```

没有组内 preference signal，因此 reference implementation 返回全 0 advantage。

---

# 4　Policy ratio

设 rollout 是旧 policy $\pi_{old}$ 产生的，当前 policy 为 $\pi_\theta$。

对已经 reduction 成 sequence-level 的 log-prob：

$$
\rho_{i,j}
=
\exp\left(
\log\pi_\theta(y_{i,j}|x_i)
-
\log\pi_{old}(y_{i,j}|x_i)
\right).
$$

代码：

```python
ratios = torch.exp(new_logprobs - old_logprobs)
```

Shape：

```text
new_logprobs [B,G]
old_logprobs [B,G]
ratios       [B,G]
advantages   [B,G]
```

---

# 5　Clipped surrogate

当前 teaching objective 使用 PPO/GRPO-style clipping：

$$
L_{policy}
=
-\mathbb E\left[
\min\left(
\rho A,
\operatorname{clip}(\rho,1-\epsilon,1+\epsilon)A
\right)
\right].
$$

实现：

```python
unclipped = ratios * advantages
clipped = clipped_ratios * advantages
policy_loss = -torch.minimum(unclipped, clipped).mean()
```

这一步不应该被简单记成“GRPO 就是这三行”。真实训练还包含 rollout generation、token-level masking、old-policy snapshots、reference/KL、batching、reward processing、distributed updates 等。

---

# 6　一个有意思的手算不变量：loss 可以是 0，但 gradient 不为 0

若当前：

$$
\pi_\theta=\pi_{old},
$$

则：

$$
\rho=1.
$$

而 group-normalized advantage 满足：

$$
\frac1G\sum_j A_j=0.
$$

所以 objective 的**当前标量值**可能：

$$
L=0.
$$

但每个 rollout log-prob 是不同变量：

$$
\frac{\partial L}{\partial \log\pi(y_j)}
\propto -A_j.
$$

因此：

```text
high-reward sample → gradient descent raises its log-prob
low-reward sample  → gradient descent lowers its log-prob
```

`test_grpo_style_objective_is_zero_value_but_has_useful_gradient_at_old_policy` 就在检查这个现象。

---

# 7　Reference KL penalty

如果提供 frozen reference log-prob，当前教学实现使用一个非负 estimator：

$$
D=\exp(\Delta)-\Delta-1,
$$

其中：

$$
\Delta=\log\pi_{ref}-\log\pi_\theta.
$$

然后：

$$
L=L_{policy}+\beta\,\mathbb E[D].
$$

其目的不是宣称这是唯一 KL 写法，而是让：

```text
policy improvement pressure
vs
reference drift control
```

同时出现在同一个可检查 objective 中。

---

# 8　当前还没有实现什么

`rlvr.py` 当前输入已经是：

```text
rollout sequence log-probs
rewards
```

因此完整训练链仍缺：

```text
prompt dataset
→ sample G rollouts
→ environment/verifier
→ reward records
→ old-policy log-probs
→ token/sequence policy log-probs
→ group-relative objective
→ optimization
→ held-out evaluator
```

特别是：

> **training verifier 不能和 held-out evaluator 混成同一个东西。**

否则很容易把 reward hacking 当成 reasoning improvement。

---

# 9　运行

```bash
cd '代码-code/从零构建Astra与Codex级系统'
pip install -e '.[dev]'
pytest tests/test_rlvr.py -q
```

下一步的真正里程碑是把：

```text
SFT
→ DPO
→ verifier-generated reward
→ group-relative policy update
```

串在同一个 tiny model / held-out task 上，画出训练曲线、reward 曲线与真正 held-out success 曲线。
