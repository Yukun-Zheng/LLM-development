# A6　多智能体、角色分工与并行工作树

> **核心问题**：多智能体什么时候真的有用？把同一个模型调用三次为什么不自动变成“团队”？如何处理任务分解、依赖、并行、通信、冲突、评审和合并？

---

# 1　单 Agent 的结构性瓶颈

单 Agent 在长任务中常遇到：

- context 被多个子任务污染；
- 一个策略同时承担 planner / executor / reviewer；
- 不同子任务可并行但被串行执行；
- 自己修改的结果又由自己审查；
- 同一工作目录难以隔离试验分支。

Multi-Agent 的合理动机不是“模拟社会”，而是**分解状态、并行工作和引入独立验证视角**。

---

# 2　Multi-Agent 的最小形式化

设有 $N$ 个 agent：

$$
\Pi=\{\pi_1,\pi_2,\ldots,\pi_N\}.
$$

每个 agent 拥有局部状态 $h_i$，通过通信消息 $m_{i\rightarrow j}$ 交换信息：

$$
a_i\sim \pi_i(a_i\mid h_i,g_i),
$$

$$
h_j\leftarrow h_j\oplus m_{i\rightarrow j}.
$$

真正的问题是：

- $g_i$ 如何分配？
- 哪些 state 共享？
- 哪些 artifact 隔离？
- 冲突怎样发现？
- 谁拥有最终 merge 权限？

---

# 3　常见拓扑

## Coordinator–Worker

```text
Coordinator
├─ Worker A
├─ Worker B
└─ Worker C
```

适合：明确子任务、可并行、最后汇总。

## Planner–Executor–Reviewer

```text
Planner → Executor → Reviewer
   ↑                    │
   └──── replan ←───────┘
```

适合：需要独立验证。

## Peer-to-Peer

多个 agent 自主通信。灵活，但容易形成：

- message explosion；
- responsibility ambiguity；
- duplicated work；
- deadlock / circular debate。

---

# 4　任务图比“角色扮演”更重要

多 Agent 调度应建立 task DAG：

$$
G=(V,E).
$$

只有当依赖节点完成后，任务才 ready。

例如：

```text
inspect API ──┐
              ├─> implement adapter ─> integration test
inspect tests ┘

docs audit ────────────────> documentation update
```

这样 scheduler 才能知道哪些任务可以真正并行。

---

# 5　Coding Agent 中的 Worktree 隔离

对软件工程，最自然的 worker isolation 是 Git worktree：

```text
repo/main
├─ ../worker-a   branch agent/a
├─ ../worker-b   branch agent/b
└─ ../review     branch review
```

worker 的输出不是一段聊天，而是：

- commit；
- diff；
- tests；
- evidence；
- notes。

这使 reviewer 可以基于 artifact 而不是语言描述做合并判断。

---

# 6　通信应该传“决策充分信息”

最差的多 Agent 系统会互相发送完整 transcript。

更合理的消息：

```json
{
  "task":"implement cache position fix",
  "status":"done",
  "commit":"abc123",
  "tests":["test_cache.py::test_decode_parity"],
  "risks":["only tested on CPU"],
  "notes":["position must start at past_len"]
}
```

也就是说 communication 本身需要 schema 和 provenance。

---

# 7　并行不等于更快

若子任务高度耦合，多 worker 会增加：

- merge conflicts；
- duplicate exploration；
- communication overhead；
- reviewer burden。

因此多 Agent 是否值得，应比较：

$$
\text{speedup}=\frac{T_{single}}{T_{multi}},
$$

以及总计算开销：

$$
C_{multi}=\sum_i C_i+C_{coord}+C_{review}.
$$

不能只看 wall-clock time，而忽略 token / compute 翻倍。

---

# 8　原始研究与框架

多智能体 LLM 系统的早期代表包括：

- CAMEL: https://arxiv.org/abs/2303.17760
- AutoGen: https://arxiv.org/abs/2308.08155

这些工作展示了角色交互与多 Agent conversation 的可能性，但教材会进一步强调：**对工程任务而言，文件系统隔离、task graph、artifact verification、merge policy 往往比“角色设定”更关键。**

---

# 9　源码映射与下一步

当前已有：

- `worktree.py`：Git worktree primitive；
- `memory.py`：可持久化事件；
- `coding.py`：单 worker coding loop。

后续代码升级：

1. `planning.py`：task graph；
2. `multi_agent.py`：coordinator / worker scheduler；
3. worker result schema；
4. reviewer protocol；
5. merge / conflict handling；
6. budget accounting；
7. parallel execution tests。

下一章讨论怎样评测这些系统，而不是只展示几个成功 demo。
