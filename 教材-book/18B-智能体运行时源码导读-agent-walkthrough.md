# 第十八篇附录 B　智能体运行时源码导读：从 Tool Call 到 Codex-class 仓库闭环

> 对应工程：`structured.py`、`tools.py`、`agent.py`、`editing.py`、`repo_map.py`、`coding.py`、`memory.py`、`worktree.py`

---

# 1　把 Agent 从营销词还原成状态机

最小 agent 可以写成：

$$
s_{t+1}=F(s_t,a_t,o_{t+1}),
$$

其中：

- $s_t$：历史消息、任务状态、memory；
- $a_t$：模型输出的 action；
- $o_{t+1}$：环境执行 action 后返回的 observation。

ReAct 原始资料：https://arxiv.org/abs/2210.03629

这比“让模型多思考几步”更精确，因为环境是闭环的一部分。

---

# 2　`structured.py`：模型文本怎样变成可执行动作

当前协议：

```json
{"tool":"filesystem","arguments":{"action":"read","path":"README.md"}}
```

解析层必须区分：

```text
ordinary text → final answer
valid tool JSON → action
invalid schema → failed observation
```

后续再升级为 token-level grammar constraints，而不是一开始把关键机制藏进 vendor API。

---

# 3　`tools.py`：工具系统是执行边界

当前工具：

- rooted filesystem；
- shell；
- Git；
- HTTP GET browser。

每个工具都有 schema、executor、timeout/error behavior。

必须特别强调：`cwd=repo` **不是 security sandbox**。真正安全执行代码需要 container / VM / namespace / network / secret boundaries。

---

# 4　`editing.py`：为什么第一版不用“模糊编辑”

第一版编辑器要求 `old` 在文件中**恰好出现一次**。

原因：coding agent 最危险的失败之一，是修改了“看起来相似但其实错误”的位置。

exact edit 失败时强迫 agent：

```text
read more context
→ construct more specific old text
→ retry
```

这是比“尽量帮它猜”更适合研究正确性的策略。

---

# 5　`repo_map.py`：先建立结构，再读仓库

大仓库不能把所有文件全塞进 context。

第一版 repo map 从：

- Python AST 的 class / function / method；
- Markdown headings

生成低成本结构地图。

下一阶段再进入 tree-sitter、LSP、reference graph、import graph。

---

# 6　`agent.py`：失败不是异常终点

工具错误会被序列化为 observation，再交回 policy。

因此：

```text
pytest failed
```

不是 Python agent process crash，而是下一步决策的数据。

这正是长程任务 recovery 的基础。

---

# 7　`coding.py`：Codex-class 的第一个闭环

目前已经可以表达：

```text
repo_map
  ↓
search/read
  ↓
exact edit
  ↓
pytest/lint/build
  ↓
inspect failure
  ↓
edit again
  ↓
git diff
  ↓
final answer
```

“会生成 Python”与“能在仓库里闭环修问题”是两个不同的能力层级。

SWE-bench 原始资料：https://arxiv.org/abs/2310.06770

---

# 8　`memory.py`：长上下文不等于长期状态

上下文窗口是模型本次 forward 能看到多少 token；长期任务状态则需要：

- persistent event log；
- notes；
- searchable history；
- compaction；
- checkpoints。

第一版使用 SQLite append-only events，确保任务状态可以跨进程存在。

---

# 9　`worktree.py`：为什么并行 coding agent 需要文件系统隔离

多个 agent 在同一个 working tree 同时编辑会互相覆盖。

Git worktree 提供：

```text
main repo
├─ worker-A worktree / branch agent/A
├─ worker-B worktree / branch agent/B
└─ reviewer
```

下一阶段会在此之上实现 task graph、worker scheduler、reviewer 和 merge policy。

---

# 10　这一层之后才轮到“像 Codex”

真正需要继续补齐的是：

1. public-checkpoint backend；
2. stronger sandbox；
3. patch / AST / LSP editing；
4. browser / computer use；
5. parallel worktrees；
6. verifier；
7. SWE-bench harness；
8. long-horizon checkpoint/resume；
9. multi-agent review / merge。

这些会按源码和测试继续推进，不把接口占位符算作完成。
