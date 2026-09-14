# A5　编程智能体：从代码生成到仓库闭环

> **核心问题**：会生成代码为什么远远不等于会做软件工程？一个真正的 Coding Agent 需要哪些感知、编辑、执行、验证与版本控制能力？

---

# 1　代码生成与软件工程任务不是一回事

单次代码生成通常是：

$$
y\sim p_\theta(y\mid prompt).
$$

真实仓库任务则是：

```text
understand issue
→ inspect repo
→ locate symbols
→ infer constraints
→ edit
→ run tests/build/typecheck
→ inspect failures
→ edit again
→ review diff
→ possibly commit
```

其中最难的部分常常不是“写出一段语法正确的代码”，而是：

- 找对修改位置；
- 理解跨文件依赖；
- 不破坏已有行为；
- 在不完整测试下选择验证方式；
- 处理环境和依赖；
- 根据真实失败继续修复。

---

# 2　Repository Context 是第一瓶颈

大仓库不可能完整塞进 context。

因此 coding agent 需要构建 repo representation：

```text
file tree
imports
symbols
classes/functions
references
README / docs
build system
tests
recent Git history
```

当前工程使用 Python AST + Markdown headings 构造最小 repo map；后续扩展到 tree-sitter / LSP / reference graph。

---

# 3　编辑动作需要“可失败”

一个危险的编辑器会在上下文模糊时“猜”。

我们当前使用 exact-match edit：

```text
old snippet occurs exactly once → replace
0 matches → fail
>1 matches → fail
```

这样强迫 Agent 先获得更具体上下文。

后续可扩展：

- unified diff；
- AST rewrite；
- symbol-level edit；
- LSP-assisted rename；
- code formatter integration。

---

# 4　测试是环境 feedback

Coding Agent 的一个关键优势是软件环境天然拥有大量 verifier：

```text
unit test
integration test
compiler
linter
type checker
formatter
benchmark
snapshot test
```

因此代码 Agent 的闭环可以非常“科学”：

$$
change\rightarrow test\rightarrow evidence\rightarrow next\ change.
$$

模型说“看起来没问题”不能替代真实 test result。

---

# 5　SWE-bench 为什么重要

SWE-bench 将真实 GitHub issue 与真实 repository state 结合，要求系统生成能让测试通过的 patch。

原始资料：Jimenez et al., **SWE-bench: Can Language Models Resolve Real-World GitHub Issues?**  
https://arxiv.org/abs/2310.06770

它把评测从：

```text
write a function from prompt
```

推进到：

```text
understand existing codebase
+ issue
+ tests
+ patch
```

这更接近 Codex-class 系统的真实能力边界。

---

# 6　SWE-agent：Agent-Computer Interface 也很关键

SWE-agent 强调 Agent 与计算机环境之间的接口设计会显著影响软件工程表现。

原始资料：Yang et al., **SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering**  
https://arxiv.org/abs/2405.15793

这说明：

> 同一个 base model，换一套更合适的 observation/action interface，系统能力也可能显著改变。

因此不能把 coding-agent benchmark 全部归因于模型参数。

---

# 7　Git 是 Agent 状态管理工具

一个成熟 Coding Agent 应理解：

- `git status`；
- `git diff`；
- commit history；
- branches；
- worktrees；
- conflict；
- revert；
- review surface。

Git 不只是“最后提交一下”，而是长程任务的状态隔离与可追踪性基础。

---

# 8　并行 Coding Agent 为什么需要 Worktree

若两个 worker 在同一 working tree 修改：

```text
worker A edit file.py
worker B edit file.py
```

状态会互相污染。

更合理：

```text
main repository
├─ worktree-A / branch agent-A
├─ worktree-B / branch agent-B
└─ reviewer / merger
```

每个 worker 有独立文件系统视图，然后通过 diff / commit 进行 review 与 merge。

---

# 9　源码映射

当前工程已经有：

- `repo_map.py`：仓库结构感知；
- `editing.py`：exact safe edit；
- `tools.py`：filesystem / shell / Git；
- `coding.py`：Coding Agent contract；
- `worktree.py`：worktree primitive；
- `agent.py`：闭环执行。

当前缺口明确包括：

- tree-sitter / LSP；
- unified diff parser；
- test selection；
- coverage-guided verification；
- parallel worker scheduler；
- reviewer / merge agent；
- SWE-bench harness。

---

# 10　最终 Coding Agent 数据流

```text
Issue
 ↓
Repo Map / Search
 ↓
Hypothesis
 ↓
Read precise context
 ↓
Edit
 ↓
Test / Build / Lint
 ↓
Observation
 ├─ pass → review diff
 └─ fail → traceback / replan / edit
 ↓
Git diff / verifier
 ↓
Finish
```

---

# 11　原始资料

- SWE-bench: https://arxiv.org/abs/2310.06770
- SWE-agent: https://arxiv.org/abs/2405.15793
- ReAct: https://arxiv.org/abs/2210.03629

这条线最终直接汇入本书的 Codex-class capstone。
