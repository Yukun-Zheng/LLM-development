# C04　Prompt、AGENTS.md、Context、Compaction 与 Memory

> **Primary sources**  
> - Base instructions: https://github.com/openai/codex/blob/main/codex-rs/protocol/src/prompts/base_instructions/default.md  
> - Config fallback filenames: https://github.com/openai/codex/blob/main/codex-rs/config/src/config_toml.rs  
> - Turn/context loop: https://github.com/openai/codex/blob/main/codex-rs/core/src/session/turn.rs  
> - Compaction budget: https://github.com/openai/codex/blob/main/codex-rs/core/src/compact_token_budget.rs  
> - Remote compaction: https://github.com/openai/codex/blob/main/codex-rs/core/src/compact_remote_v2.rs  
> - Memories crates: https://github.com/openai/codex/tree/main/codex-rs/memories

---

# 1　Agent 的“上下文”不是一个 prompt 字符串

成熟 runtime 中，模型最终看到的是多个来源合成的 model-visible state：

```text
base/system instructions
+ developer policy
+ repository AGENTS.md
+ nested scoped instructions
+ skills
+ plugin / connector context
+ environment / cwd context
+ conversation history
+ tool observations
+ memory / notes
+ current user input
```

所以 context manager 真正的问题是：

$$
C_t = \mathrm{Assemble}(S,D,R,K,E,H,M,U_t),
$$

而不是：

```python
prompt = system + history + user
```

---

# 2　AGENTS.md 是 scoped instruction system

OpenAI Codex 的 base instructions 明确说明：

- `AGENTS.md` 中关于 code style / structure / naming 的指令对其 scope 内代码生效；
- 更深层目录的 `AGENTS.md` 在冲突时优先；
- system/developer/user 级别指令仍具有更高优先关系。

抽象成路径继承：

```text
repo/AGENTS.md
    ↓ applies broadly
repo/src/AGENTS.md
    ↓ overrides inside src/
repo/src/compiler/AGENTS.md
    ↓ overrides inside compiler/
```

因此 repository instructions 是一种**层级作用域配置**。

这比把整个仓库所有说明文件一次性塞给模型更精确，也更省 context。

---

# 3　Instruction source 本身应该可追踪

理想上下文不能只有：

```text
"use snake_case"
```

还应该知道：

```text
source = repo/src/AGENTS.md
scope = repo/src/**
priority = nested-project-instruction
loaded_at = turn N
```

这对 debug 很重要。

当模型行为异常时，需要回答：

> 到底是哪条 instruction 影响了它？

所以我们的后续 context fragment 结构应包含 provenance。

---

# 4　为什么 Context Window 和 Task Lifetime 必须分离

长任务可能经历：

```text
Task lifetime: 2 hours
Model context window A: 20 min
Model context window B: 25 min
Model context window C: ...
```

官方 `run_turn` 已经把 compaction 融入 task/turn lifecycle：达到 token threshold 后，可以 compact 再继续 follow-up。

因此：

$$
\text{Task state} \supsetneq \text{Current context window}.
$$

一个 context window 只是长期任务的临时工作集。

---

# 5　Compaction 不是简单截断最旧消息

最粗暴策略：

```text
history = history[-N:]
```

会丢掉：

- 用户真正目标；
- 已做的架构决策；
- 修改过哪些文件；
- 测试失败证据；
- 尚未完成的 todo；
- sandbox / environment facts。

更合理的 compaction 要提炼**继续执行所需的 sufficient state**：

```text
goal
constraints
decisions
modified artifacts
observed failures
verification state
pending work
important environment facts
```

这就是 Agent 长程记忆问题与普通聊天摘要的差别。

---

# 6　World State 与 Conversation State 要区分

模型 history 说：

> “我已经创建 `foo.py`。”

不意味着文件系统里 `foo.py` 仍然存在。

因此：

```text
conversation belief
≠
environment truth
```

Compaction 后尤其如此。

长期 Agent 应重新观察关键 world state：

```text
filesystem
Git diff
running processes
tests
remote task status
browser state
```

而不是完全相信摘要。

---

# 7　Memory 至少分四类

```text
Working Memory
  当前 context 内临时状态

Episodic Memory
  发生过哪些 action / observation / failure

Semantic Memory
  提炼出的长期事实、规则、用户/项目约束

Artifact State
  文件、Git commit、patch、test report 等外部真实对象
```

其中 Artifact State 通常比自然语言 memory 更可靠。

对于 coding agent：

```text
Git commit hash
pytest report
file contents
```

往往比“我记得测试通过了”更可信。

---

# 8　我们的当前实现与缺口

现有：

```text
memory.py       SQLite persistent event memory
agent.py        message history
codex_harness.py turn events
planning.py     explicit task DAG
verification.py evidence
```

下一阶段需要统一成：

```text
ThreadStore
├─ events
├─ turns
├─ compacted windows
├─ instruction fragments
├─ task graph
├─ artifact references
├─ verifier evidence
└─ resume checkpoints
```

并建立：

```text
ContextAssembler
ContextBudgeter
Compactor
CheckpointManager
```

---

# 9　Context 的信息论视角

如果完整历史是随机变量 $H$，compact state 为 $Z$，未来完成任务所需信息为 $Y$，我们真正希望的是：

$$
Z=f(H)
$$

在压缩 $Z$ 的同时，尽量保留与未来行动相关的信息：

$$
I(Z;Y) \text{ 尽可能大},
$$

同时让 token cost：

$$
|Z| \ll |H|.
$$

这比“总结得通顺”更接近 Agent compaction 的目标。

---

# 10　必须做的实验

### M1 scoped AGENTS resolution

构造三层目录，每层 AGENTS.md 有冲突规则，验证目标文件只收到正确的继承链。

### M2 compact-resume equivalence

同一 scripted task：

```text
full history run
vs
mid-task compact + resume
```

比较最终 action sequence 与 verification result。

### M3 stale-memory attack

memory 写“tests pass”，随后人为让测试失败；agent 必须通过真实 verifier 更新 belief。

### M4 context budget

记录每类 fragment 的 token 占比，验证 compaction 前后重要 constraint 没丢。

最终我们要让 memory / compaction 成为可测系统，而不是 prompt engineering 经验谈。
