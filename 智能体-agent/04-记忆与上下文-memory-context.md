# A4　记忆、上下文管理与长期任务状态

> **核心问题**：Context Window、RAG、Memory、Task State、Checkpoint 是五个不同概念。为什么把聊天记录无限塞回 prompt 并不等于“有记忆”？

---

# 1　先区分五种状态

| 名称 | 主要作用 |
|---|---|
| Context Window | 本次模型 forward 可见的 token |
| Retrieval | 从外部索引取回相关信息 |
| Working Memory | 当前任务暂时需要保持的中间状态 |
| Long-Term Memory | 跨步骤、跨会话持久保存的信息 |
| Checkpoint | 系统恢复执行所需的可重建状态 |

它们可以互相连接，但不能混为一谈。

---

# 2　为什么完整历史不能一直塞进 Context

若每轮都追加 observation：

$$
H_t=H_{t-1}\oplus a_{t-1}\oplus o_t,
$$

则 context 长度近似单调增长：

$$
|H_t|=O(t).
$$

长程任务最终会遇到：

- context limit；
- attention cost；
- 无关历史干扰；
- 关键事实被埋没；
- 重复 observation；
- 旧状态与新状态冲突。

因此需要 context management。

---

# 3　Memory 应该存什么

Agent 不需要“记住一切”，而是需要保留**影响未来决策的状态**。

典型内容：

```text
goal / constraints
important user decisions
files already inspected
files changed
commands executed
known failures
successful tests
artifacts
unresolved questions
plan state
external facts + provenance
```

可以把 memory write 视为一个压缩函数：

$$
m_{t+1}=U(m_t,h_t,o_{t+1}).
$$

好的 $U$ 不是摘要所有文本，而是提取决策充分信息。

---

# 4　Event Log 与 Semantic Memory

## Event Log

按时间追加：

```text
step 1 read config.py
step 2 edit cache.py
step 3 pytest failed
step 4 inspect traceback
```

优点：可审计、可 replay、容易 checkpoint。

## Semantic Memory

按主题组织：

```text
KV cache invariant:
position offset must equal past length
```

优点：检索方便，但写入时需要更强归纳。

实践中通常两者并存。

---

# 5　MemGPT 的思想价值

MemGPT 将有限 context 视为“主存”，将外部 memory 视为更大但需要管理的存储层，强调 agent 自己管理上下文与记忆交换。

原始资料：Packer et al., **MemGPT: Towards LLMs as Operating Systems**  
https://arxiv.org/abs/2310.08560

这类系统的重要启发是：**长上下文不是长期记忆问题的唯一解。**

---

# 6　Context Compaction

可以把历史压缩成：

$$
C_t=\mathrm{Compress}(H_{0:t},M_t,G),
$$

然后只把决策所需内容送入模型。

一个实用策略：

```text
System Contract
+ Goal
+ Current Plan
+ Recent Observations
+ Retrieved Relevant Memories
+ Current Artifact State
```

而不是整个历史 transcript。

---

# 7　Checkpoint 与 Resume

一个真正长程 Agent 应该能：

```text
process exits
→ restart
→ load checkpoint
→ reconstruct task state
→ continue
```

Checkpoint 至少包括：

- task id；
- goal；
- plan；
- completed / failed steps；
- relevant memory ids；
- working directory / branch / worktree；
- tool policy；
- artifact state；
- remaining budget。

这与“聊天记录还在”完全不是同一层工程能力。

---

# 8　Memory 的错误类型

智能体 memory 也会出错：

1. **stale memory**：旧状态已失效；
2. **false memory**：模型把推测写成事实；
3. **over-retention**：保留过多无关信息；
4. **under-retention**：关键约束丢失；
5. **poisoning**：外部内容向 memory 注入恶意指令；
6. **privacy leak**：不应持久化的信息被保存。

因此 memory write 本身也是一个需要 policy 和 provenance 的动作。

---

# 9　源码映射

当前工程 `memory.py` 已有 SQLite append-only event memory：

- `append()`；
- `recent()`；
- `search()`；
- metadata JSON。

下一步会增加：

- typed task state；
- importance / provenance；
- compaction；
- checkpoint / resume；
- retrieval ranking；
- stale-state invalidation。

---

# 10　原始资料

- MemGPT: https://arxiv.org/abs/2310.08560
- Generative Agents: https://arxiv.org/abs/2304.03442
- Retrieval-Augmented Generation: https://arxiv.org/abs/2005.11401

下一章进入目前最重要的 Agent 垂直方向之一：Coding Agent。
