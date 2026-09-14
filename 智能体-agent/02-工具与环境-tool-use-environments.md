# A2　工具调用、动作空间与环境接口

> **核心问题**：模型输出怎样从字符串变成真实世界动作？Tool Calling 为什么本质上是“动作空间设计 + 协议验证 + 环境执行 + observation 回传”？

---

# 1　把 Tool Call 写成形式化动作

设工具集合：

$$
\mathcal A=\{a^{(1)},a^{(2)},\ldots,a^{(K)}\}.
$$

每个工具不是一个名字，而是：

$$
a^{(k)}=(name,schema,executor,policy).
$$

其中：

- `name`：动作标识；
- `schema`：参数约束；
- `executor`：执行函数；
- `policy`：权限、超时、网络、文件根目录等边界。

模型真正需要生成的是：

$$
(name,args)\in \mathcal A.
$$

随后系统验证 `args`，执行 executor，得到 observation。

---

# 2　为什么 schema 是安全和可靠性的第一道边界

若模型输出：

```json
{"tool":"filesystem","arguments":{"action":"read","path":"README.md"}}
```

系统至少要验证：

1. tool 是否存在；
2. arguments 是否是 object；
3. required fields 是否齐全；
4. type 是否正确；
5. 是否允许 additional properties；
6. path 是否越过 allowed root；
7. action 是否在允许集合。

因此：

```text
LLM text
↓ parse
candidate action
↓ schema validation
valid action
↓ policy check
authorized action
↓ execution
observation
```

不能把“模型生成了 JSON”当作“动作可以直接执行”。

---

# 3　工具设计决定了 Agent 的可解问题空间

一个 Agent 的能力不仅由 model 决定，还由 action space 决定。

如果只有：

```text
read_file(path)
```

它永远无法真实修改仓库。

若加入：

```text
search_code(query)
edit_file(path, old, new)
run_shell(command)
git_diff()
```

就出现了新的闭环能力。

因此工具集合可以看成 agent 的“效应器（effectors）”。

---

# 4　Toolformer：工具使用能否被模型学进去

Toolformer 研究了模型如何自监督地学习调用外部 API，而不是纯靠人工 prompt 规定何时调用工具。

原始资料：Schick et al., **Toolformer: Language Models Can Teach Themselves to Use Tools**  
https://arxiv.org/abs/2302.04761

这里要区分两层：

- **tool-use capability in model**：模型是否知道什么时候、怎样调用；
- **tool runtime in system**：真实系统是否能安全、可靠地执行。

模型能力强不意味着 runtime 安全；runtime 完善也不意味着模型会正确选工具。

---

# 5　环境 observation 必须保真

执行工具后，observation 至少应保存：

- status / exit code；
- stdout；
- stderr；
- changed files；
- structured result；
- timeout / permission error；
- provenance。

代码任务中：

```text
pytest failed
```

远远不够，真正有用的是：

```text
exit_code=1
failed_test=test_cache_decode_parity
traceback=...
```

因为 observation 的信息质量直接决定下一轮 policy 是否能纠错。

---

# 6　Browser / Computer Use 是更复杂的环境

Browser tool 的 observation 可能是 DOM、文本、URL、下载结果。

Computer Use 更进一步：

```text
screenshot
→ detect UI state
→ mouse / keyboard action
→ new screenshot
```

这时 observation 已经是多模态的，动作也从离散 API 扩展为坐标、按键、拖拽等连续/组合操作。

因此 Computer Agent 与传统 function calling 并不是同一个难度等级。

---

# 7　安全边界必须落到执行层

仅在 system prompt 写：

> “不要执行危险命令。”

不是可靠安全机制。

执行层至少需要：

```text
allowed filesystem roots
command timeout
network allow/deny policy
secret isolation
process isolation
resource quota
destructive-action confirmation
audit log
```

对 coding agent，未来应使用 container / VM / namespace 等隔离，而不仅是 `cwd=repository`。

---

# 8　源码映射

当前工程中：

- `structured.py`：tool-call parsing 与 schema subset；
- `tools.py`：filesystem / shell / Git 等工具；
- `general_tools.py`：HTTP browser 等通用工具；
- `editing.py`：安全 exact-match 编辑；
- `agent.py`：执行失败作为 observation 回到循环。

---

# 9　原始资料

- MRKL Systems: https://arxiv.org/abs/2205.00445
- WebGPT: https://arxiv.org/abs/2112.09332
- Toolformer: https://arxiv.org/abs/2302.04761
- ReAct: https://arxiv.org/abs/2210.03629
- WebArena: https://arxiv.org/abs/2307.13854
- OSWorld: https://arxiv.org/abs/2404.07972

下一章讨论：拥有工具之后，Agent 怎样规划、验证、反思和从失败中恢复。
