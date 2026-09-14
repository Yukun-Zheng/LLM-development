# C03　Tools、Exec、Patch、Approval 与 Sandbox：安全边界不是 Prompt

> **Primary sources**  
> - `core/src/exec_policy.rs`: https://github.com/openai/codex/blob/main/codex-rs/core/src/exec_policy.rs  
> - `core/src/tools/approvals.rs`: https://github.com/openai/codex/blob/main/codex-rs/core/src/tools/approvals.rs  
> - `core/src/tools/sandboxing.rs`: https://github.com/openai/codex/blob/main/codex-rs/core/src/tools/sandboxing.rs  
> - `apply-patch/`: https://github.com/openai/codex/tree/main/codex-rs/apply-patch  
> - `sandboxing/`: https://github.com/openai/codex/tree/main/codex-rs/sandboxing  
> - `linux-sandbox/`: https://github.com/openai/codex/tree/main/codex-rs/linux-sandbox

---

# 1　三个经常被混在一起的东西

真实 Coding Agent 至少有三层不同约束：

```text
1. Tool schema
   模型“可以请求什么”

2. Approval policy
   某个动作“是否需要额外同意”

3. Sandbox / permission enforcement
   即使模型想做、用户也同意，进程“实际上能做什么”
```

它们不能合并成一句：

> “请不要删除危险文件。”

Prompt 是 policy communication；sandbox 才是 execution boundary。

---

# 2　官方源码确实把 approval 与 sandbox 分开

公开源码中 `AskForApproval`、permission profile、sandbox manager/policy 在不同模块中流动。

例如 `core/src/tools/sandboxing.rs` 会根据 approval policy 判断是否可以请求一次 no-sandbox execution；`core/src/exec_policy.rs` 同时接收 command、approval policy 与 permission profile。

这验证了一个重要系统设计：

$$
\text{AllowedExecution}
\neq
f(\text{LLM intent only}).
$$

更接近：

$$
\text{Decision}
=
f(\text{requested action},\text{policy},\text{permission profile},\text{sandbox},\text{review}).
$$

---

# 3　Tool Router 的责任

工具层不能只做：

```python
if name == "shell": subprocess.run(args)
```

成熟路径更接近：

```text
model-visible ToolCall
        ↓
parse / schema validate
        ↓
resolve tool handler
        ↓
permission / policy review
        ↓
approval if needed
        ↓
sandbox transformation
        ↓
execute
        ↓
collect stdout/stderr/metadata
        ↓
structured observation
```

任何一层失败，都应该有明确 failure semantics。

---

# 4　为什么 Patch 应该是一等动作

直接让模型执行：

```bash
python -c 'rewrite whole file...'
```

虽然万能，但可审计性差。

专门的 patch/edit action 可以提供：

- 变更范围；
- preimage / context；
- ambiguity detection；
- conflict；
- diff preview；
- approval boundary；
- rollback / review evidence。

OpenAI Codex workspace 单独维护 `apply-patch` crate，本身就是很强的工程信号：**代码修改接口值得成为独立 subsystem**。

我们目前有 `editing.py` 的 exact-match safe edit，后续应演化为 unified diff / semantic patch engine。

---

# 5　Approval 是异步 protocol interaction

approval 不只是同步 `input("y/n")`。

它应该表现成：

```text
Agent Core
   ↓
ApprovalRequested event
   ↓
UI / reviewer / policy service
   ↓
ApprovalDecision submission
   ↓
Core resumes suspended action
```

这样 approval reviewer 可以是：

```text
human user
auto-review policy
organization policy engine
security reviewer agent
```

这也是为什么 protocol 层必须是一等公民。

---

# 6　“Never ask”不等于“无限权限”

这是非常容易误解的一点。

```text
approval_policy = never
```

合理含义是：

> 不通过交互式 approval 请求扩大权限。

而不是：

> 关闭 sandbox，想干什么都行。

安全设计必须保持：

```text
interaction policy
≠
permission boundary
```

否则 headless/CI 模式会天然变成 root shell，这是不可接受的抽象。

---

# 7　Sandbox 要真正作用在 OS / process 边界

我们最终需要研究的约束维度至少包括：

```text
filesystem read roots
filesystem write roots
network access
process spawn
syscall capability
environment variables / secrets
home directory
Git credentials
SSH agent
IPC
resource limits
timeouts
```

Agent 安全不是只做一个 path prefix check。

我们的 `RootedPaths` 是教学第一步，但不是 production sandbox。

---

# 8　我们的 Clean-room v1

`codex_harness.py` 当前实现：

```text
TurnSettings
├─ sandbox_policy metadata
└─ approval_required_tools
```

以及：

```text
APPROVAL_REQUESTED
→ approval handler
→ APPROVAL_DECIDED
→ allow: execute
→ deny: structured observation
```

测试已经明确验证：**deny 时 tool body 根本不会运行**。

但当前 `SandboxPolicy` 只携带状态，不执行真实隔离。源码里专门写明这一点，防止把“policy enum 存在”冒充“sandbox 已实现”。

---

# 9　后续实现顺序

```text
P0  PermissionProfile 数据结构
P0  action risk classification
P0  async approval broker
P1  subprocess environment scrub
P1  filesystem allowlist
P1  network policy
P1  resource limits
P2  Linux namespace / bubblewrap/container backend
P2  secret broker
P2  audit log
P3  auto-review / guardian policy
```

每层都必须独立测试。

---

# 10　安全实验

### S1 Path escape

请求读取 `../../etc/passwd`，确认文件层拒绝。

### S2 Approval bypass

模型请求 approval-gated shell tool；deny 后确认 subprocess 从未启动。

### S3 Prompt vs enforcement

删掉 system prompt 中的“不要越权”文字，安全测试仍必须通过。

### S4 Secret isolation

给 parent process 一个 fake secret，运行 sandbox command，验证 child 不可见。

### S5 Network denial

sandbox network=off 时，即使模型和用户都请求 curl，也必须在 execution boundary 失败。

这组实验能把“Agent 安全”从文本约束推进到系统安全。
