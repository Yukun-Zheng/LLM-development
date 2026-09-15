# Lesson 12　AGENTS.md 作用域与指令来源：从“读一个文件”到可审计的 Repository Instructions

Coding Agent 的行为并不只由一个全局 system prompt 决定。真实仓库经常需要：

```text
repository-wide rules
        ↓
subdirectory rules
        ↓
working-directory rules
```

OpenAI Codex 的公开源码已经把 `AGENTS.md` 作为 project-level documentation 机制，并明确从 project root 到当前 cwd 逐层发现、按顺序拼接，同时支持 `AGENTS.override.md`、fallback filenames 与总 byte budget。当前核验的一手源码：[`openai/codex` `codex-rs/core/src/agents_md.rs`](https://github.com/openai/codex/blob/fc269b66adc37f3c855df222ad80b02733355c46/codex-rs/core/src/agents_md.rs)。

本项目据此做了独立 Python clean-room reference implementation：

```text
src/astra_codex/instructions.py
src/astra_codex/coding.py
```

测试：

```text
tests/test_instructions.py
```

---

## 1　真实问题不是“找到 AGENTS.md”

最粗糙的实现可能是：

```python
Path(repo) / "AGENTS.md"
```

但这无法表达大型 monorepo 的局部规则。例如：

```text
repo/
├── AGENTS.md
├── backend/
│   └── AGENTS.md
└── frontend/
    └── AGENTS.md
```

当 cwd 在：

```text
repo/backend/api/
```

正确输入应该包含：

```text
repo/AGENTS.md
repo/backend/AGENTS.md
```

而不应该泄漏：

```text
repo/frontend/AGENTS.md
```

所以这是一个**作用域解析问题**。

---

## 2　从 project root 到 cwd，而不是 filesystem root 到 cwd

公开 Codex 源码的关键规则之一是：先找到项目根，再只在：

```text
project root
→ ...
→ cwd
```

这条祖先链上查找。cite source: openai/codex agents_md.rs, verified snapshot fc269b66...

本项目的 reference resolver 默认以：

```text
.git
```

作为 project root marker。

如果没有找到 marker，保守行为是：

```text
只检查 cwd
```

而不是继续一路读到：

```text
/home
/
```

否则上层目录中的无关/恶意指令可能被意外注入模型上下文。

---

## 3　每个目录只选择一个 candidate

当前优先级：

```text
AGENTS.override.md
        ↓
AGENTS.md
        ↓
configured fallback files
```

例如同一个目录有：

```text
AGENTS.override.md
AGENTS.md
WORKFLOW.md
```

只加载：

```text
AGENTS.override.md
```

这对应公开 Codex 当前的 candidate ordering 思路，而不是把同一目录所有说明文件全部叠加。

---

## 4　为什么需要 provenance

模型最终看到的可能只是：

```text
root rule

package rule

local rule
```

但 runtime/debugger 必须知道每一段来自哪里。

所以我们的 `InstructionSource` 不只保存文本，还保存：

```text
source_path
scope_directory
candidate_name
contents
truncated
bytes_loaded
```

于是当 Agent 做出一个奇怪决策时，可以追问：

```text
哪条 instruction 影响了模型？
它来自哪个目录？
它是 AGENTS.md 还是 override？
是否因为 byte budget 被截断？
```

这与我们整本教材的核心原则完全一致：

> **进入模型上下文的状态也必须有 provenance。**

---

## 5　总 byte budget 为什么是全层级共享

假设：

```text
root AGENTS.md      20 KB
package AGENTS.md   20 KB
local AGENTS.md     20 KB
```

如果限制对每个文件分别应用 32 KB，最终可能进入 60 KB。

更合理的是维护：

```text
remaining = max_total_bytes
```

逐层读取：

```text
root consumes 20K
remaining = 12K
package only receives 12K
remaining = 0
local not loaded
```

我们的 resolver 因此记录：

```text
truncated = True / False
```

让截断成为显式事实，而不是静默丢文本。

---

## 6　指令进入 Coding Agent 的真实数据流

现在 `build_coding_agent(...)` 不再只有固定的：

```text
CODING_SYSTEM_PROMPT
```

而是：

```text
repository_root
working_directory
       ↓
ProjectInstructionResolver
       ↓
ResolvedInstructions
├─ sources/provenance
└─ resolved text
       ↓
CODING_SYSTEM_PROMPT
+ Project instructions
       ↓
model-visible system message
```

自动测试建立了 sibling negative control：

```text
root instruction      → present
api instruction       → present
web sibling rule      → absent
```

这比只测试“能读到 AGENTS.md”强，因为它同时证明了**应加载**与**不应加载**的边界。

---

## 7　Instruction hierarchy 不等于安全权限

这一点必须严格区分：

```text
AGENTS.md
= model-visible behavioral instruction
```

而：

```text
PermissionProfile
Bearer Authorization
OS Sandbox
= execution boundary
```

即使 AGENTS.md 写：

> 不要访问 `.env`

也绝不能把它当成 filesystem security policy。

正确结构仍然是：

```text
Instruction
→ influences proposal

Permission / Sandbox
→ constrains actual execution
```

Prompt 不是安全边界。

---

## 8　与官方 Codex 当前实现的边界

我们的实现已经对齐了公开源码中最核心的教学语义：

```text
project root discovery
root → cwd ordering
never walk above root
override > AGENTS > fallback
global byte budget
source provenance
```

但没有宣称复制完整 Codex production behavior。当前还没有复现：

```text
remote ExecutorFileSystem abstraction
multiple environment labeling
project trust gating
host user/thread instructions
sandbox-aware remote metadata probes
all config layering semantics
```

这些会继续在 Codex Source Anatomy 课程里逐项拆。

---

## 9　验收标准

这一层的第一版验收不是“成功找到一个文件”，而是：

```text
root → cwd hierarchy order                  ✓
override priority                           ✓
fallback only when primary absent           ✓
total byte budget + explicit truncation      ✓
no project marker → cwd only                 ✓
empty marker list → parent traversal off     ✓
never walk above nearest project root        ✓
sibling instructions do not leak             ✓
resolved instructions enter coding prompt    ✓
source/scope provenance remains inspectable  ✓
```

下一步应把 instruction provenance 进一步接入：

```text
ContextStore
→ INSTRUCTION fragments
→ event / rollout trace
→ final task artifact provenance
```

这样我们最终可以从某次 Agent 行为反向追到：模型当时到底收到了哪些仓库规则。
