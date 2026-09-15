# Lesson 14　受限进程沙箱与真实隔离边界

这一课解决一个很容易被 Agent 项目模糊掉的问题：

> **“我限制了 tool 参数”不等于“我已经安全执行不可信代码”。**

截至这一阶段，项目已经有三层不同的安全机制：

```text
Control-plane AuthN/AuthZ
        ↓
Tool Permission / Approval
        ↓
Restricted Process Execution
        ↓
[仍缺]
OS namespace / container / VM isolation
```

这四层解决的是不同问题，任何一层都不能冒充另一层。

---

## 1　为什么原来的 `ShellTool` 不能叫 sandbox

旧教学工具的执行路径本质上是：

```python
subprocess.run(
    ["bash", "-lc", command],
    cwd=repo,
    env=os.environ,
)
```

它对学习 Agent Loop 很直观，但从安全角度几乎等价于把当前进程的主机权限交给命令：

```text
模型输出 shell string
      ↓
bash parser
      ↓
主机进程权限
      ↓
当前用户能访问的 filesystem / network / secrets
```

即使 `cwd` 指向仓库目录，也不代表命令不能读取 `/etc`、`$HOME`、SSH key、环境变量或访问网络。

因此本项目现在明确保留两种模式：

- `ShellTool`：用于最早期透明教学；
- `SandboxExecTool`：用于逐步进入安全 runtime 的 argv-only 执行路径。

---

## 2　第一步：去掉 shell parser

新的 `SandboxExecTool` 不接受：

```text
"pytest && rm -rf ..."
```

而接受：

```json
{
  "argv": ["pytest", "tests/test_x.py", "-q"]
}
```

这一步非常重要，因为：

```text
command string
→ shell syntax
→ |  >  &&  ;  $(...)  glob  expansion
```

而 argv execution 是：

```text
[program, arg1, arg2, ...]
→ execve-style process launch
```

少掉了一整层隐式解释器。

但注意：**argv-only 仍然不是 sandbox**。如果你允许执行 `python`，Python 程序仍然可以主动打开任意宿主机文件或 socket。

---

## 3　Workspace cwd boundary

`RestrictedSubprocessSandbox` 会先解析 cwd：

```text
workspace_root = /repo
cwd = /repo/subdir       → allow
cwd = /repo/../outside   → deny
```

这里防的是控制面直接把工作目录切出项目根目录。

它并不意味着子进程看不到 workspace 之外的文件。

也就是说：

```text
cwd confinement
!=
filesystem confinement
```

真正 filesystem confinement 需要 mount namespace、chroot/container rootfs、Landlock 或其他 OS enforcement。

---

## 4　Executable allowlist

策略显式记录：

```python
allowed_executables = {
    "python3",
    "pytest",
    "git",
    "ruff",
}
```

运行前只看 executable basename：

```text
pytest ...  → allow
python3 ... → allow
sh ...      → deny
```

这一层的作用是降低“模型可以随意启动任意宿主程序”的攻击面。

但是仍然不能把它理解为能力证明。例如允许：

```text
python3 -c <arbitrary code>
```

本身已经是很强的计算能力。

因此 allowlist 必须和真正 OS isolation 配合，而不是取代 isolation。

---

## 5　Environment allowlist：默认不把 secret 传进 child

旧模式常见做法是：

```python
env={**os.environ, ...}
```

这会让 Agent command 自动继承：

```text
API keys
cloud credentials
proxy credentials
SSH-related variables
service tokens
internal endpoints
```

新 runner 改成：

```text
parent environment
      ↓
explicit allowlist
      ↓
child environment
```

默认只保留少量运行环境键，例如：

```text
PATH
LANG
LC_ALL
PYTHONUNBUFFERED
```

测试会在父进程设置一个假 secret，然后验证 child 看到的是 `<missing>`。

这仍然不是完整 credential isolation。生产系统还需要：

- per-tool credential broker；
- short-lived credentials；
- scope-bound tokens；
- secret never entering model context；
- audit trail。

---

## 6　Resource limits

当前 reference runner 在 POSIX 上使用 `setrlimit` 类约束：

```text
CPU time
file size
open files
process count
optional address space
core dump = 0
```

同时父进程有 wall-clock timeout：

```text
spawn process group
      ↓
communicate(timeout=T)
      ↓ timeout
kill process group
```

为什么同时需要 CPU time 和 wall time？

因为一个进程可以：

```text
sleep / block on I/O
```

几乎不消耗 CPU，却永远不结束。

反过来，一个程序也可以在很短 wall time 内疯狂消耗 CPU。

两者不是同一个指标。

Python 原始资料：

- `subprocess`: https://docs.python.org/3/library/subprocess.html
- `resource`: https://docs.python.org/3/library/resource.html

---

## 7　Linux `no_new_privs`

Linux 提供 `PR_SET_NO_NEW_PRIVS`。

当前 runner 在 Linux child 中执行：

```text
prctl(PR_SET_NO_NEW_PRIVS, 1)
```

它保证 exec 后不能通过某些机制获得比当前进程更多的 privilege。

测试不会只检查“代码调用过 prctl”，而是让 child 自己读取：

```text
/proc/self/status
```

并验证：

```text
NoNewPrivs: 1
```

Linux kernel 原始说明：

- https://docs.kernel.org/userspace-api/no_new_privs.html

但必须再次强调：

```text
no_new_privs
!=
seccomp
!=
filesystem namespace
!=
network namespace
```

---

## 8　这一版到底真正 enforce 了什么

当前 `sandbox.py` 的事实边界是：

```text
✅ no shell parser
✅ argv-only execution
✅ executable allowlist
✅ cwd cannot escape workspace
✅ environment allowlist
✅ wall timeout
✅ process-group kill on timeout
✅ POSIX rlimits
✅ Linux no_new_privs
✅ output clipping
```

而没有实现：

```text
❌ mount namespace
❌ read-only root filesystem
❌ network namespace
❌ seccomp syscall filter
❌ capability drop
❌ container image/rootfs
❌ VM boundary
❌ kernel exploit resistance
```

所以 capability 的名字故意是：

> **Restricted Subprocess Sandbox Guards**

而不是：

> **Secure Untrusted-Code Sandbox**

---

## 9　为什么要做“负能力声明”

安全系统最危险的错误之一不是“没有某个功能”，而是：

> **系统以为自己有。**

如果文档写“sandboxed shell”，但实际上只是 `cwd=/repo`，上层 Agent 会错误地允许更高风险动作。

因此项目统一采用：

```text
implemented mechanism
+
explicit non-goals
+
negative tests
+
capability manifest
```

来定义安全边界。

---

## 10　Coding Agent 如何切换

默认仍然可以构建最透明的教学 Agent：

```python
agent = build_coding_agent(backend, repo)
```

它继续使用早期 `ShellTool`。

进入安全实验后：

```python
policy = SandboxPolicy(
    workspace_root=repo,
    allowed_executables=frozenset({"python3", "pytest", "git", "ruff"}),
)
runner = RestrictedSubprocessSandbox(policy)

agent = build_coding_agent(
    backend,
    repo,
    execution_sandbox=runner,
)
```

此时工具表中：

```text
shell        → removed
sandbox_exec → present
```

这是一个可测试的结构变化，而不是 prompt 里一句“请安全执行”。

---

## 11　下一层：真正 OS / container sandbox

下一阶段安全链应继续推进为：

```text
Model proposes action
      ↓
Permission / Approval
      ↓
Credential Scope
      ↓
Container / namespace sandbox
├─ isolated rootfs
├─ workspace mount
├─ network policy
├─ no_new_privs
├─ capability drop
├─ syscall policy
├─ pids / cpu / memory
└─ audit
      ↓
Tool execution
      ↓
Artifact / observation
```

真正的验收不能只是“容器启动成功”，而要有 escape-oriented negative tests，例如：

```text
read host secret     → must fail
write outside mount  → must fail
connect network      → must fail under network-none
spawn too many pids  → must fail
exceed memory        → must terminate
privilege gain       → must fail
```

只有这一层真正落地后，项目才能把 `security.os_container_sandbox` 从 `planned/partial` 提升为 `validated`。

---

## 12　这一课的验收标准

看完这一课，应该能明确回答：

1. AuthN、AuthZ、Approval、Process Restriction、OS Sandbox 各自保护什么？
2. 为什么 `cwd` 在 repo 内不能阻止读取 `/etc/passwd`？
3. 为什么 argv-only 比 `bash -lc` 更可控，但仍不能运行不可信代码？
4. 为什么 secret scrubbing 必须发生在 process boundary，而不是只写进 system prompt？
5. `no_new_privs` 能防什么、不能防什么？
6. 为什么一个安全 capability 必须同时记录“做到了什么”和“没做到什么”？

源码对应：

```text
src/astra_codex/sandbox.py
src/astra_codex/coding.py
tests/test_sandbox.py
```
