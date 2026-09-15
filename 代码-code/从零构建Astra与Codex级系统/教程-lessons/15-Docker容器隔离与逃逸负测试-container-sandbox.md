# Lesson 15　Docker 容器隔离与逃逸负测试

Lesson 14 只把命令执行从“裸 `bash -lc`”推进到 **restricted subprocess**：argv-only、allowlist、secret scrubbing、rlimit、timeout、`no_new_privs`。这一课继续跨过真正重要的一层：

> **把 Agent 运行的代码从宿主进程权限域移进独立容器，并用失败导向的测试证明隔离边界确实存在。**

这里的关键词不是“Docker 能启动”，而是：

```text
host secret        → container must not see
read-only workspace→ write must fail
read-only rootfs   → write must fail
network none       → host loopback must be unreachable
cap-drop ALL       → CapEff must be 0
no-new-privileges  → NoNewPrivs must be 1
parent secret env  → child must not inherit
```

只有这些**负测试**真的通过，容器隔离才有可审计证据。

---

## 1　为什么 Bubblewrap 路线保留，但不能伪装成“已验证”

项目先实现了 `bubblewrap_sandbox.py`，目标是：

```text
minimal filesystem view
+ user/mount/PID/IPC/UTS namespaces
+ optional network namespace
+ capability drop
+ workspace bind
```

但 GitHub-hosted Ubuntu 24.04 runner 的真实结果是：

```text
network namespace path:
RTM_NEWADDR: Operation not permitted

shared-network path:
setting up uid map: Permission denied
```

这说明：

```text
bwrap installed
!=
bwrap namespace execution permitted by host kernel/policy
```

所以测试被拆成两类：

1. **hard assertions**：命令构造、allowlist、mount policy 等纯契约必须通过；
2. **runtime namespace evidence**：宿主 policy 不允许 user/network namespace 时明确 `skip`，并把 capability 留在 `partial`。

这比“CI 绿了所以 sandbox 验证完成”重要得多。

---

## 2　为什么再加 Docker backend

GitHub-hosted runner 自带可工作的 Docker Engine，因此我们可以得到另一条真正能在 CI 执行的隔离路径：

```text
Agent argv
   ↓
Host-side fixed Docker policy
   ↓
docker run
   ↓
container namespaces/cgroups
   ↓
read-only image root
   ↓
only /workspace bind-mounted
   ↓
network none
   ↓
process
```

源码：

```text
src/astra_codex/docker_sandbox.py
```

核心对象：

```text
DockerSandboxPolicy
DockerSandbox
DockerSandboxExecTool
```

---

## 3　不能让模型直接控制 Docker flags

如果模型可以生成：

```text
docker run --privileged -v /:/host ...
```

那“用了 Docker”几乎没有安全意义。

因此当前设计把两种输入严格分开。

Host policy 决定：

```text
image
workspace mount
network policy
read-only rootfs
capabilities
security-opt
pids limit
memory limit
cpu limit
uid/gid
environment
```

Model 只决定：

```text
allowlisted guest executable
+ argv
+ workspace-relative cwd
```

也就是：

```text
model-controlled data
!=
runtime security configuration
```

---

## 4　当前固定的容器安全参数

Reference backend 当前构造的核心 Docker contract 是：

```text
--rm
--init
--user <host uid>:<host gid>
--cap-drop ALL
--security-opt no-new-privileges:true
--pids-limit N
--memory BYTES
--cpus N
--read-only
--network none
--mount type=bind,source=<workspace>,target=/workspace[,readonly]
--tmpfs /tmp:rw,nosuid,nodev,size=...
--workdir /workspace/...
--env <explicit entries only>
```

这里每一项解决的问题不同。

例如：

```text
--read-only
```

保护 image rootfs 不被普通任务写入，但如果 `/workspace` 是 rw bind mount，Agent 仍然能修改项目；这正是 coding agent 需要的权限模型。

而：

```text
--network none
```

解决的是网络出站/host-loopback 暴露，不解决 filesystem。

---

## 5　Host secret invisibility

测试会在宿主创建：

```text
<tmp>/outside/host-secret.txt
```

并只把：

```text
<tmp>/workspace
```

bind 到容器的 `/workspace`。

容器内部尝试访问宿主 secret 的原绝对路径，必须得到：

```text
exists() == False
```

与此同时，它又必须能在 rw workspace 中创建：

```text
/workspace/created.txt
```

这同时证明：

```text
隔离不是“整个任务什么都不能做”
```

而是：

```text
只暴露任务真正需要的资源
```

也就是 capability-oriented execution 的基本思想。

---

## 6　Read-only workspace 与 read-only rootfs 是两件事

CI 分别检查：

```text
/workspace/forbidden.txt
/etc/forbidden.txt
```

在 read-only workspace policy 下，前者必须失败；在 `--read-only` rootfs 下，后者也必须失败。

因此：

```text
workspace permission
```

和：

```text
container root filesystem permission
```

不能合并成一个布尔概念。

未来我们还会继续细化：

```text
workspace read-only
workspace write
specific-path write
artifact-output-only
```

---

## 7　Network-none 不能只看 Docker 参数

最弱的测试是：

```python
assert "--network none" in docker_command
```

这只能证明我们**请求了**隔离。

当前测试进一步在宿主启动一个真实 TCP listener：

```text
host 127.0.0.1:<random port>
```

然后让 container 尝试连接相同地址。

验收：

```text
container connect_ex != 0
host listener never accepts connection
```

所以证据从：

```text
configuration inspection
```

升级成：

```text
behavioral negative test
```

---

## 8　Secret isolation 也必须做行为测试

宿主 pytest 进程设置：

```text
ASTRA_CODEX_CONTAINER_SECRET=do-not-leak
```

容器只接收 policy 中显式声明的 `--env`：

```text
HOME
LANG
PYTHONUNBUFFERED
```

容器读取 secret 时必须得到：

```text
<missing>
```

这证明默认行为不是：

```text
inherit os.environ
```

而是：

```text
explicit environment capability
```

生产系统下一层会把这一点从静态环境变量进一步升级成 short-lived credential broker。

---

## 9　直接读取 `/proc/self/status`

测试不是相信 Docker command line，而是在容器内部读取 Linux process status：

```text
CapEff
NoNewPrivs
```

并要求：

```text
CapEff == 0
NoNewPrivs == 1
```

因此：

```text
--cap-drop ALL
--security-opt no-new-privileges:true
```

不仅存在于 command builder，而且在受测 container process 上形成了实际 observable state。

---

## 10　CI 的真实证据

专用 workflow：

```text
.github/workflows/sandbox-security.yml
```

Sandbox security run 6 使用：

```text
GitHub-hosted Ubuntu 24.04
Docker Engine 28.0.4
containerd 2.3.4
runc 1.5.1
python:3.12-slim
```

当次拉取记录：

```text
registry digest:
sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea

local image id:
sha256:ec7d6c95cd3692a2e2d228a8b1ca74e4025b54121fcc4c5da6f09cfa473315ad
```

Docker container isolation tests：

```text
7 passed
Ruff: All checks passed
```

整个 Sandbox security workflow 两个 jobs 均成功：

```text
namespace-sandbox         → success（受 host-policy 限制的 runtime case 明确 skip）
docker-container-sandbox → success（真实 container negative tests 执行）
```

---

## 11　为什么现在仍不应该说“安全运行任意恶意代码”

Docker backend 比 restricted subprocess 强很多，但仍然共享宿主 Linux kernel：

```text
container
   ↓
namespace / cgroup / LSM / runtime
   ↓
shared host kernel
```

因此它不能抵抗所有：

```text
kernel vulnerability
container-runtime vulnerability
Docker daemon compromise
image supply-chain compromise
side-channel
hardware/firmware attack
```

当前也还没有自定义：

```text
seccomp profile
AppArmor/SELinux profile
credential broker
per-tool network allowlist
read-only dependency cache architecture
signed/pinned image policy
```

所以 machine-readable capability 应该区分：

```text
Docker tested isolation backend → validated for stated threat model
General hostile-code sandbox     → partial
VM-grade isolation               → not implemented
```

---

## 12　镜像 tag 与 digest

当前 policy 默认示例使用：

```text
python:3.12-slim
```

CI 会记录本次实际 registry digest 和 local image id，但 tag 自身仍然可能随未来发布移动。

因此生产化下一步应该变成：

```text
human-readable tag
+
pinned immutable digest
+
image provenance / SBOM
+
signature verification
```

教材不能把“今天拉到的 image”与“永远相同的 environment”混为一谈。

---

## 13　与 Coding Agent 的最终关系

安全 Coding Agent 的目标路径应该是：

```text
Model
 ↓
structured action
 ↓
Permission / Approval
 ↓
Container policy
 ↓
DockerSandboxExecTool
 ↓
container
 ↓
workspace mutation
 ↓
Artifact snapshot
 ↓
Verifier / hidden tests
```

而不是：

```text
Model → bash -lc → host
```

下一步会继续把 `build_coding_agent(...)` 的 execution backend 从 restricted subprocess 扩展到 container backend，同时保留透明的早期教学模式用于理解最小闭环。

---

## 14　原始资料

Docker 官方资料：

- `docker run`：https://docs.docker.com/reference/cli/docker/container/run/
- Bind mounts：https://docs.docker.com/engine/storage/bind-mounts/
- Network none：https://docs.docker.com/engine/network/drivers/none/
- Runtime resource constraints：https://docs.docker.com/engine/containers/resource_constraints/
- Security：https://docs.docker.com/engine/security/

Linux 边界继续参见：

- `no_new_privs`：https://docs.kernel.org/userspace-api/no_new_privs.html

源码与测试：

```text
src/astra_codex/docker_sandbox.py
tests/test_docker_sandbox.py
.github/workflows/sandbox-security.yml
```

---

## 15　本课毕业标准

读者应能解释并亲手验证：

1. 为什么“用了 Docker”本身不是安全证明？
2. 为什么 Docker flags 必须由 host policy 固定，而不能交给模型？
3. host secret invisibility 和 env secret scrubbing 有什么区别？
4. read-only rootfs 与 read-only workspace 为什么必须分开？
5. `--network none` 为什么要通过真实 socket negative test 验证？
6. `CapEff=0` 与 `NoNewPrivs=1` 分别说明什么？
7. 为什么容器成功通过这些测试后，仍不能宣称达到 VM-grade hostile-code containment？

能回答这些问题并读懂 `docker_sandbox.py`，才算真正理解 Agent execution sandbox 的这一层。
