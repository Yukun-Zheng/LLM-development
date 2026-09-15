# Lesson 11　控制面认证与授权：谁可以控制哪个 Agent Thread？

当 App Server 从同进程对象调用扩展到真实 HTTP 后，一个新的问题立刻变成 P0：

> **谁可以调用控制面？即使身份合法，他又可以控制哪些 Thread、执行哪些方法？凭据泄露或过期以后又会发生什么？**

如果这一步没有进入 runtime，而只是依赖“URL 很难猜”“只给可信客户端”，那么拥有 shell、filesystem、Git、browser 等能力的 Agent 会把一个普通 API 权限漏洞放大成环境执行权限漏洞。

对应源码：

```text
src/astra_codex/control_auth.py
src/astra_codex/app_server.py
src/astra_codex/runtime.py
src/astra_codex/runtime_queue.py
src/astra_codex/http_app_server.py
```

测试：

```text
tests/test_control_auth.py
```

---

## 1　Authentication 与 Authorization 必须分开

认证（Authentication）回答：

```text
你是谁？
```

授权（Authorization）回答：

```text
你可以做什么？
你可以作用于哪个 Thread？
```

当前最小模型：

```text
Bearer Token
     ↓
Authentication
     ↓
Principal
├─ subject
├─ allowed_methods
└─ thread_ids
     ↓
Authorization
     ↓
App Server dispatch
     ↓
Runtime / Queue enforcement
```

因此“token 正确”绝不等于“所有 endpoint 都可以调用”。

---

## 2　Principal 的两个正交 scope

一个只读 `thr_a` 的客户端可以被定义为：

```text
subject = reader-a
allowed_methods = {server/discover, thread/get, event/poll}
thread_ids = {thr_a}
```

于是：

```text
thread/get(thr_a)      ✓
event/poll(thr_a)      ✓
thread/get(thr_b)      ✗
thread/submit(thr_a)   ✗
event/poll(all)        ✗
```

权限因此同时约束**动作**和**资源**。

---

## 3　授权必须下沉到 Queue Claim

假设 FIFO 顺序：

```text
work_b  thread=thr_b  created=1
work_a  thread=thr_a  created=2
```

worker principal 只允许：

```text
thread_ids = {thr_a}
```

错误设计是：

```text
global claim work_b
→ 之后才发现不允许
→ 再拒绝
```

此时 capability 已经泄漏到 lease 阶段。

现在 `DurableWorkQueue.claim(...)` 在同一个 `BEGIN IMMEDIATE` 事务中直接加入：

```text
AND thread_id IN (authorized_threads)
```

所以数据流是：

```text
Principal({thr_a})
→ runtime/runOne(threadId=thr_a)
→ App Server AuthZ
→ runtime.run_one(allowed_thread_ids={thr_a})
→ SQL-filtered queue.claim(thread_ids={thr_a})
→ 只能 lease thr_a
```

自动负对照故意让 `thr_b` 更早入队，最终 `thr_a` 被执行，而 `thr_b` 仍保持 `PENDING`。

---

## 4　Scoped worker 的 API 契约

全局 admin 可以：

```text
runtime/runOne({workerId: admin-worker})
```

thread-scoped principal 必须：

```text
runtime/runOne({workerId: worker-a, threadId: thr_a})
```

缺失或越权 `threadId` 都返回 Forbidden。

这里存在两层防线：

```text
App Server AuthZ
        ↓
Queue SQL Claim Filter
```

后者位于真正发生任务所有权转移的位置。

---

## 5　Scoped Thread creation / fork

如果 principal 只允许：

```text
{thr_a, thr_a_child}
```

那么随机生成一个新 Thread ID 会让资源归属变得不清楚。

所以 thread-scoped principal 创建或 fork 时必须显式给出目标 ID，并且 parent / child 都必须落在授权集合内。

---

## 6　Bearer Credential 不只是 `token → Principal`

当前实现已经把 bearer credential 生命周期显式化为：

```text
BearerCredential
├─ principal
├─ issued_at
├─ expires_at
└─ revoked_at
```

原始 secret 不作为 lookup value 保存，而是：

$$
H=\mathrm{SHA256}(token)
$$

然后：

```text
digest → BearerCredential
```

这不是 password KDF，也不能替代 secret manager；目标只是避免 authorizer 表直接保留可复制使用的明文 bearer secret。

---

## 7　Expiry、Revocation 与 Rotation

### Expiry

注册短期凭据：

```text
issued_at = 100
expires_at = 110
```

则：

```text
auth(now=109.999) → pass
auth(now=110.000) → expired
```

边界采用：

```text
now >= expires_at
```

即到期时刻本身已经不可用。

### Revocation

```text
active
→ revoke(token)
→ revoked_at = t
→ authenticate → rejected
```

重复 revoke 保持第一次 `revoked_at`，不会不断改写安全审计时间。

### Rotation

```text
old token
→ verify old is active
→ register new token with same Principal scope
→ revoke old token
```

因此权限不会因为换 token 意外扩大。

自动测试还覆盖一个关键 failure case：如果新 token 已经存在，rotation 失败时旧 token **不能被提前吊销**。

---

## 8　HTTP / SSE 只携带身份，权限逻辑保持统一

HTTP 请求：

```text
Authorization: Bearer <token>
```

统一进入：

```text
BearerTokenAuthorizer
→ Principal
→ method/thread authorization
→ runtime / queue enforcement
```

因此：

```text
InProcess JSON-RPC
HTTP JSON-RPC
SSE event stream
```

共享同一套 principal/thread policy，不各写一份权限判断。

---

## 9　错误边界

App Server 区分：

```text
-32001  Unauthenticated
-32003  Forbidden
-32000  Runtime error
-32600  Invalid Request
-32601  Method not found
-32602  Invalid params
-32603  Internal error
```

于是：

```text
身份无效 / 已过期 / 已吊销
≠
身份有效但权限不足
≠
runtime 执行失败
```

客户端可以据此决定重新登录、请求授权，还是处理任务失败。

---

## 10　当前仍不是生产身份系统

现在已经有：

```text
Bearer digest lookup
expiry
revocation
rotation
method scope
thread scope
queue-level claim fencing
HTTP/SSE policy reuse
```

但仍然没有：

```text
secure credential persistence
multi-process revocation propagation
OIDC / workload identity
mTLS
central identity provider
security audit service
rate limiting
CSRF / Origin policy
TLS termination
```

尤其：

> **Bearer Token + 明文公网 HTTP 仍然是不安全的。**

当前 HTTP/SSE reference server 默认只服务 loopback。

---

## 11　验收标准

现在自动测试要求：

```text
missing token                 → Unauthenticated
invalid token                 → Unauthenticated
expired token                 → rejected
revoked token                 → rejected
rotation                      → old revoked, new active
failed rotation               → old remains active
wrong method                  → Forbidden
wrong thread                  → Forbidden
unscoped event feed           → Forbidden for scoped principal
scoped runOne without thread  → Forbidden
scoped runOne wrong thread    → Forbidden
scoped runOne allowed thread  → only that thread can be leased
older unauthorized work       → remains PENDING
admin global worker           → allowed
HTTP / SSE bearer             → same resource policy
```

截至这一版进入全量回归，Fast CPU CI run 143：

```text
112 passed, 1 warning in 7.53s
Ruff correctness lint: All checks passed
```

下一阶段身份侧重点不再是继续加静态字段，而是**持久 credential/session store、跨进程 revocation、OIDC/workload identity 与安全审计**；执行安全则继续独立推进真正的 OS/container sandbox。
