# Lesson 11　控制面认证与授权：谁可以控制哪个 Agent Thread？

当 App Server 从同进程对象调用扩展到真实 HTTP 后，一个新的问题立刻变成 P0：

> **谁可以调用控制面？即使身份合法，他又可以控制哪些 Thread、执行哪些方法？**

如果这一步没有进入 runtime，而只是依赖“URL 很难猜”“只给可信客户端”，那么拥有 shell、filesystem、Git、browser 等能力的 Agent 会把一个普通 API 权限漏洞放大成环境执行权限漏洞。

本课对应源码：

```text
src/astra_codex/control_auth.py
src/astra_codex/app_server.py
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

本项目当前最小模型：

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
```

因此“token 正确”绝不等于“所有 endpoint 都可以调用”。

---

## 2　Principal 的两个正交 scope

当前 `Principal` 有两个主要维度：

```text
allowed_methods
thread_ids
```

例如只读某一任务的 IDE client 可以被定义为：

```text
subject = reader-a

allowed_methods = {
    server/discover,
    thread/get,
    event/poll
}

thread_ids = {
    thr_a
}
```

于是它可以：

```text
thread/get(thr_a)      ✓
event/poll(thr_a)      ✓
```

但不能：

```text
thread/get(thr_b)      ✗
thread/submit(thr_a)   ✗
event/poll(all)        ✗
```

这比“reader role”这种只有角色名、没有具体资源边界的设计更可审计。

---

## 3　为什么 scoped principal 不能调用全局 `runtime/runOne`

当前 `runtime/runOne` 的语义是：

```text
claim next global WorkItem
```

它没有：

```text
threadId = ...
```

因此如果 principal 只被授权访问：

```text
thr_a
```

但我们允许它调用全局：

```text
runtime/runOne
```

那么 queue 可能返回：

```text
thr_b 的 WorkItem
```

这会绕过 Thread ACL。

因此当前授权器采取保守规则：

> **只要 principal 是 thread-scoped，就禁止调用无法安全映射到 Thread 的全局 worker endpoint。**

这不是功能缺陷，而是 capability boundary。

未来正确做法是增加：

```text
authorized queue claim
```

例如 worker principal 能提交一组允许的 Thread/tenant filter，由 Work Queue 在 SQL claim 阶段就施加约束。

---

## 4　为什么 scoped Thread creation 必须显式给 ID

若 principal 的 scope 是：

```text
{thr_a, thr_a_child}
```

却允许：

```text
thread/create({})
```

由服务器随机生成：

```text
thr_8fa...
```

那么创建后这个新 Thread 是否属于 principal scope 就变得模糊。

所以当前规则是：

```text
thread-scoped principal
→ thread/create 必须显式 new threadId
→ new threadId 必须已经属于授权集合
```

`thread/fork` 同理：parent 和 child 都必须落在 scope 内。

---

## 5　Bearer Token 为什么不以明文保存在 authorizer 中

当前教学实现把 token 转成：

$$
H=\mathrm{SHA256}(token)
$$

内存映射只保存：

```text
digest → Principal
```

这不是密码哈希方案，也不能代替 secret manager；目的只是避免授权表本身长期保留可直接复制使用的明文 token。

真正生产身份系统仍需要：

```text
short-lived token
expiry
rotation
revocation
OIDC / workload identity
mTLS
secret manager
central policy/audit
```

---

## 6　HTTP 层只负责携带身份，不负责决定权限

HTTP 请求：

```text
Authorization: Bearer <token>
```

经过：

```text
HTTP handler
→ extract bearer
→ AgentAppServer.handle(..., bearer_token=...)
→ BearerTokenAuthorizer
→ Principal
→ method/thread policy
→ dispatch
```

这样授权逻辑不会散落成：

```text
if header == ...
```

写在每一个 endpoint 内。

同样的 `AgentAppServer` policy 可以用于：

```text
InProcess transport
HTTP transport
future SSE/WebSocket transport
```

---

## 7　错误边界

当前 App Server 区分：

```text
-32001  Unauthenticated
-32003  Forbidden
-32000  Runtime error
-32600  Invalid Request
-32601  Method not found
-32602  Invalid params
-32603  Internal error
```

因此调用方可以区分：

```text
没有身份
≠
身份存在但权限不足
≠
runtime 本身执行失败
```

这对 IDE、自动重试和安全审计都很重要。

---

## 8　当前实现仍然不能安全暴露公网

目前 HTTP server 默认只绑定：

```text
127.0.0.1
```

即使已经有 bearer auth，仍然没有：

```text
TLS
OIDC
expiry / rotation
session management
rate limit
CSRF / Origin policy
multi-tenant policy database
security audit sink
```

尤其是：

> **Bearer Token + 明文公网 HTTP = token 可以被窃听。**

所以当前能力只能定义为：

```text
local authenticated reference control plane
```

不是 production internet service。

---

## 9　验收

这一层第一版必须证明：

```text
missing token       → Unauthenticated
invalid token       → Unauthenticated
wrong method        → Forbidden
wrong thread        → Forbidden
unscoped event feed → Forbidden for scoped principal
global worker claim → Forbidden for scoped principal
admin principal     → can run global worker
HTTP bearer header  → same policy as in-process transport
```

只有这些是自动测试过的，才可以说认证/授权进入了 runtime，而不是“README 写了权限设计”。

下一阶段：**SSE/WebSocket replay + session identity + queue-level thread/tenant filtering**，以及更底层、完全不同问题的 **OS/container sandbox**。
