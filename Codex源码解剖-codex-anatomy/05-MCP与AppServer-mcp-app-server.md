# C05　MCP、App Server 与外部能力互操作

> **Primary sources**  
> - Codex MCP crate: https://github.com/openai/codex/tree/main/codex-rs/codex-mcp  
> - MCP source directory: https://github.com/openai/codex/tree/main/codex-rs/codex-mcp/src  
> - RMCP client: https://github.com/openai/codex/tree/main/codex-rs/rmcp-client  
> - App Server: https://github.com/openai/codex/tree/main/codex-rs/app-server  
> - App Server protocol: https://github.com/openai/codex/tree/main/codex-rs/app-server-protocol

---

# 1　Codex 中 MCP 不是“把一个 HTTP API 包成 tool”

官方 workspace 中 MCP 已经被拆成独立的 `codex-mcp` 与 `rmcp-client` 等 crate。

当前 `codex-mcp/src` 可以看到独立模块处理：

```text
auth changes
auth elicitation
binding
binding clients
catalog
client capabilities
client tool catalog
Codex apps
...
```

这说明真实 MCP integration 至少需要考虑：

```text
server discovery
capability negotiation
tool catalog
binding
transport
authentication
elicitation
lifecycle
error propagation
model-visible tool conversion
```

而不只是 `tools/list` + `tools/call`。

---

# 2　MCP 在 Agent Stack 中处于哪一层

```text
Model
  ↓
Tool Router
  ↓
MCP Client Adapter
  ↓
Transport / Auth
  ↓
MCP Server
  ↓
External resource / tool
```

MCP 本身不是 planner，也不决定什么时候使用工具。

它解决的是：

> **如何标准化发现、描述、调用外部能力与上下文。**

Policy 仍在 Agent runtime。

---

# 3　为什么 MCP server availability 会进入 Turn Context

官方 `run_turn` 会收集当前输入所要求的 MCP servers / plugins，并在建立 step context 时考虑这些依赖。

这意味着工具目录不是永恒常量：

```text
User mentions capability
      ↓
resolve required MCP server
      ↓
ensure startup / dependencies
      ↓
build step context
      ↓
advertise tools to model
```

这与我们当前静态 `ToolRegistry` 的差距很明显。

后续需要：

```text
ToolCatalogSnapshot
McpServerState
CapabilitySource
AvailabilityReason
AuthState
```

---

# 4　App Server 与 MCP 是两个不同的 protocol boundary

不要混淆：

```text
App Server:
external UI/client ↔ Codex runtime

MCP:
Codex runtime ↔ external tool/context server
```

所以整体是：

```text
IDE / Desktop / CLI client
          ↓
     App Server
          ↓
      Codex Core
          ↓
       MCP Client
          ↓
      MCP Servers
```

这两个协议方向完全不同。

---

# 5　我们现有 minimal MCP 的定位

本仓库已经实现：

```text
JSON-RPC 2.0 educational envelope
server/discover
tools/list
tools/call
ToolRegistry adapter
in-process transport
client wrapper
unit tests
```

入口：

[`../代码-code/从零构建Astra与Codex级系统/src/astra_codex/mcp.py`](../代码-code/从零构建Astra与Codex级系统/src/astra_codex/mcp.py)

这适合学习 protocol 的最小数据流，但和官方 Codex MCP subsystem 仍有巨大差距。

---

# 6　下一阶段不应该直接跳到“支持全部 MCP”

推荐按可验证层次推进：

```text
M0 in-process request/response       已完成
M1 stdio framed transport
M2 subprocess lifecycle
M3 timeout / cancellation
M4 auth state
M5 capability negotiation
M6 resources / prompts
M7 elicitation
M8 HTTP transport
M9 reconnect / server failure recovery
M10 dynamic tool catalog snapshot
```

每层做 protocol transcript tests。

---

# 7　为什么要做 transcript parity

Protocol implementation 最适合对比：

```text
input JSON-RPC transcript
→ official/expected state transition
→ output transcript
```

而不是比较自然语言回答。

例如：

```text
client connects
→ discovers server
→ tools/list
→ call X
→ server error
→ runtime maps error into model observation
```

我们可以把每段 transcript 保存成 golden fixture。

---

# 8　App Server clean-room 目标

我们最终也要写自己的 UI-neutral adapter：

```text
astra-codex app-server
├─ thread/start
├─ thread/resume
├─ thread/fork
├─ turn/start
├─ turn/interrupt
├─ approval/respond
├─ event stream
└─ state query
```

但它只负责 protocol translation，不复制 core decision logic。

---

# 9　Auth 与 Permission 不能由 MCP server 自己决定全部

外部 server 可能要求 credential，但 host runtime 还必须决定：

```text
这个 server 是否可信？
允许暴露哪些 secrets？
tool call 是否需要 approval？
网络是否允许？
输出能否写入 model context？
是否可能有 prompt injection？
```

因此 MCP integration 还会和：

```text
permission manager
secret broker
sandbox
prompt-injection defense
```

交叉。

---

# 10　最终实验矩阵

| 实验 | 要证明什么 |
|---|---|
| stdio echo server | transport framing 正确 |
| server crash | Agent 不随 server 一起崩溃 |
| tool schema update | 新 snapshot 对模型可见 |
| auth required | credential flow 与 tool loop 解耦 |
| malicious tool description | tool catalog 不应绕过 system policy |
| timeout/cancel | 长调用能被 turn interrupt |
| two servers same tool name | namespace/disambiguation 正确 |

做到这里，MCP 才从“会调 API”进入 Agent infrastructure。
