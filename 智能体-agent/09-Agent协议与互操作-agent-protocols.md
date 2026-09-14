# A9　Agent Protocols：MCP、A2A 与智能体互操作

> **本章主线**：当 Agent 从“一个进程里调用几个 Python 函数”发展到跨进程、跨主机、跨组织协作时，工具调用本身就变成协议问题。协议层决定能力如何发现、参数怎样描述、任务怎样传递、权限怎样声明、状态怎样关联、错误怎样返回。

---

# 1　为什么 Agent 需要协议层

最小 Tool Agent 可以写成：

```text
LLM
↓
JSON tool call
↓
Python function
```

但真实系统更接近：

```text
Agent Host
├─ local filesystem tool
├─ remote database server
├─ browser runtime
├─ enterprise SaaS connector
├─ MCP server
└─ another remote agent
```

所以问题从“函数签名是什么”升级成：

- 服务怎样发现？
- capability 怎样声明？
- 请求/响应格式是什么？
- transport 是 stdio 还是 HTTP？
- authorization 放在哪里？
- 长任务怎样表示？
- streaming 怎样处理？
- Agent-to-Agent 怎样交付 artifact？

---

# 2　先理解 JSON-RPC 2.0

MCP 与 A2A 都大量使用 RPC 思想。JSON-RPC 的最小请求：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "search",
    "arguments": {"q": "attention"}
  }
}
```

成功响应：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {...}
}
```

错误响应：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32601,
    "message": "Method not found"
  }
}
```

关键点是：**协议把模型决策与能力实现解耦。**

---

# 3　MCP：Model Context Protocol

官方规范入口：

- https://modelcontextprotocol.io/specification/
- 2026-07-28 版本发布说明： https://blog.modelcontextprotocol.io/posts/2026-07-28/

截至 2026-07-28，MCP 已进一步朝 stateless protocol core 演进；新版本不再要求传统 initialize/session handshake，每个请求可以携带协议版本、客户端身份与 capability metadata，并新增 `server/discover` 用于可选能力发现。

教材必须注意版本差异：早期 2025 MCP 与 2026 版本的 session/lifecycle 语义并不完全相同。

## 3.1　MCP 解决什么

MCP 不是 Agent planner，也不是 LLM。

它主要定义：

```text
Host / Client
      ↕
MCP Protocol
      ↕
Server
├─ tools
├─ resources
├─ prompts
└─ other protocol capabilities
```

也就是说，它解决的是**模型/Agent 如何以统一协议连接外部上下文和能力**。

## 3.2　Tool discovery

概念上：

```text
tools/list
↓
[
  {
    name,
    description,
    inputSchema
  }
]
```

Agent 不应该依赖 server 内部 Python 类名，而只依赖协议契约。

## 3.3　Tool call

```text
tools/call
├─ name
└─ arguments
```

这与我们当前 `ToolRegistry` 非常接近，因此非常适合从内部接口逐步升级成真正协议接口。

---

# 4　为什么“用了 MCP”不等于“有了 Agent”

MCP 只回答：

> 怎么访问能力？

但不回答：

- 应该什么时候调用？
- 先调用哪个？
- 调用失败怎么办？
- 怎样规划？
- 怎样验证？
- 什么时候停止？

所以完整栈应该是：

```text
Policy / Planner
       ↓
Agent Runtime
       ↓
MCP Client
       ↓
MCP Server
       ↓
Environment
```

协议层只是 Agent OS 的一层。

---

# 5　A2A：Agent2Agent

官方规范：

- https://a2a-protocol.org/

A2A 关注的不是“Agent 调工具”，而是：

> **Agent 与 Agent 怎样跨边界协作。**

其核心对象包括：

- AgentCard；
- Message；
- Task；
- TaskStatus；
- Artifact；
- streaming task update。

一个 Agent 可以公开自己的能力和 endpoint，另一个 Agent 发送 message，服务端返回 Message 或 Task。

## 5.1　Tool 与 Agent 的区别

Tool 更像：

```text
call(arguments)
→ result
```

Agent 更像：

```text
goal / message
→ possibly long-running task
→ status updates
→ artifacts
→ completion / failure
```

因此 Agent-to-Agent 协议需要更强的 task lifecycle。

---

# 6　AgentCard 为什么重要

AgentCard 本质上是一种 capability advertisement：

```text
Who are you?
What can you do?
Where are you?
Which transport do you support?
Do you support streaming?
How should I authenticate?
```

它让多智能体系统不必把所有 worker 逻辑硬编码在一个 coordinator 中。

未来我们自己的 multi-agent runtime 应逐步从：

```python
workers = {"a": worker_a, "b": worker_b}
```

升级到：

```text
discover remote agent
→ inspect AgentCard
→ negotiate capability
→ send task
→ stream updates
→ receive artifact
```

---

# 7　MCP 与 A2A 的分工

一个简单但有用的区分：

```text
MCP:
Agent / Host ↔ Tools / Context Servers

A2A:
Agent ↔ Agent
```

但现实系统可能组合：

```text
Coordinator Agent
      │
      ├─ MCP → Git server
      ├─ MCP → browser server
      │
      ├─ A2A → coding worker
      └─ A2A → research worker
```

所以最终 Agent platform 需要同时理解 tool protocol 和 agent protocol。

---

# 8　安全不是协议之外的问题

协议一旦允许远程能力，就必须考虑：

- server identity；
- client identity；
- authorization；
- least privilege；
- capability spoofing；
- schema injection；
- prompt injection through tool output；
- secret leakage；
- confused deputy；
- audit log。

**“工具 schema 合法”并不代表调用安全。**

一个高质量 Agent runtime 应分离：

```text
Model proposes action
        ↓
Protocol validates shape
        ↓
Permission layer checks authority
        ↓
Execution layer performs action
        ↓
Audit layer records evidence
```

---

# 9　从零实现 Minimal MCP

本项目源码目标：

```text
src/astra_codex/mcp.py
```

第一阶段只实现教学子集：

1. JSON-RPC 2.0 request / response；
2. `tools/list`；
3. `tools/call`；
4. stateless server dispatch；
5. client wrapper；
6. tool schema 映射；
7. deterministic unit tests。

不把第一版包装成“完整 MCP SDK”。

## 9.1　为什么先实现 stateless in-process transport

先把协议语义写清楚：

```text
request bytes
→ JSON decode
→ RPC validation
→ method dispatch
→ result/error
→ JSON encode
```

再替换 transport：

```text
in-process
→ stdio
→ HTTP
```

如果一开始就使用官方 SDK，读者会只会“配置 MCP”，而不会理解 protocol。

---

# 10　未来代码路线

```text
minimal_mcp/
├── jsonrpc
├── tool discovery
├── tool call
├── protocol metadata
├── stdio transport
├── HTTP transport
├── auth layer
└── interoperability tests

minimal_a2a/
├── AgentCard
├── Message
├── Task
├── Artifact
├── message/send
├── tasks/get
└── streaming
```

最终目标不是重新造一个生产 SDK，而是能读懂任何 Agent protocol 的 wire format、lifecycle 与安全边界。