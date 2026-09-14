# OpenAI Codex 官方源码索引 / Primary-Source Map

> **Canonical repository**：https://github.com/openai/codex  
> **License**：Apache-2.0（见官方 README / LICENSE）。  
> **教材核验快照**：`d77ebc72237a639b6d877f2edc3b20b54631f25e`，2026-09-14。  
> **原则**：涉及 Codex harness 的机制，优先引用官方源码/官方仓库文档，而不是二手博客。

---

# 1　入口与架构

1. Repository README  
   https://github.com/openai/codex/blob/main/README.md
2. Rust workspace / crate graph  
   https://github.com/openai/codex/blob/main/codex-rs/Cargo.toml
3. Core protocol mental model  
   https://github.com/openai/codex/blob/main/codex-rs/docs/protocol_v1.md
4. Repository-level contributor/agent instructions  
   https://github.com/openai/codex/blob/main/AGENTS.md

---

# 2　核心 Agent Loop

1. Turn loop  
   https://github.com/openai/codex/blob/main/codex-rs/core/src/session/turn.rs
2. Session module  
   https://github.com/openai/codex/tree/main/codex-rs/core/src/session
3. Task implementations  
   https://github.com/openai/codex/tree/main/codex-rs/core/src/tasks
4. Thread abstraction  
   https://github.com/openai/codex/blob/main/codex-rs/core/src/codex_thread.rs
5. Tool routing / registry  
   https://github.com/openai/codex/tree/main/codex-rs/core/src/tools

关键 claim：官方 `run_turn` 源码注释明确描述 function-call → execute → next sampling，以及 assistant-only message → record → complete 的循环语义。

---

# 3　Protocol / App Server

1. App Server  
   https://github.com/openai/codex/tree/main/codex-rs/app-server
2. App Server protocol  
   https://github.com/openai/codex/tree/main/codex-rs/app-server-protocol
3. ThreadSettings TypeScript schema  
   https://github.com/openai/codex/blob/main/codex-rs/app-server-protocol/schema/typescript/v2/ThreadSettings.ts
4. TurnStartParams schema  
   https://github.com/openai/codex/blob/main/codex-rs/app-server-protocol/schema/typescript/v2/TurnStartParams.ts
5. Thread start/resume/fork generated schemas  
   https://github.com/openai/codex/tree/main/codex-rs/app-server-protocol/schema/typescript/v2

---

# 4　执行、审批、Patch 与 Sandbox

1. Exec policy  
   https://github.com/openai/codex/blob/main/codex-rs/core/src/exec_policy.rs
2. Tool approvals  
   https://github.com/openai/codex/blob/main/codex-rs/core/src/tools/approvals.rs
3. Tool sandboxing  
   https://github.com/openai/codex/blob/main/codex-rs/core/src/tools/sandboxing.rs
4. Common sandboxing crate  
   https://github.com/openai/codex/tree/main/codex-rs/sandboxing
5. Linux sandbox  
   https://github.com/openai/codex/tree/main/codex-rs/linux-sandbox
6. Windows sandbox service  
   https://github.com/openai/codex/tree/main/codex-rs/windows-sandbox-service
7. Apply patch  
   https://github.com/openai/codex/tree/main/codex-rs/apply-patch
8. Shell command  
   https://github.com/openai/codex/tree/main/codex-rs/shell-command

---

# 5　Instructions、Context、Compaction 与 Memory

1. Base instructions  
   https://github.com/openai/codex/blob/main/codex-rs/protocol/src/prompts/base_instructions/default.md
2. Config / project-doc fallback  
   https://github.com/openai/codex/blob/main/codex-rs/config/src/config_toml.rs
3. Context fragments  
   https://github.com/openai/codex/tree/main/codex-rs/context-fragments
4. Prompt infrastructure  
   https://github.com/openai/codex/tree/main/codex-rs/prompts
5. Compaction code  
   https://github.com/openai/codex/tree/main/codex-rs/core/src
6. `compact_token_budget.rs`  
   https://github.com/openai/codex/blob/main/codex-rs/core/src/compact_token_budget.rs
7. `compact_remote_v2.rs`  
   https://github.com/openai/codex/blob/main/codex-rs/core/src/compact_remote_v2.rs
8. Memories  
   https://github.com/openai/codex/tree/main/codex-rs/memories
9. History  
   https://github.com/openai/codex/tree/main/codex-rs/history
10. Thread store  
    https://github.com/openai/codex/tree/main/codex-rs/thread-store

---

# 6　MCP / Connectors / Extensions

1. Codex MCP  
   https://github.com/openai/codex/tree/main/codex-rs/codex-mcp
2. Codex MCP source modules  
   https://github.com/openai/codex/tree/main/codex-rs/codex-mcp/src
3. RMCP client  
   https://github.com/openai/codex/tree/main/codex-rs/rmcp-client
4. Connectors  
   https://github.com/openai/codex/tree/main/codex-rs/connectors
5. MCP extension  
   https://github.com/openai/codex/tree/main/codex-rs/ext/mcp
6. Skills  
   https://github.com/openai/codex/tree/main/codex-rs/skills
7. Plugins / core plugins  
   https://github.com/openai/codex/tree/main/codex-rs/core-plugins

---

# 7　Multi-Agent 与 Worktree

1. Multi-agent role/context logic  
   https://github.com/openai/codex/blob/main/codex-rs/core/src/session/multi_agents.rs
2. Multi-agent tool handlers  
   https://github.com/openai/codex/tree/main/codex-rs/core/src/tools/handlers/multi_agents
3. Multi-agent v2 handlers  
   https://github.com/openai/codex/tree/main/codex-rs/core/src/tools/handlers/multi_agents_v2
4. Agent graph store  
   https://github.com/openai/codex/tree/main/codex-rs/agent-graph-store
5. Agent identity  
   https://github.com/openai/codex/tree/main/codex-rs/agent-identity
6. Agent roles  
   https://github.com/openai/codex/tree/main/codex-rs/agent-roles
7. Worktree  
   https://github.com/openai/codex/tree/main/codex-rs/worktree

当前公开源码中的 multi-agent usage hints 明确描述 `spawn_agent`、`followup_task`、`send_message`，并说明 child agents 可以继续 spawn sub-agents。

---

# 8　Trace / Evaluation / Observability

1. Rollout  
   https://github.com/openai/codex/tree/main/codex-rs/rollout
2. Rollout trace  
   https://github.com/openai/codex/tree/main/codex-rs/rollout-trace
3. Analytics  
   https://github.com/openai/codex/tree/main/codex-rs/analytics
4. OpenTelemetry  
   https://github.com/openai/codex/tree/main/codex-rs/otel
5. Core tests  
   https://github.com/openai/codex/tree/main/codex-rs/core/tests
6. App Server tests  
   https://github.com/openai/codex/tree/main/codex-rs/app-server/tests

---

# 9　Code Mode 与扩展能力

1. Code mode  
   https://github.com/openai/codex/tree/main/codex-rs/code-mode
2. Code mode host  
   https://github.com/openai/codex/tree/main/codex-rs/code-mode-host
3. Code mode protocol  
   https://github.com/openai/codex/tree/main/codex-rs/code-mode-protocol
4. Code mode runtime  
   https://github.com/openai/codex/tree/main/codex-rs/code-mode-runtime
5. V8 proof-of-concept  
   https://github.com/openai/codex/tree/main/codex-rs/v8-poc
6. Web search extension  
   https://github.com/openai/codex/tree/main/codex-rs/ext/web-search
7. Guardian extensions  
   https://github.com/openai/codex/tree/main/codex-rs/ext/guardian-v2

---

# 10　教材使用方法

Codex 相关章节默认按：

```text
Pinned official source
→ source claim
→ architecture diagram
→ clean-room implementation
→ automated test
→ observable parity
→ gap report
```

进行。

对应课程：[`../Codex源码解剖-codex-anatomy/README.md`](../Codex源码解剖-codex-anatomy/README.md)
