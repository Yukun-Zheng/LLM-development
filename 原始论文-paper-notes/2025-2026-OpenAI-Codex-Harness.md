# Source Card：OpenAI Codex 开源 Harness / CLI（2025–2026）

> **对象**：OpenAI 官方开源仓库 `openai/codex`  
> https://github.com/openai/codex  
> **教材核验快照**：`d77ebc72237a639b6d877f2edc3b20b54631f25e`（2026-09-14）  
> **许可**：Apache-2.0，见官方仓库 README / LICENSE。  
> **课程位置**：[`../Codex源码解剖-codex-anatomy/README.md`](../Codex源码解剖-codex-anatomy/README.md)  
> **我们自己的 clean-room 实现**：[`../代码-code/从零构建Astra与Codex级系统/src/astra_codex/codex_harness.py`](../代码-code/从零构建Astra与Codex级系统/src/astra_codex/codex_harness.py)

---

# 1　这张卡片研究的到底是什么

这里的“Codex 开源”特指：

```text
Codex CLI
+ agent harness / runtime
+ protocol / execution / sandbox / MCP / multi-agent 等公开源码
```

它**不等于**：

```text
frontier Codex 模型权重开源
training recipe 开源
OpenAI 云端 Codex 全部生产基础设施开源
```

这是全书后续引用 Codex 时最重要的证据边界。

---

# 2　核心一手资料

- Repository README：  
  https://github.com/openai/codex/blob/main/README.md
- Rust workspace / crate graph：  
  https://github.com/openai/codex/blob/main/codex-rs/Cargo.toml
- Protocol mental model：  
  https://github.com/openai/codex/blob/main/codex-rs/docs/protocol_v1.md
- Core turn loop：  
  https://github.com/openai/codex/blob/main/codex-rs/core/src/session/turn.rs
- Tool approvals：  
  https://github.com/openai/codex/blob/main/codex-rs/core/src/tools/approvals.rs
- Tool sandboxing：  
  https://github.com/openai/codex/blob/main/codex-rs/core/src/tools/sandboxing.rs
- Base instructions / AGENTS.md semantics：  
  https://github.com/openai/codex/blob/main/codex-rs/protocol/src/prompts/base_instructions/default.md
- Multi-agent context / roles：  
  https://github.com/openai/codex/blob/main/codex-rs/core/src/session/multi_agents.rs
- App Server protocol：  
  https://github.com/openai/codex/tree/main/codex-rs/app-server-protocol
- MCP subsystem：  
  https://github.com/openai/codex/tree/main/codex-rs/codex-mcp

完整源码地图：[`../参考文献-references/03-OpenAI-Codex官方源码索引-codex-source-map.md`](../参考文献-references/03-OpenAI-Codex官方源码索引-codex-source-map.md)

---

# 3　Claim Ledger

## CODEX-HARNESS-C01

- **类型**：open-source boundary
- **精确转述**：OpenAI 官方 `openai/codex` 仓库公开 Codex CLI / agent harness 相关源码，并使用 Apache-2.0 许可证。
- **原始证据**：官方 repository README / LICENSE。
- **不能推出**：Codex frontier model weights、训练数据、训练 recipe 或全部云端生产基础设施已经开源。
- **状态**：supported with explicit boundary

## CODEX-HARNESS-C02

- **类型**：runtime architecture
- **精确转述**：官方 protocol 文档把 Session、Task、Turn 区分为不同层级；一个 Task 可以包含多个 Turn，而一个 Turn 可以包含模型请求、streamed response、工具/命令执行与 approval interaction。
- **原始证据**：`codex-rs/docs/protocol_v1.md`。
- **意义**：长期任务生命周期不能与一次 model context / 一次 sampling 调用混为一谈。
- **状态**：supported

## CODEX-HARNESS-C03

- **类型**：agent loop
- **精确转述**：公开 `run_turn` 实现的核心语义是：模型产生 function/tool call 时执行该调用，并把结果送入下一次 sampling；如果只产生 assistant message，则记录历史并完成当前 turn。
- **原始证据**：`codex-rs/core/src/session/turn.rs` 源码与函数注释。
- **适用条件**：这是公开 harness 当前实现的核心循环，不代表所有内部 OpenAI agent runtime 永远采用完全相同细节。
- **状态**：supported

## CODEX-HARNESS-C04

- **类型**：protocol / systems
- **精确转述**：公开协议把 UI → core 的 submission 与 core → UI 的 event 分开，使核心 agent runtime 可以被 CLI/TUI/IDE/App 等不同客户端驱动。
- **原始证据**：`codex-rs/docs/protocol_v1.md` 与 `app-server-protocol/`。
- **状态**：supported

## CODEX-HARNESS-C05

- **类型**：security architecture
- **精确转述**：公开源码把 approval policy 与 sandbox / permission policy 作为可区分的 runtime concern，而不是只依赖 prompt 中的自然语言约束。
- **原始证据**：`core/src/tools/approvals.rs`、`core/src/tools/sandboxing.rs`、App Server `ThreadSettings` schema。
- **不能推出**：我们的教学版 `SandboxPolicy` enum 已经实现 OS 级安全隔离；当前并没有。
- **状态**：supported; our reproduction partial

## CODEX-HARNESS-C06

- **类型**：instruction/context semantics
- **精确转述**：Codex base instructions 明确描述 `AGENTS.md` 的目录作用域与嵌套优先关系；更深层目录的指令可覆盖较高层目录的冲突规则，而 system/developer/user instructions 仍具有各自优先关系。
- **原始证据**：`codex-rs/protocol/src/prompts/base_instructions/default.md`。
- **状态**：supported

## CODEX-HARNESS-C07

- **类型**：multi-agent runtime
- **精确转述**：公开 Codex multi-agent runtime 提供 spawn/follow-up/message/wait/interrupt/list 等不同协作语义，并允许 child agent 继续产生 sub-agent。
- **原始证据**：`core/src/session/multi_agents.rs` 与 multi-agent tool handlers。
- **不能推出**：任意增加 agent 数量都会提高任务成功率或降低总成本。
- **状态**：supported; performance benefit task-dependent

---

# 4　我们当前的 clean-room 复现证据

当前 `codex_harness.py` 独立实现了一个小型状态机：

```text
TURN_STARTED
→ model sampling
→ optional tool call
→ optional approval request/decision
→ tool execution
→ structured observation
→ follow-up sampling
→ TURN_COMPLETED / TURN_STOPPED
```

对应自动测试明确覆盖：

```text
tool → observation → follow-up → final
approval deny → tool body never executes
approval allow → tool executes
step limit → explicit stop event
```

这证明的是**我们已经重建了公开架构的一小段行为契约**；并不等同于完整复刻官方 Codex。

---

# 5　后续需要继续建立的 Claim / Parity

下一批重点：

```text
CODEX-HARNESS-C08  pending user steering
CODEX-HARNESS-C09  compaction + continuation
CODEX-HARNESS-C10  resume / fork
CODEX-HARNESS-C11  MCP lifecycle / tool catalog snapshot
CODEX-HARNESS-C12  enforced sandbox
CODEX-HARNESS-C13  AgentGraph / mailbox / concurrency
CODEX-HARNESS-C14  worktree isolation + reviewer/merge
CODEX-HARNESS-C15  App Server transcript parity
```

每个 claim 都要求：官方源码位置 + 我们自己的实现 + 自动测试/协议 transcript，而不是只写一段架构描述。
