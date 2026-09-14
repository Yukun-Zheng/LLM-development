# 实现状态：从“教材蓝图”到真正可运行系统

> 本文件只记录**已经落地并有证据的代码**。没有实现的能力不打勾，不用“规划完成”冒充“代码完成”。

## 当前已实现

### Model Runtime

- [x] UTF-8 byte tokenizer
- [x] 教学版 byte-level BPE training / encode / decode
- [x] `ModelConfig` 与 tensor contract
- [x] Embedding / RMSNorm / RoPE / causal mask
- [x] Llama/Hugging Face half-split RoPE 与 interleaved RoPE
- [x] Grouped-Query Attention (GQA)
- [x] SwiGLU / residual Transformer blocks / LM head
- [x] per-layer KV Cache / prefill / incremental decode
- [x] greedy / temperature / top-k / top-p sampling / repetition penalty
- [x] safetensors raw loader / checkpoint key remap / shape report
- [x] full-forward vs cached-decode parity
- [x] **公开真实 checkpoint adapter：`HuggingFaceTB/SmolLM2-135M`**
- [x] **自己的 Transformer runtime 与 HF reference logits parity**

### 真实 checkpoint parity 证据

目标 checkpoint：`HuggingFaceTB/SmolLM2-135M`。

```text
public config.json + raw safetensors
        ↓
public_checkpoint.py key mapping
        ↓
our DecoderOnlyTransformer
        ↓
our logits

same input ids
        ↓
Hugging Face AutoModelForCausalLM
        ↓
reference logits
```

GitHub Actions 受测 CPU float32 / eager-reference 结果：

```json
{"max_abs": 0.0, "mean_abs": 0.0, "argmax_agreement": 1.0}
```

这个结论只证明当前受测模型/输入/数值设置，不外推到所有 Llama-family checkpoint。

### Agent Runtime

- [x] explicit JSON tool-call protocol + JSON-schema subset validator
- [x] tool registry / dispatch / error-as-observation
- [x] rooted filesystem / repository search / exact-match editing
- [x] shell execution + exit-code observation
- [x] Git status / diff / log / show
- [x] Python AST + Markdown heading repo map
- [x] minimal observe → act → observe loop
- [x] coding-agent system contract
- [x] SQLite persistent event memory
- [x] Git worktree manager primitive
- [x] minimal public HTTP text browser
- [x] deterministic scripted backend
- [x] typed task DAG / dependency-aware `PlanGraph`
- [x] failure → downstream blocked-state propagation
- [x] external verifier + file / argv-command / composite verification
- [x] deterministic coordinator primitive

### Agent Protocol Runtime

- [x] JSON-RPC 2.0 educational envelope
- [x] minimal MCP server discovery / `tools/list` / `tools/call`
- [x] `ToolRegistry` → MCP tool schema mapping
- [x] in-process MCP transport + client wrapper
- [x] protocol and tool-failure tests

当前 `mcp.py` 是**教学子集**，不是完整生产 MCP SDK。

### OpenAI Codex Source-Anatomy / Clean-room Runtime

官方源码课程：[`../../../Codex源码解剖-codex-anatomy/README.md`](../../../Codex源码解剖-codex-anatomy/README.md)

当前从公开 `openai/codex` 源码提取行为契约后，已经独立实现：

- [x] `codex_harness.py` clean-room educational turn loop
- [x] explicit `TurnSettings`
- [x] `SandboxPolicy` **metadata contract**（不是 OS enforcement）
- [x] approval-required tool gate
- [x] approval allow / deny state transition
- [x] typed harness events：turn/model/approval/tool/complete/stop
- [x] tool result → model observation → follow-up sampling
- [x] explicit max-model-step stop condition
- [x] four dedicated Codex-harness tests

已经自动验证：

```text
tool → observation → follow-up → final
approval deny → real tool body never executes
approval allow → tool executes
step limit → TURN_STOPPED
```

这只是 MiniCodex v1，不声称 API-compatible 或 production-compatible，也不声称实现了 OpenAI 未公开部分。

## 当前明确未实现

### Model / Inference

- [ ] high-performance FlashAttention / SDPA backend
- [ ] paged KV / prefix cache / chunked prefill
- [ ] continuous batching scheduler
- [ ] speculative decoding
- [ ] grammar-level constrained decoding
- [ ] native multimodal vision encoder / projector
- [ ] 更多真实 architecture/checkpoint parity matrix

### Codex-style Runtime / Agent Infrastructure

- [ ] async Submission Queue / Event Queue
- [ ] durable ThreadStore / TurnId / resume / fork
- [ ] pending user steering during active task
- [ ] scoped `AGENTS.md` resolver + instruction provenance
- [ ] context budget / compaction / rollover continuation
- [ ] OS/container security sandbox
- [ ] permission profiles / async approval broker
- [ ] App Server / external JSON-RPC control plane
- [ ] MCP stdio / HTTP / auth / resources / prompts / elicitation
- [ ] minimal A2A runtime
- [ ] AgentGraph / stable AgentId / mailbox
- [ ] parallel worker scheduler
- [ ] reviewer / merge agent
- [ ] unified diff / semantic patch engine
- [ ] tree-sitter / LSP semantic code intelligence
- [ ] test selection / coverage-guided verifier
- [ ] rollout trace / artifact store

### General Agent

- [ ] JavaScript browser automation
- [ ] DOM / accessibility-tree observation
- [ ] screenshot / mouse / keyboard computer use
- [ ] visual grounding
- [ ] Browser / Computer Use verifier
- [ ] indirect prompt-injection defenses
- [ ] SWE-bench / WebArena / OSWorld-style harnesses
- [ ] long-horizon general-agent benchmark
- [ ] online Agentic RL / policy-parameter update

## 自动测试状态

最新 Fast CI 对包含 `codex_harness.py` 与 `test_codex_harness.py` 的提交真实运行：

```text
30 passed, 1 warning in 3.02s
Ruff correctness lint: All checks passed
```

Fast CI 使用 **CPU-only PyTorch + pip cache**；真实 checkpoint parity 使用独立 workflow 并缓存公开权重。

## 成熟度规则

每个能力只有达到以下标准才算“实现”：

1. 有源码；
2. 有明确输入/输出契约；
3. 有至少一个自动测试；
4. 教材能解释对应数学或系统机制；
5. 若声称与工业/公开实现等价，必须有 parity / benchmark 证据。

智能体通识：[`../../智能体-agent/README.md`](../../智能体-agent/README.md)  
Codex 官方源码解剖：[`../../../Codex源码解剖-codex-anatomy/README.md`](../../../Codex源码解剖-codex-anatomy/README.md)