# 实现状态：从“教材蓝图”到真正可运行系统

> 本文件只记录**已经落地并有证据的代码**。没有实现的能力不打勾，不用“规划完成”冒充“代码完成”。

## 当前已实现

### Model Runtime

- [x] UTF-8 byte tokenizer
- [x] 教学版 byte-level BPE training / encode / decode
- [x] `ModelConfig` 与 tensor contract
- [x] Embedding
- [x] RMSNorm
- [x] RoPE
- [x] Llama/Hugging Face half-split RoPE 与 interleaved RoPE 两种布局
- [x] causal mask
- [x] Grouped-Query Attention (GQA)
- [x] SwiGLU
- [x] residual Transformer blocks
- [x] LM head / tied embeddings
- [x] per-layer KV Cache
- [x] prefill
- [x] one-token incremental decode
- [x] greedy / temperature / top-k / top-p sampling
- [x] repetition penalty
- [x] safetensors raw loader
- [x] checkpoint key remap / shape report
- [x] full-forward vs cached-decode parity test
- [x] **公开真实 checkpoint adapter：`HuggingFaceTB/SmolLM2-135M`**
- [x] **自己的 Transformer runtime 与官方参考实现 logits parity**

### 真实 checkpoint parity 证据

目标 checkpoint：`HuggingFaceTB/SmolLM2-135M`。

执行路径：

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

GitHub Actions 实测：

```json
{
  "max_abs": 0.0,
  "mean_abs": 0.0,
  "argmax_agreement": 1.0
}
```

这意味着在当前测试输入、CPU float32 和 reference eager attention 设置下，我们从 raw public weights 加载到自己的 forward 后，输出与公开参考实现逐元素完全一致。

注意：这证明的是**当前受测模型/输入/数值设置的 forward parity**，不应夸大成“兼容所有 Llama-family checkpoint”。

### Agent Runtime

- [x] explicit JSON tool-call protocol
- [x] JSON-schema subset validator
- [x] tool registry / dispatch / error-as-observation
- [x] rooted filesystem access
- [x] repository text search
- [x] exact-match safe editing
- [x] shell execution + exit-code observation
- [x] Git status / diff / log / show
- [x] Python AST + Markdown heading repo map
- [x] observe → act → observe loop
- [x] coding-agent system contract
- [x] SQLite persistent event memory
- [x] Git worktree manager primitive
- [x] minimal public HTTP text browser
- [x] deterministic scripted backend for tests
- [x] typed task DAG / dependency-aware `PlanGraph`
- [x] plan failure → downstream blocked-state propagation
- [x] external verifier protocol + file / argv-command verifiers
- [x] composite verification with structured evidence
- [x] deterministic coordinator primitive over task graph

### Agent Protocol Runtime

- [x] JSON-RPC 2.0 educational request / response envelope
- [x] MCP 2026-07-28 protocol version metadata
- [x] stateless `server/discover`
- [x] MCP `tools/list`
- [x] MCP `tools/call`
- [x] existing `ToolRegistry` → MCP tool-schema mapping
- [x] in-process MCP transport
- [x] MCP client wrapper
- [x] protocol and tool-failure unit tests

当前 `mcp.py` 明确是**教学子集**，不是生产级 MCP SDK。

## 当前明确未实现

这些不能因为“接口留好了”就写成已经完成：

### Model / Inference

- [ ] 高性能 FlashAttention / SDPA backend
- [ ] paged KV cache
- [ ] prefix cache
- [ ] chunked prefill
- [ ] continuous batching scheduler
- [ ] speculative decoding
- [ ] grammar-level constrained decoding
- [ ] native multimodal vision encoder / projector
- [ ] 更多真实 checkpoint / architecture adapter 的 parity matrix

### Protocol / Agent Runtime

- [ ] MCP stdio transport
- [ ] MCP HTTP transport
- [ ] MCP auth / permission integration
- [ ] MCP resources / prompts / extensions 完整覆盖
- [ ] minimal A2A AgentCard / Message / Task / Artifact runtime
- [ ] JavaScript browser automation
- [ ] DOM / accessibility-tree observation
- [ ] screenshot / mouse / keyboard computer use
- [ ] visual grounding
- [ ] OS/container security sandbox
- [ ] permission manager / approval gates
- [ ] tree-sitter / LSP semantic code intelligence
- [ ] unified diff / semantic patch engine
- [ ] test selection / coverage-guided verifier
- [ ] **parallel** worker scheduler
- [ ] reviewer / merge agent
- [ ] task checkpoint / resume across processes
- [ ] SWE-bench evaluation harness
- [ ] WebArena / OSWorld-style evaluation harness
- [ ] long-horizon general-agent benchmark
- [ ] online Agentic RL / policy-parameter update

## 自动测试状态

当前单元测试已经覆盖 model/cache、tokenizer、structured tools、editing/memory/repo map、planning/verification/coordinator、public-checkpoint adapter 与 minimal MCP。

Fast CI 使用 **CPU-only PyTorch**，避免普通单元测试无意义下载整套 CUDA wheel；真实 checkpoint parity 使用独立 workflow 并缓存公开权重。

## 成熟度规则

每个能力只有达到以下标准才算“实现”：

1. 有源码；
2. 有明确输入/输出契约；
3. 有至少一个自动测试；
4. 教材能解释对应数学或系统机制；
5. 若声称与工业/公开实现等价，必须有 parity / benchmark 证据。

智能体主线理论入口：[`../../智能体-agent/README.md`](../../智能体-agent/README.md)