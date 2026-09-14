# 实现状态：从“教材蓝图”到真正可运行系统

> 本文件只记录**已经落地的代码**。没有实现的能力不打勾，不用“规划完成”冒充“代码完成”。

## 当前已实现

### Model Runtime

- [x] UTF-8 byte tokenizer
- [x] 教学版 byte-level BPE training / encode / decode
- [x] `ModelConfig` 与 tensor contract
- [x] Embedding
- [x] RMSNorm
- [x] RoPE
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

## 当前明确未实现

这些不能因为“接口留好了”就写成已经完成：

- [ ] 与某个公开真实 checkpoint 的 logits parity adapter
- [ ] 高性能 FlashAttention / SDPA backend
- [ ] paged KV cache
- [ ] continuous batching scheduler
- [ ] speculative decoding
- [ ] grammar-level constrained decoding
- [ ] native multimodal vision encoder / projector
- [ ] JavaScript browser automation
- [ ] screenshot / mouse / keyboard computer use
- [ ] OS/container security sandbox
- [ ] tree-sitter / LSP semantic code intelligence
- [ ] unified diff parser
- [ ] test selection / coverage-guided verifier
- [ ] **parallel** worker scheduler
- [ ] reviewer / merge agent
- [ ] task checkpoint / resume across processes
- [ ] SWE-bench evaluation harness
- [ ] long-horizon general-agent benchmark
- [ ] online Agentic RL / policy-parameter update

## 成熟度规则

每个能力只有达到以下标准才算“实现”：

1. 有源码；
2. 有明确输入/输出契约；
3. 有至少一个自动测试；
4. 教材能解释对应数学或系统机制；
5. 若声称与工业实现等价，必须有 parity / benchmark 证据。

智能体主线理论入口：[`../../智能体-agent/README.md`](../../智能体-agent/README.md)
