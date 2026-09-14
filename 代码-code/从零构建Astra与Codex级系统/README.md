# 从零构建 Astra-class 与 Codex-class 系统

这是整本教材的**终极代码工程**：不是调用高层 Agent framework 拼一个 demo，而是从模型 forward、权重加载、推理缓存一路写到工具、环境、规划、验证、Coding Agent、Computer Use 与 Multi-Agent。

> **边界**：不要求自己训练 frontier 权重；可以使用 tiny random weights 做机制测试，使用公开 checkpoint 验证真实实现。Astra / Codex 的完整私有内部实现并未公开，因此本项目构建的是 **Astra-class / Codex-class** 同类系统，不声称复刻 OpenAI 私有源码。

实时实现状态：[`实现状态-STATUS.md`](实现状态-STATUS.md)  
全书质量看板：[`../../质量看板-QUALITY.md`](../../质量看板-QUALITY.md)  
智能体理论主线：[`../../智能体-agent/README.md`](../../智能体-agent/README.md)

---

# 1　现在已经真正落地的源码

```text
src/astra_codex/
├── config.py             # 模型结构约束
├── tokenizer.py          # UTF-8 byte tokenizer + 从零 BPE
├── model.py              # RMSNorm / RoPE / GQA / SwiGLU / Transformer
├── cache.py              # per-layer KV Cache
├── sampling.py           # greedy / temperature / top-k / top-p
├── engine.py             # prefill / incremental decode / streaming
├── weights.py            # raw safetensors / key remap / shape audit
├── public_checkpoint.py  # 公开 Llama-family checkpoint adapter
│
├── structured.py         # JSON tool call / schema validation
├── tools.py              # filesystem / shell / Git
├── editing.py            # ambiguity-safe exact edit
├── repo_map.py           # Python AST + Markdown structure map
├── memory.py             # SQLite persistent event memory
├── planning.py           # typed task DAG
├── verification.py       # external verifier
├── worktree.py           # Git worktree primitive
├── multi_agent.py        # coordinator primitive
├── general_tools.py      # minimal HTTP text browser
├── mcp.py                # minimal stateless MCP teaching subset
├── agent.py              # observe → act → observe loop
└── coding.py             # repository coding agent assembly
```

另外还有：

```text
examples/
tests/
教程-lessons/
```

---

# 2　最重要的新里程碑：真实公开模型已经完全对齐

我们不再只证明 tiny random model “能跑”。现在已经把真实公开 checkpoint：

```text
HuggingFaceTB/SmolLM2-135M
```

的 `config.json + raw safetensors` 直接加载进**我们自己的** `DecoderOnlyTransformer`。

数据流：

```text
SmolLM2 config.json
        ↓
ModelConfig
        ↓
raw safetensors
        ↓
llama_key_map
        ↓
our RMSNorm / RoPE / GQA / SwiGLU / Transformer
        ↓
our logits
```

参考端仅使用 Hugging Face `AutoModelForCausalLM` 作为独立 oracle，不用于实现我们的 forward。

GitHub Actions 的真实 CPU float32 parity 结果：

```json
{
  "max_abs": 0.0,
  "mean_abs": 0.0,
  "argmax_agreement": 1.0
}
```

也就是说，在当前受测模型、固定输入与 reference eager attention 设置下：

> **我们的 raw-weight loader + 自己写的 Transformer forward 与公开参考实现逐元素完全一致。**

运行：

```bash
pip install -e '.[parity]'
python examples/02_smollm2_parity.py
```

这只是第一个模型族。下一步会把 parity 扩成矩阵，而不是把一次成功外推成“兼容所有模型”。

---

# 3　模型运行时完整数据流

```text
Text
 ↓
Tokenizer / Token IDs
 ↓
Embedding
 ↓
RMSNorm
 ↓
Q / K / V
 ↓
RoPE
 ↓
Grouped-Query Causal Attention
 ↓
Residual
 ↓
RMSNorm
 ↓
SwiGLU
 ↓
Residual
 ↓
... N blocks
 ↓
Final RMSNorm
 ↓
LM Head
 ↓
Logits
```

自回归推理：

```text
Prompt
 ↓
Prefill
 ↓
per-layer KV Cache
 ↓
1-token Decode
 ↓
Sampling
 ↓
append token
 ↓
repeat
```

当前已经测试：

$$
\mathrm{logits}_{cached}(x_T)
\approx
\mathrm{logits}_{full}(x_{1:T}).
$$

---

# 4　Agent Runtime 数据流

```text
User Goal
   ↓
Context / Persistent Memory
   ↓
Policy / Model Backend
   ↓
Structured Action
   ↓
Tool / Protocol Runtime
   ├─ repo map
   ├─ filesystem
   ├─ edit
   ├─ shell
   ├─ Git
   ├─ HTTP browser
   └─ MCP
   ↓
Environment
   ↓
Observation
   ↓
Planner / Task DAG
   ↓
Verifier
   ├─ pass
   ├─ fail → retry / replan
   └─ blocked
   ↓
Continue / Finish
```

当前 Agent 不是纸面 ReAct 图，而能真实执行文件读写、shell、Git、编辑，并把失败作为 observation 返回。

---

# 5　Minimal MCP 已经从零实现

当前 `mcp.py` 实现一个**教学用 MCP 子集**：

```text
JSON-RPC 2.0
→ request / response envelope
→ server/discover
→ tools/list
→ tools/call
→ ToolRegistry adapter
→ in-process transport
→ MCPClient
```

并有独立测试：

- capability discovery；
- tool listing；
- filesystem tool round-trip；
- unknown RPC method；
- tool failure → structured observation。

这不是完整 MCP SDK。后续仍要实现：

```text
stdio transport
→ HTTP transport
→ auth / permissions
→ resources / prompts / extensions
→ minimal A2A
```

理论：[`../../智能体-agent/09-Agent协议与互操作-agent-protocols.md`](../../智能体-agent/09-Agent协议与互操作-agent-protocols.md)

---

# 6　从 0 的工程阶段

| 阶段 | 目标 | 当前状态 |
|---|---|---:|
| 00 | config / tensor contract / tests | ✅ |
| 01 | byte tokenizer + BPE | ✅ |
| 02 | modern decoder-only Transformer | ✅ |
| 03 | raw safetensors + checkpoint mapping | ✅ |
| 04 | real public checkpoint parity | ✅ SmolLM2-135M |
| 05 | KV Cache / prefill / decode / sampling | ✅ |
| 06 | structured generation / tool call | ✅ 第一版 |
| 07 | persistent memory / task state | ✅ primitive |
| 08 | filesystem / shell / Git / edit / browser | ✅ 第一版 |
| 09 | planning / verifier / recovery | ✅ primitive |
| 10 | coding-agent repository loop | ✅ 第一版 |
| 11 | protocol interoperability | 🟡 minimal MCP 已有 |
| 12 | multi-agent | 🟡 task DAG / coordinator / worktree 已有 |
| 13 | computer use | 🔴 待实现 |
| 14 | production inference scheduler | 🔴 待实现 |
| 15 | real benchmarks | 🔴 待实现 |

---

# 7　下一阶段模型系统

接下来不再只堆 feature，而按 reference → optimized → parity 顺序推进：

```text
Explicit Attention
→ PyTorch SDPA parity
→ FlashAttention backend

Contiguous KV Cache
→ Paged KV Cache
→ Prefix Cache

Single request
→ Continuous Batching
→ Chunked Prefill

Normal decoding
→ Speculative Decoding

JSON parser
→ grammar-level constrained decoding
```

每个优化都必须保留朴素 reference implementation，避免“跑得快但不知道函数是否被改了”。

---

# 8　下一阶段 Coding Agent

```text
text search
→ tree-sitter
→ symbol table
→ LSP definitions/references
→ import / dependency graph
→ semantic patch
→ test impact analysis
→ reviewer
```

最终 Coding Agent 要真正完成：

```text
understand issue
→ map repo
→ locate symbols
→ edit
→ select tests
→ run
→ diagnose
→ repair
→ full verification
→ review diff
→ finish
```

而不是“根据 prompt 写一段代码”。

---

# 9　下一阶段 Computer Use

当前只有 HTTP text fetch；真正 Astra-class general agent 还需要：

```text
JS-capable browser
→ DOM / Accessibility Tree
→ screenshot
→ visual grounding
→ click / type / scroll
→ state verification
→ desktop applications
→ cross-application workflow
```

理论：[`../../智能体-agent/10-浏览器与计算机使用-browser-computer-use.md`](../../智能体-agent/10-浏览器与计算机使用-browser-computer-use.md)

---

# 10　强制实现原则

### 先自己写，再用工业框架作对照

Attention 先写：

```python
scores = q @ k.transpose(-2, -1)
scores = scores / math.sqrt(head_dim)
scores = scores.masked_fill(~mask, float("-inf"))
probs = torch.softmax(scores, dim=-1)
out = probs @ v
```

然后才研究 SDPA、FlashAttention、vLLM。

### 每个模块都要能写出 shape

GQA：

```text
q: [B, Hq, Tq, Dh]
k: [B, Hkv, Tk, Dh]
v: [B, Hkv, Tk, Dh]
```

### 每个优化都有 parity

```text
Reference Implementation
       vs
Optimized Implementation
       ↓
numerical / behavioral parity
```

### Agent 成功必须来自环境证据

```text
模型说 tests pass        ≠ tests pass
shell exit code 0 + verifier = evidence
```

### 安全边界最终必须是系统机制

prompt 不是 sandbox。最终需要 permission、approval、process/container isolation、network/secrets boundary 与 audit log。

---

# 11　学习入口

```text
教程-lessons/00-从字符串到Logits-string-to-logits.md
教程-lessons/01-KVCache与增量解码-kv-cache.md
教程-lessons/02-工具协议与Agent循环-tool-agent-loop.md
教程-lessons/03-Codex级仓库闭环-coding-agent.md
../../教材-book/18A-模型运行时源码导读-runtime-walkthrough.md
../../教材-book/18B-智能体运行时源码导读-agent-walkthrough.md
```

本地基础测试：

```bash
cd '代码-code/从零构建Astra与Codex级系统'
pip install -e '.[dev]'
pytest
python examples/00_tiny_generate.py
python examples/01_scripted_coding_agent.py
```

真实公开 checkpoint parity：

```bash
pip install -e '.[parity]'
python examples/02_smollm2_parity.py
```

---

# 12　最终毕业验收

### Model Runtime

- 多个公开 checkpoint logits parity；
- KV Cache / Paged KV / Prefix Cache；
- streaming / batching / speculative decoding；
- profiling 与 serving。

### Codex-class

真实 Git repository + issue：

```text
inspect → localize → edit → test → diagnose → fix → verify → review → finish
```

### Astra-class

真实跨工具长程任务：

```text
research
→ browser/computer
→ filesystem / shell / Python
→ artifacts
→ failure recovery
→ verification
→ final result
```

### Multi-Agent

```text
coordinator
├─ worker A / worktree A
├─ worker B / worktree B
└─ reviewer
     ↓
verified merge
```

总理论入口：[`../../教材-book/18-从零构建Astra与Codex级系统-capstone.md`](../../教材-book/18-从零构建Astra与Codex级系统-capstone.md)

**本工程的终点不是“代码很多”，而是每一层都能解释、运行、测试、对齐、失败恢复并接受真实任务。**