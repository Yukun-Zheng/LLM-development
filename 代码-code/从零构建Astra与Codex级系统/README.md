# 从零构建 Astra-class 与 Codex-class 系统

这是整本《大语言模型发展史与技术原理》的**终极代码项目**。

目标不是调用现成 agent framework，而是从最底层的模型 forward 开始，逐步搭出：

1. 自己的 decoder-only Transformer runtime；
2. 自己的 KV Cache 与推理循环；
3. structured output / tool calling；
4. context / notes / searchable history；
5. shell / filesystem / Python / browser 工具运行时；
6. coding agent；
7. general-purpose agent；
8. worktree-based multi-agent；
9. evaluation / verifier / safety boundaries。

> 训练权重不在范围内。模型参数使用 tiny random weights 做单元测试，或加载公开 checkpoint 做 parity 验证。核心机制不能用高层框架一行隐藏掉。

---

## 最终数据流

```text
User Goal
   ↓
Context Manager
   ↓
Model Runtime
   ├─ tokenizer
   ├─ embedding
   ├─ Transformer
   ├─ KV Cache
   ├─ sampling
   └─ structured generation
   ↓
Action
   ├─ text
   └─ tool call
        ↓
Tool Runtime
   ├─ filesystem
   ├─ shell
   ├─ Python
   ├─ browser
   ├─ Git
   └─ computer
        ↓
Observation
        ↓
Agent State
        ↓
Verifier / Continue / Finish
```

---

## 实现阶段

| 阶段 | 目录 | 目标 | 对应理论 |
|---|---|---|---|
| 00 | `00-基础设施/` | config、tensor contract、tests | tensor / shape / dtype |
| 01 | `01-tokenizer/` | byte tokenizer + BPE | Tokenization |
| 02 | `02-model-core/` | decoder-only Transformer | Attention / RoPE / GQA / SwiGLU |
| 03 | `03-weight-loader/` | 加载公开 safetensors | checkpoint / weight tying |
| 04 | `04-inference-engine/` | KV Cache、prefill、decode、sampling | LLM inference |
| 05 | `05-structured-generation/` | schema / tool call | constrained generation |
| 06 | `06-context-memory/` | compaction、notes、search | long context / memory |
| 07 | `07-tool-runtime/` | shell/files/Python/browser | tool use / security |
| 08 | `08-agent-core/` | agent loop、planner、verifier | ReAct / agent systems |
| 09 | `09-codex-class/` | repo edit / test / Git / worktree | coding agents |
| 10 | `10-astra-class/` | browser/computer/artifacts | general agents |
| 11 | `11-multi-agent/` | task graph / parallel workers | multi-agent systems |
| 12 | `12-evals/` | parity / task / safety evals | evaluation |

---

## 强制原则

### 1. 先自己写，再使用工业框架对照

例如 attention：

先写：

```python
scores = q @ k.transpose(-2, -1)
scores = scores / math.sqrt(head_dim)
scores = scores + mask
probs = torch.softmax(scores, dim=-1)
out = probs @ v
```

再研究 PyTorch SDPA、FlashAttention、vLLM 在哪里优化。

不能反过来只写：

```python
AutoModelForCausalLM.from_pretrained(...)
```

然后声称“实现了大模型”。

### 2. 每个模块必须写 shape

例如：

```text
q: [B, Hq, Tq, Dh]
k: [B, Hkv, Tk, Dh]
v: [B, Hkv, Tk, Dh]
```

### 3. 每个优化都必须先有 reference implementation

例如 KV Cache：

```text
Reference: full recomputation
Optimized: incremental decode
```

二者 logits 必须做 parity test。

### 4. Agent 必须有真实 environment feedback

不能只模拟：

```text
Thought → fake Action → fake Observation
```

而要真正执行：

```text
edit → pytest → traceback → next edit
```

### 5. 安全边界必须是代码，不是 prompt

工具层需要：

- allowed roots；
- command timeout；
- network policy；
- destructive-operation guard；
- audit log。

---

## 最终验收任务

### Model Runtime

- 加载一个公开 checkpoint；
- 与 reference implementation 做 logits parity；
- KV Cache decode 与 full recompute 对齐；
- 支持 streaming generation。

### Codex-class

输入一个真实 Git repository + issue，系统能够：

```text
inspect repo
→ search code
→ edit
→ run tests
→ read failure
→ fix
→ rerun
→ review diff
→ finish
```

### Astra-class

输入跨工具任务，系统能够：

```text
research
→ browser
→ filesystem
→ Python
→ shell
→ artifact
→ verify
→ final result
```

### Multi-Agent

至少完成一次：

```text
coordinator
├─ worker A / worktree A
├─ worker B / worktree B
└─ reviewer
     ↓
merge
```

---

## 与教材的关系

总理论说明见：

[`../../教材-book/18-从零构建Astra与Codex级系统-capstone.md`](../../教材-book/18-从零构建Astra与Codex级系统-capstone.md)

从现在开始，每一个理论章节都应该最终落到这个工程中的某个模块。