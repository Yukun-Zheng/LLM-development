# 从零构建 Astra-class 与 Codex-class 系统

这是整本《大语言模型发展史与技术原理》的**终极代码项目**。

目标不是调用现成 agent framework，而是从最底层的模型 forward 开始，逐步搭出一套可以真正运行、测试和继续扩展的现代模型运行时与智能体系统。

> **边界**：训练 frontier 权重不在范围内。我们自己实现模型结构、推理与 agent runtime；参数可用 tiny random weights 做单元测试，也可加载公开 checkpoint 做 parity。Astra / Codex 的完整私有内部实现并未公开，因此本项目构建的是 **Astra-class / Codex-class** 同类系统，而不是声称复刻 OpenAI 私有源码。

---

## 现在已经不是蓝图：第一版源码已经落地

核心 Python package：

```text
src/astra_codex/
├── config.py          # 模型结构约束
├── tokenizer.py       # byte tokenizer + 从零训练 BPE
├── model.py           # RMSNorm / RoPE / GQA / SwiGLU / decoder-only Transformer
├── cache.py           # per-layer KV Cache
├── sampling.py        # greedy / temperature / top-k / top-p / repetition penalty
├── engine.py          # prefill / incremental decode / streaming generation
├── weights.py         # raw safetensors / key remap / shape audit
├── structured.py      # JSON tool call + schema validation
├── tools.py           # filesystem / shell / Git
├── editing.py         # ambiguity-safe exact edit
├── repo_map.py        # Python AST + Markdown structure map
├── memory.py          # SQLite persistent event memory
├── worktree.py        # Git worktree primitive
├── general_tools.py   # 最小 HTTP text browser
├── agent.py           # observe → act → observe loop
└── coding.py          # Codex-class repository agent assembly
```

自动测试位于：

```text
tests/
```

示例位于：

```text
examples/
```

理论—源码对应实验位于：

```text
教程-lessons/
```

实时实现清单见：[`实现状态-STATUS.md`](实现状态-STATUS.md)。

---

## 最终数据流

```text
User Goal
   ↓
Context / Memory
   ↓
Model Runtime
   ├─ tokenizer
   ├─ embedding
   ├─ RMSNorm / RoPE / GQA / SwiGLU
   ├─ Transformer
   ├─ KV Cache
   ├─ prefill / decode
   └─ sampling / structured generation
   ↓
Action
   ├─ text
   └─ tool call
        ↓
Tool Runtime
   ├─ repo map
   ├─ filesystem
   ├─ exact edit / patch
   ├─ shell
   ├─ Git / worktree
   ├─ browser
   └─ computer（后续）
        ↓
Observation
        ↓
Agent State / Persistent Memory
        ↓
Verifier
   ├─ test
   ├─ lint
   ├─ build
   ├─ diff review
   └─ task-specific checks
        ↓
Continue / Recover / Finish
```

---

## 从 0 的实现阶段

| 阶段 | 目标 | 当前状态 | 对应理论 |
|---|---|---:|---|
| 00 | config、tensor contract、tests | ✅ 已有 | tensor / shape / dtype |
| 01 | byte tokenizer + BPE | ✅ 已有 | Tokenization |
| 02 | decoder-only Transformer | ✅ 已有第一版 | Attention / RoPE / GQA / SwiGLU |
| 03 | 加载公开 safetensors | ✅ primitive | checkpoint / weight mapping |
| 04 | KV Cache、prefill、decode、sampling | ✅ 已有 | LLM inference |
| 05 | schema / structured tool call | ✅ 第一版 | constrained generation |
| 06 | persistent memory / searchable history | ✅ primitive | context / memory |
| 07 | shell / files / Git / HTTP browser | ✅ 第一版 | tool use / security |
| 08 | agent loop / failure recovery | ✅ 第一版 | ReAct / agent systems |
| 09 | repo map / edit / test / Git | ✅ 第一版 | coding agents |
| 10 | browser / computer / artifacts | 🟡 HTTP primitive；computer 待做 | general agents |
| 11 | worktree / task graph / parallel workers | 🟡 worktree primitive | multi-agent systems |
| 12 | parity / repo tasks / safety evals | 🟡 unit tests 已有 | evaluation |

---

## 已验证的关键性质

### 1. cached decode 与 full forward 数值对齐

单元测试会比较：

$$
\mathrm{logits}_{cached}(x_T)
\approx
\mathrm{logits}_{full}(x_{1:T}).
$$

这是 KV Cache 正确性的底线，而不是“代码能跑”就算完成。

### 2. tokenizer round-trip

UTF-8 byte tokenizer 与教学 BPE 都必须满足 encode/decode 可逆。

### 3. 编辑必须失败得安全

`ExactEditTool` 要求旧文本在目标文件中恰好出现一次；0 次或多次都拒绝修改，强迫 agent 读取更多上下文，而不是模糊猜位置。

### 4. 工具错误进入 observation

文件不存在、schema 错误、命令失败等不会直接摧毁 agent loop，而会被反馈给 policy，支持下一步恢复。

### 5. CI 真跑测试

GitHub Actions 的 `Capstone runtime tests` 会安装该工程并执行 `pytest`。第一版已经通过 **12 项测试**。

---

## 强制原则

### 1. 先自己写，再用工业框架作对照

Attention 先显式写：

```python
scores = q @ k.transpose(-2, -1)
scores = scores / math.sqrt(head_dim)
scores = scores.masked_fill(~mask, float("-inf"))
probs = torch.softmax(scores, dim=-1)
out = probs @ v
```

然后才研究 SDPA、FlashAttention、vLLM 如何优化同一计算。

不能反过来只写：

```python
AutoModelForCausalLM.from_pretrained(...)
```

然后声称“实现了大模型”。

### 2. 每个模块必须知道 shape

例如 GQA：

```text
q: [B, Hq, Tq, Dh]
k: [B, Hkv, Tk, Dh]
v: [B, Hkv, Tk, Dh]
```

### 3. 每个优化先有 reference implementation

例如：

```text
Reference: full recomputation
Optimized: KV-cached incremental decode
```

必须做 parity test。

### 4. Agent 必须有真实 environment feedback

目标是：

```text
inspect
→ edit
→ pytest
→ read traceback
→ fix
→ rerun
→ review diff
→ finish
```

不是 Thought / Action / Observation 的纸面流程图。

### 5. 安全边界必须最终落到系统层

当前 rooted filesystem 与 timeout 只是第一层。最终还必须有：

- container / VM isolation；
- network policy；
- destructive-operation guard；
- resource quota；
- secrets boundary；
- audit log；
- permission / approval model。

---

## 从这里开始学习

建议按以下顺序边读边跑：

1. [`教程-lessons/00-从字符串到Logits-string-to-logits.md`](教程-lessons/00-从字符串到Logits-string-to-logits.md)
2. [`教程-lessons/01-KVCache与增量解码-kv-cache.md`](教程-lessons/01-KVCache与增量解码-kv-cache.md)
3. [`教程-lessons/02-工具协议与Agent循环-tool-agent-loop.md`](教程-lessons/02-工具协议与Agent循环-tool-agent-loop.md)
4. [`教程-lessons/03-Codex级仓库闭环-coding-agent.md`](教程-lessons/03-Codex级仓库闭环-coding-agent.md)
5. [`../../教材-book/18A-模型运行时源码导读-runtime-walkthrough.md`](../../教材-book/18A-模型运行时源码导读-runtime-walkthrough.md)
6. [`../../教材-book/18B-智能体运行时源码导读-agent-walkthrough.md`](../../教材-book/18B-智能体运行时源码导读-agent-walkthrough.md)

本地运行：

```bash
cd '代码-code/从零构建Astra与Codex级系统'
pip install -e '.[dev]'
pytest
python examples/00_tiny_generate.py
python examples/01_scripted_coding_agent.py
```

---

## 最终毕业验收

### Model Runtime

- 加载公开真实 checkpoint；
- 与 reference implementation 做 logits parity；
- KV Cache decode 与 full recompute 对齐；
- streaming / batching / paged cache；
- structured generation；
- 性能与显存 profiling。

### Codex-class

输入一个真实 Git repository + issue，系统独立完成：

```text
understand issue
→ map repo
→ inspect code
→ edit
→ run tests
→ diagnose
→ fix
→ rerun
→ inspect diff
→ verify no regression
→ finish
```

### Astra-class

输入跨工具长程任务：

```text
research
→ browser/computer
→ filesystem
→ Python/shell
→ generate artifact
→ inspect artifact
→ recover from failures
→ verify
→ final result
```

### Multi-Agent

至少完成：

```text
coordinator
├─ worker A / worktree A
├─ worker B / worktree B
└─ reviewer
     ↓
verified merge
```

总理论入口：[`../../教材-book/18-从零构建Astra与Codex级系统-capstone.md`](../../教材-book/18-从零构建Astra与Codex级系统-capstone.md)。

从这一版开始，教材中的理论不再以“解释完”为终点，而以**能否在这个工程里亲手实现、测试和验证**为终点。