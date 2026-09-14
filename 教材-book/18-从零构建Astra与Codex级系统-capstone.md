# 第十八篇（Part XVIII）　从零构建 Astra-class 通用智能系统与 Codex-class 编程智能体

> **终极目标**：学完本书后，不以“知道大模型名词”为终点，而是从一个空目录开始，自己写出一套完整、可运行、可测试的现代大模型推理与智能体系统。训练权重本身不在本项目范围内，但从 tokenizer、Transformer forward、KV Cache、sampling，到工具调用、代码执行、仓库编辑、测试验证、长程记忆、并行智能体与计算机使用接口，核心代码都要亲手搭出来。

---

# 0　边界：我们“复现”的是什么，不是什么

OpenAI 的 GPT-6 Astra 与 Codex 属于前沿闭源系统。公开资料可以告诉我们其部分能力边界、产品行为、接口与评测，但**不能支持我们声称掌握其全部内部 architecture、训练数据、训练 recipe、隐藏推理系统或生产基础设施**。

因此本篇使用两个术语：

- **Astra-class**：功能层面对齐现代通用 reasoning / tool-use / computer-use 模型系统；
- **Codex-class**：功能层面对齐现代 agentic coding system，包括仓库操作、shell、测试、patch、Git、并行 agent、长程任务与验证闭环。

目标是**从公开原理出发实现同一类系统**，而不是伪造 OpenAI 私有源码。

OpenAI 公开资料入口：

- GPT-6 Astra: https://openai.com/index/gpt-6-astra/
- GPT-6 Astra API model page: https://developers.openai.com/api/docs/models/gpt-6-astra
- Codex: https://openai.com/codex/
- Codex app: https://openai.com/index/introducing-the-codex-app/

本篇所有关于闭源系统的表述都遵守第十七篇的“证据边界”原则。

---

# 1　最后到底要搭出什么

最终系统分成两层：

```text
┌─────────────────────────────────────────────────────────────┐
│                    应用层 / Agent Layer                     │
│                                                             │
│  Codex-class Coding Agent     Astra-class General Agent     │
│  ├─ repository context        ├─ tools                      │
│  ├─ shell / tests             ├─ browser / computer         │
│  ├─ patch / Git               ├─ retrieval / memory         │
│  ├─ worktree / parallelism    ├─ long-horizon task loop     │
│  ├─ verifier                  └─ artifact generation        │
│  └─ long-term task state                                     │
└─────────────────────────▲───────────────────────────────────┘
                          │ structured actions / observations
┌─────────────────────────┴───────────────────────────────────┐
│                 模型与推理运行时 / Model Runtime            │
│                                                             │
│ tokenizer → embedding → Transformer → logits                │
│             RoPE / RMSNorm / GQA / SwiGLU                   │
│             KV Cache / prefill / decode                     │
│             sampling / structured output                    │
│             optional MoE / multimodal adapter               │
└─────────────────────────────────────────────────────────────┘
```

不做权重训练，不代表跳过模型本身。我们仍要自己实现：

1. tokenizer 数据接口；
2. embedding；
3. RMSNorm；
4. RoPE；
5. causal attention；
6. MHA / GQA；
7. SwiGLU / MLP；
8. Transformer block；
9. LM head；
10. KV Cache；
11. prefill / decode；
12. sampling；
13. structured output；
14. tool protocol；
15. long-context state management。

区别只是：**参数可以使用随机小模型进行单元测试，或者加载公开权重验证实现；不要求自己花算力训练一个 frontier model。**

---

# 2　总工程：从空目录开始

最终工程目录规划：

```text
代码-code/从零构建Astra与Codex级系统/
│
├── 00-基础设施/
│   ├── tensor.py
│   ├── config.py
│   └── tests/
│
├── 01-tokenizer/
│   ├── bpe.py
│   ├── byte_bpe.py
│   └── sentencepiece_adapter.py
│
├── 02-model-core/
│   ├── embedding.py
│   ├── rmsnorm.py
│   ├── rope.py
│   ├── attention.py
│   ├── gqa.py
│   ├── swiglu.py
│   ├── block.py
│   └── model.py
│
├── 03-weight-loader/
│   ├── safetensors_reader.py
│   ├── mapping.py
│   └── reference_parity.py
│
├── 04-inference-engine/
│   ├── kv_cache.py
│   ├── prefill.py
│   ├── decode.py
│   ├── sampling.py
│   ├── batching.py
│   └── scheduler.py
│
├── 05-structured-generation/
│   ├── schema.py
│   ├── tool_call.py
│   └── constrained_decode.py
│
├── 06-context-memory/
│   ├── message_store.py
│   ├── compaction.py
│   ├── notes.py
│   ├── search.py
│   └── retrieval.py
│
├── 07-tool-runtime/
│   ├── registry.py
│   ├── shell.py
│   ├── filesystem.py
│   ├── browser.py
│   ├── python_exec.py
│   └── permissions.py
│
├── 08-agent-core/
│   ├── state.py
│   ├── loop.py
│   ├── planner.py
│   ├── executor.py
│   ├── verifier.py
│   └── recovery.py
│
├── 09-codex-class/
│   ├── repo_index.py
│   ├── code_search.py
│   ├── patch.py
│   ├── test_runner.py
│   ├── git.py
│   ├── worktree.py
│   ├── reviewer.py
│   └── coding_agent.py
│
├── 10-astra-class/
│   ├── general_agent.py
│   ├── computer_use.py
│   ├── browser_use.py
│   ├── artifact_pipeline.py
│   └── multimodal_adapter.py
│
├── 11-multi-agent/
│   ├── task_graph.py
│   ├── worker.py
│   ├── coordinator.py
│   └── merge.py
│
├── 12-evals/
│   ├── model_parity/
│   ├── tool_use/
│   ├── coding/
│   ├── long_horizon/
│   └── safety/
│
└── README.md
```

这个目录不是“一次写完”，而是整本教材的实验主线。每学完一个理论模块，就在这里补上一块。

---

# 3　阶段 0：不用任何大模型框架，先建立最小计算契约

这一阶段不碰 Transformer。

需要掌握：

- tensor shape；
- dtype；
- device；
- contiguous / stride；
- batch / sequence / hidden 三个维度；
- deterministic unit test；
- config 与 state 的分离。

代码目标：

```python
@dataclass
class ModelConfig:
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    max_position_embeddings: int
```

整个后续工程中，每一个模块都必须明确输入与输出 shape。

验收标准：

> 任何函数只看类型与 shape 文档，就知道数据怎样流过系统。

---

# 4　阶段 1：自己写 tokenizer

理论对应第十三篇。

先实现最简单的 byte tokenizer，再实现 BPE 的训练算法与 encode/decode，即使最终运行大型开放权重时会读取现成 tokenizer 文件，也必须先理解 tokenizer 在做什么。

数据流：

```text
UTF-8 string
   ↓ bytes / pre-tokenization
symbol sequence
   ↓ merge rules
subword tokens
   ↓ vocabulary lookup
input_ids [T]
```

至少完成：

- byte-level round trip；
- BPE merge；
- special tokens；
- BOS / EOS；
- padding；
- batch encoding；
- tokenizer parity tests。

验收：

```text
decode(encode(text)) == text
```

并解释中文、英文、代码在 token fertility 上为什么不同。

---

# 5　阶段 2：从矩阵开始手写现代 decoder-only Transformer

这是整个工程的核心。

禁止一开始使用：

```python
AutoModelForCausalLM.from_pretrained(...)
```

核心 forward 必须自己写。

数据流：

```text
input_ids                [B,T]
   ↓ embedding
hidden                   [B,T,C]
   ↓ N × TransformerBlock
hidden                   [B,T,C]
   ↓ final RMSNorm
hidden                   [B,T,C]
   ↓ LM head
logits                   [B,T,V]
```

每个 block：

```text
x [B,T,C]
│
├─ RMSNorm
├─ Q/K/V projections
├─ RoPE
├─ causal attention / GQA
├─ output projection
└─ residual

x [B,T,C]
│
├─ RMSNorm
├─ SwiGLU
└─ residual
```

理论必须同步掌握：

- 为什么 attention 是 $QK^T$；
- 为什么缩放 $1/\sqrt{d_h}$；
- causal mask；
- 为什么 RoPE 作用于 Q/K；
- RMSNorm 和 LayerNorm 的差别；
- SwiGLU 的门控；
- GQA 如何减少 KV heads；
- residual stream 的意义。

验收：

1. tiny random model forward 正确；
2. full-sequence logits shape 正确；
3. 与一个公开模型的小尺寸 reference implementation 对齐到数值误差范围。

---

# 6　阶段 3：自己加载公开权重

不训练模型，但要理解一个 frontier-scale checkpoint 到底是什么。

需要实现：

```text
config.json
   ↓
architecture config

*.safetensors
   ↓
weight tensors
   ↓ name mapping
our modules
```

学习内容：

- safetensors 格式；
- parameter naming；
- weight tying；
- sharded checkpoints；
- dtype；
- tensor parallel checkpoint 与普通 checkpoint 的差别。

最终要求：

> 同一公开 checkpoint，在官方 reference implementation 与我们自己的 model core 上，对相同输入产生近似一致的 logits。

这一步是“我真的写出了模型”与“我写了一个看起来像 Transformer 的玩具”的分界线。

---

# 7　阶段 4：把模型改造成真正的 inference engine

训练 forward 和生产推理不是一回事。

必须实现：

## 7.1 Prefill

```text
prompt [B,T]
   ↓ parallel forward
K/V for every layer
   ↓
KV Cache
```

## 7.2 Decode

每次只输入一个新 token：

```text
x_t
 ↓
new Q/K/V
 ↓
append K/V to cache
 ↓
attention against cached history
 ↓
logits_t
```

必须亲手验证：

```text
full recomputation logits
≈
KV-cache incremental logits
```

然后继续实现：

- greedy；
- temperature；
- top-k；
- top-p；
- repetition handling；
- stop tokens；
- streaming；
- continuous batching 的简化版；
- request scheduler。

理论对应第十篇与第十六篇。

---

# 8　阶段 5：Structured Output 与 Tool Calling

从这一阶段开始，模型不再只是输出文字。

定义统一消息：

```python
class Message:
    role: str
    content: list[ContentPart]
```

统一动作：

```python
class ToolCall:
    name: str
    arguments: dict
```

统一观察：

```python
class ToolResult:
    call_id: str
    output: str
    is_error: bool
```

Agent 基本循环：

```text
messages
   ↓
model
   ↓
text OR tool_call
   ↓
execute tool
   ↓
tool result
   ↓
append observation
   └──────────────→ model
```

必须理解为什么 Agent 的能力不等于 base model 的能力。

---

# 9　阶段 6：上下文、压缩、笔记与可搜索历史

长程智能体真正困难的不是“context window 数字很大”，而是**怎样保存任务状态**。

需要实现四种互相独立的机制：

```text
Raw Event Log
    │
    ├── recent context
    ├── compaction summary
    ├── persistent notes
    └── searchable history index
```

重点比较：

- full history；
- sliding window；
- summary compaction；
- explicit notes；
- retrieval from old context。

必须做失败实验：

> 刻意让 summary 丢掉一个关键测试结果，看 agent 是否因为历史压缩而重复犯错。

这比只讲“百万 token context”更接近真实长程系统。

---

# 10　阶段 7：工具运行时与权限系统

实现：

```text
Tool Registry
├── filesystem.read
├── filesystem.write
├── shell.exec
├── python.exec
├── search
├── browser
└── git
```

每个工具必须包含：

- JSON schema；
- validation；
- timeout；
- stdout/stderr；
- error semantics；
- permissions；
- audit log。

不能写成：

```python
os.system(model_output)
```

而要建立真正的 capability boundary。

理论内容包括：

- least privilege；
- sandbox；
- prompt injection；
- tool-result trust boundary；
- filesystem scope；
- network scope；
- destructive action confirmation。

---

# 11　阶段 8：Codex-class Coding Agent

到这里开始搭真正的编程智能体。

它至少需要：

```text
User Task
   ↓
Repository Discovery
   ↓
Search / Read Relevant Files
   ↓
Plan
   ↓
Edit
   ↓
Run Tests / Lint / Typecheck
   ↓
Observe Failure
   ↓
Diagnose
   ↓
Edit Again
   ↓
Git Diff Review
   ↓
Final Verification
```

关键模块：

## 11.1 Repository Map

读取：

- 文件树；
- package manifests；
- README；
- tests；
- CI config；
- Git status。

建立 code context，而不是一开始把整个仓库塞进 prompt。

## 11.2 Code Search

至少实现：

- exact symbol search；
- text grep；
- file-type filtering；
- dependency-oriented navigation；
- later: embedding / semantic retrieval。

## 11.3 Patch Engine

支持：

- whole-file replacement；
- exact patch；
- unified diff；
- conflict detection；
- rollback。

## 11.4 Test Loop

Agent 必须把测试当作 observation：

```text
edit
 ↓
pytest
 ↓
traceback
 ↓
reason
 ↓
edit
```

而不是“生成代码以后结束”。

## 11.5 Git / Worktree

实现：

- status；
- diff；
- branch；
- commit；
- worktree；
- merge conflict detection。

## 11.6 Verifier / Reviewer

把“写代码”和“审代码”分开。

```text
Builder Agent
   ↓
Patch
   ↓
Tests
   ↓
Reviewer / Verifier
   ↓
accept / request_changes
```

最终再扩展到多个 agent 并行工作。

---

# 12　阶段 9：Astra-class 通用 Agent

Codex 是软件工程专用环境；Astra-class 系统要进一步把环境扩展到：

```text
Browser
Computer UI
Filesystem
Python
Shell
Search
Documents
Spreadsheets
Slides
External APIs
```

核心不是增加几个 tool name，而是解决：

- heterogeneous observations；
- screenshots / text / DOM / files；
- long-horizon state；
- task switching；
- uncertainty；
- asking clarification vs continuing work；
- irreversible actions；
- artifact verification。

统一环境接口可写成：

```python
class Environment(Protocol):
    def observe(self) -> Observation: ...
    def act(self, action: Action) -> Observation: ...
```

这样 browser、terminal、desktop、IDE 都只是不同 environment adapter。

---

# 13　阶段 10：Computer Use

这一阶段不把“视觉点鼠标”神秘化。

一个最小 computer-use loop：

```text
screenshot
   ↓
state representation
   ↓
policy
   ↓
click / type / scroll / keypress
   ↓
new screenshot
```

至少实现：

- screenshot observation；
- coordinates；
- click；
- type；
- scroll；
- keypress；
- action history；
- timeout；
- visual verification。

更高级版本再加入：

- accessibility tree；
- DOM；
- OCR；
- visual grounding；
- action proposals；
- confidence；
- recovery。

理论上要讨论 POMDP、state aliasing 与 active perception，而不是只写 Selenium 脚本。

---

# 14　阶段 11：多智能体与并行任务

现代 coding agent 的一个重要方向是把任务拆成多个独立工作单元。

需要构造：

```text
Task Graph
   │
   ├── Agent A → worktree A
   ├── Agent B → worktree B
   └── Agent C → worktree C

results
   ↓
review / merge
```

必须解决：

- task decomposition；
- dependency graph；
- shared context；
- independent worktree；
- duplicate work；
- merge conflicts；
- reviewer；
- cancellation；
- retry。

并做对照实验：

> 单 agent 串行 vs 多 agent 并行，到底什么时候真的更快？

---

# 15　阶段 12：Evaluator 才能让系统真正进化

没有 eval，就无法知道自己搭的是“智能体”还是一堆脚本。

至少建立：

## 模型层

- logit parity；
- KV Cache parity；
- generation determinism；
- throughput；
- memory。

## Tool 层

- schema correctness；
- timeout；
- malformed calls；
- permission violations。

## Coding Agent

- issue → patch success；
- unit tests；
- regression tests；
- code quality；
- number of iterations；
- tokens / tool calls；
- wall-clock time。

## Long Horizon

- state retention；
- repeated failure avoidance；
- recovery after interruption；
- context rollover。

## Safety

- scope adherence；
- destructive actions；
- prompt injection；
- secret handling；
- network boundary。

最终所有 architecture 修改都必须回答：

> 它究竟改善了哪个 measurable failure mode？

---

# 16　理论学习与代码实现必须一一对应

本项目采用严格映射：

| 理论 | 必须写出的代码 |
|---|---|
| Cross Entropy / Causal LM | logits 与 next-token 接口 |
| BPE / SentencePiece | tokenizer |
| Attention | `attention.py` |
| RoPE | `rope.py` |
| RMSNorm | `rmsnorm.py` |
| SwiGLU | `swiglu.py` |
| GQA | `gqa.py` |
| KV Cache | `kv_cache.py` |
| Sampling | `sampling.py` |
| Tool Use | `tool_call.py` + registry |
| ReAct / Agent Loop | `loop.py` |
| RAG | retrieval / context builder |
| Long Context | event log / notes / search |
| Code Agent | repo/search/patch/test/git |
| Multi-Agent | task graph / worktrees |
| Computer Use | environment / screenshot / action |
| Safety | permission / policy / audit |

如果一个理论学完后无法指向一段代码，就说明还没有真正掌握。

反过来，如果一段代码解释不出数学和系统机制，也不能算完成。

---

# 17　最终验收：什么叫“我从零搭出来了”

最终项目必须满足以下条件。

## A. Model Runtime

不能依赖 Hugging Face 的高层 `AutoModelForCausalLM` 完成核心推理。

要求：

- 自己实现 decoder-only model core；
- 自己实现 RoPE / RMSNorm / GQA / SwiGLU；
- 自己实现 KV Cache；
- 自己实现 sampling；
- 能加载至少一个公开 checkpoint；
- 与 reference logits 做 parity test。

## B. Inference Engine

- prefill；
- decode；
- streaming；
- batch；
- scheduler；
- cache accounting；
- profiling。

## C. Tool Runtime

- typed tool schema；
- shell / filesystem / Python / browser；
- timeout；
- permission；
- audit log。

## D. Codex-class Agent

给它一个真实代码仓库和 issue，它能：

1. 读仓库；
2. 找相关代码；
3. 修改；
4. 跑测试；
5. 读失败；
6. 再修改；
7. 检查 diff；
8. 直到通过或明确说明 blocker。

## E. Astra-class Agent

给它跨工具任务，它能：

1. 维护目标；
2. 使用浏览器 / shell / files / Python；
3. 在多步执行中保存状态；
4. 必要时询问关键澄清；
5. 对结果进行验证；
6. 在权限边界内停止。

## F. Multi-Agent

- 多 worktree；
- task graph；
- parallel workers；
- reviewer；
- merge / conflict handling。

## G. Explainability

最关键的一条：

> 你能够从输入的一段文本开始，逐层讲清它怎样变成 token、hidden state、Q/K/V、KV Cache、logits、tool call、environment action，最后变成一个真实的 Git diff 或计算机操作。

到这里，才真正完成了“从 0 学懂并搭出现代大模型系统”。

---

# 18　本篇与整本书的关系

本书后续不再把“教材正文”和“代码项目”分成两条互不相干的线。

新的学习闭环是：

```text
原始论文 / 官方实现
        ↓
理论与数学
        ↓
最小手算
        ↓
自己写模块
        ↓
与 reference 对齐
        ↓
组合成系统
        ↓
真实任务
        ↓
失败分析
        ↓
回到理论
```

第十八篇是整本书的**总验收项目**。

如果前面某一章学完，不能帮助我们把这个系统向前推进，那么那一章就还需要继续重写。