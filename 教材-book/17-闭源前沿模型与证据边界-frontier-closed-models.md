# 第十七篇（Part XVII）　闭源前沿模型与证据边界：怎样在信息不完整时做严谨技术分析

> **本章主线**：2024–2026 年越来越多最强模型没有公开完整 architecture、parameter count、data mixture 或 training recipe。研究这些模型最危险的错误，是把 benchmark、产品体验和社区猜测包装成内部机制事实。本章建立一套“只写公开证据能支持的内容”的分析方法，并用 OpenAI、Anthropic、Google DeepMind、xAI 等公开 system card / model card 做案例。

---

# 1　为什么闭源模型需要单独的方法论

开源论文通常允许我们检查：

```text
paper
  ↓
config
  ↓
model code
  ↓
weights
  ↓
training / inference implementation
```

闭源前沿模型往往只能得到：

```text
system card / model card
release notes
API docs
selected evaluations
product behavior
```

因此研究问题必须改变。

对闭源模型，我们应问：

> **公开资料能证明什么？不能证明什么？**

而不是：

> “它内部大概就是某种 Transformer + 某种 RL。”

---

# 2　证据边界表

| 信息 | 开放模型 | 闭源模型常见公开程度 |
|---|---:|---:|
| 参数量 | 常见 | 常未公开 |
| 层数 / hidden size | 常见 | 常未公开 |
| attention 结构 | 常见 | 常未公开 |
| tokenizer | 常见 | 部分公开 |
| context window | 常见 | 通常公开 |
| pretraining data mixture | 部分 | 通常只给类别描述 |
| post-training 方法 | 部分 | 常给高层描述 |
| safety training | 部分 | system card 常披露 |
| benchmark | 常见 | 常见 |
| weights | 有 | 无 |
| official code | 常有 | 无 |

因此，对闭源模型不能使用与开放模型同样精细的逆向叙事。

---

# 3　OpenAI o1：公开证据支持到哪里

OpenAI 在 2024 年发布 o1 系列，并公开 **Learning to reason with LLMs** 与 **o1 System Card**。

原始资料：

- https://openai.com/index/learning-to-reason-with-llms/
- https://openai.com/index/openai-o1-system-card/

官方明确支持的结论包括：

1. o1 是 reasoning model；
2. 使用 reinforcement learning 训练复杂推理；
3. 性能会随更多 train-time RL compute 提升；
4. 性能也会随更多 test-time thinking compute 提升；
5. system card 讨论了 chain-of-thought、deliberative alignment 与安全评测。

但这些资料**不能**证明：

- o1 的参数量；
- 层数；
- 是否采用某个社区猜测的 hidden architecture；
- RL 的完整 reward 设计；
- 具体 rollout infrastructure。

专业教材应把这一条边界写清楚。

---

# 4　GPT‑5：模型变成“系统”

OpenAI 2025 年公开 **GPT‑5 System Card**：

https://openai.com/index/gpt-5-system-card/

一个重要公开信息是：GPT‑5 当时不是单一静态模型概念，而是一个统一系统，包括：

- fast / main model；
- deeper reasoning model；
- router；
- mini variants。

官方 system card 描述 router 根据：

- conversation type；
- complexity；
- tool needs；
- explicit user intent；

选择路径。

这带来一个教材层面的关键变化：

> **“一个产品的智能”不再等于“一个 checkpoint 的智能”。**

系统能力可能来自：

```text
model family
+ router
+ tool stack
+ prompting
+ safety layer
+ memory / context management
```

所以 benchmark 与实际产品表现之间的映射变得更复杂。

---

# 5　GPT‑5.4 / 5.5 / 5.6：2026 年的系统卡应该怎样读

OpenAI 2026 年继续发布多个系统卡：

- GPT‑5.4 Thinking  
  https://openai.com/index/gpt-5-4-thinking-system-card/
- GPT‑5.5  
  https://openai.com/index/gpt-5-5-system-card/
- GPT‑5.6  
  https://deploymentsafety.openai.com/gpt-5-6

这些资料适合研究：

- capability evaluation；
- safety evaluation；
- tool / agentic behavior；
- chain-of-thought monitoring；
- deployment safeguards；
- model family positioning。

但仍不应据此虚构未披露 architecture。

例如 GPT‑5.6 system card 可以支持：

> 这是包含 Sol / Terra / Luna 的模型家族，并公开了 preparedness / safety information。

却不能仅从名字、速度、价格反推出内部 parameter count。

---

# 6　System Card 与 Technical Report 的区别

这两种文档目的不同。

### Technical Report

通常更关注：

- architecture；
- training；
- data；
- ablation；
- benchmark。

### System Card

通常更关注：

- capabilities；
- risks；
- safety evaluations；
- mitigations；
- deployment context。

因此：

> system card 是一手资料，但不是 architecture paper。

不能因为“一手”就让它承担它没有提供的信息。

---

# 7　Anthropic：从 Constitutional AI 到 System Cards

Anthropic 在 alignment 方向有公开论文：

- Bai et al., **Constitutional AI**  
  https://arxiv.org/abs/2212.08073

同时维护官方 system card 总入口：

https://www.anthropic.com/system-cards

以及 Transparency Hub：

https://www.anthropic.com/transparency

这些资料让我们能够研究：

- Claude family 的部署安全；
- Constitutional AI 的历史思想；
- computer use / agentic coding 风险；
- capability / safety evaluation。

但对于具体 Claude 型号的内部层数、参数量等，如果没有官方披露，就应明确写：

> **未公开。**

---

# 8　为什么“Claude 是某种 MoE”不能靠社区猜测写进教材

常见推断路径：

```text
速度 / 价格 / benchmark
        ↓
猜参数量
        ↓
猜 MoE / dense
        ↓
猜 experts
```

这种推断可以作为研究假设，但不能作为教材事实。

若无：

- 官方论文；
- model card；
- verified source；

就必须标：

> **Unknown / 未公开。**

这比一个“看起来很专业”的错误架构表更有价值。

---

# 9　Google DeepMind：Model Card 体系

Google DeepMind 维护官方 Gemini model card 页面：

https://deepmind.google/models/model-cards/

截至 2026 年，页面覆盖 Gemini 2.5、Gemini 3.x 等多个型号。

model card 通常提供：

- model description；
- inputs / outputs；
- data overview；
- implementation / sustainability；
- evaluations；
- intended use / limitations；
- safety。

这类资料适合回答：

> 某个产品型号公开支持哪些 modality、context、reasoning / agentic feature？

但仍不自动等价于完整 architecture disclosure。

---

# 10　Gemini 2.5：“thinking model”这个说法来自哪里

Google 2025 年官方发布 Gemini 2.5 时，明确把它描述为 thinking model：

https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-model-thinking-updates-march-2025/

官方说明把提升归因于：

- enhanced base model；
- improved post-training；
- reasoning / thinking capability。

因此教材可以写：

> Google 官方将 Gemini 2.5 定位为 thinking model，并公开强调 base model 与 post-training 的共同提升。

但若官方没有给出 RL objective 的完整公式，就不能自行补一个“Gemini 使用某某 RL 算法”的结论。

---

# 11　Gemini 3.x：型号快速迭代为什么需要日期

到了 2026，Gemini 3.x 已出现多个 model card。

例如 model card 总入口会随时间增加：

- Pro；
- Flash；
- Flash-Lite；
- Audio；
- Image；
- Computer Use；
- Deep Think 等。

这意味着教材不能写：

> “Gemini 3 的 context window 是 X。”

更专业的写法是：

> “Gemini 3.5 Flash-Lite（2026-07 model card）公开支持……”。

**型号 + 日期 + 来源** 必须一起出现。

---

# 12　xAI Grok 4：RL 与 Native Tool Use 的公开证据

xAI 2025 年官方 Grok 4 发布：

https://x.ai/news/grok-4

并公开 Grok 4 Model Card：

https://data.x.ai/2025-08-20-grok-4-model-card.pdf

官方公开材料支持：

- reasoning model 定位；
- scaling reinforcement learning；
- native tool use；
- web / code 等工具能力；
- safety evaluations。

xAI 官方还披露其训练使用大规模 GPU cluster 的信息。

但仍应区分：

> 官方发布中的自报 benchmark

与

> 第三方独立评测。

---

# 13　“官方说 SOTA”为什么不等于科学比较结束

任何厂商 benchmark 都必须检查：

```text
model version
prompt
sampling
reasoning effort
tools
parallel sampling
pass@k
scaffold
dataset version
```

例如：

```text
No tools
```

与：

```text
Python + web + parallel thinking
```

不能放在一列里直接比较。

厂商 benchmark 是一手资料，但通常不是独立证据。

---

# 14　Meta Llama 4：开放模型提供更细机制入口

Meta 的 Llama 4 官方资源：

https://ai.meta.com/llama/get-started/

官方公开页面与 model card 提供了例如：

- Llama 4 Scout；
- Llama 4 Maverick；
- multimodal；
- MoE；
- active / total parameters；
- context information。

由于开放程度更高，我们可以比闭源 API 模型进一步分析：

- config；
- weight shapes；
- tokenizer；
- inference implementation。

所以教材应按“证据可得程度”决定分析粒度，而不是对所有模型强行用同样模板。

---

# 15　“Open Source”与“Open Weight”必须分开

即使权重可下载，也可能缺少：

- training data；
- training code；
- full recipe；
- permissive license。

因此模型开放度至少拆成：

```text
weights
architecture
code
training recipe
data
license
```

把所有可下载权重模型都叫“完全开源”是不严谨的。

---

# 16　如何分析一个新发布的闭源模型

标准流程：

```text
1. 找官方 technical report / system card / model card
2. 找 API docs
3. 找官方 release notes
4. 抽取明确披露的事实
5. 把未知项写成 Unknown
6. 分离官方 benchmark 与独立 benchmark
7. 检查工具 / reasoning effort / pass@k
8. 记录版本与发布日期
9. 对后续更新重新审计
```

---

# 17　证据表模板

以后任何闭源模型都可以建立：

| 项目 | 已公开 | 来源 | 是否可验证 |
|---|---|---|---|
| 参数量 | Unknown | — | 否 |
| Context | X | API / model card | 是 |
| Multimodal | Yes | model card | 是 |
| RL reasoning | 官方高层描述 | system card | 部分 |
| Architecture | Unknown | — | 否 |
| Tool use | Yes | official docs | 是 |
| Benchmark | 官方结果 | system card | 需独立验证 |

这个表比社区流传的“猜架构表”更科学。

---

# 18　产品系统的表现不能简单归因给 base model

现代 ChatGPT / Claude / Gemini / Grok 等产品可能包括：

```text
router
system prompt
memory
retrieval
web search
code execution
computer use
safety filters
context compression
parallel sampling
```

所以用户体验上的提升可能来自：

- checkpoint；
- post-training；
- tools；
- orchestration；
- product layer。

教材必须区分这几层。

---

# 19　闭源模型研究的最高纪律

只遵守一句话：

> **不知道就写不知道。**

在前沿研究中，“Unknown”不是缺点，而是信息边界。

真正危险的是：

```text
community guess
   ↓
repeated by blogs
   ↓
becomes common belief
   ↓
textbook writes it as fact
```

本书必须阻断这条链。

---

# 20　原始资料包

## OpenAI

- o1 System Card  
  https://openai.com/index/openai-o1-system-card/
- Learning to reason with LLMs  
  https://openai.com/index/learning-to-reason-with-llms/
- GPT-5 System Card  
  https://openai.com/index/gpt-5-system-card/
- GPT-5.4 Thinking System Card  
  https://openai.com/index/gpt-5-4-thinking-system-card/
- GPT-5.5 System Card  
  https://openai.com/index/gpt-5-5-system-card/
- GPT-5.6 System Card  
  https://deploymentsafety.openai.com/gpt-5-6

## Anthropic

- Model System Cards  
  https://www.anthropic.com/system-cards
- Transparency Hub  
  https://www.anthropic.com/transparency
- Constitutional AI  
  https://arxiv.org/abs/2212.08073

## Google DeepMind

- Gemini Model Cards  
  https://deepmind.google/models/model-cards/
- Gemini 2.5 introduction  
  https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-model-thinking-updates-march-2025/

## xAI

- Grok 4  
  https://x.ai/news/grok-4
- Grok 4 Model Card  
  https://data.x.ai/2025-08-20-grok-4-model-card.pdf
- Safety / Model Cards  
  https://x.ai/safety

## Meta

- Llama model resources / model cards  
  https://ai.meta.com/llama/get-started/

---

# 21　最终目标

学完本章，你看到一个新模型发布时，不应该第一反应是：

> “它多少参数？是不是 MoE？”

而应该先问：

> **这些信息公开了吗？证据在哪里？**

只有建立这种信息纪律，才可能真正研究快速变化的前沿模型，而不是追随传闻。
