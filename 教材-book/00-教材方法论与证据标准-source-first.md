# 教材方法论与证据标准：Source-First、Mechanism-First、Reproducibility-First

> **本文件是全书的最高写作规范。** 后续任何章节、公式、图、模型对比、历史结论、性能数字和工程结论，都应遵守这里的证据标准。

这本书的目标不是“收集尽可能多的模型名字”，而是建立一套能长期经受技术迭代的知识体系：**每一个重要结论都尽量回到原始论文、官方技术报告、官方代码、模型卡、系统卡、数据集论文或可复现实验本身。**

如果一个结论无法从一手材料中得到，我们就必须明确标注它来自哪里，以及它属于“事实”“作者解释”“经验总结”还是“推断”。

---

# 1　教材的四条主线

全书同时沿四条线推进，而不是只按年份罗列模型：

1. **历史线**：当时遇到了什么问题，为什么旧方法不够？
2. **机制线**：数学目标、数据流、张量形状、架构与训练算法究竟改了什么？
3. **系统线**：算力、显存、通信、kernel、并行、serving 如何决定“理论上能做”和“实际上能做”的边界？
4. **证据线**：原论文声称什么？原代码真正实现什么？后来工作复现或修正了什么？

因此，每个核心技术都应回答：

> **问题从哪里来 → 原始工作如何定义问题 → 原公式/原架构是什么 → 原实验如何验证 → 后续工作哪里继承、哪里推翻 → 今天我们应该怎样理解。**

---

# 2　证据等级

## S0：原始研究论文 / 原始技术报告

最高优先级。包括：

- 原论文；
- 官方技术报告；
- 原始 system card / model card；
- 数据集原始论文；
- benchmark 原始论文。

例如讲 Transformer 时，主证据必须回到 Vaswani et al. (2017)，而不能只引用博客对 Transformer 的解释。

## S1：官方实现与原始代码

包括：

- 作者官方 GitHub；
- 官方 release；
- checkpoint config；
- tokenizer config；
- training recipe；
- 推理实现；
- CUDA / Triton kernel。

**论文写了什么和代码真正做了什么有时并不完全相同。** 本书在可能时会同时检查 S0 与 S1。

## S2：官方模型卡、系统卡、发布说明与产品文档

这类来源对闭源模型尤其重要。例如 OpenAI、Anthropic、Google DeepMind、xAI 等很多前沿模型并不公开完整训练细节，因此公开可验证边界可能只到 system card / model card。

本书会明确写：

> “官方公开材料只能支持到这里，剩余训练细节未公开。”

而不是根据 benchmark 或营销文案反推不存在的架构细节。

## S3：独立复现与严格消融

当原论文结论存在争议时，优先寻找：

- 独立复现；
- 多 seed 实验；
- 控制变量消融；
- 重新实现；
- cross-dataset 验证。

这一级证据用于回答“原作者报告的结果是否稳健”。

## S4：高质量综述、课程与教材

用于：

- 建立背景；
- 对比不同术语；
- 帮助组织知识。

但它们不能代替原始来源支撑核心技术结论。

## S5：新闻、博客、论坛与社区经验

可用于记录产业史、工程经验、社区观点，但必须明确标注为二手或经验性材料，不能包装成实验事实。

---

# 3　每一章的固定结构

对于重要章节，原则上采用以下模板。

## 3.1 原始问题

先回答：**研究者当时到底想解决什么？**

不能从“某论文提出了某方法”开始，而应从前一代方法的瓶颈开始。

## 3.2 原始资料包

列出本章最关键的一手材料：

- Paper / technical report；
- Official code；
- Model card / system card；
- Dataset / benchmark；
- 必要的后续复现。

## 3.3 原模型数据流

优先画数据流和 shape，再推公式。

例如：

```text
input ids [B,T]
      ↓
embedding [B,T,d]
      ↓
Q,K,V [B,h,T,dh]
      ↓
attention scores [B,h,T,T]
      ↓
context [B,T,d]
```

## 3.4 原始数学定义

公式必须区分：

- 原论文明确给出的；
- 本书为了教学重新推导的；
- 后续主流实现修改后的。

若为了可读性改写符号，应说明符号对应关系。

## 3.5 手算与最小例子

每个核心机制尽可能给出：

- 小矩阵手算；
- toy example；
- 极简 PyTorch 实现。

## 3.6 原始实验

不只报告 benchmark 结果，还要说明：

- 数据集；
- baseline；
- 参数量；
- 训练预算；
- evaluation protocol；
- 是否使用工具；
- 是否使用 test-time compute；
- 是否存在不同 prompt / scaffold。

## 3.7 原始结论与后来修正

单独区分：

> **原作者当时认为发生了什么**

与

> **今天我们有更多证据后如何理解。**

这一步是教材区别于论文摘要集合的关键。

## 3.8 从原代码读机制

若开源，应指出关键代码路径，例如：

```text
model.py
  └─ TransformerBlock
      ├─ Attention
      └─ MLP
```

并追踪 tensor shape、缓存、并行和 kernel 路径。

## 3.9 最小复现实验

要求能回答一个可证伪问题，例如：

- 去掉 causal mask 会怎样？
- 把 MHA 换 GQA，KV cache 如何变化？
- 相同 compute 下扩大参数还是扩大 token 更有效？
- RLVR 的收益是否来自更多 rollout 而非 policy learning？

## 3.10 现代位置

最后回答：

- 今天还在不在用？
- 被谁替代？
- 哪些思想留下来了？
- 哪些历史结论已经过时？

---

# 4　数字与 benchmark 的引用规则

任何精确数字都要能追溯。

例如：

> “模型有 671B 总参数、每 token 激活 37B”

必须直接链接到对应官方技术报告或模型卡。

禁止把以下数字直接横向比较：

- prompt 不同；
- sampling 不同；
- pass@1 与 pass@k 不同；
- 有工具与无工具不同；
- test-time compute 不同；
- benchmark 版本不同；
- contamination policy 不同。

表格必须尽量增加“测试设置”一列。

---

# 5　闭源模型的写作原则

闭源模型是最容易被教材写错的一部分。

对于 GPT、Claude、Gemini、Grok 等没有公开完整架构和训练 recipe 的模型，本书只写公开资料能够支持的内容。

例如：

- 可以写 system card 明确披露的训练范式；
- 可以写官方 API 明确披露的 context window；
- 可以写官方 benchmark，但必须标“官方报告”；
- **不能因为某模型性能类似某架构，就声称它内部一定采用该架构。**

当信息不可知时，最专业的写法就是：

> **未公开。**

---

# 6　原始论文并不等于最终真理

“Source-First”不是“作者说什么就照抄什么”。

原论文提供的是历史上的第一手证据，但科学结论还要经历：

```text
原论文
  ↓
复现
  ↓
消融
  ↓
跨模型验证
  ↓
反例
  ↓
新的解释
```

因此本书会刻意加入：

- **Original Claim：原始主张**
- **Evidence：证据**
- **Later Evidence：后续证据**
- **Current View：当前较稳健理解**

四层结构。

---

# 7　图的标准

所有重要机制图尽量由本书重新绘制，而不是直接复制网络图片。

图至少满足以下之一：

1. 数据怎么流；
2. tensor shape 怎么变；
3. 训练和推理哪里不同；
4. GPU / memory / communication 怎么流；
5. agent 与 environment 怎么闭环。

图注必须注明：

> 据 X et al. (year) / 官方代码重绘整理。

---

# 8　代码与复现标准

教材代码分三层。

### Level 1：教学实现

目标是可读，不追求速度。

### Level 2：忠实复现

尽可能对齐原论文或官方代码：

- 参数；
- 数据处理；
- optimizer；
- schedule；
- evaluation。

### Level 3：工业实现

学习现代系统为什么这样写：

- fused kernels；
- FlashAttention；
- tensor / pipeline / expert parallel；
- continuous batching；
- paged KV cache；
- quantized inference。

三者不能混在一起。

---

# 9　全书必须覆盖的知识维度

一本真正完整的 LLM 教材不能只讲模型结构，还必须覆盖：

- 数学基础；
- tokenization；
- data curation；
- pretraining objective；
- architecture；
- scaling；
- optimization；
- distributed training；
- hardware；
- post-training；
- preference learning；
- reasoning RL；
- multimodality；
- retrieval；
- tool use；
- agents；
- inference systems；
- evaluation；
- safety；
- interpretability；
- data contamination；
- memorization；
- synthetic data；
- continual / online learning；
- long context；
- memory；
- model editing；
- open-weight ecosystem；
- Chinese LLM ecosystem；
- frontier closed models；
- economics / compute / energy；
- post-Transformer research。

缺其中任何一大片，都只能算“LLM 入门教程”，而不是全面教材。

---

# 10　章节成熟度标记

后续章节可使用四级状态：

- **L0 目录级**：只有问题与资料索引；
- **L1 教材级**：机制、公式、历史与引用完整；
- **L2 复现级**：含代码、实验和结果；
- **L3 研究级**：含争议、反例、最新证据与开放问题。

本书最终目标不是所有章节“有文字”，而是核心章节尽可能达到 **L2–L3**。

---

# 11　引用格式

正文第一次出现关键技术时，就近链接一手资料：

> Vaswani et al. (2017), *Attention Is All You Need* — 原论文链接。

章末再提供“原始资料包”，按：

1. 论文；
2. 官方代码；
3. 官方模型卡/系统卡；
4. 独立复现；
5. 延伸阅读；

分类。

不要只在全书末尾放 bibliography，让读者无法判断某句话的来源。

---

# 12　本书最终应该训练出的能力

读者面对任何新模型时，应能完成下面的反向工程：

```text
新模型公告
   ↓
找原始技术报告 / model card / code
   ↓
识别 objective / architecture / data / optimizer
   ↓
画数据流与 shape
   ↓
判断训练 compute 与推理 compute
   ↓
检查 benchmark protocol
   ↓
复现关键模块
   ↓
设计反例与消融
   ↓
判断“真正创新在哪里”
```

这才是这本书的终点：**不是记住大模型历史，而是获得独立读懂、复现、质疑和创造下一代模型的能力。**
