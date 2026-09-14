# 第十三篇（Part XIII）　词元化、语料与数据工程：模型看到的世界从哪里来

> **本章主线**：大模型不是直接阅读“自然语言”，而是读取 token 序列；它也不是从“互联网”抽象地学习，而是从经过抓取、过滤、去重、混合和采样后的具体训练分布中学习。本章把 tokenizer 与数据工程作为模型机制的一部分，而不是预处理细节。

---

# 1　为什么 tokenizer 不是无关紧要的前处理

设原始字符串为 $s$，tokenizer 是一个映射：

$$
\tau(s)=(x_1,x_2,\ldots,x_T),\qquad x_i\in\{1,\ldots,V\}.
$$

语言模型真正优化的是 token 序列上的条件概率：

$$
p_\theta(x_1,\ldots,x_T)=\prod_{t=1}^T p_\theta(x_t\mid x_{\lt t}).
$$

因此 tokenizer 会直接改变：

- 序列长度 $T$；
- 词表大小 $V$；
- embedding / LM head 参数量；
- 不同语言的压缩效率；
- 数字、代码、空格与 Unicode 的边界；
- context window 中实际能容纳的信息量。

这意味着：**相同的 128K token context，对不同 tokenizer 并不代表相同数量的字符、汉字、代码或语义信息。**

---

# 2　从 word-level 到 subword

早期 word-level vocabulary 面临一个基本问题：自然语言词形组合巨大，永远存在 OOV（out-of-vocabulary）。

一种极端方案是 character-level：

```text
robot → r o b o t
```

优点是词表小、没有 OOV；缺点是序列很长。

另一极端是 word-level：

```text
robot → [robot]
```

序列短，但词表巨大，长尾严重。

Subword 的目标是在两者之间找到折中。

---

# 3　BPE：从原始算法到神经机器翻译

Sennrich, Haddow & Birch (2015/2016) 将 Byte Pair Encoding（BPE）用于神经机器翻译中的 rare words。[原论文](https://arxiv.org/abs/1508.07909)

教学化地看，BPE 反复做：

```text
统计相邻符号 pair
      ↓
找到频率最高的 pair
      ↓
把它合并为一个新符号
      ↓
重复
```

假设初始 corpus：

```text
low low lower newest widest
```

开始时符号可能按字符切分：

```text
l o w
l o w
l o w e r
...
```

若 `l o` 经常相邻，就合并成 `lo`；再可能将 `lo w` 合成 `low`。

最终得到一个介于 character 与 word 之间的 vocabulary。

### 关键点

BPE 并不是“理解词根”。它只是一个基于频率的离散压缩过程。某些 merge 会与形态学结构吻合，只是统计分布的结果。

---

# 4　SentencePiece：把空格也当数据

Kudo & Richardson (2018) 提出的 SentencePiece 将 tokenization 看成从原始 Unicode 字符串直接学习子词模型，而不要求语言先经过 whitespace tokenization。[原论文](https://arxiv.org/abs/1808.06226)

这对中文、日文等没有天然空格分词边界的语言尤其重要。

SentencePiece 的一个核心工程思想是：

> **不要假设语言天然先被切成“词”。**

它支持 BPE 与 unigram language model 等模型。

---

# 5　GPT 系列与 byte-level BPE

GPT-2 采用 byte-level BPE 路线，使模型能够表示任意 UTF-8 文本，同时避免巨大的 Unicode 字符词表。

原始资料应同时看：

- GPT-2 技术报告：  
  https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- OpenAI tokenizer 实现脉络；
- `tiktoken` 官方实现：  
  https://github.com/openai/tiktoken

Byte-level 的核心好处是：任何文本最终都可以退化到 byte 表示，不会真正出现无法编码的字符。

但代价是 token 边界可能非常不“人类语言学”。

---

# 6　Tokenizer 的真实数据流

```text
UTF-8 string
    ↓ normalization / regex pre-tokenization
text chunks
    ↓ byte / unicode mapping
base symbols
    ↓ merge ranks / unigram model
subword pieces
    ↓ vocab lookup
Token IDs [T]
    ↓ embedding table
X [T,d]
```

很多教材直接从 embedding 开始，这会让读者误以为 token ID 是天然存在的。

实际上 tokenization 决定了模型输入坐标系。

---

# 7　中文为什么尤其应该检查 token fertility

可以定义一种简单的 fertility：

$$
F=\frac{\text{token 数}}{\text{字符或词的数量}}.
$$

如果同一段中文在 tokenizer A 中平均每个汉字需要更多 token，那么：

- context window 的中文有效容量更低；
- prefill FLOPs 更多；
- KV cache 占用更高；
- 训练数据中的中文 token 比例与字符比例不同。

因此“多语言能力”不仅与数据量有关，也与 tokenizer 的压缩效率和 vocabulary allocation 有关。

---

# 8　词表越大越好吗？

词表扩大，通常会减少平均序列长度，但 embedding 与输出层规模增加。

若 embedding dimension 为 $d$，词表大小为 $V$，最朴素 embedding 参数量约：

$$
N_{emb}=Vd.
$$

若 input embedding 与 LM head 不共享参数，则词表扩大还会扩大输出投影。

所以 vocabulary size 是一个系统权衡：

```text
更大 vocabulary
  ├─ 更短序列
  ├─ 更高语义压缩
  └─ 更大 embedding / softmax 成本
```

---

# 9　真正决定模型的不是“互联网”，而是数据管线

训练语料通常经历：

```text
raw sources
   ↓
crawl / collection
   ↓
parsing
   ↓
language ID
   ↓
quality filtering
   ↓
safety / privacy filtering
   ↓
deduplication
   ↓
document / domain mixture
   ↓
tokenization
   ↓
sampling / curriculum
   ↓
training batches
```

因此，“模型在 Common Crawl 上训练”不是一个足够精确的描述。

真正的问题是：**Common Crawl 的哪一部分，以什么规则被保留下来？**

---

# 10　C4：过滤本身就是建模选择

T5 的 C4（Colossal Clean Crawled Corpus）展示了一条重要路线：从 Common Crawl 中构造更干净的训练语料。[T5 原论文](https://arxiv.org/abs/1910.10683)

这里最值得学习的不是 C4 的绝对大小，而是：

> 同一个原始 web crawl，通过不同过滤器，会得到不同的模型世界。

过滤器本身隐含价值判断与 distribution shift。

---

# 11　The Pile：显式 mixture 的思想

Gao et al. (2020) 的 The Pile 将多个不同来源组织成显式 mixture。[原论文](https://arxiv.org/abs/2101.00027)

这提醒我们：

$$
P_{train}(x)=\sum_k \pi_k P_k(x),
$$

其中 $P_k$ 是某个数据域，$\pi_k$ 是采样权重。

改变 $\pi_k$ 就是在改变模型所见世界的先验分布。

代码、论文、书籍、论坛、网页、对话，不是“混在一起就行”，而是一个 mixture design 问题。

---

# 12　去重：不是省磁盘，而是改变学习动力学

Lee et al. (2021) 系统研究了训练数据去重，指出重复样本会影响记忆、泛化和 benchmark contamination。[原论文](https://arxiv.org/abs/2107.06499)

最简单 exact dedup：

```text
hash(document)
```

但互联网更常见的是 near-duplicate：

```text
同一文章
只改标题 / 广告 / HTML / 少量句子
```

所以实际系统常使用：

- n-gram shingles；
- MinHash；
- locality-sensitive hashing；
- embedding similarity；
- document clustering。

去重阈值太宽会误删不同内容；太窄又留住模板化重复。

---

# 13　FineWeb、Dolma 与 DataComp-LM：数据研究本身成为主战场

开放数据研究在 2024 前后明显成熟。

重要原始资料包括：

- Soldaini et al., **Dolma**  
  https://arxiv.org/abs/2402.00159
- Penedo et al., **The FineWeb Datasets**  
  https://arxiv.org/abs/2406.17557
- Li et al., **DataComp-LM**  
  https://arxiv.org/abs/2406.11794

这些工作共同推动一个观念：

> **数据 recipe 可以像模型 architecture 一样被系统研究。**

---

# 14　质量过滤不是“高质量分类器打一分”这么简单

假设质量模型给文档分数 $q(x)$。

一种做法是阈值过滤：

$$
\mathcal D'=\{x\in\mathcal D:q(x)>\tau\}.
$$

但这会造成新的分布偏差。

例如一个“像 Wikipedia 的文本就是高质量”的 classifier，可能系统性降低：

- 口语；
- 对话；
- 某些小语种；
- 非正式代码讨论；
- 特殊领域文本。

因此 quality filtering 本质上是一个 precision / diversity / coverage trade-off。

---

# 15　Benchmark contamination

如果 benchmark 样本或其解答出现在预训练数据中，模型高分可能部分来自记忆。

污染至少分三层：

1. exact match；
2. near duplicate；
3. semantic leakage / solution leakage。

因此真正严谨的评测需要同时考虑：

```text
benchmark release date
training cutoff
web replication
solution availability
near-duplicate matching
```

“模型不知道测试集”不能仅靠官方一句声明。

---

# 16　合成数据：数据不再只来自人类互联网

当模型能力提升后，一个新循环出现：

```text
model
 ↓ generate
synthetic examples
 ↓ filter / verify
new training set
 ↓ train
new model
```

但 synthetic data 有两个完全不同的世界：

- **有 verifier 的合成数据**：数学、代码、结构化环境；
- **没有可靠 verifier 的合成数据**：风格、开放问答、价值偏好。

前者更容易规模化，后者容易出现 self-reinforcing bias。

---

# 17　数据时代的 Scaling Law 应如何理解

Chinchilla 提醒行业：固定 compute 下，很多早期大模型训练 token 不够。[Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)

但“更多 token”也不是无限成立：

- 高质量数据有限；
- 重复 epoch 带来收益递减；
- domain distribution 影响 transfer；
- synthetic data 改变有效数据定义。

所以更现代的表达是：

$$
\text{Capability}=f(N,D,C,Q,M,O),
$$

其中除了参数 $N$、数据量 $D$、算力 $C$，还要考虑数据质量 $Q$、mixture $M$ 和 optimization $O$。

---

# 18　原始资料包

### Tokenization

- Sennrich et al. (2015/2016), BPE for NMT  
  https://arxiv.org/abs/1508.07909
- Kudo & Richardson (2018), SentencePiece  
  https://arxiv.org/abs/1808.06226
- OpenAI `tiktoken`  
  https://github.com/openai/tiktoken

### Data

- T5 / C4  
  https://arxiv.org/abs/1910.10683
- The Pile  
  https://arxiv.org/abs/2101.00027
- Deduplicating Training Data Makes Language Models Better  
  https://arxiv.org/abs/2107.06499
- Dolma  
  https://arxiv.org/abs/2402.00159
- FineWeb  
  https://arxiv.org/abs/2406.17557
- DataComp-LM  
  https://arxiv.org/abs/2406.11794

---

# 19　最小实验

建议做四个实验：

1. 同一中英双语语料分别用不同 tokenizer，统计 token fertility；
2. 固定模型与 compute，只改变 vocabulary size；
3. 构造含 near-duplicate 的 toy corpus，比较记忆与 validation loss；
4. 固定 token budget，改变 code / prose mixture，观察迁移任务变化。

这四个实验会让你真正理解：**数据不是喂给模型的燃料，而是模型世界观的一部分。**
