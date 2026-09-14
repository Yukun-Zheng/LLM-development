# 第十五篇（Part XV）　可解释性与机制研究：模型内部到底发生了什么

> **本章主线**：大模型可以输出正确答案，但“为什么会得到这个答案”仍是独立问题。本章区分 attribution、representation analysis、causal intervention、mechanistic interpretability、model editing 与 sparse feature discovery，强调只有真正改变内部变量并观察行为变化，才更接近因果机制解释。

---

# 1　“可解释”到底在解释什么

解释对象至少有五种：

1. **输入归因**：哪些 token / feature 影响了输出？
2. **表示分析**：某层 hidden state 编码了什么？
3. **电路分析**：哪些 head、MLP、feature 共同实现某个算法？
4. **因果定位**：改变哪个内部变量会改变行为？
5. **全局机制**：模型如何形成 in-context learning、检索、推理、规划等能力？

把这些混为一谈，会导致“看到了相关性”却声称“解释了模型”。

---

# 2　Activation 只是内部状态，不天然有语义

对于第 $l$ 层 hidden state：

$$
h^{(l)}_t\in\mathbb R^d,
$$

我们可以读取它，但读取不等于理解。

如果某个神经元在法语文本上激活很高，最多说明：

> 该激活与法语输入相关。

要声称它“负责法语”，至少还需要 intervention：抑制或修改它后，行为是否发生系统变化？

---

# 3　Linear Probe：表示里“可线性读出”什么

给定 hidden states $h$ 和标签 $y$，训练一个线性 probe：

$$
\hat y=Wh+b.
$$

若准确率高，说明 $y$ 在表示中至少是 linearly decodable。

但这不能直接推出模型在计算中真的使用了该信息。

一个变量可能：

- 可以被 probe 读出；
- 但下游预测完全不依赖它。

所以 probe 更适合回答“信息是否存在”，而不是“信息是否因果参与”。

---

# 4　Attention visualization 为什么不能等于 explanation

Attention matrix：

$$
A=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right).
$$

它确实表示一个 attention head 在当前 forward pass 中怎样加权 value。

但最终输出还经过：

- 多个 head；
- output projection；
- residual stream；
- MLP；
- 后续很多层。

因此“某个 token attention 权重最大”并不自动意味着它对最终答案贡献最大。

Attention map 是机制的一部分，但不是完整因果解释。

---

# 5　Residual Stream：Transformer 电路的共享总线

在 decoder-only Transformer 中，一个有用的机制视角是 residual stream：

$$
x_{l+1}=x_l+\Delta x_l^{attn}+\Delta x_l^{mlp}.
$$

每一层不是完全替换表示，而是向 residual stream 写入增量。

于是可以把模型理解为：

```text
residual stream
   ↑      ↑
 attention heads
   ↑      ↑
 MLP features
   ↓
next layer reads again
```

Elhage et al. 的 Transformer Circuits 工作系统化了这种视角。[原始资料](https://transformer-circuits.pub/2021/framework/index.html)

---

# 6　Logit Lens：中间层“已经想输出什么”

若最终 LM head 为：

$$
\mathrm{logits}=W_U h^{(L)},
$$

一种简单分析是把中间层 hidden state 也投影到 vocabulary：

$$
\mathrm{logits}^{(l)}=W_U h^{(l)}.
$$

这称为 logit lens 一类方法。

它能帮助观察：

- 某个 token 的概率何时开始上升；
- 事实答案在哪层出现；
- 某些错误是否在早期已经形成。

但它隐含“中间层可以直接用最终 unembedding 解码”的假设，因此也需要谨慎解释。

---

# 7　Activation Patching：从相关走向因果

假设有两个输入：

- clean input：模型行为正确；
- corrupted input：模型行为错误。

我们可以把 clean run 某层激活替换到 corrupted run：

```text
corrupted run
    ↓
layer l
    ↓ replace activation with clean activation
continue forward
    ↓
behavior recovers?
```

如果替换某个 activation 后答案恢复，说明这个位置/组件对行为具有更强的因果关联。

这类方法比单纯看 attention map 更接近机制定位。

---

# 8　Causal Tracing 与事实记忆定位

Meng et al. (2022) 的 ROME 工作通过 causal tracing 等方法研究事实关联在 GPT 中的定位，并进一步提出 model editing。[原论文](https://arxiv.org/abs/2202.05262)

典型思路：

1. clean prompt 得到正确事实；
2. 给关键 token representation 加噪声，使预测失败；
3. 在某层恢复 clean activation；
4. 检查事实预测是否恢复。

这比“哪个层相关”更接近：

> 哪个内部状态对该事实预测是必要或关键的？

---

# 9　Model Editing：定位与修改是两回事

ROME、MEMIT 等方法尝试在不完整重训模型的情况下修改某些事实关联。

例如想把：

```text
The Eiffel Tower is in Paris
```

局部改为另一个目标关系。

但成功编辑必须同时检查：

- edit success；
- paraphrase generalization；
- neighborhood specificity；
- unrelated knowledge preservation；
- multi-hop consequences。

如果只让一个 prompt 输出改变，不代表“模型知识”真的被干净修改。

---

# 10　Induction Heads：一个经典可研究电路

Olsson et al. (2022) 研究 induction heads 与 in-context learning 的关系。[原论文](https://arxiv.org/abs/2209.11895)

一种典型模式：

```text
... A B ... A
            ↓
模型倾向继续预测 B
```

一个 induction-like circuit 可以利用前文出现过的 token pattern，把“上次 A 后面是什么”复制到当前上下文。

这类工作重要的地方不是“找到了一个神奇 attention head”，而是展示了：

> 某些 transformer 行为可以被分解成较小的可组合计算结构。

---

# 11　Circuit 的标准应该多严格

若声称一个 circuit 实现某行为，至少希望验证：

### Sufficiency

保留该 circuit 是否还能实现行为？

### Necessity

移除它是否显著破坏行为？

### Specificity

它是否只影响目标行为，而不是把模型整体破坏？

### Generalization

是否跨 prompt、seed、model size、task 仍成立？

只展示几个漂亮 attention 图，不满足这些标准。

---

# 12　Superposition：为什么一个 neuron 不等于一个概念

大模型 hidden dimension 有限，但可能需要表示远多于 $d$ 个 feature。

一个核心假设是 superposition：多个 feature 以非正交方式共享同一表示空间。

这会导致：

> 单个 neuron 往往是 polysemantic 的。

因此“神经元语义”可能不是最好的解释单位。

Olah 等 circuits 工作与后续 sparse autoencoder 研究都围绕这一问题发展。

---

# 13　Sparse Autoencoder：把 dense activation 分解成稀疏 feature

给定 activation $x\in\mathbb R^d$，训练 encoder：

$$
z=f(W_e x+b_e),
$$

decoder：

$$
\hat x=W_d z+b_d,
$$

并对 $z$ 施加稀疏约束：

$$
\mathcal L
=
\lVert x-\hat x\rVert_2^2
+\lambda\lVert z\rVert_1.
$$

目标是得到大量较稀疏、可能更接近单一语义的 feature。

Anthropic 2024 年的 Scaling Monosemanticity 工作将这一路线扩展到 Claude 3 Sonnet 的内部表示分析。[原始资料](https://transformer-circuits.pub/2024/scaling-monosemanticity/)

---

# 14　“可解释 feature”仍然不等于完整机制

即使某 SAE feature 看起来对应：

```text
Golden Gate Bridge
```

还需要问：

- 它在哪些输入上激活？
- 激活是否有 false positive？
- 增强它会改变输出吗？
- 抑制它会怎样？
- 它通过哪些 downstream components 影响 logits？

所以 feature discovery 只是开始。

---

# 15　Mechanistic Interpretability 的研究对象应从“神经元”走向“计算图”

更成熟的机制问题是：

```text
输入模式
  ↓
feature A
  ↓
head B writes to residual stream
  ↓
MLP C transforms representation
  ↓
head D retrieves another token
  ↓
logit direction changes
```

这更像程序分析，而不是可视化。

---

# 16　Reasoning Model 给可解释性带来的新问题

推理模型增加了新的层次：

```text
internal computation
      ↓
visible reasoning trace
      ↓
final answer
```

一个关键问题是：

> 可见的 reasoning trace 是否忠实反映内部计算？

不能把 chain-of-thought 文本自动当成内部机制解释。

需要区分：

- faithfulness；
- usefulness；
- monitorability；
- controllability。

对于闭源 reasoning models，本书只使用 system card / 官方研究中明确支持的结论。

---

# 17　Interpretability 工具链

常见工具包括：

- forward hooks；
- activation cache；
- head ablation；
- activation patching；
- attribution patching；
- logit lens；
- probing；
- causal tracing；
- sparse autoencoders。

TransformerLens 是一个重要开源工具：  
https://github.com/TransformerLensOrg/TransformerLens

但使用工具不等于完成解释，实验设计仍然最重要。

---

# 18　一个标准机制实验

以“模型为什么预测 Paris”为例：

```text
Prompt
"The Eiffel Tower is located in"
      ↓
baseline: Paris high probability
      ↓
corrupt subject representation
      ↓
Paris probability falls
      ↓
patch layer/head/MLP activations
      ↓
find which intervention restores Paris
      ↓
trace downstream path to logits
```

最后还要用 paraphrase 与 control facts 检查 specificity。

---

# 19　原始资料包

- Olah et al. (2020), **Zoom In: An Introduction to Circuits**  
  https://distill.pub/2020/circuits/zoom-in/
- Elhage et al. (2021), **A Mathematical Framework for Transformer Circuits**  
  https://transformer-circuits.pub/2021/framework/index.html
- Olsson et al. (2022), **In-context Learning and Induction Heads**  
  https://arxiv.org/abs/2209.11895
- Meng et al. (2022), **Locating and Editing Factual Associations in GPT**  
  https://arxiv.org/abs/2202.05262
- Anthropic (2024), **Scaling Monosemanticity**  
  https://transformer-circuits.pub/2024/scaling-monosemanticity/
- TransformerLens  
  https://github.com/TransformerLensOrg/TransformerLens

---

# 20　最小实验

推荐从一个两层或四层 tiny GPT 开始：

1. 可视化 attention pattern；
2. 做 single-head ablation；
3. 做 clean/corrupted activation patching；
4. 用 logit lens 看答案在哪层形成；
5. 训练一个小 SAE，检查 feature sparsity 与 intervention。

只有当“观察 → 干预 → 行为变化”形成闭环时，可解释性才开始从漂亮图走向真正的机制科学。
