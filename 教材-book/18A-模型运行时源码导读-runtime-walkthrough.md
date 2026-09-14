# 第十八篇附录 A　模型运行时源码导读：从 Token ID 到增量解码

> 对应工程：`代码-code/从零构建Astra与Codex级系统/src/astra_codex/`
>
> 本章不重新讲一遍 Transformer，而是把理论精确映射到我们自己写的运行时代码。

---

# 1　为什么“不会训练”也必须会写 forward

训练一个 frontier model 需要巨额数据和算力；但理解模型结构不需要重新训练它。

我们把任务拆开：

```text
训练问题：如何得到参数 θ？
运行时问题：给定 θ 和 token IDs，如何精确计算 logits？
```

本项目暂时跳过第一问的超大规模部分，但第二问必须从头实现。

---

# 2　`config.py`：架构首先是一组约束

关键约束包括：

$$
d_{model} \bmod H_q = 0,
$$

$$
H_q \bmod H_{kv} = 0.
$$

前者决定 head dimension，后者决定一个 KV head 被多少 query heads 共享。

代码中的 `head_dim` 与 `kv_repeat` 不是辅助变量，而是架构关系的可执行表达。

---

# 3　`tokenizer.py`：先从完全透明的 byte tokenizer 开始

生产 tokenizer 很复杂，因此教学上先写最小可逆系统：UTF-8 byte $\leftrightarrow$ token ID。

随后再自己训练 BPE：

1. 统计相邻 token pair；
2. 找频率最高 pair；
3. 创建新 token；
4. 全语料替换；
5. 重复。

这样读者真正知道 vocabulary 是怎样“长出来”的。

原始资料：

- Sennrich et al. 2015/2016: https://arxiv.org/abs/1508.07909
- SentencePiece: https://arxiv.org/abs/1808.06226

---

# 4　`model.py`：现代 decoder-only block

我们的 block 是：

```text
x
│
├─ RMSNorm
│   ↓
│  GQA + RoPE + causal mask
│   ↓
├─ residual add
│
├─ RMSNorm
│   ↓
│  SwiGLU
│   ↓
└─ residual add
```

它不是任何闭源 GPT 的内部声明，而是公开现代 LLM 中常见组件的透明组合。

## 4.1 RMSNorm

原始资料：https://arxiv.org/abs/1910.07467

## 4.2 RoPE

原始资料：https://arxiv.org/abs/2104.09864

## 4.3 GQA

原始资料：https://arxiv.org/abs/2305.13245

## 4.4 SwiGLU

原始资料：https://arxiv.org/abs/2002.05202

---

# 5　`cache.py` + `engine.py`：推理系统真正开始的地方

如果只会写 `model.forward()`，还没有进入现代 serving。

生成被拆成：

```text
prefill(prompt)
  ↓
KV cache + next-token logits
  ↓
decode(token_1)
  ↓
append KV
  ↓
decode(token_2)
  ↓
...
```

最重要的单元测试是数值等价：

$$
\mathrm{logits}_{cached}(x_T)
\approx
\mathrm{logits}_{full}(x_{1:T}).
$$

若这一步不成立，后面所有 sampling、tool call、agent 都建立在错误模型运行时上。

---

# 6　`sampling.py`：logits 不是答案

我们显式实现：

- greedy；
- temperature；
- top-k；
- nucleus / top-p；
- repetition penalty。

这样可以看清：**模型参数决定 logits，sampling policy 决定从分布中怎么选 token。**

不要把 sampling 行为误写成模型“知识”或“推理能力”。

---

# 7　`weights.py`：公开 checkpoint 不是一行 `from_pretrained`

真正加载公开权重要解决：

- shard；
- tensor key；
- shape；
- transposition；
- fused/unfused QKV；
- tied weights；
- tokenizer / config 对齐。

当前源码先实现 raw safetensors loading、key remap 和 shape report。下一阶段选择一个公开小模型做 logits parity。

---

# 8　毕业验收

这一层完成时，读者必须能回答并亲手验证：

1. 每个 tensor shape 为什么是这样？
2. RoPE 在 Q/K 哪个维度上工作？
3. GQA 为什么减少 KV Cache？
4. causal mask 在 prefill 与 decode 时有什么不同？
5. 为什么 cache parity 是必须的？
6. logits 与 token sampling 是什么关系？
7. 一个 safetensors checkpoint 怎样映射到自己定义的 module？
