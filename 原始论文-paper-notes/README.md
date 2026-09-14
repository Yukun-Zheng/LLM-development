# 原始论文卡片与 Claim Ledger / Paper Notes

> **目标**：不再只在正文末尾列论文链接，而是为核心原始工作建立可审计的研究卡片：**原问题 → 原始 claim → 原公式 → 原实验 → 官方代码 → 后续修正 → 教材位置 → 我们的复现**。

这个目录是 Source-First 方法论的进一步工程化。

---

# 1　为什么需要 Paper Card

普通参考文献表只能回答：

> “这篇论文在哪里？”

Paper Card 还要回答：

```text
论文到底声称了什么？
证据是哪张表 / 哪个实验？
公式如何定义？
代码真正如何实现？
哪些结论后来被复现？
哪些后来被修正？
本教材在哪些章节使用了它？
我们的代码是否复现了相关机制？
```

因此 Paper Card 是**引用与解释之间的中间层**。

---

# 2　Claim ID

核心结论使用稳定 ID，例如：

```text
TRF-2017-C01
TRF-2017-C02
GPT3-2020-C01
CHIN-2022-C01
```

一个 Claim 记录：

```text
ID
├─ 原始表述 / 我们的精确转述
├─ 类型：method / empirical / systems / interpretation
├─ 原始证据
├─ 适用条件
├─ 后续证据
├─ 当前状态
└─ 教材引用位置
```

状态可以是：

- `supported`：后续证据总体支持；
- `conditional`：只在特定条件成立；
- `revised`：后续工作显著修正；
- `disputed`：证据冲突；
- `historical`：主要作为历史原始主张保留。

---

# 3　卡片模板

模板：[`TEMPLATE-paper-card.md`](TEMPLATE-paper-card.md)

首张完整示例：

- [`2017-Attention-Is-All-You-Need.md`](2017-Attention-Is-All-You-Need.md)

后续优先补齐：

```text
2018-GPT1
2018-BERT
2019-GPT2
2020-Scaling-Laws
2020-GPT3
2022-Chinchilla
2022-InstructGPT
2022-CoT
2022-FlashAttention
2023-LLaMA
2023-DPO
2023-ReAct
2024-DeepSeek-V2
2024-Mamba-2
2024-SWE-bench
2025-DeepSeek-R1
2025-LLaDA
```

---

# 4　与其他目录的关系

```text
参考文献-references/
    ↓ 找到原始来源

原始论文-paper-notes/
    ↓ 审计 claim 与证据

教材-book/ / 智能体-agent/
    ↓ 形成解释与推导

代码-code/
    ↓ 实现机制

tests / CI
    ↓ 验证
```

全局映射：[`../学习地图-MAP.md`](../学习地图-MAP.md)  
成熟度：[`../质量看板-QUALITY.md`](../质量看板-QUALITY.md)

---

# 5　原则

Paper Card 不是论文摘要。它必须尽量区分：

1. **作者明确写出的 claim**；
2. **我们从公式/实验得出的解释**；
3. **后续论文对它的支持或修正**；
4. **社区后来形成但原论文没有声称的说法**。

任何无法定位到原始证据的“著名结论”，都不应该因为被重复很多次就自动升级为事实。