# Paper Card：<YEAR> <PAPER TITLE>

> **原始来源**：<paper URL>  
> **官方代码**：<official code URL, if any>  
> **教材位置**：<links to chapters>  
> **代码位置**：<links to our implementation>  
> **最后核验日期**：YYYY-MM-DD

---

# 1　原始问题

论文出现之前，研究问题是什么？旧方法的瓶颈必须按当时历史语境描述，避免用后来知识重写历史。

---

# 2　方法的最小定义

用作者自己的定义、原公式与必要的张量形状描述方法。

不要先写社区简化版。

---

# 3　Claim Ledger

## `<ID>-C01`

- **类型**：method / empirical / systems / interpretation
- **精确转述**：
- **原始证据**：section / equation / table / figure
- **适用条件**：
- **后续证据**：
- **状态**：supported / conditional / revised / disputed / historical
- **教材使用位置**：

## `<ID>-C02`

同上。

---

# 4　原始实验设置

必须记录会改变结论的设置：

- data；
- model size；
- optimizer；
- training compute；
- evaluation metric；
- baseline；
- hardware；
- seed / variance（若公开）。

---

# 5　原始结果

只记录作者实际报告的结果。不要把不同论文、不同数据、不同评测条件下的数字强行拼成横向榜单。

---

# 6　官方实现与论文差异

如果有官方代码，检查：

```text
paper equation
        ↓
official implementation
        ↓
config / defaults
```

记录任何实际差异。

---

# 7　后续复现、修正与反例

至少回答：

1. 核心结果是否被独立复现？
2. 哪些结论只在特定规模/数据/compute 下成立？
3. 是否有后续论文给出更强解释？
4. 是否出现反例或负结果？

---

# 8　我们自己的复现

| 项目 | 状态 | 证据 |
|---|---|---|
| 公式实现 | ☐ | |
| 单元测试 | ☐ | |
| 数值 parity | ☐ | |
| 原实验复现 | ☐ | |
| 关键消融 | ☐ | |

---

# 9　最终应该记住什么

用 3–5 条写“当前较稳健理解”，而不是论文宣传语。

---

# 10　不能从这篇论文推出什么

主动列出常见过度推断。