# 评测实验室 / Evaluation Lab

> **定位**：把“pytest 通过”与“智能体能力真的变强”严格分开。

单元测试回答：

> 代码是否按接口契约工作？

能力评测回答：

> 在真实任务和真实环境中，系统完成了多少工作，花了多少时间、token、工具调用与人工干预，失败在哪里？

二者不能互相替代。

---

# 1　统一评测对象

本项目最终统一记录以下层次：

```text
Model Eval
├─ logits parity
├─ perplexity / task accuracy
└─ architecture / numerical ablation

Inference Eval
├─ TTFT
├─ TPOT / ITL
├─ throughput
├─ memory
└─ scheduler fairness

Agent Eval
├─ task success
├─ partial success
├─ wall-clock time
├─ model steps
├─ tool calls
├─ retries
├─ context compactions
├─ subagents
├─ approvals
├─ policy violations
└─ verifier evidence
```

第一版机器可用指标实现位于：

[`../代码-code/从零构建Astra与Codex级系统/src/astra_codex/evaluation.py`](../代码-code/从零构建Astra与Codex级系统/src/astra_codex/evaluation.py)

---

# 2　Agent benchmark 不能只看最终一句话

我们不接受：

```text
assistant: "任务已经完成。"
```

作为完成证据。

至少需要：

```text
trajectory
+ environment state
+ artifact
+ verifier / grader
```

例如 Coding Agent：

```text
issue
→ inspected files
→ patch
→ targeted tests
→ broader tests
→ git diff
→ verifier
```

Browser / Computer Agent：

```text
initial environment state
→ actions
→ final environment state
→ deterministic / hidden grader
```

---

# 3　实验对照原则

任何 Agent 架构改动都尽量保留 controlled comparison。

Multi-Agent 不是“agent 越多越先进”，至少比较：

| 配置 | Success | Wall time | Steps | Tool calls | Tokens | Cost | Conflicts | Human approvals |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 agent | | | | | | | | |
| 2 agents | | | | | | | | |
| 4 agents | | | | | | | | |
| 8 agents | | | | | | | | |

Memory、planning、compaction、reviewer、computer-use observation 同理：必须与简单 baseline 对照。

---

# 4　计划接入的 benchmark 家族

后续按适配难度推进：

```text
Level 0  deterministic local fixtures
Level 1  repository coding fixtures
Level 2  SWE-bench adapter
Level 3  browser environment adapter
Level 4  OS/computer environment adapter
Level 5  long-horizon cross-application tasks
```

评测适配器必须记录 benchmark version、environment image、tool budget、timeout、model settings 和 grader version，避免不同运行条件被误写成模型能力差异。

---

# 5　最终原则

```text
Unit Test != Capability Eval
Benchmark Score != Scientific Explanation
Model Quality != Agent Runtime Quality
Agent Count != Multi-Agent Benefit
Long Context != Long-Horizon Reliability
```

本目录最终会成为整个项目的“实验事实层”：任何声称“新模块更强”的结论，都应该能在这里找到可重复的对照实验。
