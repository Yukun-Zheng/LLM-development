# 评测实验室 / Evaluation Lab

> **定位**：把“pytest 通过”“模型输出看起来不错”和“智能体在真实环境里真的完成任务”严格分开。

单元测试回答：**实现是否满足代码契约？**  
能力评测回答：**系统在给定环境、预算和 grader 下完成多少工作，代价和失败模式是什么？**

---

# 1　现在已经有可执行评测 substrate

源码：

- [`evaluation.py`](../代码-code/从零构建Astra与Codex级系统/src/astra_codex/evaluation.py)：trajectory metrics；
- [`benchmark.py`](../代码-code/从零构建Astra与Codex级系统/src/astra_codex/benchmark.py)：case / executor / grader / record；
- [`test_scheduler_benchmark.py`](../代码-code/从零构建Astra与Codex级系统/tests/test_scheduler_benchmark.py)：确定性 reference tests。

当前数据流：

```text
BenchmarkCase
    ↓
Executor
    ↓
Trajectory / Final Result
    ↓
Grader
    ↓
Grade
    ↓
BenchmarkRecord
    ↓
AggregateMetrics
```

`BenchmarkHarness` 刻意不把“assistant 最后说了什么”直接当作成功；成功需要 grader/verifier 决定。

Fast CI 已验证一个两个 case 的 toy suite：一个通过、一个失败，aggregate `success_rate=0.5`。这证明评测框架的分层行为，不代表真实 Agent benchmark 能力。

---

# 2　统一指标层

当前 `TrajectoryMetrics` 可记录：

```text
success
verifier_passed
model_steps
tool_calls
tool_failures
approvals_requested
stopped_by_limit
wall_time_s
input_tokens
output_tokens
cost_usd
```

最终扩展：

```text
context_compactions
retrievals
subagents_spawned
worktree_conflicts
retries
human_interventions
policy_violations
sandbox_denials
environment_errors
artifact_count
```

Inference 侧统一记录：

```text
TTFT
TPOT / ITL
throughput
prompt/decode tokens
KV/state memory
queueing delay
batch occupancy
scheduler fairness
```

---

# 3　成功必须来自环境/Artifact 证据

不接受：

```text
assistant: "任务已经完成。"
```

作为完成证据。

Coding Agent 至少应形成：

```text
initial repository
→ issue
→ trajectory
→ patch
→ tests
→ final git diff
→ deterministic/hidden grader
```

Browser / Computer Agent：

```text
initial environment state
→ actions
→ final environment state
→ hidden/deterministic grader
```

研究 Agent：

```text
question
→ sources / files / computation
→ artifact
→ factual / structural / reproducibility grader
```

---

# 4　Benchmark Case 规范下一步

真实 benchmark adapter 每个 case 至少固定：

```text
case_id
benchmark_version
environment/image version
initial state
goal
tool/action budget
timeout
model settings
permission profile
sandbox profile
grader version
expected artifacts
```

否则：

```text
score A > score B
```

无法知道到底来自模型、harness、环境、tool budget 还是 grader 差异。

---

# 5　Controlled Comparison 是核心，而不是排行榜

任何新模块优先回答“它究竟贡献了什么”。

例如：

| 对照 | 主要测量 |
|---|---|
| memory vs no-memory | success / context tokens / stale retrieval |
| raw history vs compaction | success / tokens / evidence loss |
| planner vs reactive | success / steps / replans |
| 1 vs 2 vs 4 vs 8 agents | success / wall time / tokens / conflicts / cost |
| reviewer vs no-reviewer | regressions / false positives / cost |
| DOM vs screenshot | grounding success / latency / token/image cost |
| contiguous vs paged KV | logits parity / memory / fragmentation |
| scheduler A vs B | TTFT / TPOT / throughput / fairness |
| normal vs speculative decode | distribution parity / latency / accepted tokens |

---

# 6　Benchmark 路线

```text
Level 0  deterministic local toy cases        ✅ substrate 已有
Level 1  repository fixtures                  ← 当前下一步
Level 2  SWE-bench adapter
Level 3  browser environment adapter
Level 4  OS/computer environment adapter
Level 5  long-horizon cross-application tasks
```

Level 1 不需要先追求大规模 benchmark。先建立几个完全可复现的小仓库：bug、feature、refactor、test failure、multi-file dependency，让 patch + tests + artifact grader 的闭环稳定，再接 SWE-bench。

---

# 7　评测与训练的接口

随着 Agentic RL 加入，评测必须和 reward 严格区分：

```text
training verifier / reward
            ≠
held-out evaluator
```

否则系统会直接优化 grader，而不是任务本身。

未来 RLVR/Agentic RL 实验至少保留：

```text
train environments
held-out environments
reward function version
evaluation grader version
```

并报告 reward hacking / overfitting。

---

# 8　原则

```text
Unit Test != Capability Eval
Benchmark Score != Scientific Explanation
Model Quality != Agent Runtime Quality
Agent Count != Multi-Agent Benefit
Long Context != Long-Horizon Reliability
Final Answer != Final Environment State
Training Reward != Held-out Evaluation
```

本目录最终是整个项目的**实验事实层**：任何“更强、更快、更安全、更可靠”的结论，都必须能在这里找到可重复的任务、环境、grader、指标与对照。
