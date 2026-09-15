# 评测实验室 / Evaluation Lab

> **定位**：这是全项目的**实验事实层**。把“pytest 通过”“模型最后说自己完成了”和“智能体真的改变了正确的环境状态”严格分开。

单元测试回答：**实现是否满足代码契约？**  
能力评测回答：**系统在给定环境、预算和 grader 下完成多少真实工作，代价与失败模式是什么？**

---

# 1　当前可执行评测栈

```text
Trajectory Metrics          evaluation.py
        ↓
Generic Case / Grader       benchmark.py
        ↓
Repository Final-state Eval repository_eval.py
        ↓
[next] hidden/multi-file repository suite
        ↓
[next] SWE-bench / Browser / OS adapters
```

源码：

- [`evaluation.py`](../代码-code/从零构建Astra与Codex级系统/src/astra_codex/evaluation.py)
- [`benchmark.py`](../代码-code/从零构建Astra与Codex级系统/src/astra_codex/benchmark.py)
- [`repository_eval.py`](../代码-code/从零构建Astra与Codex级系统/src/astra_codex/repository_eval.py)

测试：

- [`test_scheduler_benchmark.py`](../代码-code/从零构建Astra与Codex级系统/tests/test_scheduler_benchmark.py)
- [`test_repository_eval.py`](../代码-code/从零构建Astra与Codex级系统/tests/test_repository_eval.py)

---

# 2　通用 Case / Grader 分层

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

`BenchmarkHarness` 刻意不把 assistant final answer 当作 success。是否成功由外部 grader 决定。

当前 toy suite 已自动验证：两个 case 中一个通过、一个失败，aggregate `success_rate=0.5`。

---

# 3　Repository Fixture：第一个真正的环境终态评测

`RepositoryFixture` 当前固定：

```text
case_id
goal
initial files
verify argv
optional expected final files
timeout
```

Harness：

```text
materialize repository
→ git init
→ Coding Agent executes
→ Agent stops
→ independent verify command
→ expected-file check
→ RepositoryGrade
```

最重要的是 grading 发生在 Agent 完成以后，并且不信任其自然语言结论。

自动测试故意构造：

```text
Case A
Agent: "I fixed it and all tests pass."
但没有调用任何工具，也没有修改 solution.py
→ external command fails
→ expected file mismatch
→ FAIL

Case B
Agent performs real exact edit
→ runs command
→ external harness independently reruns verifier
→ final file matches expected state
→ PASS
```

这使本项目第一次拥有：

> **“模型自述成功”与“最终环境成功”可被系统性区分的 coding-agent benchmark。**

---

# 4　统一 Trajectory Metrics

当前已经能记录：

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

后续扩展：

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

Inference 侧统一：

```text
TTFT
TPOT / ITL
throughput
queueing delay
batch occupancy
KV/state memory
scheduler fairness
```

---

# 5　Benchmark Case 的复现信息

真实 adapter 每个 case 最终必须固定：

```text
case_id
benchmark version
environment/image version
initial state
goal
tool/action budget
timeout
model/backend settings
permission profile
sandbox profile
grader version
expected artifacts
random seeds / nondeterminism policy
```

否则两次 score 无法科学比较。

---

# 6　Controlled Comparison 优先于排行榜

本项目后续最重要的评测不是“我们的 Agent 排第几”，而是：

| 对照 | 主要问题 |
|---|---|
| memory vs no-memory | Memory 是否真的提高成功率，还是只增加 context/cost？ |
| raw history vs compaction | 压缩节省多少 token，又丢多少证据？ |
| planner vs reactive | plan 是否减少失败/重试？ |
| 1 vs 2 vs 4 vs 8 agents | 并行是否值得额外 token/conflict/merge cost？ |
| reviewer vs no-reviewer | reviewer 能减少多少 regression？ |
| DOM vs screenshot | 哪种 observation 对 GUI grounding 更有效？ |
| contiguous vs paged KV | 数值等价后，memory/fragmentation 收益多少？ |
| scheduler A vs B | TTFT/TPOT/throughput/fairness trade-off？ |
| normal vs speculative | 分布是否一致，实际 latency 是否下降？ |
| no durability vs durable runtime | crash 后 completion/lost work/duplicate side effect 如何变化？ |

---

# 7　Benchmark 路线

```text
Level 0   deterministic toy case / grader              ✅
Level 1a  single-file repository final-state fixture   ✅
Level 1b  multi-file + hidden-test repository suite    ← next
Level 2   SWE-bench adapter
Level 3   browser environment adapter
Level 4   OS/computer environment adapter
Level 5   long-horizon cross-application + crash/recovery
```

Level 1b 应先覆盖：

```text
bug fix
feature addition
multi-file dependency
refactor with behavior preservation
hidden regression
failing test localization
```

并保存 patch / test / final-state artifacts。

---

# 8　评测与 RL 的隔离

随着 verifier RL 加入，必须强制：

```text
training reward / verifier
!=
held-out evaluator
```

否则可能得到：

```text
training reward ↑
held-out task success ↔ / ↓
```

而误以为 reasoning 变强。

所以未来 SFT→DPO→RLVR 实验必须同时记录：

```text
train environment set
training reward version
held-out environment set
evaluator/grader version
reward curve
held-out success curve
```

---

# 9　与 Durable Runtime 的交叉评测

长程 Agent 的可靠性不能靠“支持 resume”这句话证明。

未来 resilience benchmark：

```text
start task
→ execute side effects
→ kill worker at controlled point
→ lease expires / runtime restarts
→ replay/checkpoint recovery
→ continue
→ final-state grader
```

测量：

```text
completion rate
recovery latency
lost work
duplicate side effects
manual interventions
final artifact correctness
```

这会直接检验 durable runtime 的设计是否真正有价值。

---

# 10　原则

```text
Unit Test != Capability Eval
Final Answer != Final Environment State
Benchmark Score != Scientific Explanation
Model Quality != Runtime Quality
Agent Count != Multi-Agent Benefit
Long Context != Long-Horizon Reliability
Training Reward != Held-out Evaluation
```

任何“更强、更快、更可靠、更安全”的结论，最终都必须在本层找到**可重复任务、固定环境、grader、指标、artifact 与对照组**。
