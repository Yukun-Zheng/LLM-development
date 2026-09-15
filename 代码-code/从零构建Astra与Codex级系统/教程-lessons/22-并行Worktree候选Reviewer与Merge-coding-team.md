# Lesson 22：并行 Worktree 候选、Independent Verifier、Reviewer 与 Merge

> Lesson 21 解决了“多个 Agent 如何持久存在并真正并行”。本课进一步解决 Codex-class coding team 最关键的工程问题：**多个 worker 不能并发写同一个 working tree；每个候选修改必须先变成独立 patch artifact，再经过外部验证与 reviewer 选择，最后才允许进入 coordinator worktree。**

---

## 1　为什么共享一个工作目录会毁掉多 Agent 语义

假设两个 worker 同时做：

```text
worker-A → 修改 parser.py
worker-B → 修改 parser.py
```

如果它们共享同一个 working tree，会立刻出现：

```text
谁写的哪一行？
B 的测试是在 A 的改动上跑的吗？
最终 diff 属于哪个 Agent？
失败时该回滚谁？
Reviewer 审的是哪个候选？
```

因此 coding team 必须先建立：

```text
same base commit
      │
      ├─ worktree-A / branch agent/A
      ├─ worktree-B / branch agent/B
      └─ worktree-C / branch agent/C
```

每个 Agent 的 filesystem state 独立，Git object database 仍可共享。

---

## 2　当前 Reference Pipeline

源码：

```text
src/astra_codex/coding_team.py
src/astra_codex/worktree.py
src/astra_codex/parallel_agents.py
src/astra_codex/artifacts.py
src/astra_codex/verification.py
```

完整数据流：

```text
Persistent AgentGraph
        ↓
ParallelAgentCoordinator
        ↓
worker-specific Git Worktree
        ↓
worker modifies files
        ↓
git diff --binary BASE
        ↓
Independent CommandVerifier
        ↓
immutable candidate-patch Artifact
        ↓
ReviewerDecision
        ↓
git apply --check
        ↓
apply selected patch to coordinator tree
        ↓
post-merge verification
        ├─ PASS → keep patch
        └─ FAIL → reverse patch / rollback
```

这里最重要的原则是：

> **Worker 的自然语言 final answer 不构成 merge 权限。**

---

## 3　候选对象是什么

当前 `CodingCandidate` 包含：

```text
candidate
├─ task_id
├─ agent_id
├─ branch
├─ worktree_path
├─ patch_artifact_id
├─ verification
├─ worker_summary
└─ worker_evidence
```

注意真正可 merge 的不是：

```text
worker_summary
```

而是：

```text
patch_artifact_id
+ verification
```

自然语言 summary 只帮助人/Reviewer 理解意图。

---

## 4　为什么 Patch 必须进入 Artifact Store

worker 结束时我们执行：

```text
git diff --binary <base_commit> --
```

再把 patch bytes 写入 content-addressed `ArtifactStore`。

metadata 记录：

```text
task_id
agent_id
base_commit
verification_verdict
```

于是 candidate patch 具有：

```text
immutable bytes
SHA-256
producer identity
base identity
verification result
```

它比“当前 worktree 里大概还有一份改动”强很多。

后续即使 worker worktree 被删除，Reviewer 仍能基于 artifact 审查候选。

---

## 5　Independent Verifier 是 Merge Gate

worker callback 完成后，框架自己运行：

```python
CommandVerifier(worktree_path, verify_argv).verify()
```

因此：

```text
Agent says "fixed"
        ↓
independent command returns non-zero
        ↓
candidate.verification = FAIL
        ↓
Reviewer eligibility = false
```

测试专门构造：

```text
candidate_good → answer.txt = 42 → verifier PASS
candidate_bad  → answer.txt = 41 → verifier FAIL
```

Reviewer 只能选择 PASS candidate。

---

## 6　Reviewer 目前是什么

当前 `review(...)` 首先做硬过滤：

```text
verification != PASS
        ↓
reject
```

通过者进入 eligible set。

如果没有额外 score：

```text
选择第一个 passing candidate
```

如果提供：

```python
score(candidate) -> float
```

则：

```text
max(score)
```

决定选择。

这只是 deterministic reviewer substrate，不等于已经实现 frontier LLM Reviewer。

后续 Reviewer 可以把：

```text
patch artifact
hidden-test evidence
static analysis
code review model
risk policy
```

组合成真正 review score。

---

## 7　为什么 merge 前还要 `git apply --check`

Candidate 在自己的 worktree 中成立，不代表 coordinator 当前 tree 一定还能接收它。

因此 merge 分两步：

```text
git apply --check patch
        ↓
只有 check 成功
        ↓
git apply patch
```

这至少检测：

```text
context mismatch
already changed lines
basic patch conflict
```

后续多候选组合 merge 还需要更强的 dependency/conflict analysis。

---

## 8　Post-Merge Verification 为什么必须再跑一次

即使 candidate worktree 里 verifier PASS，也必须在 coordinator tree 再跑：

```text
candidate verification PASS
        ↓
patch apply
        ↓
post-merge verification
```

原因包括：

```text
coordinator tree 可能已经包含其他 accepted patch
组合后行为可能改变
环境可能不同
merge 本身可能引入错误
```

Reference implementation 如果 post-merge verifier FAIL：

```text
git apply -R patch
```

把刚才的 patch 反向应用，恢复 coordinator tree。

测试明确验证：

```text
candidate verifier = permissive PASS
merge verifier      = strict FAIL
        ↓
MergeResult.applied = false
        ↓
answer.txt 恢复到 base 内容
        ↓
git diff == empty
```

---

## 9　为什么这还不是“自动 merge 多个 Agent 的成果”

当前 Reviewer 选：

```text
one candidate
```

而不是把多个 passing patch 自动拼在一起。

这是刻意的。

多 patch merge 需要进一步定义：

```text
file overlap
symbol overlap
dependency order
semantic conflict
test interaction
merge conflict resolution
```

如果太早写成：

```text
for patch in patches:
    git apply(patch)
```

只是在制造难以追责的组合状态。

因此当前顺序是：

```text
single-candidate review/merge correctness
        ↓
independent non-overlapping patch merge
        ↓
conflict-aware merge planner
        ↓
Reviewer / Merge Agent
```

---

# 10　如何接 Rollout Trace

下一步应该把 coding team 的关键决策全部变成 trace edge：

```text
worker task assigned
      ↓
candidate patch artifact
      ↓
independent verification
      ↓
reviewer decision
      ↓
merge attempt
      ↓
post-merge verification
```

最终可以形成：

```text
artifact A ─┐
verifier A ─┼→ reviewer.selected(A)
artifact B ─┘
                │
                ▼
           merge.applied
                │
                ▼
       post_merge_verification
```

这样“为什么最终用了 A 而没有用 B”可以被完整审计。

---

# 11　与真正 Codex-class 团队还差哪些层

当前已经有：

```text
persistent Agent identity
mailbox / lease
real parallel callbacks
Git worktree isolation
candidate patch artifact
independent verifier
reviewer selection substrate
merge apply + rollback
```

仍未完成：

```text
tree-sitter / LSP semantic context
hidden tests / SWE-bench adapter
multiple patch conflict graph
LLM Reviewer
merge agent
cross-candidate dependency discovery
process/remote worker execution
A2A transport
per-agent model/token/tool accounting
1/2/4/8-agent controlled benchmark
```

---

# 12　测试验收

当前 `tests/test_coding_team.py` 验证两个关键闭环。

### 闭环 A：好/坏候选

```text
same base commit
├─ candidate_good → modifies answer.txt to 42 → PASS
└─ candidate_bad  → modifies answer.txt to 41 → FAIL

Reviewer
→ selects only candidate_good

Merge
→ git apply --check
→ git apply
→ verifier PASS
→ coordinator answer.txt == 42
```

### 闭环 B：Merge-time Regression

```text
candidate verifier PASS
        ↓
Reviewer selects candidate
        ↓
apply patch
        ↓
stricter post-merge verifier FAIL
        ↓
reverse patch
        ↓
coordinator tree returns clean
```

这两条测试把一个非常重要的工程原则落成代码：

> **“一个 Agent 的候选方案通过它自己的执行阶段”与“这个候选最终允许进入主工作树”是两个不同的验证事件。**

---

# 13　下一步

从当前状态继续，应按：

```text
Worktree candidate / reviewer / merge       ← 本课
        ↓
Reviewer + merge decisions → Rollout Trace
        ↓
Multi-file hidden-test fixtures
        ↓
SWE-bench adapter
        ↓
semantic repo map (tree-sitter / LSP)
        ↓
conflict-aware multi-patch merge
        ↓
1/2/4/8-agent controlled benchmark
```

最终只有在 controlled benchmark 中真实看到：

```text
success ↑
wall time ↓
且 cost / conflict / duplicate work 可接受
```

我们才有资格说：

> **多 Agent 对这类 coding task 真正产生了系统收益。**
