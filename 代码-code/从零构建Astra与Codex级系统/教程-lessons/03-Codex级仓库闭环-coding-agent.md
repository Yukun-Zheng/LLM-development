# Lab 03　Codex-class 的第一个真实闭环：Read → Edit → Test → Inspect → Fix

> 对应源码：`coding.py`、`tools.py`、`agent.py`

一个 coding agent 的价值不在于“能生成代码片段”，而在于它能在真实仓库状态上闭环工作。

## 1　最小工具面

第一阶段只给三个 primitive：

```text
filesystem
shell
git
```

这已经足够完成：

```text
搜索代码
→ 阅读上下文
→ 写入修改
→ pytest / lint / build
→ 阅读 traceback
→ 再修改
→ git diff
→ 最终验证
```

## 2　为什么 verifier 必须是一等公民

错误的 coding agent：

```text
写代码 → “应该没问题” → 完成
```

正确闭环：

```text
写代码
 ↓
运行 test / typecheck / lint / build
 ↓
读取真实 exit code 与日志
 ↓
失败则继续修
 ↓
成功后才能声称完成
```

因此系统 prompt 明确禁止“没有 observation 就声称测试通过”。

## 3　下一层能力

后续逐个自己实现：

- unified diff / patch application；
- syntax-aware repo map；
- tree-sitter / LSP；
- test selection；
- Git worktree；
- parallel workers；
- reviewer agent；
- merge conflict handling；
- long-running task checkpoint；
- browser / issue / PR context；
- sandboxed computer use。

## 4　评测不能只看 HumanEval

代码补全 benchmark 不能代表仓库级软件工程。最终评测要把：

- unit-level coding；
- repo-level issue resolution；
- test pass rate；
- regression；
- patch minimality；
- wall-clock / tool calls；
- recovery after failure

分开测。

SWE-bench 原始资料：Jimenez et al., 2023/2024, https://arxiv.org/abs/2310.06770
