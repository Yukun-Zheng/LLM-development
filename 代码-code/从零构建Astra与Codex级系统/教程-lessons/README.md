# 从 0 搭到 Astra-class / Codex-class：代码课程顺序

这组 Lab 是整本教材的“毕业实验”。原则是：**每学一个机制，就在源码里找到它；每写一段源码，就能回到公式和原始论文解释它。**

| Lab | 主题 | 主要代码 | 验收 |
|---|---|---|---|
| 00 | 字符串 → tokenizer → Transformer → logits | `tokenizer.py`, `model.py` | shape 全部正确 |
| 01 | KV Cache / prefill / decode | `cache.py`, `engine.py` | cached logits 与 full logits 对齐 |
| 02 | structured tool call / agent loop | `structured.py`, `tools.py`, `agent.py` | 工具失败也能成为 observation |
| 03 | repository coding loop | `editing.py`, `repo_map.py`, `coding.py` | read → edit → verify 闭环 |
| 04 | checkpoint loading | `weights.py` | shape mapping / missing key 可审计 |
| 05 | memory / compaction | `memory.py` | 长任务跨进程恢复 |
| 06 | Git worktree workers | `worktree.py` | 两个任务隔离修改 |
| 07 | browser / computer adapters | `general_tools.py` + 后续模块 | 可验证环境 observation |
| 08 | high-performance inference | 后续 engine | batching / paged cache |
| 09 | coding intelligence | 后续 tree-sitter/LSP | repo-level issue 修复 |
| 10 | multi-agent | 后续 scheduler | task graph + reviewer |
| 11 | evaluation | 后续 evals | SWE-bench / agent task metrics |

目前 Lab 00–03 已有完整源码和测试；04–07 已有第一批 primitive，后续继续提升到工业级实现。
