# Lab 02　Tool Calling 与 Agent Loop：从“会说”到“会做”

> 对应源码：`structured.py`、`tools.py`、`agent.py`

## 1　Agent 不是一句 prompt

最小闭环是：

```text
Goal
 ↓
Model
 ↓
Action / Tool Call
 ↓
Environment
 ↓
Observation
 ↓
Model
 ↓
...
 ↓
Finish
```

ReAct 的核心价值之一就是把 reasoning 与 acting 放进同一个交互轨迹。原始资料：Yao et al., https://arxiv.org/abs/2210.03629

## 2　为什么要结构化 action

如果模型只是输出：

> “我现在去读 README。”

环境并不知道该执行什么。

本项目先规定最透明的协议：

```json
{"tool":"filesystem","arguments":{"action":"read","path":"README.md"}}
```

随后：

1. JSON parse；
2. schema validation；
3. tool dispatch；
4. result 转成 observation；
5. 写回消息历史。

生产系统可以用 grammar-constrained decoding 或 API 原生 tool-call channel，但状态机本质不变。

## 3　工具不是“函数名列表”

工具至少需要：

- name；
- description；
- JSON schema；
- executor；
- error model；
- permissions；
- timeout；
- observation serialization。

源码中的 `ToolSpec`、`ToolRegistry` 把这些接口显式化。

## 4　为什么失败也要变成 observation

真实 agent 会遇到：文件不存在、测试失败、命令超时、schema 错误。

正确行为不是 Python exception 直接把整个 agent 进程打死，而是把失败反馈给 policy，让它决定修正动作。

## 5　安全边界

当前 `FilesystemTool` 已阻止 `../` 越出仓库根目录；`ShellTool` 仍然只是教学级边界，不是安全 sandbox。

真正 Codex-class 系统必须进一步使用：

- container / VM isolation；
- seccomp / namespace；
- network policy；
- resource quota；
- secrets boundary；
- approval / permission model。

这部分后续单独实现，不能把“cwd=root”误写成安全隔离。
