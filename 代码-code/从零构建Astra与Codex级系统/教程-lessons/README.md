# 从 0 搭到 Astra-class / Codex-class：代码课程顺序

这组 Lab 是整本教材的“毕业实验”。原则是：**每学一个机制，就在源码里找到它；每写一段源码，就能回到公式、状态机和原始资料解释它。**

这里不再按“未来想做什么”编号，而按仓库中**已经存在的可运行实验**排序。

| Lesson | 主题 | 主要代码 | 核心验收 |
|---|---|---|---|
| 00 | 字符串 → tokenizer → Transformer → logits | `tokenizer.py`, `model.py` | shape / causal data flow 正确 |
| 01 | KV Cache / prefill / decode | `cache.py`, `engine.py` | cached logits 与 full logits 对齐 |
| 02 | structured tool call / Agent loop | `structured.py`, `tools.py`, `agent.py` | action → observation 闭环 |
| 03 | Codex 级仓库闭环 | `editing.py`, `repo_map.py`, `coding.py` | read → edit → verify |
| 04 | 从 SFT 到 DPO | `posttraining.py` | objective 数值与真实 optimizer step |
| 05 | Paged KV 与调度 | `paged_cache.py`, `scheduler.py`, `batch_executor.py`, `prefix_cache.py` | cache / batch parity + TTFT/TPOT 生命周期 |
| 06 | 从 Agent Loop 到 Durable Runtime | `durable.py`, `runtime_queue.py`, `runtime.py` | restart / replay / lease / checkpoint |
| 07 | 从 Verifier 到 Group-relative RL | `rlvr.py`, `verification.py` | reward group normalization / gradient direction |
| 08 | 副作用幂等与崩溃恢复 | `tool_journal.py`, `runtime.py` | completed side effect 不重复执行，in-doubt 不盲重试 |
| 09 | 实时 Steering、租约与 App Server | `steering.py`, `runtime_control.py`, `artifacts.py`, `app_server.py` | mid-turn steering / heartbeat / artifact / JSON-RPC |
| 10 | 可重放事件流与 HTTP 控制面 | `event_stream.py`, `http_app_server.py`, `app_server.py` | cursor replay / live harness events / real localhost HTTP |
| 11 | 控制面认证与授权 | `control_auth.py`, `app_server.py`, `runtime_queue.py` | bearer identity / method scope / thread scope / queue-level claim fencing |
| 12 | AGENTS.md 作用域与来源 | `instructions.py`, `coding.py`, `context.py` | hierarchy / override / budget / sibling negative control / provenance |
| 13 | SSE 实时事件推送与断线重连 | `sse_events.py`, `event_stream.py`, `control_auth.py` | cursor catch-up / Last-Event-ID / live delivery / auth-scoped stream |

对应文件：

```text
00-从字符串到Logits-string-to-logits.md
01-KVCache与增量解码-kv-cache.md
02-工具协议与Agent循环-tool-agent-loop.md
03-Codex级仓库闭环-coding-agent.md
04-从SFT到DPO-post-training.md
05-PagedKV与调度-serving-runtime.md
06-从AgentLoop到DurableRuntime-durable-agent.md
07-从Verifier到GroupRelativeRL-rlvr.md
08-副作用幂等与崩溃恢复-idempotency.md
09-实时Steering租约与AppServer-live-control-plane.md
10-可重放事件流与HTTP控制面-event-stream-http.md
11-控制面认证与授权-control-plane-auth.md
12-AGENTS作用域与指令来源-agents-md-provenance.md
13-SSE实时事件推送与断线重连-sse-reconnect.md
```

## 当前硬证据

截至 SSE 第一版进入全量回归的 Fast CPU CI run 138：

```text
108 passed, 1 warning in 8.59s
Ruff correctness lint: All checks passed
```

这意味着 Lesson 13 不只是协议说明：cursor replay、`Last-Event-ID`、thread/topic filter、连接后新事件实时到达、401/403 认证授权边界都已经进入自动测试。

## 课程成熟度规则

一个 Lesson 不因为 Markdown 已经写完就算完成。至少需要：

```text
理论/状态机
+ 对应源码
+ 自动测试
+ 明确失败边界
```

若声称与真实工业系统等价，还必须进一步增加：

```text
protocol parity
numerical parity
benchmark / real environment evidence
```

因此当前这些 Lesson 的定位是 **reference system curriculum**：把每一层做成可观察、可测试、可继续优化的最小正确实现，再逐渐替换成更强的工业级机制。
