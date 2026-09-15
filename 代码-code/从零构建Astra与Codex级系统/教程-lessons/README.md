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
| 14 | 受限进程沙箱与真实隔离边界 | `sandbox.py`, `coding.py` | argv-only / executable+env guard / timeout+rlimit / Linux no_new_privs / negative controls |
| 15 | Docker 容器隔离与逃逸负测试 | `docker_sandbox.py`, `coding.py` | host-secret invisibility / ro workspace+rootfs / network-none / CapEff=0 / NoNewPrivs=1 / Coding Agent container path |
| 16 | 物理 KV 块分配器与 Tensor Slab | `kv_block_allocator.py`, `kv_tensor_pool.py` | free-list / refcount / COW / fragmentation / physical K/V parity |
| 17 | 物理前缀缓存与计算复用 | `physical_prefix_cache.py`, `kv_tensor_pool.py` | exact hit 0 forward / suffix-only compute / block sharing / COW / source release |
| 18 | Page-Aware Attention 与异长 Decode | `page_aware_decode.py` | 禁止 materialize K/V / mixed length / logits+cache parity |
| 19 | Continuous Batching 调度闭环 | `continuous_batching.py`, `scheduler.py` | decode+new prefill 同轮推进 / terminal block release / TTFT+TPOT |
| 20 | 统一 Rollout Trace 与证据链 | `rollout_trace.py`, `rollout_trace_collector.py` | model/tool/steering/artifact/verifier provenance + hash-chain integrity |
| 21 | 持久 AgentGraph、Mailbox 与真正并行 Worker | `agent_graph.py`, `parallel_agents.py` | hierarchy/restart + message lease/reclaim + real worker overlap + one-task-per-agent |
| 22 | 并行 Worktree 候选、Reviewer 与 Merge | `coding_team.py`, `worktree.py`, `artifacts.py`, `verification.py` | isolated candidates + independent verification + patch artifact + review + merge rollback |

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
14-受限进程沙箱与真实隔离边界-process-sandbox.md
15-Docker容器隔离与逃逸负测试-container-sandbox.md
16-物理KV块分配器与前缀共享-kv-block-allocator.md
17-物理前缀缓存与计算复用-physical-prefix-cache.md
18-PageAwareAttention与异长Decode-page-aware-decode.md
19-ContinuousBatching调度闭环-continuous-batching.md
20-统一RolloutTrace与证据链-rollout-trace.md
21-持久AgentGraph与并行Mailbox-multi-agent-runtime.md
22-并行Worktree候选Reviewer与Merge-coding-team.md
```

## 当前硬证据

普通 Fast CPU Capstone CI run 219：

```text
165 passed, 14 skipped, 1 warning in 8.96s
Ruff correctness lint: All checks passed
```

Inference 主线现在已经实际串成：

```text
contiguous KV
→ logical paged KV
→ physical block allocator
→ physical K/V tensor slabs
→ physical prefix sharing
→ page-aware mixed-length decode
→ interleaved continuous-batching lifecycle
```

Agent / evaluation / multi-agent 主线现在已经串成：

```text
Durable Thread / WorkItem
→ model + tool runtime events
→ steering consumption
→ immutable artifacts
→ independent verifier
→ Rollout Trace
→ hash-chain integrity check

Persistent AgentGraph
→ durable mailbox
→ lease / reclaim / stale-owner rejection
→ parallel coordinator
→ one-task-per-agent execution
→ worker-specific Git worktree
→ candidate patch artifact
→ independent verifier
→ reviewer selection
→ merge apply / rollback
```

关键自动验收包括：

```text
physical block decode logits == full recomputation
exact physical prefix hit → zero extra model forward
partial longest-prefix hit → only suffix forward
page-aware decode → no historical K/V materialization
mixed cached lengths → one decode API
page-aware logits + per-layer K/V == full recomputation
old request decode + newly arrived request prefill → same serving iteration
finished/cancelled request → physical blocks released
scheduler metrics → TTFT / TPOT / total latency
rollout trace close/reopen → evidence persists
same work-item → cannot rebind to another thread
trace payload tampering → hash-chain verification fails
second trace sync → zero duplicate evidence
verifier result → explicit parent edge to artifact evidence
AgentGraph close/reopen → hierarchy/status/mailbox persist
message lease expiry → another consumer can reclaim
stale mailbox owner → cannot ACK
subtree cancellation → descendants stop without killing siblings
two parallel workers → barrier proves real wall-clock overlap
per-agent peak active tasks == 1
worker exception → failed agent isolated, surviving worker continues
same base commit → independent Git worktrees
bad candidate verifier FAIL → reviewer cannot select
selected binary patch → immutable artifact → git apply --check
post-merge verifier FAIL → reverse patch → coordinator tree clean
```

这里仍然是 reference serving/agent system，而不是 production vLLM/SGLang/Codex/Astra 等价物。当前 prefill 仍是逐请求 reference path，page-aware attention 仍有 Python request/block loops，也没有 chunked prefill、preemption/swap、GPU fused kernel 或真实并发 workload benchmark。Rollout Trace 的 SHA-256 链只能提供 tamper-evident 语义；如果攻击者能重写整个数据库并重新计算所有 hash，仍需要外部可信 digest anchor 才能建立更强审计边界。

当前 Multi-Agent 并发使用单进程 ThreadPool 作为**可观察的参考并发层**，持久 SQLite 状态由 coordinator thread 更新。Coding team 已经有 worker-specific Git worktree、candidate-patch artifact、独立 verifier、deterministic reviewer substrate 与单 patch merge rollback，但仍没有 remote actor runtime、LLM Reviewer、多 patch conflict graph、A2A 或跨机器执行。因此“并行已经存在”不等于“多 Agent 已证明优于单 Agent”。后续必须通过 1/2/4/8-agent controlled benchmark 给出 success/cost/wall-time/conflict 的实验结果。

Docker/Bubblewrap runtime tests 会按环境能力跳过，因此另有专门的 security workflow。Docker tested boundary 使用 shared Linux kernel，不能宣传成 VM-grade hostile-code containment；Bubblewrap backend 在 runner 禁止 namespace 时会显式 skip，而不是把“无法启动”当成“隔离成功”。

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

因此当前这些 Lesson 的定位是 **reference system curriculum**：先把每一层做成可观察、可测试、可继续优化的最小正确实现，再逐渐替换成更强的工业级机制。
