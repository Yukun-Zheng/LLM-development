# 教学代码 / Code

本目录存放与教材正文配套的**可运行代码**。目标不是调用现成框架“跑通即可”，而是把关键机制写到足够透明，让读者能够把公式、张量形状（shape）与代码逐行对应起来。

## 终极目标

整本书的代码终点不是若干孤立 demo，而是：

> **从空目录开始，逐层写出一个 Astra-class 通用智能系统与 Codex-class 编程智能体。训练权重本身不作为目标，但模型结构、权重加载、推理、KV Cache、工具调用、Agent loop、代码执行、Git/worktree、长程记忆、browser/computer interface、多智能体与评测都要亲手实现。**

理论有两条主线共同汇入这个工程：

- **LLM 模型主线**：[`../教材-book/README.md`](../教材-book/README.md)
- **智能体系统主线**：[`../智能体-agent/README.md`](../智能体-agent/README.md)

总工程入口：

- [`从零构建Astra与Codex级系统/README.md`](从零构建Astra与Codex级系统/README.md)
- 理论总章：[`../教材-book/18-从零构建Astra与Codex级系统-capstone.md`](../教材-book/18-从零构建Astra与Codex级系统-capstone.md)

## 当前代码

| 文件 / 项目 | 中文说明 | 覆盖内容 |
|---|---|---|
| [`minigpt.py`](minigpt.py) | **从零实现一个小型 GPT** | 字符级 tokenizer、Embedding、RMSNorm、因果多头注意力、SwiGLU、Transformer Block、交叉熵训练、自回归生成 |
| [`从零构建Astra与Codex级系统/`](从零构建Astra与Codex级系统/) | **全书终极工程** | Model Runtime → Inference Engine → Tool Runtime → Agent Planning / Verification → Codex-class → Astra-class → Multi-Agent → Evals |

当前终极工程已经不仅有最小 Agent loop，还加入了：

- typed task DAG / `PlanGraph`；
- dependency-aware ready / blocked state；
- external verifier protocol；
- file / command / composite verification；
- coordinator primitive；
- evidence handoff。

这些模块对应 `智能体-agent/` 中的规划、验证和多智能体理论，而不是孤立脚本。

## 代码阅读原则

教材代码优先保证：

1. **数据流清楚**：每一步输入/输出 shape 尽量可追踪；
2. **机制透明**：核心模块尽量自己实现，而不是把关键逻辑藏在高层 API 中；
3. **可以运行**：示例应能作为最小实验直接执行；
4. **Reference parity**：优化实现必须能与朴素实现或官方公开 reference 做数值/行为对齐；
5. **理论对应代码**：每一个重要公式都应能定位到真实实现；
6. **Agent 必须可验证**：不能把模型一句“完成了”当成任务成功，必须保存 environment observation / verifier evidence；
7. **与工业实现区分**：教学实现不等同于生产级高性能实现，涉及 FlashAttention、vLLM、Megatron 等系统优化时会在正文中单独解释。

后续代码不再以“多放几个独立脚本”为主要组织方式，而是优先推动 `从零构建Astra与Codex级系统/` 这个主工程逐阶段完成。
