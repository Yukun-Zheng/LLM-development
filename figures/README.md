# 教材图谱：机制图、数据流图与时间线

> 本目录采用**作者重绘**的方式组织教材插图。图的目的不是装饰，而是让读者在看到公式之前先建立数据流、shape 和因果关系。  
> GitHub 原生支持 Mermaid；后续如导出 PDF/网页，可将这些图统一渲染为 SVG/PDF。  
> 每幅图均注明主要来源；图是本书根据来源重新组织后的解释性示意，不等同于原论文原图。

---

# 图 1　从 Transformer 到 Agentic AI 的技术主线

```mermaid
flowchart LR
    A[2017 Transformer] --> B[GPT / decoder-only]
    A --> C[BERT / encoder]
    B --> D[GPT-2]
    D --> E[Scaling Laws]
    E --> F[GPT-3 / ICL]
    F --> G[Instruction Tuning]
    G --> H[RLHF / ChatGPT]
    E --> I[Chinchilla]
    I --> J[LLaMA / Open-weight]
    J --> K[LoRA / QLoRA]
    J --> L[RoPE / GQA / FlashAttention]
    L --> M[MoE / Efficient Scaling]
    H --> N[DPO / Preference Optimization]
    F --> O[Chain-of-Thought]
    O --> P[Test-time Compute]
    P --> Q[RLVR / Reasoning RL]
    M --> Q
    Q --> R[Tool Use / Agentic RL]
    R --> S[Long-horizon Agents]
    L --> T[Native Multimodality]
    T --> S
```

**主要来源**：Vaswani et al. (2017); Brown et al. (2020); Hoffmann et al. (2022); Ouyang et al. (2022); Touvron et al. (2023); DeepSeek-AI (2025)。

---

# 图 2　Decoder-only Transformer 的主数据流

```mermaid
flowchart TD
    A[Token IDs\nB×T] --> B[Token Embedding\nB×T×C]
    B --> C[+ Position / RoPE]
    C --> D[RMSNorm / LayerNorm]
    D --> E[Causal Self-Attention]
    E --> F[Residual Add]
    F --> G[Norm]
    G --> H[MLP / SwiGLU]
    H --> I[Residual Add]
    I --> J{More Blocks?}
    J -- yes --> D
    J -- no --> K[Final Norm]
    K --> L[LM Head]
    L --> M[Logits\nB×T×V]
    M --> N[Softmax / Sampling]
    N --> O[Next Token]
```

**主要来源**：Vaswani et al. (2017); GPT/LLaMA 系列公开架构。

---

# 图 3　Self-Attention 的 shape 图

```mermaid
flowchart LR
    X[X\nB×T×C] --> Q[Q\nB×h×T×d]
    X --> K[K\nB×h×T×d]
    X --> V[V\nB×h×T×d]
    Q --> S[QKᵀ / √d\nB×h×T×T]
    K --> S
    S --> M[Causal Mask]
    M --> A[Softmax\nB×h×T×T]
    A --> O[A·V\nB×h×T×d]
    V --> O
    O --> C[Concat Heads\nB×T×C]
    C --> P[Output Projection\nB×T×C]
```

公式：

\[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V.
\]

**主要来源**：Vaswani et al., 2017, https://arxiv.org/abs/1706.03762

---

# 图 4　预训练 → SFT → Preference → RL 的后训练栈

```mermaid
flowchart TD
    A[Massive Raw / Curated Corpus] --> B[Pretraining\nNext-token Prediction]
    B --> C[Base Model]
    C --> D[Instruction Data]
    D --> E[SFT]
    E --> F[Instruction Model]
    F --> G{Preference / Reward Signal}
    G --> H[Pairwise Preferences]
    H --> I[DPO / Offline Preference Optimization]
    G --> J[Reward Model]
    J --> K[PPO / Online RLHF]
    G --> L[Automatic Verifier]
    L --> M[GRPO / RLVR / Reasoning RL]
    I --> N[Post-trained Model]
    K --> N
    M --> N
```

**主要来源**：Ouyang et al. (2022); Rafailov et al. (2023); Shao et al. (2024); DeepSeek-AI (2025)。

---

# 图 5　Scaling 不是一个旋钮，而是三个资源变量

```mermaid
flowchart TD
    C[Training Compute C] --> Q{Compute Budget Allocation}
    Q --> N[Model Parameters N]
    Q --> D[Training Tokens D]
    N --> L[Validation Loss / Capability]
    D --> L
    R[Data Quality & Distribution] --> L
    A[Architecture & Optimizer] --> L
```

粗略地，对 dense Transformer：

\[
C\propto ND.
\]

但最终 loss 不只由 \(N,D,C\) 决定，还受数据质量、架构、优化器和训练稳定性影响。

**主要来源**：Kaplan et al. (2020); Hoffmann et al. (2022)。

---

# 图 6　MHA → GQA → MQA：KV Cache 为什么会下降

```mermaid
flowchart TB
    subgraph MHA[Multi-Head Attention]
      Q1[Q heads: h] --> K1[K heads: h]
      K1 --> V1[V heads: h]
    end

    subgraph GQA[Grouped-Query Attention]
      Q2[Q heads: h] --> K2[K/V heads: g, g < h]
      K2 --> V2[Queries share KV by groups]
    end

    subgraph MQA[Multi-Query Attention]
      Q3[Q heads: h] --> K3[1 K head]
      K3 --> V3[1 V head]
    end
```

KV Cache 的每层规模可粗略理解为：

\[
O(B\cdot T\cdot h_{kv}\cdot d_h),
\]

因此降低 \(h_{kv}\) 可以显著减少 decode 阶段缓存。

**主要来源**：Shazeer (2019); Ainslie et al. (2023)。

---

# 图 7　MoE：总参数大，但每个 token 只激活部分专家

```mermaid
flowchart LR
    X[Token Hidden State] --> R[Router]
    R --> E1[Expert 1]
    R --> E2[Expert 2]
    R --> E3[Expert 3]
    R --> E4[Expert ...]
    R --> EN[Expert N]
    E1 --> G[Weighted Combine]
    E2 --> G
    E3 --> G
    E4 --> G
    EN --> G
    G --> Y[Output Hidden State]
```

Top-k routing 的核心思想：

\[
\text{Total Parameters}\gg\text{Activated Parameters per Token}.
\]

真正的系统难点包括 expert load balance、all-to-all communication、capacity、routing stability 与 expert parallelism。

**主要来源**：Shazeer et al. (2017); Switch Transformer (2021); Mixtral (2024); DeepSeekMoE (2024)。

---

# 图 8　KV Cache：训练并行与自回归推理的分水岭

```mermaid
sequenceDiagram
    participant P as Prompt tokens
    participant M as Transformer
    participant C as KV Cache
    participant O as Output

    P->>M: Prefill all prompt tokens
    M->>C: Store K/V for every layer
    M->>O: logits for next token
    O->>M: generated token t+1
    C->>M: reuse previous K/V
    M->>C: append new K/V only
    M->>O: logits for token t+2
    O->>M: generated token t+2
    C->>M: reuse cached history
```

没有 KV Cache 时，生成第 \(t\) 个 token 会重复计算之前所有 token 的 K/V；有 cache 后，历史 K/V 被复用，但显存占用随上下文长度增长。

**主要来源**：标准 Transformer 自回归推理；现代 serving 系统文献，尤其 vLLM / PagedAttention。

---

# 图 9　Reasoning RL / RLVR 的闭环

```mermaid
flowchart LR
    P[Prompt / Problem] --> M[Policy Model]
    M --> R1[Rollout 1]
    M --> R2[Rollout 2]
    M --> R3[Rollout ...]
    R1 --> V[Verifier / Reward]
    R2 --> V
    R3 --> V
    V --> A[Advantages / Relative Scores]
    A --> U[Policy Update]
    U --> M
```

当 verifier 是数学答案、单元测试或环境成功信号时，训练可以在不依赖人工逐条打分的情况下扩展。但 verifier 设计本身会成为新的瓶颈和可攻击面。

**主要来源**：DeepSeekMath (2024); DeepSeek-R1 (2025); process/outcome supervision literature。

---

# 图 10　Agent 不等于 LLM：它是一个闭环系统

```mermaid
flowchart TD
    U[User / Goal] --> M[LLM Policy / Reasoner]
    M --> A[Action / Tool Call]
    A --> T[Tools\nBrowser / Code / Search / API]
    T --> E[Environment Result]
    E --> O[Observation]
    O --> M
    M --> MEM[Memory / State]
    MEM --> M
    M --> F[Final Answer / Task Completion]
```

Agent 评测必须拆分：

- base model 能力；
- tool interface；
- prompt/scaffold；
- planning / memory；
- environment；
- retry budget；
- test-time compute。

**主要来源**：WebGPT (2021); ReAct (2022/2023); Toolformer (2023) 及后续 agent literature。

---

# 图 11　RAG：检索与生成是两个子系统

```mermaid
flowchart LR
    D[Documents] --> C[Chunking]
    C --> E[Embedding]
    E --> I[Vector / Hybrid Index]
    Q[User Query] --> QE[Query Encoding]
    QE --> RET[Retriever]
    I --> RET
    RET --> RR[Reranker]
    RR --> CTX[Selected Context]
    Q --> GEN[Generator LLM]
    CTX --> GEN
    GEN --> ANS[Answer + Attribution]
```

RAG 失败可能来自 retrieval，也可能来自 generation；两者必须分别评测。

**主要来源**：Lewis et al. (2020); DPR (2020); RETRO (2021/2022)。

---

# 图 12　大模型训练的多维并行

```mermaid
flowchart TD
    M[One Huge Model + Dataset] --> DP[Data Parallel]
    M --> TP[Tensor Parallel]
    M --> PP[Pipeline Parallel]
    M --> CP[Sequence / Context Parallel]
    M --> EP[Expert Parallel]
    DP --> H[Hybrid Parallel Training]
    TP --> H
    PP --> H
    CP --> H
    EP --> H
```

现实中的 trillion-scale 或大 MoE 训练通常是多维并行组合，并同时处理 optimizer states、activation memory、communication topology 和 fault tolerance。

**主要来源**：Megatron-LM; ZeRO; FSDP; GShard / MoE systems。

---

# 插图规范

后续正文新增图时遵循：

1. **先画数据流，再写公式**；
2. shape 尽可能标明；
3. 区分训练与推理；
4. 区分“数学变化”和“系统优化”；
5. 重绘论文机制时标 `据 X et al. (year) 重绘/整理`；
6. 不直接复制论文整图，除非许可证明确允许且确有必要；
7. 不使用来历不明的网络示意图；
8. 时效性产品架构图必须写版本和日期。

这套图谱会随着各章扩写继续增加。