# 教材图谱：机制图、数据流图与时间线 / Figures

> 本目录采用**作者重绘**的方式组织教材插图。图的目的不是装饰，而是让读者在看到公式之前先建立数据流、张量形状（shape）和因果关系。  
> GitHub 原生支持 Mermaid；后续如导出 PDF/网页，可将这些图统一渲染为 SVG/PDF。  
> 每幅图均注明主要来源；图是本书根据来源重新组织后的解释性示意，不等同于原论文原图。

---

# 图 1　从 Transformer 到智能体 AI（Agentic AI）的技术主线

```mermaid
flowchart LR
    A[2017 Transformer<br/>变换器] --> B[GPT<br/>仅解码器 decoder-only]
    A --> C[BERT<br/>编码器 encoder]
    B --> D[GPT-2]
    D --> E[Scaling Laws<br/>规模定律]
    E --> F[GPT-3 / ICL<br/>上下文学习]
    F --> G[Instruction Tuning<br/>指令微调]
    G --> H[RLHF / ChatGPT<br/>人类反馈强化学习]
    E --> I[Chinchilla<br/>计算最优训练]
    I --> J[LLaMA / Open-weight<br/>开放权重生态]
    J --> K[LoRA / QLoRA<br/>参数高效微调]
    J --> L[RoPE / GQA / FlashAttention<br/>现代注意力组件]
    L --> M[MoE / Efficient Scaling<br/>混合专家与高效扩展]
    H --> N[DPO / Preference Optimization<br/>偏好优化]
    F --> O[Chain-of-Thought<br/>思维链]
    O --> P[Test-Time Compute<br/>测试时计算]
    P --> Q[RLVR / Reasoning RL<br/>推理强化学习]
    M --> Q
    Q --> R[Tool Use / Agentic RL<br/>工具使用与智能体强化学习]
    R --> S[Long-Horizon Agents<br/>长程智能体]
    L --> T[Native Multimodality<br/>原生多模态]
    T --> S
```

**主要来源**：Vaswani et al. (2017); Brown et al. (2020); Hoffmann et al. (2022); Ouyang et al. (2022); Touvron et al. (2023); DeepSeek-AI (2025)。

---

# 图 2　仅解码器 Transformer（Decoder-only Transformer）的主数据流

```mermaid
flowchart TD
    A[Token IDs<br/>词元编号 B×T] --> B[Token Embedding<br/>词元嵌入 B×T×C]
    B --> C[Position / RoPE<br/>位置信息]
    C --> D[RMSNorm / LayerNorm<br/>归一化]
    D --> E[Causal Self-Attention<br/>因果自注意力]
    E --> F[Residual Add<br/>残差相加]
    F --> G[Norm<br/>归一化]
    G --> H[MLP / SwiGLU<br/>前馈网络]
    H --> I[Residual Add<br/>残差相加]
    I --> J{More Blocks?<br/>还有更多层?}
    J -- 是 --> D
    J -- 否 --> K[Final Norm<br/>最终归一化]
    K --> L[LM Head<br/>语言模型输出头]
    L --> M[Logits<br/>未归一化分数 B×T×V]
    M --> N[Softmax / Sampling<br/>概率化与采样]
    N --> O[Next Token<br/>下一个词元]
```

**主要来源**：Vaswani et al. (2017); GPT/LLaMA 系列公开架构。

---

# 图 3　自注意力（Self-Attention）的张量形状图

```mermaid
flowchart LR
    X[X<br/>输入 B×T×C] --> Q[Q 查询<br/>B×h×T×d]
    X --> K[K 键<br/>B×h×T×d]
    X --> V[V 值<br/>B×h×T×d]
    Q --> S[QKᵀ / √d<br/>注意力分数 B×h×T×T]
    K --> S
    S --> M[Causal Mask<br/>因果掩码]
    M --> A[Softmax<br/>注意力权重 B×h×T×T]
    A --> O[A·V<br/>加权值 B×h×T×d]
    V --> O
    O --> C[Concat Heads<br/>拼接多头 B×T×C]
    C --> P[Output Projection<br/>输出投影 B×T×C]
```

公式：

$$
\mathrm{Attention}(Q,K,V)
=
\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V.
$$

**主要来源**：Vaswani et al., 2017, https://arxiv.org/abs/1706.03762

---

# 图 4　预训练 → 监督微调（SFT）→ 偏好学习 → 强化学习的后训练栈

```mermaid
flowchart TD
    A[Massive Raw / Curated Corpus<br/>大规模原始与清洗语料] --> B[Pretraining<br/>预训练：下一个词元预测]
    B --> C[Base Model<br/>基础模型]
    C --> D[Instruction Data<br/>指令数据]
    D --> E[SFT<br/>监督微调]
    E --> F[Instruction Model<br/>指令模型]
    F --> G{Preference / Reward Signal<br/>偏好或奖励信号}
    G --> H[Pairwise Preferences<br/>成对偏好]
    H --> I[DPO<br/>离线偏好优化]
    G --> J[Reward Model<br/>奖励模型]
    J --> K[PPO / Online RLHF<br/>在线强化学习]
    G --> L[Automatic Verifier<br/>自动验证器]
    L --> M[GRPO / RLVR / Reasoning RL<br/>推理强化学习]
    I --> N[Post-trained Model<br/>后训练模型]
    K --> N
    M --> N
```

**主要来源**：Ouyang et al. (2022); Rafailov et al. (2023); Shao et al. (2024); DeepSeek-AI (2025)。

---

# 图 5　规模扩展（Scaling）不是一个旋钮，而是三个资源变量

```mermaid
flowchart TD
    C[Training Compute C<br/>训练计算量] --> Q{Compute Budget Allocation<br/>计算预算分配}
    Q --> N[Model Parameters N<br/>模型参数量]
    Q --> D[Training Tokens D<br/>训练词元数]
    N --> L[Validation Loss / Capability<br/>验证损失与能力]
    D --> L
    R[Data Quality & Distribution<br/>数据质量与分布] --> L
    A[Architecture & Optimizer<br/>架构与优化器] --> L
```

粗略地，对稠密 Transformer（dense Transformer）：

$$
C\propto ND.
$$

但最终损失（loss）不只由 $N,D,C$ 决定，还受数据质量、架构、优化器和训练稳定性影响。

**主要来源**：Kaplan et al. (2020); Hoffmann et al. (2022)。

---

# 图 6　MHA → GQA → MQA：为什么键值缓存（KV Cache）会下降

```mermaid
flowchart TB
    subgraph MHA[Multi-Head Attention<br/>多头注意力 MHA]
      Q1[Q heads: h<br/>h 个查询头] --> K1[K heads: h<br/>h 个键头]
      K1 --> V1[V heads: h<br/>h 个值头]
    end

    subgraph GQA[Grouped-Query Attention<br/>分组查询注意力 GQA]
      Q2[Q heads: h<br/>h 个查询头] --> K2[K/V heads: g, g < h<br/>g 组键值头]
      K2 --> V2[Queries share KV by groups<br/>查询按组共享 KV]
    end

    subgraph MQA[Multi-Query Attention<br/>多查询注意力 MQA]
      Q3[Q heads: h<br/>h 个查询头] --> K3[1 K head<br/>1 个键头]
      K3 --> V3[1 V head<br/>1 个值头]
    end
```

KV Cache 的每层规模可粗略理解为：

$$
O(B\cdot T\cdot h_{kv}\cdot d_h),
$$

因此降低 $h_{kv}$ 可以显著减少解码（decode）阶段缓存。

**主要来源**：Shazeer (2019); Ainslie et al. (2023)。

---

# 图 7　混合专家（MoE）：总参数大，但每个词元只激活部分专家

```mermaid
flowchart LR
    X[Token Hidden State<br/>词元隐藏状态] --> R[Router<br/>路由器]
    R --> E1[Expert 1<br/>专家 1]
    R --> E2[Expert 2<br/>专家 2]
    R --> E3[Expert 3<br/>专家 3]
    R --> E4[Expert ...<br/>更多专家]
    R --> EN[Expert N<br/>专家 N]
    E1 --> G[Weighted Combine<br/>加权合并]
    E2 --> G
    E3 --> G
    E4 --> G
    EN --> G
    G --> Y[Output Hidden State<br/>输出隐藏状态]
```

Top-k 路由（routing）的核心思想：

$$
\text{Total Parameters}\gg\text{Activated Parameters per Token}.
$$

真正的系统难点包括专家负载均衡（expert load balance）、全互联通信（all-to-all communication）、容量（capacity）、路由稳定性（routing stability）与专家并行（expert parallelism）。

**主要来源**：Shazeer et al. (2017); Switch Transformer (2021); Mixtral (2024); DeepSeekMoE (2024)。

---

# 图 8　键值缓存（KV Cache）：训练并行与自回归推理的分水岭

```mermaid
sequenceDiagram
    participant P as 提示词元 Prompt Tokens
    participant M as Transformer 模型
    participant C as KV Cache 键值缓存
    participant O as 输出 Output

    P->>M: Prefill：一次处理全部提示词元
    M->>C: 为每一层保存 K/V
    M->>O: 输出下一个词元的 logits
    O->>M: 生成词元 t+1
    C->>M: 复用历史 K/V
    M->>C: 只追加新的 K/V
    M->>O: 输出词元 t+2 的 logits
    O->>M: 生成词元 t+2
    C->>M: 继续复用缓存历史
```

没有 KV Cache 时，生成第 $t$ 个词元会重复计算之前所有词元的 K/V；有缓存后，历史 K/V 被复用，但显存占用随上下文长度增长。

**主要来源**：标准 Transformer 自回归推理；现代推理服务（serving）系统文献，尤其 vLLM / PagedAttention。

---

# 图 9　推理强化学习（Reasoning RL / RLVR）的闭环

```mermaid
flowchart LR
    P[Prompt / Problem<br/>提示与问题] --> M[Policy Model<br/>策略模型]
    M --> R1[Rollout 1<br/>轨迹 1]
    M --> R2[Rollout 2<br/>轨迹 2]
    M --> R3[Rollout ...<br/>更多轨迹]
    R1 --> V[Verifier / Reward<br/>验证器与奖励]
    R2 --> V
    R3 --> V
    V --> A[Advantages / Relative Scores<br/>优势与相对得分]
    A --> U[Policy Update<br/>策略更新]
    U --> M
```

当验证器（verifier）是数学答案、单元测试或环境成功信号时，训练可以在不依赖人工逐条打分的情况下扩展。但 verifier 设计本身会成为新的瓶颈和可攻击面。

**主要来源**：DeepSeekMath (2024); DeepSeek-R1 (2025); 过程/结果监督（process/outcome supervision）相关文献。

---

# 图 10　智能体（Agent）不等于 LLM：它是一个闭环系统

```mermaid
flowchart TD
    U[User / Goal<br/>用户与目标] --> M[LLM Policy / Reasoner<br/>模型策略与推理器]
    M --> A[Action / Tool Call<br/>动作与工具调用]
    A --> T[Tools<br/>浏览器 / 代码 / 搜索 / API]
    T --> E[Environment Result<br/>环境返回结果]
    E --> O[Observation<br/>观测]
    O --> M
    M --> MEM[Memory / State<br/>记忆与状态]
    MEM --> M
    M --> F[Final Answer / Task Completion<br/>最终回答与任务完成]
```

智能体评测必须拆分：

- 基础模型（base model）能力；
- 工具接口（tool interface）；
- 提示与脚手架（prompt / scaffold）；
- 规划与记忆（planning / memory）；
- 环境（environment）；
- 重试预算（retry budget）；
- 测试时计算（test-time compute）。

**主要来源**：WebGPT (2021); ReAct (2022/2023); Toolformer (2023) 及后续智能体（agent）文献。

---

# 图 11　检索增强生成（RAG）：检索与生成是两个子系统

```mermaid
flowchart LR
    D[Documents<br/>文档] --> C[Chunking<br/>切块]
    C --> E[Embedding<br/>嵌入]
    E --> I[Vector / Hybrid Index<br/>向量或混合索引]
    Q[User Query<br/>用户查询] --> QE[Query Encoding<br/>查询编码]
    QE --> RET[Retriever<br/>检索器]
    I --> RET
    RET --> RR[Reranker<br/>重排序器]
    RR --> CTX[Selected Context<br/>选中上下文]
    Q --> GEN[Generator LLM<br/>生成模型]
    CTX --> GEN
    GEN --> ANS[Answer + Attribution<br/>回答与来源归因]
```

RAG 失败可能来自检索（retrieval），也可能来自生成（generation）；两者必须分别评测。

**主要来源**：Lewis et al. (2020); DPR (2020); RETRO (2021/2022)。

---

# 图 12　大模型训练的多维并行

```mermaid
flowchart TD
    M[One Huge Model + Dataset<br/>一个超大模型与数据集] --> DP[Data Parallel<br/>数据并行]
    M --> TP[Tensor Parallel<br/>张量并行]
    M --> PP[Pipeline Parallel<br/>流水线并行]
    M --> CP[Sequence / Context Parallel<br/>序列 / 上下文并行]
    M --> EP[Expert Parallel<br/>专家并行]
    DP --> H[Hybrid Parallel Training<br/>混合并行训练]
    TP --> H
    PP --> H
    CP --> H
    EP --> H
```

现实中的万亿参数级（trillion-scale）或大型 MoE 训练通常是多维并行组合，并同时处理优化器状态（optimizer states）、激活内存（activation memory）、通信拓扑（communication topology）和容错（fault tolerance）。

**主要来源**：Megatron-LM; ZeRO; FSDP; GShard / MoE systems。

---

# 插图规范

后续正文新增图时遵循：

1. **先画数据流，再写公式**；
2. 张量形状（shape）尽可能标明；
3. 区分训练与推理；
4. 区分“数学变化”和“系统优化”；
5. 重绘论文机制时标 `据 X et al. (year) 重绘/整理`；
6. 不直接复制论文整图，除非许可证明确允许且确有必要；
7. 不使用来历不明的网络示意图；
8. 时效性产品架构图必须写版本和日期。

这套图谱会随着各章扩写继续增加。
