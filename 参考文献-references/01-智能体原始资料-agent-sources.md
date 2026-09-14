# 智能体原始资料索引 / Agent Primary-Source Map

> **用途**：为 `智能体-agent/` 提供一手资料脊柱。优先原论文、原始 benchmark、官方代码与系统文档；二手综述只用于导航。

---

# A　工具使用与闭环交互

1. Nakano et al. (2021), **WebGPT: Browser-assisted question-answering with human feedback**  
   https://arxiv.org/abs/2112.09332
2. Karpas et al. (2022), **MRKL Systems: A modular, neuro-symbolic architecture that combines large language models, external knowledge sources and discrete reasoning**  
   https://arxiv.org/abs/2205.00445
3. Yao et al. (2022/2023), **ReAct: Synergizing Reasoning and Acting in Language Models**  
   https://arxiv.org/abs/2210.03629
4. Schick et al. (2023), **Toolformer: Language Models Can Teach Themselves to Use Tools**  
   https://arxiv.org/abs/2302.04761

---

# B　规划、搜索、反思与自我改进

1. Wang et al. (2022), **Self-Consistency Improves Chain of Thought Reasoning in Language Models**  
   https://arxiv.org/abs/2203.11171
2. Yao et al. (2023), **Tree of Thoughts: Deliberate Problem Solving with Large Language Models**  
   https://arxiv.org/abs/2305.10601
3. Shinn et al. (2023), **Reflexion: Language Agents with Verbal Reinforcement Learning**  
   https://arxiv.org/abs/2303.11366
4. Madaan et al. (2023), **Self-Refine: Iterative Refinement with Self-Feedback**  
   https://arxiv.org/abs/2303.17651

---

# C　记忆与长期状态

1. Park et al. (2023), **Generative Agents: Interactive Simulacra of Human Behavior**  
   https://arxiv.org/abs/2304.03442
2. Packer et al. (2023), **MemGPT: Towards LLMs as Operating Systems**  
   https://arxiv.org/abs/2310.08560

阅读时要区分：context management、retrieval、persistent state、episodic memory、semantic memory 并非同一概念。

---

# D　编程智能体与软件工程

1. Jimenez et al. (2023/2024), **SWE-bench: Can Language Models Resolve Real-World GitHub Issues?**  
   https://arxiv.org/abs/2310.06770
2. Yang et al. (2024), **SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering**  
   https://arxiv.org/abs/2405.15793

对应研究问题：

- repo understanding；
- issue localization；
- editing interface；
- test feedback；
- patch verification；
- agent-computer interface；
- benchmark budget fairness。

---

# E　浏览器、计算机使用与通用环境

1. Zhou et al. (2023), **WebArena: A Realistic Web Environment for Building Autonomous Agents**  
   https://arxiv.org/abs/2307.13854
2. Liu et al. (2023), **AgentBench: Evaluating LLMs as Agents**  
   https://arxiv.org/abs/2308.03688
3. Xie et al. (2024), **OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments**  
   https://arxiv.org/abs/2404.07972

这些 benchmark 的环境、工具接口、任务预算和 reset 策略必须与模型能力分开分析。

---

# F　多智能体系统

1. Li et al. (2023), **CAMEL: Communicative Agents for Mind Exploration of Large Scale Language Model Society**  
   https://arxiv.org/abs/2303.17760
2. Wu et al. (2023), **AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation**  
   https://arxiv.org/abs/2308.08155

研究多 Agent 时，本教材重点关注：task graph、communication cost、state isolation、artifact handoff、review / merge，而不把“多个角色聊天”本身当作充分创新。

---

# G　环境学习与 Agentic RL 邻接资料

1. WebGPT: https://arxiv.org/abs/2112.09332
2. DeepSeekMath: https://arxiv.org/abs/2402.03300
3. DeepSeek-R1: https://arxiv.org/abs/2501.12948
4. ReAct: https://arxiv.org/abs/2210.03629

这一部分快速演化。引用“Agentic RL”时必须具体说明：

- reward 从哪里来；
- 是在线 RL、离线 trajectory learning，还是只更新 memory；
- policy 参数是否真的更新；
- environment 是否可复现；
- success verifier 是否可靠。

---

# H　阅读顺序

推荐：

```text
WebGPT
→ ReAct
→ Toolformer
→ Reflexion / ToT
→ MemGPT
→ SWE-bench / SWE-agent
→ WebArena / OSWorld / AgentBench
→ CAMEL / AutoGen
→ Agentic RL
```

对应教材入口：[`../智能体-agent/README.md`](../智能体-agent/README.md)
