# 附录 / Appendices

本目录集中存放正文中需要反复查阅、但不适合打断历史主线的数学推导、实现细节与速查材料。

| 附录 | 中文名称 | 主要内容 | 文件 |
|---|---|---|---|
| A | **数学基础** | 线性代数、概率、信息论、优化，以及 LLM 公式中最常见的数学对象 | [`A-math.md`](A-math.md) |
| B | **从零实现 MiniGPT** | 词元化、Embedding、Attention、FFN、训练、损失函数、自回归推理 | [`B-minigpt.md`](B-minigpt.md) |
| C | **后训练数学速查** | RLHF、PPO、DPO、GRPO、RLVR 的目标函数、推导与关系 | [`C-post-training-math.md`](C-post-training-math.md) |

## 使用方式

正文中遇到公式看不懂时，优先回到附录 A；希望把 Transformer 真正落到代码时，读附录 B；进入 ChatGPT、推理模型和强化学习后训练阶段时，把附录 C 当作公式手册使用。
