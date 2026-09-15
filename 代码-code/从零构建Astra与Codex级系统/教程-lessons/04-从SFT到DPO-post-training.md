# Lab 04：从 SFT 到 DPO——把后训练公式变成 Tensor 与 Optimizer Step

> **目标**：不使用 Hugging Face Trainer / TRL 等高层训练框架，从最小 tensor 开始理解：SFT 到底监督哪些 token？DPO 到底比较哪四个 log-prob？为什么 reference model 必须冻结？

对应源码：[`../src/astra_codex/posttraining.py`](../src/astra_codex/posttraining.py)  
对应测试：[`../tests/test_posttraining.py`](../tests/test_posttraining.py)  
理论入口：[`../../../教材-book/03-对齐与ChatGPT-alignment-chatgpt.md`](../../../教材-book/03-对齐与ChatGPT-alignment-chatgpt.md)

原始资料：

- Ouyang et al., **Training language models to follow instructions with human feedback**, 2022: https://arxiv.org/abs/2203.02155
- Rafailov et al., **Direct Preference Optimization: Your Language Model is Secretly a Reward Model**, 2023: https://arxiv.org/abs/2305.18290

---

# 1　先从 SFT 的最小数据流开始

假设一条对话被拼成：

```text
<User> 2+2 等于多少？
<Assistant> 4
```

token IDs：

```text
input_ids: [u0, u1, u2, a0, a1]
```

Decoder-only LM 输出：

```text
logits: [B, T, V]
```

其中：

```text
logits[:, 0] predicts token 1
logits[:, 1] predicts token 2
...
logits[:, T-2] predicts token T-1
```

所以 causal LM objective 必须显式 shift：

$$
\mathcal L_{\mathrm{CE}}
=-\frac{1}{N}\sum_{t\in\mathcal S}
\log p_\theta(x_t\mid x_{\lt t}),
$$

其中 $\mathcal S$ 是真正希望监督的位置。

Instruction tuning 通常不希望把 user prompt 也当成 assistant target。因此代码里用：

```python
IGNORE_INDEX = -100
```

把 prompt/user target mask 掉：

```text
labels:
[-100, -100, ..., assistant_token_0, assistant_token_1, ...]
```

源码入口：

```python
causal_lm_loss(logits, labels)
```

其核心不是神秘 Trainer，而只是：

```text
logits[:, :-1]
labels[:, 1:]
      ↓
cross_entropy(ignore_index=-100)
```

---

# 2　一个可以手算的 SFT 检查

假设 vocabulary size：

$$
V=5,
$$

模型所有 logits 都是 0。

softmax 后每个 token 概率：

$$
p(x)=\frac{1}{5}.
$$

单个 supervised token 的 NLL：

$$
-\log\frac{1}{5}=\log 5.
$$

即使一条序列只监督两个 token，只要 PyTorch CE 默认对有效 target 求 mean，loss 仍应是：

$$
\mathcal L=\log 5.
$$

`test_causal_lm_loss_masks_prompt_targets` 就在检查这个不变量。

这类测试非常重要：它验证的不是“程序能跑”，而是**代码里的 tensor reduction 与数学定义一致**。

---

# 3　SFT step 到底做了什么

当前：

```python
sft_step(
    model,
    optimizer,
    input_ids,
    labels,
    max_grad_norm=1.0,
)
```

实际数据流：

```text
input_ids
   ↓
model forward
   ↓
logits [B,T,V]
   ↓
causal shift + mask
   ↓
CE scalar
   ↓
loss.backward()
   ↓
optional clip_grad_norm_
   ↓
optimizer.step()
```

没有“训练魔法”。

但当前实现也**不是完整 SFT pipeline**，因为还缺：

```text
conversation template
→ dataset
→ sequence packing
→ batching
→ distributed sampler
→ optimizer schedule
→ checkpoint
→ validation
→ model export
```

教材必须把“objective 已实现”和“生产训练系统已实现”分开。

---

# 4　DPO 的四个核心 log-prob

一个 preference pair：

```text
prompt x
├─ chosen   y_w
└─ rejected y_l
```

需要两个模型：

```text
policy    πθ
reference πref
```

以及四个 sequence log-prob：

$$
\log \pi_\theta(y_w\mid x),
$$

$$
\log \pi_\theta(y_l\mid x),
$$

$$
\log \pi_{\mathrm{ref}}(y_w\mid x),
$$

$$
\log \pi_{\mathrm{ref}}(y_l\mid x).
$$

当前 `sequence_logprobs()` 会先对每个 supervised response token 取：

$$
\log p(x_t\mid x_{\lt t}),
$$

再沿 sequence 求和。

Shape：

```text
logits              [B, T, V]
shifted labels      [B, T-1]
token log-probs     [B, T-1]
sequence total      [B]
```

---

# 5　DPO margin

定义：

$$
r_w=\beta\left[
\log\pi_\theta(y_w\mid x)
-\log\pi_{\mathrm{ref}}(y_w\mid x)
\right],
$$

$$
r_l=\beta\left[
\log\pi_\theta(y_l\mid x)
-\log\pi_{\mathrm{ref}}(y_l\mid x)
\right].
$$

margin：

$$
m=r_w-r_l.
$$

DPO loss：

$$
\mathcal L_{\mathrm{DPO}}
=-\log\sigma(m).
$$

源码：

```python
dpo_from_logprobs(...)
```

这个函数刻意与 model forward 分开，因为我们希望能直接把标量公式和代码逐项比对。

---

# 6　最重要的 DPO 手算不变量

如果：

$$
\pi_\theta=\pi_{\mathrm{ref}},
$$

那么 chosen 和 rejected 的两个 policy-reference log ratio 都为 0：

$$
r_w=r_l=0.
$$

所以：

$$
m=0,
$$

$$
\mathcal L=-\log\sigma(0)=\log2.
$$

这就是当前自动测试：

```text
policy == reference
→ DPO loss == log(2)
```

如果这一点不成立，就说明 shift、mask、sequence reduction 或 DPO ratio 至少有一处错了。

---

# 7　为什么 reference 必须冻结

当前：

```python
with torch.no_grad():
    reference_chosen = ...
    reference_rejected = ...
```

DPO 更新的是 policy：

```text
policy     θ0 → θ1
reference  fixed
```

测试会在 optimizer step 前后比较参数：

```text
policy weight changed      == True
reference weight changed   == False
```

这不是性能 benchmark，而是算法 contract。

---

# 8　实验：自己跑

```bash
cd '代码-code/从零构建Astra与Codex级系统'
pip install -e '.[dev]'
pytest tests/test_posttraining.py -q
```

当前 Fast CI 已把这些测试纳入全仓回归。

---

# 9　下一层不是“直接训大模型”

本项目下一步应该按：

```text
objective parity
→ tiny dataset
→ dataloader / packing
→ checkpointable loop
→ held-out eval
→ tiny SFT model
→ tiny DPO model
→ verifier reward
→ RLVR / group-relative policy update
```

推进。

这样即使不训练 frontier-scale GPT，我们也真正理解并亲手实现后训练算法，而不是只会调用一个 `Trainer`。
