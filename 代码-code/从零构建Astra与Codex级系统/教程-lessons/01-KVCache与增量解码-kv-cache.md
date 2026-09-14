# Lab 01　KV Cache、Prefill 与 Decode：为什么聊天模型能一个 token 一个 token 地吐出来

> 对应源码：`cache.py`、`engine.py`、`model.py`

## 1　训练与生成的计算图不同

训练时整段 token 已知，可以一次并行计算：

```text
[B,T] -> Transformer -> [B,T,V]
```

生成时第 $t+1$ 个 token 不存在，必须先得到第 $t$ 个 token。因此 decode 在时间维度上天然串行。

## 2　没有 KV Cache 会发生什么

生成到长度 $T$ 时，如果每一步都重新把全部 prefix 输入模型，历史 token 的 K/V 会被重复计算。

KV Cache 保存每层：

```text
K_cache [B,Hkv,T,D]
V_cache [B,Hkv,T,D]
```

下一步只计算新 token 的 Q/K/V，再把新 K/V append 到 cache。

## 3　Prefill

Prompt 一次送入模型：

```text
prompt [B,T]
   ↓
full forward
   ↓
last logits [B,V]
+ KV cache for all layers
```

源码：`GenerationEngine.prefill`。

## 4　Decode

以后每步只输入：

```text
next_token [B,1]
```

源码：`GenerationEngine.decode_one`。

最关键的正确性测试不是速度，而是：

> 使用 cache 的 incremental logits 应与“把完整序列重新 forward”得到的最后位置 logits 一致。

这就是 `test_incremental_cache_matches_full_forward`。

## 5　为什么 GQA 会直接影响服务成本

缓存规模近似：

$$
O(L\cdot B\cdot T\cdot H_{kv}\cdot D).
$$

所以从 MHA 的 $H_{kv}=H_q$ 降到 GQA 的少量 KV heads，会直接减少长上下文 decode 的显存压力。

## 6　从教学实现到工业实现

下一阶段再研究：

- paged KV cache；
- prefix caching；
- continuous batching；
- speculative decoding；
- tensor parallel serving。

vLLM / PagedAttention 原始资料：Kwon et al., 2023, https://arxiv.org/abs/2309.06180
