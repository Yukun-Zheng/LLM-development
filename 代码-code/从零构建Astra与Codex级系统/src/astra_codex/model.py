from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .config import ModelConfig

KVPair = tuple[torch.Tensor, torch.Tensor]


@dataclass(slots=True)
class ModelOutput:
    logits: torch.Tensor
    past_key_values: tuple[KVPair, ...] | None = None


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize by root-mean-square, without subtracting the mean.
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(variance + self.eps).to(dtype=x.dtype)
        return normalized * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    """RoPE applied to Q and K, shape [B, H, T, D]."""

    def __init__(self, head_dim: int, theta: float = 10_000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def cos_sin(
        self,
        positions: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # positions [T], inv_freq [D/2] -> angles [T, D/2]
        angles = torch.outer(positions.float(), self.inv_freq)
        angles = torch.repeat_interleave(angles, 2, dim=-1)
        return angles.cos().to(dtype=dtype), angles.sin().to(dtype=dtype)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self.cos_sin(positions, dtype=q.dtype)
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class GroupedQueryAttention(nn.Module):
    """Causal GQA with explicit tensors, masks, and optional KV cache."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        d = config.head_dim
        self.q_proj = nn.Linear(h, config.num_heads * d, bias=False)
        self.k_proj = nn.Linear(h, config.num_kv_heads * d, bias=False)
        self.v_proj = nn.Linear(h, config.num_kv_heads * d, bias=False)
        self.o_proj = nn.Linear(config.num_heads * d, h, bias=False)
        self.rope = RotaryEmbedding(d, config.rope_theta)

    def _reshape_q(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.config.num_heads, self.config.head_dim).transpose(1, 2)

    def _reshape_kv(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.config.num_kv_heads, self.config.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        *,
        past_key_value: KVPair | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVPair | None]:
        b, t, _ = x.shape
        q = self._reshape_q(self.q_proj(x))
        k = self._reshape_kv(self.k_proj(x))
        v = self._reshape_kv(self.v_proj(x))

        past_len = 0 if past_key_value is None else int(past_key_value[0].shape[-2])
        positions = torch.arange(past_len, past_len + t, device=x.device)
        q, k = self.rope(q, k, positions)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)

        present = (k, v) if use_cache else None

        # [B, H_kv, T_total, D] -> [B, H_q, T_total, D]
        if self.config.kv_repeat != 1:
            k_for_q = k.repeat_interleave(self.config.kv_repeat, dim=1)
            v_for_q = v.repeat_interleave(self.config.kv_repeat, dim=1)
        else:
            k_for_q, v_for_q = k, v

        scores = torch.matmul(q, k_for_q.transpose(-2, -1)) / math.sqrt(self.config.head_dim)

        # Query absolute positions are past_len..past_len+t-1; key positions are
        # 0..T_total-1.  This works both for full prefill and one-token decode.
        total_len = k_for_q.shape[-2]
        q_pos = torch.arange(past_len, past_len + t, device=x.device)[:, None]
        k_pos = torch.arange(total_len, device=x.device)[None, :]
        allowed = k_pos <= q_pos
        scores = scores.masked_fill(~allowed[None, None, :, :], float("-inf"))

        probs = F.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
        context = torch.matmul(probs, v_for_q)
        context = context.transpose(1, 2).contiguous().view(b, t, self.config.hidden_size)
        return self.o_proj(context), present


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        *,
        past_key_value: KVPair | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVPair | None]:
        attn_out, present = self.attn(
            self.attn_norm(x), past_key_value=past_key_value, use_cache=use_cache
        )
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, present


class DecoderOnlyTransformer(nn.Module):
    """Small but structurally modern decoder-only LM.

    Input:  input_ids [B, T]
    Output: logits    [B, T, V]
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        past_key_values: tuple[KVPair | None, ...] | None = None,
        use_cache: bool = False,
    ) -> ModelOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B, T]")
        if input_ids.shape[1] == 0:
            raise ValueError("input_ids cannot be empty")
        if past_key_values is None:
            past_key_values = (None,) * len(self.blocks)
        if len(past_key_values) != len(self.blocks):
            raise ValueError("past_key_values layer count mismatch")

        past_len = 0
        if past_key_values and past_key_values[0] is not None:
            past_len = int(past_key_values[0][0].shape[-2])
        if past_len + input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")

        x = self.embed_tokens(input_ids)
        presents: list[KVPair] = []
        for block, past in zip(self.blocks, past_key_values):
            x, present = block(x, past_key_value=past, use_cache=use_cache)
            if use_cache:
                assert present is not None
                presents.append(present)

        logits = self.lm_head(self.final_norm(x))
        return ModelOutput(logits=logits, past_key_values=tuple(presents) if use_cache else None)
