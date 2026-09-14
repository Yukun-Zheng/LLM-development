"""A small, readable decoder-only Transformer for the textbook.

This file is intentionally educational rather than production-optimized.
It implements, with plain PyTorch:

- character-level tokenization
- token + learned positional embeddings
- Pre-Norm Transformer blocks
- RMSNorm
- causal multi-head self-attention
- SwiGLU feed-forward network
- next-token cross-entropy training
- autoregressive generation

Run:
    python code/minigpt.py

Optional text corpus:
    python code/minigpt.py --text path/to/corpus.txt --steps 5000

Requirements:
    pip install torch
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_TEXT = """
Large language models learn probability distributions over token sequences.
A decoder-only Transformer predicts the next token from previous tokens.
Attention lets each token read information from earlier positions.
Training and inference are different: training predicts many positions in
parallel, while autoregressive generation produces tokens one by one.
This tiny corpus exists only so the example runs without downloading data.
For meaningful training, pass a much larger UTF-8 text file with --text.
""" * 300


class CharTokenizer:
    """Minimal tokenizer: one Unicode character = one token."""

    def __init__(self, text: str):
        self.itos = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


class RMSNorm(nn.Module):
    """RMSNorm(x) = weight * x / sqrt(mean(x^2) + eps)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(rms + self.eps)
        return self.weight * x_norm


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.dropout = cfg.dropout

        # One matrix computes Q, K and V together for efficiency/readability.
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.out_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # Lower triangular causal mask. Shape is broadcast to [B, h, T, T].
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer(
            "causal_mask",
            mask.view(1, 1, cfg.block_size, cfg.block_size),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        B, T, C = x.shape

        # [B,T,C] -> [B,T,3C] -> three tensors [B,T,C]
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # [B,T,C] -> [B,T,h,dh] -> [B,h,T,dh]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # [B,h,T,dh] @ [B,h,dh,T] -> [B,h,T,T]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Future positions are forbidden. -inf makes their softmax probability 0.
        scores = scores.masked_fill(~self.causal_mask[:, :, :T, :T], float("-inf"))

        # Row-wise distribution over key positions.
        att = F.softmax(scores, dim=-1)
        att = self.attn_dropout(att)

        # [B,h,T,T] @ [B,h,T,dh] -> [B,h,T,dh]
        y = att @ v

        # Merge heads: [B,h,T,dh] -> [B,T,h,dh] -> [B,T,C]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)
        return self.resid_dropout(y)


class SwiGLU(nn.Module):
    """Modern gated FFN: SiLU(gate(x)) * up(x), then down-project."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        # A compact educational choice; production models tune this width.
        hidden = 4 * cfg.n_embd
        self.gate = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.up = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.gate(x)) * self.up(x)
        return self.dropout(self.down(x))


class Block(nn.Module):
    """Pre-Norm Transformer block."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.final_norm = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Weight tying: input token embeddings and output vocabulary projection
        # share one parameter matrix, a common language-model design.
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # idx: [B,T]
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(
                f"sequence length {T} exceeds block_size {self.cfg.block_size}"
            )

        positions = torch.arange(T, device=idx.device)  # [T]

        # token_emb: [B,T,C], pos_emb: [T,C], broadcast over batch.
        x = self.token_emb(idx) + self.pos_emb(positions)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)  # [B,T,C]
        logits = self.lm_head(x)  # [B,T,V]

        loss = None
        if targets is not None:
            # Flatten positions so CrossEntropy sees N=BT independent targets.
            loss = F.cross_entropy(
                logits.reshape(B * T, self.cfg.vocab_size),
                targets.reshape(B * T),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        self.eval()

        for _ in range(max_new_tokens):
            # Only the most recent block_size tokens can fit in this model.
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)

            # Only the last position predicts the next token.
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None:
                k = min(top_k, next_logits.size(-1))
                threshold = torch.topk(next_logits, k).values[:, [-1]]
                next_logits = next_logits.masked_fill(
                    next_logits < threshold, float("-inf")
                )

            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # [B,1]
            idx = torch.cat([idx, next_id], dim=1)

        return idx


def make_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample shifted input/target windows.

    If a window is:
        [a, b, c, d, e]
    then:
        x = [a, b, c, d]
        y = [b, c, d, e]
    """
    if len(data) <= block_size:
        raise ValueError("corpus must be longer than block_size")

    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: MiniGPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    batch_size: int,
    eval_iters: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    out: dict[str, float] = {}

    for split, data in (("train", train_data), ("val", val_data)):
        losses = []
        for _ in range(eval_iters):
            x, y = make_batch(data, model.cfg.block_size, batch_size, device)
            _, loss = model(x, y)
            assert loss is not None
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)

    model.train()
    return out


def load_text(path: str | None) -> str:
    if path is None:
        return DEFAULT_TEXT
    return Path(path).read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--generate", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    text = load_text(args.text)
    tokenizer = CharTokenizer(text)
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    split = int(0.9 * len(ids))
    train_data = ids[:split]
    val_data = ids[split:]

    cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    model = MiniGPT(cfg).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device}")
    print(f"vocab_size={tokenizer.vocab_size}")
    print(f"parameters={n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    model.train()
    for step in range(args.steps + 1):
        if step % args.eval_interval == 0:
            losses = estimate_loss(
                model,
                train_data,
                val_data,
                args.batch_size,
                args.eval_iters,
                device,
            )
            print(
                f"step={step:5d} "
                f"train_loss={losses['train']:.4f} "
                f"val_loss={losses['val']:.4f}"
            )

        if step == args.steps:
            break

        x, y = make_batch(
            train_data,
            cfg.block_size,
            args.batch_size,
            device,
        )
        _, loss = model(x, y)
        assert loss is not None

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    # Begin generation from one known character so the tokenizer can encode it.
    prompt = text[:1]
    start = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    generated = model.generate(
        start,
        max_new_tokens=args.generate,
        temperature=0.8,
        top_k=min(40, tokenizer.vocab_size),
    )

    print("\n--- generation ---\n")
    print(tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
