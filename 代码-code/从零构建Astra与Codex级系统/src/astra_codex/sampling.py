from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    temperature: float = 0.8
    top_k: int | None = None
    top_p: float | None = 0.95
    repetition_penalty: float = 1.0

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")


def _apply_repetition_penalty(
    logits: torch.Tensor,
    history: torch.Tensor | None,
    penalty: float,
) -> torch.Tensor:
    if history is None or penalty == 1.0:
        return logits
    out = logits.clone()
    for batch_index in range(out.shape[0]):
        token_ids = torch.unique(history[batch_index])
        selected = out[batch_index, token_ids]
        selected = torch.where(selected < 0, selected * penalty, selected / penalty)
        out[batch_index, token_ids] = selected
    return out


def sample_next_token(
    logits: torch.Tensor,
    config: SamplingConfig,
    *,
    history: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample IDs from logits [B, V], returning [B, 1]."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape [B, V]")
    logits = _apply_repetition_penalty(logits, history, config.repetition_penalty)

    if config.temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / config.temperature

    if config.top_k is not None:
        k = min(config.top_k, logits.shape[-1])
        threshold = torch.topk(logits, k=k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    if config.top_p is not None and config.top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative = sorted_probs.cumsum(dim=-1)
        remove = cumulative > config.top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        filtered = torch.full_like(logits, float("-inf"))
        logits = filtered.scatter(1, sorted_indices, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)
