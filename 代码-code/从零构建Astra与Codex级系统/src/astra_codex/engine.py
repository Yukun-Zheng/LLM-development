from __future__ import annotations

from collections.abc import Iterator

import torch

from .cache import KVCache
from .model import DecoderOnlyTransformer
from .sampling import SamplingConfig, sample_next_token


class GenerationEngine:
    """Minimal prefill/decode engine using an explicit KV cache."""

    def __init__(self, model: DecoderOnlyTransformer) -> None:
        self.model = model

    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, KVCache]:
        output = self.model(input_ids, use_cache=True)
        assert output.past_key_values is not None
        cache = KVCache.empty(self.model.config.num_layers)
        cache.replace(output.past_key_values)
        return output.logits[:, -1, :], cache

    @torch.inference_mode()
    def decode_one(
        self,
        token_ids: torch.Tensor,
        cache: KVCache,
    ) -> torch.Tensor:
        if token_ids.ndim != 2 or token_ids.shape[1] != 1:
            raise ValueError("decode_one expects token_ids with shape [B, 1]")
        output = self.model(
            token_ids,
            past_key_values=cache.as_past_key_values(),
            use_cache=True,
        )
        assert output.past_key_values is not None
        cache.replace(output.past_key_values)
        return output.logits[:, -1, :]

    @torch.inference_mode()
    def stream_generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        sampling: SamplingConfig | None = None,
        stop_token_ids: set[int] | None = None,
        generator: torch.Generator | None = None,
    ) -> Iterator[torch.Tensor]:
        sampling = sampling or SamplingConfig()
        stop_token_ids = stop_token_ids or set()
        logits, cache = self.prefill(input_ids)
        history = input_ids

        for _ in range(max_new_tokens):
            next_token = sample_next_token(
                logits,
                sampling,
                history=history,
                generator=generator,
            )
            yield next_token
            history = torch.cat((history, next_token), dim=1)

            if any(int(token) in stop_token_ids for token in next_token[:, 0]):
                break
            logits = self.decode_one(next_token, cache)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        sampling: SamplingConfig | None = None,
        stop_token_ids: set[int] | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        pieces = [input_ids]
        for token in self.stream_generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            sampling=sampling,
            stop_token_ids=stop_token_ids,
            generator=generator,
        ):
            pieces.append(token)
        return torch.cat(pieces, dim=1)
