from __future__ import annotations

from dataclasses import dataclass

import torch

KVPair = tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True, slots=True)
class PrefixCacheEntry:
    token_ids: tuple[int, ...]
    past_key_values: tuple[KVPair, ...]
    last_logits: torch.Tensor

    @property
    def length(self) -> int:
        return len(self.token_ids)


@dataclass(slots=True)
class PrefixCacheStats:
    lookups: int = 0
    exact_hits: int = 0
    partial_hits: int = 0
    misses: int = 0
    reused_tokens: int = 0

    @property
    def hit_rate(self) -> float:
        if self.lookups == 0:
            return 0.0
        return (self.exact_hits + self.partial_hits) / self.lookups


class ReferencePrefixKVStore:
    """Small inspectable longest-prefix KV store for batch-size-one prompts.

    Entries keep cloned per-layer K/V tensors and the prompt's last logits. The
    implementation intentionally uses a Python dictionary and linear longest-
    prefix search; it establishes semantics before radix trees, block hashing,
    eviction, sharing across replicas, or GPU-resident prefix blocks.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple[int, ...], PrefixCacheEntry] = {}
        self.stats = PrefixCacheStats()

    @staticmethod
    def _clone_past(past_key_values: tuple[KVPair, ...]) -> tuple[KVPair, ...]:
        return tuple((key.clone(), value.clone()) for key, value in past_key_values)

    def put(
        self,
        token_ids: tuple[int, ...],
        past_key_values: tuple[KVPair, ...],
        last_logits: torch.Tensor,
    ) -> None:
        if not token_ids:
            raise ValueError("prefix token_ids must be non-empty")
        sequence_lengths = {int(key.shape[-2]) for key, _ in past_key_values}
        if sequence_lengths != {len(token_ids)}:
            raise ValueError(
                "past_key_values length must match prefix token count; "
                f"got cache lengths {sorted(sequence_lengths)} for {len(token_ids)} tokens"
            )
        self._entries[token_ids] = PrefixCacheEntry(
            token_ids=token_ids,
            past_key_values=self._clone_past(past_key_values),
            last_logits=last_logits.detach().clone(),
        )

    def longest_prefix(self, token_ids: tuple[int, ...]) -> PrefixCacheEntry | None:
        self.stats.lookups += 1
        best: PrefixCacheEntry | None = None
        for cached_tokens, entry in self._entries.items():
            if len(cached_tokens) > len(token_ids):
                continue
            if token_ids[: len(cached_tokens)] != cached_tokens:
                continue
            if best is None or entry.length > best.length:
                best = entry

        if best is None:
            self.stats.misses += 1
            return None
        self.stats.reused_tokens += best.length
        if best.length == len(token_ids):
            self.stats.exact_hits += 1
        else:
            self.stats.partial_hits += 1
        return PrefixCacheEntry(
            token_ids=best.token_ids,
            past_key_values=self._clone_past(best.past_key_values),
            last_logits=best.last_logits.clone(),
        )

    def clear(self) -> None:
        self._entries.clear()
        self.stats = PrefixCacheStats()

    def __len__(self) -> int:
        return len(self._entries)


@dataclass(frozen=True, slots=True)
class PrefixPrefillResult:
    logits: torch.Tensor
    past_key_values: tuple[KVPair, ...]
    reused_tokens: int
    computed_tokens: int
    exact_hit: bool


class ReferencePrefixCacheEngine:
    """Prefill engine that reuses the longest cached token prefix.

    Current scope is deliberately narrow: batch size one and exact token-prefix
    identity. The result is compared against a full prompt forward in tests, so
    adding cache reuse is not allowed to change model semantics.
    """

    def __init__(self, model, store: ReferencePrefixKVStore | None = None) -> None:  # type: ignore[no-untyped-def]
        self.model = model
        self.store = store or ReferencePrefixKVStore()
        self.model_forward_calls = 0

    @staticmethod
    def _tokens(input_ids: torch.Tensor) -> tuple[int, ...]:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("reference prefix cache currently expects input_ids [1, T]")
        if input_ids.shape[1] == 0:
            raise ValueError("prompt cannot be empty")
        return tuple(int(token) for token in input_ids[0].tolist())

    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor) -> PrefixPrefillResult:
        tokens = self._tokens(input_ids)
        entry = self.store.longest_prefix(tokens)

        if entry is not None and entry.length == len(tokens):
            return PrefixPrefillResult(
                logits=entry.last_logits.clone(),
                past_key_values=entry.past_key_values,
                reused_tokens=entry.length,
                computed_tokens=0,
                exact_hit=True,
            )

        if entry is None:
            suffix = input_ids
            past = None
            reused = 0
        else:
            suffix = input_ids[:, entry.length :]
            past = entry.past_key_values
            reused = entry.length

        self.model_forward_calls += 1
        output = self.model(suffix, past_key_values=past, use_cache=True)
        assert output.past_key_values is not None
        last_logits = output.logits[:, -1, :]
        self.store.put(tokens, output.past_key_values, last_logits)
        return PrefixPrefillResult(
            logits=last_logits,
            past_key_values=output.past_key_values,
            reused_tokens=reused,
            computed_tokens=len(tokens) - reused,
            exact_hit=False,
        )
