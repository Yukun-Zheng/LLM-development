from __future__ import annotations

from dataclasses import dataclass, field

import torch

KVPair = tuple[torch.Tensor, torch.Tensor]


@dataclass(slots=True)
class LayerPages:
    """Variable-length physical pages for one layer's K/V state."""

    key_pages: list[torch.Tensor] = field(default_factory=list)
    value_pages: list[torch.Tensor] = field(default_factory=list)

    @property
    def sequence_length(self) -> int:
        return sum(int(page.shape[-2]) for page in self.key_pages)

    @property
    def page_count(self) -> int:
        return len(self.key_pages)


class ReferencePagedKVCache:
    """Inspectable reference semantics for paged KV storage.

    This implementation is intentionally about *correctness*, not speed. It
    stores history in fixed-capacity logical pages and can reconstruct the
    contiguous ``past_key_values`` tuple expected by the educational model.

    The model still materializes a full K/V result today, so this class does not
    claim vLLM-like allocator efficiency yet. It establishes the page-table and
    incremental-ingest contract that the optimized allocator must preserve.
    """

    def __init__(self, num_layers: int, *, page_size: int = 16) -> None:
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        self.page_size = page_size
        self.layers = [LayerPages() for _ in range(num_layers)]

    @property
    def sequence_length(self) -> int:
        lengths = {layer.sequence_length for layer in self.layers}
        if len(lengths) != 1:
            raise RuntimeError(f"layer cache lengths diverged: {sorted(lengths)}")
        return lengths.pop()

    @property
    def page_count(self) -> int:
        return sum(layer.page_count for layer in self.layers)

    def _append_to_layer(
        self,
        layer: LayerPages,
        key_delta: torch.Tensor,
        value_delta: torch.Tensor,
    ) -> None:
        if key_delta.shape != value_delta.shape:
            raise ValueError("key/value shapes must match")
        if key_delta.ndim != 4:
            raise ValueError("paged cache expects [B, H_kv, T, D] tensors")

        offset = 0
        total = int(key_delta.shape[-2])
        while offset < total:
            if layer.key_pages and int(layer.key_pages[-1].shape[-2]) < self.page_size:
                free = self.page_size - int(layer.key_pages[-1].shape[-2])
                take = min(free, total - offset)
                key_piece = key_delta[..., offset : offset + take, :].clone()
                value_piece = value_delta[..., offset : offset + take, :].clone()
                layer.key_pages[-1] = torch.cat((layer.key_pages[-1], key_piece), dim=-2)
                layer.value_pages[-1] = torch.cat(
                    (layer.value_pages[-1], value_piece), dim=-2
                )
                offset += take
                continue

            take = min(self.page_size, total - offset)
            layer.key_pages.append(
                key_delta[..., offset : offset + take, :].clone()
            )
            layer.value_pages.append(
                value_delta[..., offset : offset + take, :].clone()
            )
            offset += take

    def ingest_full(self, present_key_values: tuple[KVPair, ...]) -> None:
        """Append only the suffix that is new relative to the current cache."""

        if len(present_key_values) != len(self.layers):
            raise ValueError("cache layer count mismatch")
        old_length = self.sequence_length

        new_lengths = {int(key.shape[-2]) for key, _ in present_key_values}
        if len(new_lengths) != 1:
            raise ValueError("present cache layers have inconsistent lengths")
        new_length = new_lengths.pop()
        if new_length < old_length:
            raise ValueError(
                f"present cache shrank from {old_length} to {new_length} tokens"
            )
        if new_length == old_length:
            return

        for layer, (key, value) in zip(self.layers, present_key_values, strict=True):
            self._append_to_layer(
                layer,
                key[..., old_length:new_length, :],
                value[..., old_length:new_length, :],
            )

    def as_past_key_values(self) -> tuple[KVPair, ...]:
        pairs: list[KVPair] = []
        for layer in self.layers:
            if not layer.key_pages:
                raise RuntimeError("paged cache is empty")
            pairs.append(
                (
                    torch.cat(layer.key_pages, dim=-2),
                    torch.cat(layer.value_pages, dim=-2),
                )
            )
        return tuple(pairs)

    def page_table(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(int(page.shape[-2]) for page in layer.key_pages)
            for layer in self.layers
        )

    def clear(self) -> None:
        for layer in self.layers:
            layer.key_pages.clear()
            layer.value_pages.clear()


class ReferencePagedGenerationEngine:
    """Generation-engine slice that proves contiguous ↔ paged cache parity."""

    def __init__(self, model, *, page_size: int = 16) -> None:
        self.model = model
        self.page_size = page_size

    @torch.inference_mode()
    def prefill(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, ReferencePagedKVCache]:
        output = self.model(input_ids, use_cache=True)
        assert output.past_key_values is not None
        cache = ReferencePagedKVCache(
            self.model.config.num_layers, page_size=self.page_size
        )
        cache.ingest_full(output.past_key_values)
        return output.logits[:, -1, :], cache

    @torch.inference_mode()
    def decode_one(
        self,
        token_ids: torch.Tensor,
        cache: ReferencePagedKVCache,
    ) -> torch.Tensor:
        if token_ids.ndim != 2 or token_ids.shape[1] != 1:
            raise ValueError("decode_one expects token_ids with shape [B, 1]")
        output = self.model(
            token_ids,
            past_key_values=cache.as_past_key_values(),
            use_cache=True,
        )
        assert output.past_key_values is not None
        cache.ingest_full(output.past_key_values)
        return output.logits[:, -1, :]
