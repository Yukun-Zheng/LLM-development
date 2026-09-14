from __future__ import annotations

from dataclasses import dataclass, field

import torch

KVPair = tuple[torch.Tensor, torch.Tensor]


@dataclass(slots=True)
class KVCache:
    """Per-layer key/value tensors used by incremental decoding.

    Each tensor has shape [B, H_kv, T, D].  The cache stores already-computed K
    and V so decode only computes the newest token rather than recomputing the
    whole prefix.
    """

    layers: list[KVPair | None] = field(default_factory=list)

    @classmethod
    def empty(cls, num_layers: int) -> "KVCache":
        return cls([None] * num_layers)

    @property
    def sequence_length(self) -> int:
        for layer in self.layers:
            if layer is not None:
                return int(layer[0].shape[-2])
        return 0

    def as_past_key_values(self) -> tuple[KVPair | None, ...]:
        return tuple(self.layers)

    def replace(self, present_key_values: tuple[KVPair, ...]) -> None:
        if len(present_key_values) != len(self.layers):
            raise ValueError("cache layer count mismatch")
        self.layers[:] = list(present_key_values)

    def clear(self) -> None:
        for i in range(len(self.layers)):
            self.layers[i] = None
