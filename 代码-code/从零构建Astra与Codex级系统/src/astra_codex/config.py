from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Modern decoder-only Transformer configuration.

    The defaults are intentionally tiny enough for CPU unit tests. Nothing in
    the implementation assumes frontier-scale dimensions.
    """

    vocab_size: int = 259
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 8
    num_kv_heads: int = 2
    intermediate_size: int = 768
    max_seq_len: int = 2048
    rope_theta: float = 10_000.0
    rope_interleaved: bool = False
    rms_norm_eps: float = 1e-6
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.hidden_size <= 0 or self.num_layers <= 0:
            raise ValueError("hidden_size and num_layers must be positive")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads for GQA")
        if self.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.head_dim % 2 != 0:
            raise ValueError("RoPE implementation requires an even head_dim")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def kv_repeat(self) -> int:
        """How many query heads share one KV head in grouped-query attention."""

        return self.num_heads // self.num_kv_heads
