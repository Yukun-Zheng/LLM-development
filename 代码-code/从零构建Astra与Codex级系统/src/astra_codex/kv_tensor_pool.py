from __future__ import annotations

from dataclasses import dataclass

import torch

from .kv_block_allocator import KVBlockAllocator

KVPair = tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True, slots=True)
class TensorPoolStats:
    allocated_blocks: int
    free_blocks: int
    logical_tokens: int
    physical_token_slots_used: int
    tensor_bytes_reserved: int


class PhysicalKVTensorPool:
    """Reference K/V tensor slab indexed by physical block ids.

    Storage shape:

        [num_layers, total_blocks, num_kv_heads, block_size, head_dim]

    for keys and values independently.

    The pool uses ``KVBlockAllocator`` for request ownership/refcounts/COW and
    performs the corresponding tensor copies/writes. The educational model still
    consumes contiguous ``past_key_values`` in older reference paths, while the
    newer page-aware decoder reads these slabs directly by block id.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        total_blocks: int,
        num_kv_heads: int,
        block_size: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        for name, value in (
            ("num_layers", num_layers),
            ("total_blocks", total_blocks),
            ("num_kv_heads", num_kv_heads),
            ("block_size", block_size),
            ("head_dim", head_dim),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        self.num_layers = num_layers
        self.total_blocks = total_blocks
        self.num_kv_heads = num_kv_heads
        self.block_size = block_size
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device)
        self.allocator = KVBlockAllocator(total_blocks, block_size=block_size)
        shape = (
            num_layers,
            total_blocks,
            num_kv_heads,
            block_size,
            head_dim,
        )
        self.keys = torch.zeros(shape, dtype=dtype, device=self.device)
        self.values = torch.zeros_like(self.keys)

    def create_request(self, request_id: str) -> None:
        self.allocator.create_request(request_id)

    def _validate_delta(self, delta: tuple[KVPair, ...]) -> int:
        if len(delta) != self.num_layers:
            raise ValueError(
                f"expected {self.num_layers} layers, received {len(delta)}"
            )
        lengths: set[int] = set()
        expected_prefix = (1, self.num_kv_heads)
        for layer, (key, value) in enumerate(delta):
            if key.shape != value.shape:
                raise ValueError(f"layer {layer}: key/value shapes differ")
            if key.ndim != 4 or tuple(key.shape[:2]) != expected_prefix:
                raise ValueError(
                    f"layer {layer}: expected [1,{self.num_kv_heads},T,{self.head_dim}]"
                )
            if int(key.shape[-1]) != self.head_dim:
                raise ValueError(
                    f"layer {layer}: head_dim {key.shape[-1]} != {self.head_dim}"
                )
            if key.dtype != self.dtype or value.dtype != self.dtype:
                raise ValueError(f"layer {layer}: dtype must match tensor pool")
            if key.device != self.device or value.device != self.device:
                raise ValueError(f"layer {layer}: device must match tensor pool")
            lengths.add(int(key.shape[-2]))
        if len(lengths) != 1:
            raise ValueError("delta layers have inconsistent sequence lengths")
        length = lengths.pop()
        if length <= 0:
            raise ValueError("delta sequence length must be positive")
        return length

    def require_batch_append_capacity(self, token_counts: dict[str, int]) -> int:
        """Preflight COW/new-block demand for a single-thread reference batch.

        No allocator state is mutated. If the summed demand cannot fit, the
        method raises before any request is appended. This gives OOM-atomic
        *preflight* semantics to the current synchronous executor.

        It is deliberately not called a distributed reservation: no block ids
        are removed from the free list and another concurrent allocator user
        could invalidate the check. A future scheduler-owned reservation token
        must solve that stronger problem.
        """

        required = 0
        for request_id, token_count in token_counts.items():
            if token_count < 0:
                raise ValueError("token_count cannot be negative")
            table = self.allocator._table(request_id)
            required += self.allocator._required_blocks_for_append(table, token_count)
        free = self.allocator.metrics().free_blocks
        if required > free:
            raise MemoryError(
                "KV block pool exhausted for batch append: "
                f"batch needs {required} free blocks, only {free} available"
            )
        return required

    def _copy_cow_if_needed(
        self,
        before: tuple,
        after: tuple,
    ) -> None:
        if not before or not after:
            return
        old_last = before[-1]
        new_last = after[min(len(before), len(after)) - 1]
        if old_last.block_id == new_last.block_id:
            return
        tokens = min(old_last.logical_tokens, new_last.logical_tokens)
        self.keys[:, new_last.block_id, :, :tokens, :].copy_(
            self.keys[:, old_last.block_id, :, :tokens, :]
        )
        self.values[:, new_last.block_id, :, :tokens, :].copy_(
            self.values[:, old_last.block_id, :, :tokens, :]
        )

    def append_delta(self, request_id: str, delta: tuple[KVPair, ...]) -> None:
        token_count = self._validate_delta(delta)
        before = self.allocator.block_table(request_id)
        old_length = self.allocator.requests[request_id].sequence_length
        self.allocator.append_tokens(request_id, token_count)
        after = self.allocator.block_table(request_id)
        self._copy_cow_if_needed(before, after)

        for offset in range(token_count):
            logical_position = old_length + offset
            entry_index = logical_position // self.block_size
            in_block = logical_position % self.block_size
            entry = after[entry_index]
            for layer_index, (key, value) in enumerate(delta):
                self.keys[
                    layer_index, entry.block_id, :, in_block, :
                ].copy_(key[0, :, offset, :])
                self.values[
                    layer_index, entry.block_id, :, in_block, :
                ].copy_(value[0, :, offset, :])

    def append_batch_delta(
        self,
        deltas: dict[str, tuple[KVPair, ...]],
    ) -> None:
        """Validate and OOM-preflight a batch before the first append mutates it."""

        lengths: dict[str, int] = {}
        for request_id, delta in deltas.items():
            if request_id not in self.allocator.requests:
                raise KeyError(f"unknown request: {request_id}")
            lengths[request_id] = self._validate_delta(delta)
        self.require_batch_append_capacity(lengths)
        for request_id, delta in deltas.items():
            self.append_delta(request_id, delta)

    def ingest_present(
        self,
        request_id: str,
        present_key_values: tuple[KVPair, ...],
    ) -> None:
        """Append only the new suffix from model-returned full present K/V."""

        if len(present_key_values) != self.num_layers:
            raise ValueError("cache layer count mismatch")
        old_length = self.allocator.requests[request_id].sequence_length
        lengths = {int(key.shape[-2]) for key, _ in present_key_values}
        if len(lengths) != 1:
            raise ValueError("present cache layers have inconsistent lengths")
        new_length = lengths.pop()
        if new_length < old_length:
            raise ValueError(
                f"present cache shrank from {old_length} to {new_length} tokens"
            )
        if new_length == old_length:
            return
        delta = tuple(
            (
                key[..., old_length:new_length, :],
                value[..., old_length:new_length, :],
            )
            for key, value in present_key_values
        )
        self.append_delta(request_id, delta)

    def fork_request(
        self,
        source_request_id: str,
        target_request_id: str,
        *,
        prefix_tokens: int | None = None,
    ) -> None:
        source_entries = self.allocator.block_table(source_request_id)
        source_length = self.allocator.requests[source_request_id].sequence_length
        length = source_length if prefix_tokens is None else prefix_tokens
        self.allocator.fork_request(
            source_request_id,
            target_request_id,
            prefix_tokens=prefix_tokens,
        )
        target_entries = self.allocator.block_table(target_request_id)

        logical_start = 0
        source_index = 0
        for target_entry in target_entries:
            while (
                source_index < len(source_entries)
                and logical_start
                >= sum(item.logical_tokens for item in source_entries[: source_index + 1])
            ):
                source_index += 1
            if source_index >= len(source_entries):
                raise RuntimeError("fork tensor mapping exceeded source block table")
            source_entry = source_entries[source_index]
            if target_entry.block_id != source_entry.block_id:
                tokens = target_entry.logical_tokens
                self.keys[:, target_entry.block_id, :, :tokens, :].copy_(
                    self.keys[:, source_entry.block_id, :, :tokens, :]
                )
                self.values[:, target_entry.block_id, :, :tokens, :].copy_(
                    self.values[:, source_entry.block_id, :, :tokens, :]
                )
            logical_start += target_entry.logical_tokens

        if self.allocator.requests[target_request_id].sequence_length != length:
            raise RuntimeError("forked tensor request length mismatch")

    def truncate(self, request_id: str, new_length: int) -> None:
        self.allocator.truncate(request_id, new_length)

    def release_request(self, request_id: str) -> None:
        block_ids = {entry.block_id for entry in self.allocator.block_table(request_id)}
        self.allocator.release_request(request_id)
        for block_id in block_ids:
            if self.allocator.block_refcount(block_id) == 0:
                self.keys[:, block_id].zero_()
                self.values[:, block_id].zero_()

    def materialize(self, request_id: str) -> tuple[KVPair, ...]:
        table = self.allocator.block_table(request_id)
        pairs: list[KVPair] = []
        for layer in range(self.num_layers):
            key_parts = [
                self.keys[
                    layer,
                    entry.block_id,
                    :,
                    : entry.logical_tokens,
                    :,
                ]
                for entry in table
            ]
            value_parts = [
                self.values[
                    layer,
                    entry.block_id,
                    :,
                    : entry.logical_tokens,
                    :,
                ]
                for entry in table
            ]
            if key_parts:
                key = torch.cat(key_parts, dim=-2).unsqueeze(0)
                value = torch.cat(value_parts, dim=-2).unsqueeze(0)
            else:
                key = torch.empty(
                    (1, self.num_kv_heads, 0, self.head_dim),
                    dtype=self.dtype,
                    device=self.device,
                )
                value = torch.empty_like(key)
            pairs.append((key, value))
        return tuple(pairs)

    def stats(self) -> TensorPoolStats:
        allocator = self.allocator.metrics()
        tensor_bytes = (
            self.keys.numel() * self.keys.element_size()
            + self.values.numel() * self.values.element_size()
        )
        return TensorPoolStats(
            allocated_blocks=allocator.allocated_blocks,
            free_blocks=allocator.free_blocks,
            logical_tokens=allocator.logical_tokens,
            physical_token_slots_used=allocator.physical_token_slots_used,
            tensor_bytes_reserved=tensor_bytes,
        )


class PhysicalBlockGenerationEngine:
    """Generation path backed by a physical block tensor pool.

    Attention still receives materialized contiguous history in this older path;
    ``HeterogeneousPageAwareDecodeReference`` is the newer direct-block reader.
    """

    def __init__(
        self,
        model,
        *,
        total_blocks: int = 128,
        block_size: int = 16,
    ) -> None:
        self.model = model
        config = model.config
        self.pool = PhysicalKVTensorPool(
            num_layers=config.num_layers,
            total_blocks=total_blocks,
            num_kv_heads=config.num_kv_heads,
            block_size=block_size,
            head_dim=config.head_dim,
            dtype=next(model.parameters()).dtype,
            device=next(model.parameters()).device,
        )

    @torch.inference_mode()
    def prefill(
        self,
        request_id: str,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        self.pool.create_request(request_id)
        output = self.model(input_ids, use_cache=True)
        assert output.past_key_values is not None
        self.pool.ingest_present(request_id, output.past_key_values)
        return output.logits[:, -1, :]

    @torch.inference_mode()
    def decode_one(
        self,
        request_id: str,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        if token_ids.ndim != 2 or token_ids.shape != (1, 1):
            raise ValueError("physical block decode currently expects token_ids [1,1]")
        output = self.model(
            token_ids,
            past_key_values=self.pool.materialize(request_id),
            use_cache=True,
        )
        assert output.past_key_values is not None
        self.pool.ingest_present(request_id, output.past_key_values)
        return output.logits[:, -1, :]
