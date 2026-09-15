from __future__ import annotations

import torch

from astra_codex.config import ModelConfig
from astra_codex.kv_tensor_pool import PhysicalBlockGenerationEngine, PhysicalKVTensorPool
from astra_codex.model import DecoderOnlyTransformer


def _synthetic_delta(
    *,
    num_layers: int = 2,
    num_kv_heads: int = 2,
    tokens: int,
    head_dim: int = 4,
    start: float = 0.0,
):
    pairs = []
    cursor = start
    for _layer in range(num_layers):
        count = num_kv_heads * tokens * head_dim
        key = torch.arange(cursor, cursor + count, dtype=torch.float32).reshape(
            1, num_kv_heads, tokens, head_dim
        )
        cursor += count
        value = torch.arange(cursor, cursor + count, dtype=torch.float32).reshape(
            1, num_kv_heads, tokens, head_dim
        )
        cursor += count
        pairs.append((key, value))
    return tuple(pairs)


def _tiny_model() -> DecoderOnlyTransformer:
    torch.manual_seed(17)
    config = ModelConfig(
        vocab_size=64,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=80,
        max_seq_len=32,
    )
    return DecoderOnlyTransformer(config).eval()


def test_tensor_pool_materializes_exact_appended_delta() -> None:
    pool = PhysicalKVTensorPool(
        num_layers=2,
        total_blocks=8,
        num_kv_heads=2,
        block_size=3,
        head_dim=4,
    )
    pool.create_request("req")
    delta = _synthetic_delta(tokens=5)
    pool.append_delta("req", delta)

    materialized = pool.materialize("req")
    for (actual_k, actual_v), (expected_k, expected_v) in zip(
        materialized, delta, strict=True
    ):
        torch.testing.assert_close(actual_k, expected_k)
        torch.testing.assert_close(actual_v, expected_v)

    stats = pool.stats()
    assert stats.allocated_blocks == 2
    assert stats.logical_tokens == 5
    assert stats.physical_token_slots_used == 5
    assert stats.tensor_bytes_reserved == pool.keys.numel() * 4 * 2


def test_tensor_pool_exact_fork_then_cow_preserves_parent_values() -> None:
    pool = PhysicalKVTensorPool(
        num_layers=2,
        total_blocks=8,
        num_kv_heads=2,
        block_size=4,
        head_dim=4,
    )
    pool.create_request("parent")
    prefix = _synthetic_delta(tokens=6)
    pool.append_delta("parent", prefix)
    pool.fork_request("parent", "child")

    parent_before = tuple((k.clone(), v.clone()) for k, v in pool.materialize("parent"))
    child_delta = _synthetic_delta(tokens=1, start=10_000.0)
    pool.append_delta("child", child_delta)

    parent_after = pool.materialize("parent")
    child_after = pool.materialize("child")
    for (actual_k, actual_v), (expected_k, expected_v) in zip(
        parent_after, parent_before, strict=True
    ):
        torch.testing.assert_close(actual_k, expected_k)
        torch.testing.assert_close(actual_v, expected_v)

    for layer, ((child_k, child_v), (prefix_k, prefix_v), (delta_k, delta_v)) in enumerate(
        zip(child_after, prefix, child_delta, strict=True)
    ):
        del layer
        torch.testing.assert_close(child_k[..., :6, :], prefix_k)
        torch.testing.assert_close(child_v[..., :6, :], prefix_v)
        torch.testing.assert_close(child_k[..., 6:, :], delta_k)
        torch.testing.assert_close(child_v[..., 6:, :], delta_v)


def test_tensor_pool_partial_prefix_fork_copies_only_requested_tail() -> None:
    pool = PhysicalKVTensorPool(
        num_layers=2,
        total_blocks=10,
        num_kv_heads=2,
        block_size=4,
        head_dim=4,
    )
    pool.create_request("source")
    full = _synthetic_delta(tokens=10)
    pool.append_delta("source", full)
    pool.fork_request("source", "prefix", prefix_tokens=6)

    target = pool.materialize("prefix")
    for (actual_k, actual_v), (source_k, source_v) in zip(target, full, strict=True):
        torch.testing.assert_close(actual_k, source_k[..., :6, :])
        torch.testing.assert_close(actual_v, source_v[..., :6, :])

    source_table = pool.allocator.block_table("source")
    target_table = pool.allocator.block_table("prefix")
    assert source_table[0].block_id == target_table[0].block_id
    assert source_table[1].block_id != target_table[1].block_id


def test_released_private_blocks_are_zeroed_before_reuse() -> None:
    pool = PhysicalKVTensorPool(
        num_layers=1,
        total_blocks=2,
        num_kv_heads=1,
        block_size=4,
        head_dim=2,
    )
    pool.create_request("a")
    pool.append_delta(
        "a",
        _synthetic_delta(
            num_layers=1, num_kv_heads=1, tokens=2, head_dim=2, start=100.0
        ),
    )
    block_id = pool.allocator.block_table("a")[0].block_id
    assert torch.count_nonzero(pool.keys[:, block_id]).item() > 0

    pool.release_request("a")
    assert torch.count_nonzero(pool.keys[:, block_id]).item() == 0
    assert torch.count_nonzero(pool.values[:, block_id]).item() == 0


def test_physical_block_generation_matches_full_recomputation() -> None:
    model = _tiny_model()
    engine = PhysicalBlockGenerationEngine(model, total_blocks=16, block_size=2)
    prefix = torch.tensor([[1, 2, 3, 4, 5]])
    next_token = torch.tensor([[6]])

    prefill = engine.prefill("req", prefix)
    reference_prefill = model(prefix).logits[:, -1, :]
    torch.testing.assert_close(prefill, reference_prefill)

    decode = engine.decode_one("req", next_token)
    full = model(torch.cat((prefix, next_token), dim=1)).logits[:, -1, :]
    torch.testing.assert_close(decode, full, atol=1e-5, rtol=1e-5)

    materialized = engine.pool.materialize("req")
    assert materialized[0][0].shape[-2] == 6
    assert engine.pool.allocator.metrics().allocated_blocks == 3
