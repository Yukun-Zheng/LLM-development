from __future__ import annotations

import torch
import pytest

from astra_codex.config import ModelConfig
from astra_codex.kv_tensor_pool import PhysicalKVTensorPool
from astra_codex.model import DecoderOnlyTransformer
from astra_codex.page_aware_decode import HeterogeneousPageAwareDecodeReference
from astra_codex.physical_prefix_cache import PhysicalPrefixCacheEngine


def _tiny_model() -> DecoderOnlyTransformer:
    torch.manual_seed(41)
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


def _pool_for(model: DecoderOnlyTransformer) -> PhysicalKVTensorPool:
    return PhysicalKVTensorPool(
        num_layers=model.config.num_layers,
        total_blocks=32,
        num_kv_heads=model.config.num_kv_heads,
        block_size=2,
        head_dim=model.config.head_dim,
        dtype=next(model.parameters()).dtype,
        device=next(model.parameters()).device,
    )


def _prefill(pool: PhysicalKVTensorPool, model: DecoderOnlyTransformer, request_id: str, ids: torch.Tensor) -> None:
    pool.create_request(request_id)
    output = model(ids, use_cache=True)
    assert output.past_key_values is not None
    pool.ingest_present(request_id, output.past_key_values)


def test_mixed_length_page_aware_decode_matches_full_recomputation_without_materialize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _tiny_model()
    pool = _pool_for(model)
    prefix_a = torch.tensor([[1, 2, 3]])
    prefix_b = torch.tensor([[4, 5, 6, 7, 8]])
    _prefill(pool, model, "a", prefix_a)
    _prefill(pool, model, "b", prefix_b)

    def forbidden_materialize(request_id: str):  # type: ignore[no-untyped-def]
        raise AssertionError(f"page-aware decode materialized K/V for {request_id}")

    monkeypatch.setattr(pool, "materialize", forbidden_materialize)
    executor = HeterogeneousPageAwareDecodeReference(model, pool)
    next_tokens = torch.tensor([[9], [10]])
    logits = executor.decode_batch(["a", "b"], next_tokens)

    reference_a = model(torch.cat((prefix_a, next_tokens[0:1]), dim=1)).logits[:, -1, :]
    reference_b = model(torch.cat((prefix_b, next_tokens[1:2]), dim=1)).logits[:, -1, :]
    torch.testing.assert_close(logits[0:1], reference_a, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(logits[1:2], reference_b, atol=1e-5, rtol=1e-5)
    assert pool.allocator.requests["a"].sequence_length == 4
    assert pool.allocator.requests["b"].sequence_length == 6


def test_page_aware_decode_writes_exact_per_layer_cache_values() -> None:
    model = _tiny_model()
    pool = _pool_for(model)
    prefixes = {
        "short": torch.tensor([[2, 3]]),
        "long": torch.tensor([[5, 6, 7, 8, 9]]),
    }
    for request_id, prefix in prefixes.items():
        _prefill(pool, model, request_id, prefix)

    executor = HeterogeneousPageAwareDecodeReference(model, pool)
    next_tokens = torch.tensor([[11], [12]])
    executor.decode_batch(["short", "long"], next_tokens)

    for index, request_id in enumerate(("short", "long")):
        full_ids = torch.cat((prefixes[request_id], next_tokens[index : index + 1]), dim=1)
        reference = model(full_ids, use_cache=True)
        assert reference.past_key_values is not None
        actual = pool.materialize(request_id)
        for (actual_k, actual_v), (expected_k, expected_v) in zip(
            actual, reference.past_key_values, strict=True
        ):
            torch.testing.assert_close(actual_k, expected_k, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(actual_v, expected_v, atol=1e-5, rtol=1e-5)


def test_page_aware_decode_preserves_shared_parent_via_copy_on_write() -> None:
    model = _tiny_model()
    prefix_engine = PhysicalPrefixCacheEngine(model, total_blocks=32, block_size=4)
    prompt = torch.tensor([[1, 2, 3, 4, 5, 6]])
    prefix_engine.prefill("parent", prompt)
    prefix_engine.prefill("child", prompt)

    parent_before = tuple(
        (key.clone(), value.clone())
        for key, value in prefix_engine.pool.materialize("parent")
    )
    executor = HeterogeneousPageAwareDecodeReference(model, prefix_engine.pool)
    decoded = executor.decode_batch(["child"], torch.tensor([[7]]))
    reference = model(torch.tensor([[1, 2, 3, 4, 5, 6, 7]])).logits[:, -1, :]
    torch.testing.assert_close(decoded, reference, atol=1e-5, rtol=1e-5)

    parent_after = prefix_engine.pool.materialize("parent")
    for (actual_k, actual_v), (expected_k, expected_v) in zip(
        parent_after, parent_before, strict=True
    ):
        torch.testing.assert_close(actual_k, expected_k)
        torch.testing.assert_close(actual_v, expected_v)

    parent_table = prefix_engine.pool.allocator.block_table("parent")
    child_table = prefix_engine.pool.allocator.block_table("child")
    assert parent_table[0].block_id == child_table[0].block_id
    assert parent_table[-1].block_id != child_table[-1].block_id


def test_page_aware_decode_rejects_duplicate_request_ids_and_bad_shape() -> None:
    model = _tiny_model()
    pool = _pool_for(model)
    _prefill(pool, model, "a", torch.tensor([[1, 2]]))
    executor = HeterogeneousPageAwareDecodeReference(model, pool)

    with pytest.raises(ValueError, match="unique"):
        executor.decode_batch(["a", "a"], torch.tensor([[3], [4]]))
    with pytest.raises(ValueError, match="shape"):
        executor.decode_batch(["a"], torch.tensor([[3, 4]]))
