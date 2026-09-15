from __future__ import annotations

import torch
import pytest

from astra_codex.config import ModelConfig
from astra_codex.kv_tensor_pool import PhysicalKVTensorPool
from astra_codex.model import DecoderOnlyTransformer
from astra_codex.page_aware_decode import HeterogeneousPageAwareDecodeReference


def _tiny_model() -> DecoderOnlyTransformer:
    torch.manual_seed(61)
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


def _pool(model: DecoderOnlyTransformer, *, total_blocks: int) -> PhysicalKVTensorPool:
    parameter = next(model.parameters())
    return PhysicalKVTensorPool(
        num_layers=model.config.num_layers,
        total_blocks=total_blocks,
        num_kv_heads=model.config.num_kv_heads,
        block_size=2,
        head_dim=model.config.head_dim,
        dtype=parameter.dtype,
        device=parameter.device,
    )


def _prefill(
    pool: PhysicalKVTensorPool,
    model: DecoderOnlyTransformer,
    request_id: str,
    ids: torch.Tensor,
) -> None:
    pool.create_request(request_id)
    output = model(ids, use_cache=True)
    assert output.past_key_values is not None
    pool.ingest_present(request_id, output.past_key_values)


def _snapshot(pool: PhysicalKVTensorPool) -> tuple[object, ...]:
    tables = {
        request_id: pool.allocator.block_table(request_id)
        for request_id in pool.allocator.requests
    }
    lengths = {
        request_id: state.sequence_length
        for request_id, state in pool.allocator.requests.items()
    }
    return (
        tables,
        lengths,
        pool.allocator.metrics(),
        pool.keys.clone(),
        pool.values.clone(),
    )


def _assert_snapshot_equal(pool: PhysicalKVTensorPool, snapshot: tuple[object, ...]) -> None:
    tables, lengths, metrics, keys, values = snapshot
    assert {
        request_id: pool.allocator.block_table(request_id)
        for request_id in pool.allocator.requests
    } == tables
    assert {
        request_id: state.sequence_length
        for request_id, state in pool.allocator.requests.items()
    } == lengths
    assert pool.allocator.metrics() == metrics
    torch.testing.assert_close(pool.keys, keys)
    torch.testing.assert_close(pool.values, values)


def test_batch_capacity_preflight_rejects_before_any_request_mutation() -> None:
    model = _tiny_model()
    pool = _pool(model, total_blocks=2)
    _prefill(pool, model, "a", torch.tensor([[1, 2]]))
    _prefill(pool, model, "b", torch.tensor([[3, 4]]))
    assert pool.allocator.metrics().free_blocks == 0
    before = _snapshot(pool)

    with pytest.raises(MemoryError, match="batch append"):
        pool.require_batch_append_capacity({"a": 1, "b": 1})

    _assert_snapshot_equal(pool, before)


def test_page_aware_decode_oom_fails_before_model_math_and_cache_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _tiny_model()
    pool = _pool(model, total_blocks=2)
    _prefill(pool, model, "a", torch.tensor([[5, 6]]))
    _prefill(pool, model, "b", torch.tensor([[7, 8]]))
    before = _snapshot(pool)

    calls = 0

    def forbidden_embed(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        raise AssertionError("model math ran before KV capacity preflight")

    monkeypatch.setattr(model.embed_tokens, "forward", forbidden_embed)
    executor = HeterogeneousPageAwareDecodeReference(model, pool)

    with pytest.raises(MemoryError, match="batch append"):
        executor.decode_batch(["a", "b"], torch.tensor([[9], [10]]))

    assert calls == 0
    _assert_snapshot_equal(pool, before)


def test_batch_delta_preflight_succeeds_then_commits_every_request() -> None:
    model = _tiny_model()
    pool = _pool(model, total_blocks=4)
    _prefill(pool, model, "a", torch.tensor([[1, 2]]))
    _prefill(pool, model, "b", torch.tensor([[3, 4]]))

    outputs = {
        "a": model(torch.tensor([[1, 2, 9]]), use_cache=True),
        "b": model(torch.tensor([[3, 4, 10]]), use_cache=True),
    }
    deltas = {}
    for request_id, output in outputs.items():
        assert output.past_key_values is not None
        deltas[request_id] = tuple(
            (key[..., -1:, :], value[..., -1:, :])
            for key, value in output.past_key_values
        )

    required = pool.require_batch_append_capacity({"a": 1, "b": 1})
    assert required == 2
    pool.append_batch_delta(deltas)

    assert pool.allocator.requests["a"].sequence_length == 3
    assert pool.allocator.requests["b"].sequence_length == 3
    assert pool.allocator.metrics().free_blocks == 0
    for request_id, output in outputs.items():
        assert output.past_key_values is not None
        actual = pool.materialize(request_id)
        for (actual_k, actual_v), (expected_k, expected_v) in zip(
            actual, output.past_key_values, strict=True
        ):
            torch.testing.assert_close(actual_k, expected_k, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(actual_v, expected_v, atol=1e-5, rtol=1e-5)
