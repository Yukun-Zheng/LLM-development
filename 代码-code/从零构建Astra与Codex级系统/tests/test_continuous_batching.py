from __future__ import annotations

import torch

from astra_codex.config import ModelConfig
from astra_codex.continuous_batching import ContinuousBatchingReferenceEngine
from astra_codex.model import DecoderOnlyTransformer
from astra_codex.scheduler import RequestStatus


def _tiny_model() -> DecoderOnlyTransformer:
    torch.manual_seed(53)
    config = ModelConfig(
        vocab_size=64,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=80,
        max_seq_len=48,
    )
    return DecoderOnlyTransformer(config).eval()


def test_new_request_is_prefilled_while_older_request_remains_in_decode() -> None:
    model = _tiny_model()
    engine = ContinuousBatchingReferenceEngine(
        model,
        total_blocks=32,
        block_size=2,
        max_batch_size=4,
    )
    engine.submit("old", torch.tensor([[1, 2, 3]]), max_new_tokens=3, now=0.0)

    first = engine.step(now=1.0)
    assert first is not None
    assert first.decode_request_ids == ()
    assert first.prefill_request_ids == ("old",)
    assert engine.scheduler.requests["old"].status is RequestStatus.DECODE

    engine.submit("new", torch.tensor([[4, 5, 6, 7, 8]]), max_new_tokens=2, now=1.5)
    second = engine.step(now=2.0)
    assert second is not None
    assert second.decode_request_ids == ("old",)
    assert second.prefill_request_ids == ("new",)
    assert set(second.emitted_tokens) == {"old", "new"}
    assert engine.scheduler.requests["old"].status is RequestStatus.DECODE
    assert engine.scheduler.requests["new"].status is RequestStatus.DECODE


def test_mixed_length_decode_lane_matches_individual_greedy_reference() -> None:
    model = _tiny_model()
    engine = ContinuousBatchingReferenceEngine(
        model,
        total_blocks=48,
        block_size=2,
        max_batch_size=8,
    )
    prompts = {
        "a": torch.tensor([[1, 2, 3]]),
        "b": torch.tensor([[7, 8, 9, 10, 11, 12]]),
    }
    engine.submit("a", prompts["a"], max_new_tokens=2, now=0.0)
    engine.submit("b", prompts["b"], max_new_tokens=2, now=0.0)

    prefill = engine.step(now=1.0)
    assert prefill is not None
    assert prefill.prefill_request_ids == ("a", "b")
    first_a = engine.output_tokens("a")[0]
    first_b = engine.output_tokens("b")[0]

    expected_second_a = int(
        model(torch.cat((prompts["a"], torch.tensor([[first_a]])), dim=1))
        .logits[:, -1, :]
        .argmax(dim=-1)[0]
    )
    expected_second_b = int(
        model(torch.cat((prompts["b"], torch.tensor([[first_b]])), dim=1))
        .logits[:, -1, :]
        .argmax(dim=-1)[0]
    )

    decode = engine.step(now=2.0)
    assert decode is not None
    assert decode.decode_request_ids == ("a", "b")
    assert decode.emitted_tokens == {
        "a": expected_second_a,
        "b": expected_second_b,
    }
    assert engine.scheduler.requests["a"].status is RequestStatus.FINISHED
    assert engine.scheduler.requests["b"].status is RequestStatus.FINISHED


def test_finished_requests_release_physical_blocks_and_metrics_are_recorded() -> None:
    model = _tiny_model()
    engine = ContinuousBatchingReferenceEngine(
        model,
        total_blocks=16,
        block_size=2,
    )
    engine.submit("req", torch.tensor([[2, 4, 6, 8]]), max_new_tokens=2, now=0.0)

    first = engine.step(now=1.0)
    assert first is not None
    assert "req" in engine.prefix_cache.pool.allocator.requests
    second = engine.step(now=3.0)
    assert second is not None

    assert engine.scheduler.requests["req"].status is RequestStatus.FINISHED
    assert "req" not in engine.prefix_cache.pool.allocator.requests
    assert engine.prefix_cache.pool.allocator.metrics().allocated_blocks == 0
    metrics = engine.metrics("req")
    assert metrics.generated_tokens == 2
    assert metrics.ttft_s == 1.0
    assert metrics.tpot_s == 2.0
    assert metrics.total_latency_s == 3.0


def test_cancel_releases_admitted_cache_but_waiting_cancel_needs_no_cache() -> None:
    model = _tiny_model()
    engine = ContinuousBatchingReferenceEngine(model, total_blocks=24, block_size=2)
    engine.submit("active", torch.tensor([[1, 3, 5]]), max_new_tokens=4, now=0.0)
    engine.step(now=1.0)
    assert "active" in engine.prefix_cache.pool.allocator.requests
    engine.cancel("active", now=1.5)
    assert engine.scheduler.requests["active"].status is RequestStatus.CANCELLED
    assert "active" not in engine.prefix_cache.pool.allocator.requests

    engine.submit("waiting", torch.tensor([[2, 4]]), max_new_tokens=2, now=2.0)
    engine.cancel("waiting", now=2.1)
    assert engine.scheduler.requests["waiting"].status is RequestStatus.CANCELLED
    assert "waiting" not in engine.prefix_cache.pool.allocator.requests


def test_run_until_idle_terminates_all_requests() -> None:
    model = _tiny_model()
    engine = ContinuousBatchingReferenceEngine(model, total_blocks=32, block_size=2)
    engine.submit("x", torch.tensor([[1, 2]]), max_new_tokens=3, now=0.0)
    engine.submit("y", torch.tensor([[3, 4, 5]]), max_new_tokens=1, now=0.0)

    iterations = engine.run_until_idle(start_time=1.0, tick_seconds=0.5)
    assert len(iterations) >= 1
    assert engine.scheduler.requests["x"].status is RequestStatus.FINISHED
    assert engine.scheduler.requests["y"].status is RequestStatus.FINISHED
    assert len(engine.output_tokens("x")) == 3
    assert len(engine.output_tokens("y")) == 1
    assert engine.prefix_cache.pool.allocator.metrics().allocated_blocks == 0
