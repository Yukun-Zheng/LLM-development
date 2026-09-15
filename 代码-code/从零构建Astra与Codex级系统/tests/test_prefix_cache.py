from __future__ import annotations

import torch

from astra_codex.config import ModelConfig
from astra_codex.model import DecoderOnlyTransformer
from astra_codex.prefix_cache import ReferencePrefixCacheEngine


def tiny_model() -> DecoderOnlyTransformer:
    torch.manual_seed(41)
    return DecoderOnlyTransformer(
        ModelConfig(
            vocab_size=64,
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            intermediate_size=72,
            max_seq_len=32,
        )
    ).eval()


def test_partial_prefix_reuse_matches_full_prompt_forward() -> None:
    model = tiny_model()
    engine = ReferencePrefixCacheEngine(model)

    prefix = torch.tensor([[1, 2, 3]])
    first = engine.prefill(prefix)
    assert first.reused_tokens == 0
    assert first.computed_tokens == 3
    assert engine.model_forward_calls == 1

    extended = torch.tensor([[1, 2, 3, 4, 5]])
    reused = engine.prefill(extended)
    full = model(extended, use_cache=True)

    assert reused.reused_tokens == 3
    assert reused.computed_tokens == 2
    assert not reused.exact_hit
    assert engine.model_forward_calls == 2
    torch.testing.assert_close(reused.logits, full.logits[:, -1, :], atol=1e-5, rtol=1e-5)
    assert len(reused.past_key_values) == len(full.past_key_values or ())
    assert full.past_key_values is not None
    for (actual_k, actual_v), (expected_k, expected_v) in zip(
        reused.past_key_values, full.past_key_values, strict=True
    ):
        torch.testing.assert_close(actual_k, expected_k, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(actual_v, expected_v, atol=1e-5, rtol=1e-5)


def test_exact_prefix_hit_avoids_another_model_forward() -> None:
    model = tiny_model()
    engine = ReferencePrefixCacheEngine(model)
    prompt = torch.tensor([[7, 8, 9, 10]])

    original = engine.prefill(prompt)
    calls_before = engine.model_forward_calls
    exact = engine.prefill(prompt)

    assert exact.exact_hit
    assert exact.reused_tokens == 4
    assert exact.computed_tokens == 0
    assert engine.model_forward_calls == calls_before
    torch.testing.assert_close(exact.logits, original.logits)
    assert engine.store.stats.exact_hits == 1
    assert engine.store.stats.misses == 1
    assert engine.store.stats.hit_rate == 0.5


def test_longest_matching_prefix_is_selected() -> None:
    model = tiny_model()
    engine = ReferencePrefixCacheEngine(model)

    engine.prefill(torch.tensor([[1, 2]]))
    engine.prefill(torch.tensor([[1, 2, 3, 4]]))
    result = engine.prefill(torch.tensor([[1, 2, 3, 4, 5]]))

    assert result.reused_tokens == 4
    assert result.computed_tokens == 1
    assert engine.store.stats.partial_hits >= 2
