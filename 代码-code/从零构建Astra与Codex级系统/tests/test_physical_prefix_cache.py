from __future__ import annotations

import torch
from torch import nn

from astra_codex.config import ModelConfig
from astra_codex.model import DecoderOnlyTransformer
from astra_codex.physical_prefix_cache import PhysicalPrefixCacheEngine, TokenPrefixIndex


def _tiny_model() -> DecoderOnlyTransformer:
    torch.manual_seed(31)
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


class CountingModel(nn.Module):
    def __init__(self, base: DecoderOnlyTransformer) -> None:
        super().__init__()
        self.base = base
        self.config = base.config
        self.calls = 0

    def forward(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        return self.base(*args, **kwargs)


def test_token_prefix_index_finds_longest_live_owner_and_unregisters() -> None:
    index = TokenPrefixIndex()
    index.register("a", (1, 2, 3, 4))
    index.register("b", (1, 2, 5))

    match = index.longest_match((1, 2, 3, 9))
    assert match is not None
    assert match.source_request_id == "a"
    assert match.prefix_tokens == 3

    index.unregister("a")
    match = index.longest_match((1, 2, 3, 9))
    assert match is not None
    assert match.source_request_id == "b"
    assert match.prefix_tokens == 2


def test_exact_prompt_hit_reuses_logits_without_another_model_forward() -> None:
    model = CountingModel(_tiny_model())
    engine = PhysicalPrefixCacheEngine(model, total_blocks=16, block_size=2)
    prompt = torch.tensor([[1, 2, 3, 4]])

    first = engine.prefill("a", prompt)
    assert model.calls == 1
    second = engine.prefill("b", prompt)

    assert model.calls == 1
    assert second.reused_from == "a"
    assert second.reused_tokens == 4
    assert second.computed_tokens == 0
    torch.testing.assert_close(second.logits, first.logits)

    table_a = engine.pool.allocator.block_table("a")
    table_b = engine.pool.allocator.block_table("b")
    assert tuple(item.block_id for item in table_a) == tuple(
        item.block_id for item in table_b
    )
    assert all(engine.pool.allocator.block_refcount(item.block_id) == 2 for item in table_a)


def test_partial_prefix_only_computes_suffix_and_matches_full_forward() -> None:
    base = _tiny_model()
    model = CountingModel(base)
    engine = PhysicalPrefixCacheEngine(model, total_blocks=24, block_size=2)
    source = torch.tensor([[7, 8, 9, 10, 11]])
    target = torch.tensor([[7, 8, 9, 20, 21, 22]])

    engine.prefill("source", source)
    calls_before = model.calls
    result = engine.prefill("target", target)

    assert model.calls == calls_before + 1
    assert result.reused_from == "source"
    assert result.reused_tokens == 3
    assert result.computed_tokens == 3
    reference = base(target).logits[:, -1, :]
    torch.testing.assert_close(result.logits, reference, atol=1e-5, rtol=1e-5)

    # Two full prefix tokens occupy a shared physical block. The third token is
    # a partial-block prefix and therefore lives in a private copied tail.
    source_table = engine.pool.allocator.block_table("source")
    target_table = engine.pool.allocator.block_table("target")
    assert source_table[0].block_id == target_table[0].block_id
    assert source_table[1].block_id != target_table[1].block_id
    assert engine.pool.allocator.block_refcount(source_table[0].block_id) == 2


def test_exact_shared_partial_tail_uses_cow_on_decode_and_preserves_source() -> None:
    base = _tiny_model()
    engine = PhysicalPrefixCacheEngine(base, total_blocks=24, block_size=4)
    prompt = torch.tensor([[1, 2, 3, 4, 5, 6]])

    engine.prefill("parent", prompt)
    engine.prefill("child", prompt)
    parent_before = tuple((k.clone(), v.clone()) for k, v in engine.pool.materialize("parent"))

    decoded = engine.decode_one("child", torch.tensor([[7]]))
    reference = base(torch.tensor([[1, 2, 3, 4, 5, 6, 7]])).logits[:, -1, :]
    torch.testing.assert_close(decoded, reference, atol=1e-5, rtol=1e-5)

    parent_after = engine.pool.materialize("parent")
    for (actual_k, actual_v), (expected_k, expected_v) in zip(
        parent_after, parent_before, strict=True
    ):
        torch.testing.assert_close(actual_k, expected_k)
        torch.testing.assert_close(actual_v, expected_v)

    parent_table = engine.pool.allocator.block_table("parent")
    child_table = engine.pool.allocator.block_table("child")
    assert parent_table[0].block_id == child_table[0].block_id
    assert parent_table[-1].block_id != child_table[-1].block_id


def test_releasing_prefix_source_keeps_shared_child_cache_valid() -> None:
    base = _tiny_model()
    engine = PhysicalPrefixCacheEngine(base, total_blocks=24, block_size=2)
    prompt = torch.tensor([[2, 4, 6, 8]])

    engine.prefill("source", prompt)
    engine.prefill("child", prompt)
    child_blocks = tuple(
        item.block_id for item in engine.pool.allocator.block_table("child")
    )
    engine.release_request("source")

    assert all(engine.pool.allocator.block_refcount(block_id) == 1 for block_id in child_blocks)
    decoded = engine.decode_one("child", torch.tensor([[10]]))
    reference = base(torch.tensor([[2, 4, 6, 8, 10]])).logits[:, -1, :]
    torch.testing.assert_close(decoded, reference, atol=1e-5, rtol=1e-5)
