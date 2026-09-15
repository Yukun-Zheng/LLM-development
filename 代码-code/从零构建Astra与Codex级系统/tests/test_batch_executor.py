from __future__ import annotations

import torch
import pytest

from astra_codex.batch_executor import HomogeneousBatchExecutor
from astra_codex.config import ModelConfig
from astra_codex.model import DecoderOnlyTransformer


def tiny_model() -> DecoderOnlyTransformer:
    torch.manual_seed(29)
    return DecoderOnlyTransformer(
        ModelConfig(
            vocab_size=64,
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            intermediate_size=72,
            max_seq_len=24,
        )
    ).eval()


def test_batched_prefill_matches_individual_forwards() -> None:
    model = tiny_model()
    executor = HomogeneousBatchExecutor(model)
    batch = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

    batched_logits, states = executor.prefill(("r1", "r2"), batch)

    expected_1 = model(batch[0:1]).logits[:, -1, :]
    expected_2 = model(batch[1:2]).logits[:, -1, :]
    torch.testing.assert_close(batched_logits[0:1], expected_1, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(batched_logits[1:2], expected_2, atol=1e-5, rtol=1e-5)
    assert [state.sequence_length for state in states] == [4, 4]


def test_batched_decode_matches_individual_full_recomputation() -> None:
    model = tiny_model()
    executor = HomogeneousBatchExecutor(model)
    prefix = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    _, states = executor.prefill(("r1", "r2"), prefix)
    next_tokens = torch.tensor([[9], [10]])

    batched_decode = executor.decode_one(states, next_tokens)

    expected_1 = model(torch.tensor([[1, 2, 3, 4, 9]])).logits[:, -1, :]
    expected_2 = model(torch.tensor([[5, 6, 7, 8, 10]])).logits[:, -1, :]
    torch.testing.assert_close(batched_decode[0:1], expected_1, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(batched_decode[1:2], expected_2, atol=1e-5, rtol=1e-5)
    assert [state.sequence_length for state in states] == [5, 5]


def test_homogeneous_executor_rejects_mixed_cache_lengths() -> None:
    model = tiny_model()
    executor = HomogeneousBatchExecutor(model)
    _, short = executor.prefill(("short",), torch.tensor([[1, 2, 3]]))
    _, long = executor.prefill(("long",), torch.tensor([[1, 2, 3, 4]]))

    with pytest.raises(ValueError, match="equal cached sequence lengths"):
        executor.decode_one(short + long, torch.tensor([[5], [6]]))
