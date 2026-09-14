import torch

from astra_codex.config import ModelConfig
from astra_codex.engine import GenerationEngine
from astra_codex.model import DecoderOnlyTransformer
from astra_codex.sampling import SamplingConfig


def tiny_model() -> DecoderOnlyTransformer:
    torch.manual_seed(0)
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


def test_forward_shape() -> None:
    model = tiny_model()
    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    output = model(ids)
    assert output.logits.shape == (2, 4, 64)


def test_incremental_cache_matches_full_forward() -> None:
    model = tiny_model()
    prefix = torch.tensor([[1, 2, 3, 4]])
    last = torch.tensor([[5]])

    full = model(torch.cat((prefix, last), dim=1)).logits[:, -1, :]
    prefill = model(prefix, use_cache=True)
    assert prefill.past_key_values is not None
    incremental = model(
        last,
        past_key_values=prefill.past_key_values,
        use_cache=True,
    ).logits[:, -1, :]

    torch.testing.assert_close(incremental, full, atol=1e-5, rtol=1e-5)


def test_generation_engine_greedy() -> None:
    model = tiny_model()
    engine = GenerationEngine(model)
    prompt = torch.tensor([[1, 2, 3]])
    output = engine.generate(
        prompt,
        max_new_tokens=4,
        sampling=SamplingConfig(temperature=0),
    )
    assert output.shape == (1, 7)
