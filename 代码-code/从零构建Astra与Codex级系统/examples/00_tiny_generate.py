"""Run the from-scratch model with random tiny weights.

Random weights will not produce useful language; this example validates the
runtime path: text -> bytes -> model -> KV cache -> tokens -> text.
"""

import torch

from astra_codex.config import ModelConfig
from astra_codex.engine import GenerationEngine
from astra_codex.model import DecoderOnlyTransformer
from astra_codex.sampling import SamplingConfig
from astra_codex.tokenizer import ByteTokenizer


def main() -> None:
    torch.manual_seed(0)
    tokenizer = ByteTokenizer()
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=160,
        max_seq_len=128,
    )
    model = DecoderOnlyTransformer(config).eval()
    engine = GenerationEngine(model)

    ids = torch.tensor([tokenizer.encode("hello", add_bos=True)])
    output = engine.generate(
        ids,
        max_new_tokens=12,
        sampling=SamplingConfig(temperature=0),
    )
    print("token ids:", output[0].tolist())
    print("decoded:", tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()
