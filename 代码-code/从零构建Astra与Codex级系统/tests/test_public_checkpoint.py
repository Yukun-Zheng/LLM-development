import torch

from astra_codex.model import rotate_half
from astra_codex.public_checkpoint import llama_config_from_dict, llama_key_map


def test_smollm2_like_config_conversion() -> None:
    config = llama_config_from_dict(
        {
            "vocab_size": 49152,
            "hidden_size": 576,
            "num_hidden_layers": 30,
            "num_attention_heads": 9,
            "num_key_value_heads": 3,
            "intermediate_size": 1536,
            "max_position_embeddings": 8192,
            "rope_theta": 100000,
            "rope_interleaved": False,
            "rms_norm_eps": 1e-5,
            "tie_word_embeddings": True,
            "attention_bias": False,
            "mlp_bias": False,
            "rope_scaling": None,
        }
    )
    assert config.head_dim == 64
    assert config.kv_repeat == 3
    assert config.rope_interleaved is False
    assert config.rope_theta == 100000


def test_llama_key_map() -> None:
    assert llama_key_map("model.embed_tokens.weight") == "embed_tokens.weight"
    assert (
        llama_key_map("model.layers.7.self_attn.q_proj.weight")
        == "blocks.7.attn.q_proj.weight"
    )
    assert (
        llama_key_map("model.layers.2.mlp.down_proj.weight")
        == "blocks.2.ffn.down_proj.weight"
    )
    assert llama_key_map("model.norm.weight") == "final_norm.weight"


def test_non_interleaved_rotate_half_matches_llama_layout() -> None:
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    rotated = rotate_half(x, interleaved=False)
    assert torch.equal(rotated, torch.tensor([-3.0, -4.0, 1.0, 2.0]))


def test_interleaved_rotate_half_remains_available() -> None:
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    rotated = rotate_half(x, interleaved=True)
    assert torch.equal(rotated, torch.tensor([-2.0, 1.0, -4.0, 3.0]))
