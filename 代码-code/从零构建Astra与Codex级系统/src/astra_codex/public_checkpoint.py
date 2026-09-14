from __future__ import annotations

"""Adapters for numerical parity against public real checkpoints.

The first target is HuggingFaceTB/SmolLM2-135M because it is a compact,
Apache-2.0 Llama-family model whose public config matches the mechanisms already
implemented in this repository: RMSNorm, non-interleaved Llama RoPE, GQA,
SwiGLU, tied embeddings, and bias-free projections.

This module does not use AutoModel to implement our forward pass.  It maps raw
public safetensors into DecoderOnlyTransformer.  A reference AutoModel is used
only by the parity script as an independent oracle.
"""

import json
import re
from pathlib import Path
from typing import Any

import torch

from .config import ModelConfig
from .model import DecoderOnlyTransformer
from .weights import load_into_model, load_safetensors, remap_state_dict, shape_report

SMOLLM2_135M_MODEL_ID = "HuggingFaceTB/SmolLM2-135M"

_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def llama_config_from_dict(raw: dict[str, Any]) -> ModelConfig:
    """Convert the public Llama-family config subset used by our runtime."""

    rope_scaling = raw.get("rope_scaling")
    if rope_scaling not in (None, {}):
        raise ValueError(
            "this educational adapter currently supports only checkpoints without rope_scaling"
        )
    hidden_size = int(raw["hidden_size"])
    num_heads = int(raw["num_attention_heads"])
    explicit_head_dim = raw.get("head_dim")
    if explicit_head_dim is not None and int(explicit_head_dim) != hidden_size // num_heads:
        raise ValueError("checkpoint uses an explicit head_dim not supported by ModelConfig")
    if bool(raw.get("attention_bias", False)):
        raise ValueError("attention projection bias is not supported by this adapter")
    if bool(raw.get("mlp_bias", False)):
        raise ValueError("MLP bias is not supported by this adapter")

    return ModelConfig(
        vocab_size=int(raw["vocab_size"]),
        hidden_size=hidden_size,
        num_layers=int(raw["num_hidden_layers"]),
        num_heads=num_heads,
        num_kv_heads=int(raw.get("num_key_value_heads", num_heads)),
        intermediate_size=int(raw["intermediate_size"]),
        max_seq_len=int(raw["max_position_embeddings"]),
        rope_theta=float(raw.get("rope_theta", 10_000.0)),
        rope_interleaved=bool(raw.get("rope_interleaved", False)),
        rms_norm_eps=float(raw.get("rms_norm_eps", 1e-6)),
        tie_embeddings=bool(raw.get("tie_word_embeddings", False)),
    )


def load_public_config(snapshot_dir: str | Path) -> tuple[dict[str, Any], ModelConfig]:
    path = Path(snapshot_dir) / "config.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("model_type") != "llama":
        raise ValueError(f"expected model_type='llama', got {raw.get('model_type')!r}")
    return raw, llama_config_from_dict(raw)


def llama_key_map(key: str) -> str | None:
    """Map Hugging Face Llama checkpoint names to our transparent runtime."""

    top_level = {
        "model.embed_tokens.weight": "embed_tokens.weight",
        "model.norm.weight": "final_norm.weight",
        "lm_head.weight": "lm_head.weight",
    }
    if key in top_level:
        return top_level[key]

    match = _LAYER_RE.match(key)
    if match is None:
        return None
    layer, suffix = match.groups()
    layer_map = {
        "input_layernorm.weight": f"blocks.{layer}.attn_norm.weight",
        "post_attention_layernorm.weight": f"blocks.{layer}.ffn_norm.weight",
        "self_attn.q_proj.weight": f"blocks.{layer}.attn.q_proj.weight",
        "self_attn.k_proj.weight": f"blocks.{layer}.attn.k_proj.weight",
        "self_attn.v_proj.weight": f"blocks.{layer}.attn.v_proj.weight",
        "self_attn.o_proj.weight": f"blocks.{layer}.attn.o_proj.weight",
        "mlp.gate_proj.weight": f"blocks.{layer}.ffn.gate_proj.weight",
        "mlp.up_proj.weight": f"blocks.{layer}.ffn.up_proj.weight",
        "mlp.down_proj.weight": f"blocks.{layer}.ffn.down_proj.weight",
    }
    return layer_map.get(suffix)


def load_llama_family_snapshot(
    snapshot_dir: str | Path,
    *,
    dtype: torch.dtype = torch.float32,
) -> DecoderOnlyTransformer:
    """Load raw public safetensors into our own DecoderOnlyTransformer."""

    _, config = load_public_config(snapshot_dir)
    model = DecoderOnlyTransformer(config).to(dtype=dtype)
    source = load_safetensors(snapshot_dir)
    mapped = remap_state_dict(source, llama_key_map)

    # Hugging Face safetensors may omit one alias when embeddings are tied.
    if config.tie_embeddings:
        if "embed_tokens.weight" not in mapped and "lm_head.weight" in mapped:
            mapped["embed_tokens.weight"] = mapped["lm_head.weight"]
        if "lm_head.weight" not in mapped and "embed_tokens.weight" in mapped:
            mapped["lm_head.weight"] = mapped["embed_tokens.weight"]

    diagnostics = shape_report(model, mapped)
    if diagnostics:
        preview = "\n".join(diagnostics[:20])
        raise ValueError(f"checkpoint does not match our runtime:\n{preview}")

    # Convert once on load; forward then runs entirely in the requested dtype.
    mapped = {key: tensor.to(dtype=dtype) for key, tensor in mapped.items()}
    missing, unexpected = load_into_model(model, mapped, strict=True)
    if missing or unexpected:  # strict=True should already reject, kept explicit for readability
        raise ValueError(f"missing={missing}, unexpected={unexpected}")
    model.eval()
    return model


def compare_logits(
    ours: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, float]:
    if ours.shape != reference.shape:
        raise ValueError(f"logit shape mismatch: ours={ours.shape}, reference={reference.shape}")
    delta = (ours.float() - reference.float()).abs()
    argmax_agreement = (ours.argmax(-1) == reference.argmax(-1)).float().mean()
    return {
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
        "argmax_agreement": float(argmax_agreement.item()),
    }
