from __future__ import annotations

"""Numerically compare our from-scratch runtime with a real public checkpoint.

This script deliberately uses Transformers only as the independent reference
oracle. Our forward pass and weight loading come from astra_codex.
"""

import argparse
import json
from pathlib import Path

import torch

from astra_codex.public_checkpoint import (
    SMOLLM2_135M_MODEL_ID,
    compare_logits,
    load_llama_family_snapshot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=SMOLLM2_135M_MODEL_ID)
    parser.add_argument("--atol", type=float, default=3e-4)
    parser.add_argument("--mean-atol", type=float, default=2e-5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise SystemExit("Install parity extras: pip install -e '.[parity]'") from exc

    snapshot = Path(
        snapshot_download(
            repo_id=args.model_id,
            allow_patterns=["config.json", "*.safetensors", "*.safetensors.index.json"],
        )
    )

    ours = load_llama_family_snapshot(snapshot, dtype=torch.float32)
    reference = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).eval()

    # Explicit IDs make this a model-runtime parity test rather than a tokenizer
    # test. Every ID is well inside SmolLM2's public 49,152-token vocabulary.
    input_ids = torch.tensor(
        [
            [1, 42, 314, 7, 1234, 4096, 17, 23],
            [2, 99, 501, 888, 1001, 2048, 33, 44],
        ],
        dtype=torch.long,
    )

    with torch.no_grad():
        our_logits = ours(input_ids).logits
        reference_logits = reference(input_ids=input_ids, use_cache=False).logits

    metrics = compare_logits(our_logits, reference_logits)
    print(json.dumps(metrics, indent=2))

    # Two conditions are intentionally required: absolute numerical closeness
    # and exact next-token argmax agreement at every tested position.
    if metrics["max_abs"] > args.atol:
        raise SystemExit(
            f"FAIL: max_abs={metrics['max_abs']:.6g} exceeds atol={args.atol:.6g}"
        )
    if metrics["mean_abs"] > args.mean_atol:
        raise SystemExit(
            f"FAIL: mean_abs={metrics['mean_abs']:.6g} exceeds mean_atol={args.mean_atol:.6g}"
        )
    if metrics["argmax_agreement"] != 1.0:
        raise SystemExit(
            f"FAIL: argmax agreement is {metrics['argmax_agreement']:.3f}, expected 1.0"
        )

    print("PASS: from-scratch runtime matches the public reference logits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
