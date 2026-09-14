from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from torch import nn

KeyMap = Callable[[str], str | None]


def load_safetensors(path: str | Path) -> dict[str, torch.Tensor]:
    """Load one safetensors file or every shard in a directory.

    This function intentionally exposes raw tensors instead of delegating to a
    model-specific high-level loader.  The reader can inspect every key/shape
    before deciding how a public checkpoint maps into our implementation.
    """

    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("install the optional 'weights' extra: pip install .[weights]") from exc

    path = Path(path)
    files = [path] if path.is_file() else sorted(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors files found under {path}")

    state: dict[str, torch.Tensor] = {}
    for file in files:
        shard = load_file(str(file), device="cpu")
        overlap = state.keys() & shard.keys()
        if overlap:
            raise ValueError(f"duplicate tensor keys across shards: {sorted(overlap)[:5]}")
        state.update(shard)
    return state


def remap_state_dict(
    source: dict[str, torch.Tensor],
    key_map: KeyMap,
) -> dict[str, torch.Tensor]:
    mapped: dict[str, torch.Tensor] = {}
    for key, tensor in source.items():
        new_key = key_map(key)
        if new_key is None:
            continue
        if new_key in mapped:
            raise ValueError(f"multiple source tensors map to {new_key}")
        mapped[new_key] = tensor
    return mapped


def shape_report(model: nn.Module, state: dict[str, torch.Tensor]) -> list[str]:
    """Return human-readable missing/unexpected/shape-mismatch diagnostics."""

    target = model.state_dict()
    lines: list[str] = []
    for key in sorted(target):
        if key not in state:
            lines.append(f"MISSING  {key:<70} target={tuple(target[key].shape)}")
        elif tuple(target[key].shape) != tuple(state[key].shape):
            lines.append(
                f"SHAPE    {key:<70} target={tuple(target[key].shape)} "
                f"source={tuple(state[key].shape)}"
            )
    for key in sorted(state.keys() - target.keys()):
        lines.append(f"EXTRA    {key:<70} source={tuple(state[key].shape)}")
    return lines


def load_into_model(
    model: nn.Module,
    state: dict[str, torch.Tensor],
    *,
    strict: bool = True,
) -> tuple[list[str], list[str]]:
    incompatible = model.load_state_dict(state, strict=strict)
    return list(incompatible.missing_keys), list(incompatible.unexpected_keys)
