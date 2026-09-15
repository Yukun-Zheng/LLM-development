from __future__ import annotations

from dataclasses import dataclass

import torch

from .cache import KVCache


@dataclass(slots=True)
class BatchedDecodeState:
    request_id: str
    cache: KVCache

    @property
    def sequence_length(self) -> int:
        return self.cache.sequence_length


class HomogeneousBatchExecutor:
    """Actual batched prefill/decode using the educational Transformer.

    This executor performs *one real model forward for multiple requests*. To
    preserve the current model's simple causal mask, all requests in a decode
    batch must have the same cached sequence length. Production continuous
    batching removes this restriction with per-request block tables / sequence
    lengths and specialized attention kernels.
    """

    def __init__(self, model) -> None:  # type: ignore[no-untyped-def]
        self.model = model

    def _new_state(
        self,
        request_id: str,
        present_key_values,
        batch_index: int,
    ) -> BatchedDecodeState:  # type: ignore[no-untyped-def]
        cache = KVCache.empty(self.model.config.num_layers)
        cache.replace(
            tuple(
                (
                    key[batch_index : batch_index + 1].clone(),
                    value[batch_index : batch_index + 1].clone(),
                )
                for key, value in present_key_values
            )
        )
        return BatchedDecodeState(request_id=request_id, cache=cache)

    @torch.inference_mode()
    def prefill(
        self,
        request_ids: tuple[str, ...],
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, list[BatchedDecodeState]]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B, T]")
        if len(request_ids) != input_ids.shape[0]:
            raise ValueError("request_ids count must match batch size")
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("request_ids must be unique within a batch")

        output = self.model(input_ids, use_cache=True)
        assert output.past_key_values is not None
        states = [
            self._new_state(request_id, output.past_key_values, index)
            for index, request_id in enumerate(request_ids)
        ]
        return output.logits[:, -1, :], states

    def _stack_past_key_values(
        self, states: list[BatchedDecodeState]
    ):  # type: ignore[no-untyped-def]
        if not states:
            raise ValueError("decode batch cannot be empty")
        lengths = {state.sequence_length for state in states}
        if len(lengths) != 1:
            raise ValueError(
                "homogeneous executor requires equal cached sequence lengths; "
                f"got {sorted(lengths)}"
            )

        num_layers = self.model.config.num_layers
        for state in states:
            if len(state.cache.layers) != num_layers:
                raise ValueError("cache layer count mismatch")

        stacked = []
        for layer_index in range(num_layers):
            pairs = [state.cache.layers[layer_index] for state in states]
            if any(pair is None for pair in pairs):
                raise ValueError("decode state contains an empty cache layer")
            keys = torch.cat([pair[0] for pair in pairs if pair is not None], dim=0)
            values = torch.cat([pair[1] for pair in pairs if pair is not None], dim=0)
            stacked.append((keys, values))
        return tuple(stacked)

    @torch.inference_mode()
    def decode_one(
        self,
        states: list[BatchedDecodeState],
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        if token_ids.ndim != 2 or token_ids.shape[1] != 1:
            raise ValueError("token_ids must have shape [B, 1]")
        if token_ids.shape[0] != len(states):
            raise ValueError("token batch size must match state count")

        past = self._stack_past_key_values(states)
        output = self.model(token_ids, past_key_values=past, use_cache=True)
        assert output.past_key_values is not None

        for index, state in enumerate(states):
            state.cache.replace(
                tuple(
                    (
                        key[index : index + 1].clone(),
                        value[index : index + 1].clone(),
                    )
                    for key, value in output.past_key_values
                )
            )
        return output.logits[:, -1, :]
