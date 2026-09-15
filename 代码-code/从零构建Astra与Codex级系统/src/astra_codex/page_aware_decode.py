from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .kv_tensor_pool import KVPair, PhysicalKVTensorPool
from .model import DecoderOnlyTransformer


class HeterogeneousPageAwareDecodeReference:
    """One-token mixed-length decode over physical K/V block tables.

    This reference path is the first inference backend in the project whose
    attention computation reads K/V history directly from ``PhysicalKVTensorPool``
    block ids. It does **not** call ``pool.materialize`` before attention.

    Q/K/V projections and FFNs run as a normal batch. The attention reduction is
    intentionally performed request-by-request because each request can have a
    different block table and cached sequence length. That makes the semantics
    inspectable before introducing a fused page-table kernel.

    Historical K/V chunks are never concatenated into one contiguous K/V tensor.
    Only attention *scores* are concatenated along the logical token dimension so
    one softmax can normalize across all blocks plus the current token.

    This is therefore a page-aware **reference** decode path, not yet a production
    continuous-batching kernel: there is no fused GPU gather, scheduler ownership,
    request preemption, or transactional multi-request reservation.
    """

    def __init__(
        self,
        model: DecoderOnlyTransformer,
        pool: PhysicalKVTensorPool,
    ) -> None:
        self.model = model
        self.pool = pool
        config = model.config
        parameter = next(model.parameters())
        expected = (
            config.num_layers,
            config.num_kv_heads,
            config.head_dim,
            parameter.dtype,
            parameter.device,
        )
        actual = (
            pool.num_layers,
            pool.num_kv_heads,
            pool.head_dim,
            pool.dtype,
            pool.device,
        )
        if actual != expected:
            raise ValueError(
                "physical K/V tensor pool does not match model layout/dtype/device: "
                f"expected={expected!r}, actual={actual!r}"
            )

    def _reshape_q(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        config = self.model.config
        return x.view(
            batch,
            tokens,
            config.num_heads,
            config.head_dim,
        ).transpose(1, 2)

    def _reshape_kv(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        config = self.model.config
        return x.view(
            batch,
            tokens,
            config.num_kv_heads,
            config.head_dim,
        ).transpose(1, 2)

    def _expand_kv_for_query_heads(self, value: torch.Tensor) -> torch.Tensor:
        repeat = self.model.config.kv_repeat
        return value if repeat == 1 else value.repeat_interleave(repeat, dim=1)

    def _attend_one_request(
        self,
        *,
        layer_index: int,
        request_id: str,
        q: torch.Tensor,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> torch.Tensor:
        """Return context [1,H_q,1,D] without materializing historical K/V."""

        scores: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        scale = 1.0 / math.sqrt(self.model.config.head_dim)

        for entry in self.pool.allocator.block_table(request_id):
            # Tensor slab layout is [L, block, H_kv, block_size, D].
            # Add only a batch dimension; do not concatenate blocks into K/V.
            k_block = self.pool.keys[
                layer_index,
                entry.block_id,
                :,
                : entry.logical_tokens,
                :,
            ].unsqueeze(0)
            v_block = self.pool.values[
                layer_index,
                entry.block_id,
                :,
                : entry.logical_tokens,
                :,
            ].unsqueeze(0)
            k_for_q = self._expand_kv_for_query_heads(k_block)
            v_for_q = self._expand_kv_for_query_heads(v_block)
            scores.append(torch.matmul(q, k_for_q.transpose(-2, -1)) * scale)
            values.append(v_for_q)

        k_current = self._expand_kv_for_query_heads(k_new)
        v_current = self._expand_kv_for_query_heads(v_new)
        scores.append(torch.matmul(q, k_current.transpose(-2, -1)) * scale)
        values.append(v_current)

        # It is safe to concatenate the scalar attention scores. The expensive
        # cached K/V tensors themselves remain in physical blocks.
        all_scores = torch.cat(scores, dim=-1)
        probabilities = F.softmax(all_scores.float(), dim=-1).to(dtype=q.dtype)

        context = torch.zeros_like(q)
        offset = 0
        for score_chunk, value_chunk in zip(scores, values, strict=True):
            length = int(score_chunk.shape[-1])
            weights = probabilities[..., offset : offset + length]
            context = context + torch.matmul(weights, value_chunk)
            offset += length
        if offset != probabilities.shape[-1]:
            raise RuntimeError("page-aware attention score/value chunk mismatch")
        return context

    @torch.inference_mode()
    def decode_batch(
        self,
        request_ids: list[str],
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Decode one token for requests with arbitrary cached sequence lengths.

        Args:
            request_ids: Request ids in batch order. Each must already have a
                physical cache in ``pool``.
            token_ids: ``[B,1]`` token ids aligned with ``request_ids``.

        Returns:
            Next-token logits ``[B,V]`` after consuming each supplied token.
        """

        if not request_ids:
            raise ValueError("request_ids cannot be empty")
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("request_ids must be unique within one decode batch")
        if token_ids.ndim != 2 or token_ids.shape != (len(request_ids), 1):
            raise ValueError("token_ids must have shape [len(request_ids),1]")

        old_lengths: list[int] = []
        for request_id in request_ids:
            try:
                length = self.pool.allocator.requests[request_id].sequence_length
            except KeyError as exc:
                raise KeyError(f"unknown physical-cache request: {request_id}") from exc
            if length + 1 > self.model.config.max_seq_len:
                raise ValueError(f"request {request_id!r} would exceed max_seq_len")
            old_lengths.append(length)

        x = self.model.embed_tokens(token_ids)
        new_pairs: dict[str, list[KVPair]] = {request_id: [] for request_id in request_ids}

        for layer_index, block in enumerate(self.model.blocks):
            normalized = block.attn_norm(x)
            q_batch = self._reshape_q(block.attn.q_proj(normalized))
            k_batch = self._reshape_kv(block.attn.k_proj(normalized))
            v_batch = self._reshape_kv(block.attn.v_proj(normalized))

            attention_outputs: list[torch.Tensor] = []
            for batch_index, (request_id, old_length) in enumerate(
                zip(request_ids, old_lengths, strict=True)
            ):
                q = q_batch[batch_index : batch_index + 1]
                k = k_batch[batch_index : batch_index + 1]
                v = v_batch[batch_index : batch_index + 1]
                position = torch.tensor([old_length], device=x.device)
                q, k = block.attn.rope(q, k, position)

                context = self._attend_one_request(
                    layer_index=layer_index,
                    request_id=request_id,
                    q=q,
                    k_new=k,
                    v_new=v,
                )
                context = (
                    context.transpose(1, 2)
                    .contiguous()
                    .view(1, 1, self.model.config.hidden_size)
                )
                attention_outputs.append(block.attn.o_proj(context))
                new_pairs[request_id].append((k, v))

            x = x + torch.cat(attention_outputs, dim=0)
            x = x + block.ffn(block.ffn_norm(x))

        logits = self.model.lm_head(self.model.final_norm(x))[:, -1, :]

        # Commit each request's one-token K/V only after all layer math succeeds.
        # This is per-request safe, but not yet an atomic reservation across the
        # whole heterogeneous batch. A production scheduler/allocator integration
        # must reserve capacity before model execution.
        for request_id in request_ids:
            self.pool.append_delta(request_id, tuple(new_pairs[request_id]))

        return logits
