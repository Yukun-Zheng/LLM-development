from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .kv_tensor_pool import PhysicalKVTensorPool


@dataclass(slots=True)
class _PrefixNode:
    children: dict[int, "_PrefixNode"] = field(default_factory=dict)
    owners: set[str] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class PrefixMatch:
    source_request_id: str
    prefix_tokens: int


class TokenPrefixIndex:
    """Reference token-prefix trie used to discover reusable physical K/V state.

    Every trie node records which live requests contain that token prefix. This
    intentionally favors inspectability over production memory efficiency. A
    radix/block-hash index can replace it later without changing the tensor-pool
    sharing contract.
    """

    def __init__(self) -> None:
        self.root = _PrefixNode()
        self._tokens: dict[str, tuple[int, ...]] = {}
        self._paths: dict[str, list[_PrefixNode]] = {}

    def register(self, request_id: str, tokens: tuple[int, ...]) -> None:
        if not request_id:
            raise ValueError("request_id cannot be empty")
        if not tokens:
            raise ValueError("tokens cannot be empty")
        if request_id in self._tokens:
            raise ValueError(f"request already indexed: {request_id}")

        node = self.root
        path: list[_PrefixNode] = []
        for token in tokens:
            node = node.children.setdefault(token, _PrefixNode())
            node.owners.add(request_id)
            path.append(node)
        self._tokens[request_id] = tokens
        self._paths[request_id] = path

    def unregister(self, request_id: str) -> None:
        try:
            path = self._paths.pop(request_id)
        except KeyError as exc:
            raise KeyError(f"request is not indexed: {request_id}") from exc
        self._tokens.pop(request_id)
        for node in path:
            node.owners.discard(request_id)

    def tokens(self, request_id: str) -> tuple[int, ...]:
        try:
            return self._tokens[request_id]
        except KeyError as exc:
            raise KeyError(f"request is not indexed: {request_id}") from exc

    def longest_match(
        self,
        tokens: tuple[int, ...],
        *,
        exclude_request_id: str | None = None,
    ) -> PrefixMatch | None:
        node = self.root
        best: PrefixMatch | None = None
        for depth, token in enumerate(tokens, 1):
            child = node.children.get(token)
            if child is None:
                break
            node = child
            candidates = child.owners
            if exclude_request_id is not None:
                candidates = candidates - {exclude_request_id}
            if candidates:
                # Deterministic owner choice keeps tests/replays stable.
                best = PrefixMatch(sorted(candidates)[0], depth)
        return best


@dataclass(frozen=True, slots=True)
class PhysicalPrefixPrefillResult:
    logits: torch.Tensor
    reused_from: str | None
    reused_tokens: int
    computed_tokens: int


class PhysicalPrefixCacheEngine:
    """Reference prefix cache backed by shared physical K/V tensor blocks.

    The engine combines three mechanisms:

    1. ``TokenPrefixIndex`` discovers the longest live token prefix;
    2. ``PhysicalKVTensorPool.fork_request`` shares complete physical blocks and
       privately copies a partial tail when required;
    3. only the unmatched suffix is forwarded through the model.

    The underlying educational Transformer still consumes contiguous history,
    so a forked block table is materialized before the suffix forward. This
    validates prefix reuse, refcounts, COW and numerical parity before a future
    page-aware attention kernel removes the gather.
    """

    def __init__(
        self,
        model,
        *,
        total_blocks: int = 128,
        block_size: int = 16,
    ) -> None:
        self.model = model
        config = model.config
        parameter = next(model.parameters())
        self.pool = PhysicalKVTensorPool(
            num_layers=config.num_layers,
            total_blocks=total_blocks,
            num_kv_heads=config.num_kv_heads,
            block_size=block_size,
            head_dim=config.head_dim,
            dtype=parameter.dtype,
            device=parameter.device,
        )
        self.index = TokenPrefixIndex()
        self._position_logits: dict[str, torch.Tensor] = {}

    @staticmethod
    def _token_tuple(input_ids: torch.Tensor) -> tuple[int, ...]:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("physical prefix cache currently expects input_ids [1,T]")
        if input_ids.shape[1] == 0:
            raise ValueError("input_ids cannot be empty")
        return tuple(int(token) for token in input_ids[0].tolist())

    @torch.inference_mode()
    def prefill(
        self,
        request_id: str,
        input_ids: torch.Tensor,
    ) -> PhysicalPrefixPrefillResult:
        if request_id in self.pool.allocator.requests:
            raise ValueError(f"request already exists: {request_id}")
        tokens = self._token_tuple(input_ids)
        match = self.index.longest_match(tokens)

        reused_from: str | None = None
        reused_tokens = 0
        target_created = False
        try:
            if match is None:
                self.pool.create_request(request_id)
                target_created = True
                output = self.model(input_ids, use_cache=True)
                assert output.past_key_values is not None
                self.pool.ingest_present(request_id, output.past_key_values)
                position_logits = output.logits[0].detach().clone()
                computed_tokens = len(tokens)
            else:
                reused_from = match.source_request_id
                reused_tokens = match.prefix_tokens
                self.pool.fork_request(
                    reused_from,
                    request_id,
                    prefix_tokens=reused_tokens,
                )
                target_created = True
                source_logits = self._position_logits[reused_from]
                prefix_logits = source_logits[:reused_tokens].clone()

                if reused_tokens == len(tokens):
                    # Causal logits at an existing prefix position are identical
                    # regardless of later tokens in the source request.
                    position_logits = prefix_logits
                    computed_tokens = 0
                else:
                    suffix = input_ids[:, reused_tokens:]
                    output = self.model(
                        suffix,
                        past_key_values=self.pool.materialize(request_id),
                        use_cache=True,
                    )
                    assert output.past_key_values is not None
                    self.pool.ingest_present(request_id, output.past_key_values)
                    suffix_logits = output.logits[0].detach().clone()
                    position_logits = torch.cat((prefix_logits, suffix_logits), dim=0)
                    computed_tokens = int(suffix.shape[1])

            if position_logits.shape[0] != len(tokens):
                raise RuntimeError("prefix-cache logits length does not match request length")
            self._position_logits[request_id] = position_logits
            self.index.register(request_id, tokens)
            return PhysicalPrefixPrefillResult(
                logits=position_logits[-1:].clone(),
                reused_from=reused_from,
                reused_tokens=reused_tokens,
                computed_tokens=computed_tokens,
            )
        except Exception:
            self._position_logits.pop(request_id, None)
            if target_created and request_id in self.pool.allocator.requests:
                self.pool.release_request(request_id)
            raise

    @torch.inference_mode()
    def decode_one(self, request_id: str, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2 or token_ids.shape != (1, 1):
            raise ValueError("physical prefix decode expects token_ids [1,1]")
        old_tokens = self.index.tokens(request_id)
        output = self.model(
            token_ids,
            past_key_values=self.pool.materialize(request_id),
            use_cache=True,
        )
        assert output.past_key_values is not None
        self.pool.ingest_present(request_id, output.past_key_values)

        self.index.unregister(request_id)
        new_tokens = old_tokens + (int(token_ids[0, 0]),)
        self.index.register(request_id, new_tokens)
        self._position_logits[request_id] = torch.cat(
            (self._position_logits[request_id], output.logits[0, -1:].detach().clone()),
            dim=0,
        )
        return output.logits[:, -1, :]

    def release_request(self, request_id: str) -> None:
        self.index.unregister(request_id)
        self._position_logits.pop(request_id)
        self.pool.release_request(request_id)
