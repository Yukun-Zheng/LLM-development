from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True, slots=True)
class SpecialTokens:
    bos: str = "<bos>"
    eos: str = "<eos>"
    pad: str = "<pad>"


class ByteTokenizer:
    """A lossless UTF-8 byte tokenizer.

    IDs 0..255 map one-to-one to bytes.  Three special tokens follow.  This is
    deliberately primitive: it lets us test an LM runtime without hiding any
    segmentation logic behind a third-party tokenizer library.
    """

    def __init__(self, specials: SpecialTokens | None = None) -> None:
        self.specials = specials or SpecialTokens()
        self.bos_id = 256
        self.eos_id = 257
        self.pad_id = 258
        self.vocab_size = 259

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = list(text.encode("utf-8"))
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int], *, errors: str = "replace") -> str:
        raw = bytes(i for i in ids if 0 <= i < 256)
        return raw.decode("utf-8", errors=errors)


class ByteBPETokenizer:
    """Minimal byte-level BPE written for teaching rather than throughput.

    Each learned token corresponds to a byte string.  Training repeatedly finds
    the most frequent adjacent token pair and replaces it with a new token ID.
    Encoding replays learned merges in training order.

    This is intentionally smaller than production GPT tokenizers: no regex
    pre-tokenizer, vocabulary heuristics, or special normalization layer is
    hidden from the reader.
    """

    def __init__(self, merges: list[tuple[int, int]] | None = None) -> None:
        self.merges: list[tuple[int, int]] = list(merges or [])
        self.token_bytes: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for left, right in self.merges:
            new_id = len(self.token_bytes)
            self.token_bytes[new_id] = self.token_bytes[left] + self.token_bytes[right]
        self._refresh_special_ids()

    def _refresh_special_ids(self) -> None:
        base = len(self.token_bytes)
        self.bos_id = base
        self.eos_id = base + 1
        self.pad_id = base + 2
        self.vocab_size = base + 3

    @staticmethod
    def _merge_pair(sequence: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        left, right = pair
        out: list[int] = []
        i = 0
        while i < len(sequence):
            if i + 1 < len(sequence) and sequence[i] == left and sequence[i + 1] == right:
                out.append(new_id)
                i += 2
            else:
                out.append(sequence[i])
                i += 1
        return out

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        *,
        vocab_size: int = 512,
        min_pair_frequency: int = 2,
    ) -> "ByteBPETokenizer":
        if vocab_size < 259:
            raise ValueError("vocab_size must leave room for 256 bytes + 3 special tokens")
        sequences = [list(text.encode("utf-8")) for text in texts]
        tokenizer = cls()
        target_base_vocab = vocab_size - 3

        while len(tokenizer.token_bytes) < target_base_vocab:
            counts: Counter[tuple[int, int]] = Counter()
            for sequence in sequences:
                counts.update(zip(sequence, sequence[1:]))
            if not counts:
                break
            pair, frequency = counts.most_common(1)[0]
            if frequency < min_pair_frequency:
                break

            new_id = len(tokenizer.token_bytes)
            tokenizer.merges.append(pair)
            tokenizer.token_bytes[new_id] = (
                tokenizer.token_bytes[pair[0]] + tokenizer.token_bytes[pair[1]]
            )
            sequences = [tokenizer._merge_pair(seq, pair, new_id) for seq in sequences]

        tokenizer._refresh_special_ids()
        return tokenizer

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        sequence = list(text.encode("utf-8"))
        for merge_index, pair in enumerate(self.merges):
            new_id = 256 + merge_index
            sequence = self._merge_pair(sequence, pair, new_id)
        if add_bos:
            sequence.insert(0, self.bos_id)
        if add_eos:
            sequence.append(self.eos_id)
        return sequence

    def decode(self, ids: Iterable[int], *, errors: str = "replace") -> str:
        pieces: list[bytes] = []
        for token_id in ids:
            piece = self.token_bytes.get(token_id)
            if piece is not None:
                pieces.append(piece)
        return b"".join(pieces).decode("utf-8", errors=errors)

    def save(self, path: str | Path) -> None:
        payload = {"merges": self.merges}
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ByteBPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(merges=[tuple(pair) for pair in payload["merges"]])
