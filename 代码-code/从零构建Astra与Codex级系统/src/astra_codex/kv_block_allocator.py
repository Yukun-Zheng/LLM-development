from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PhysicalBlock:
    block_id: int
    used_tokens: int = 0
    refcount: int = 0


@dataclass(frozen=True, slots=True)
class BlockTableEntry:
    block_id: int
    logical_tokens: int


@dataclass(slots=True)
class RequestBlockTable:
    request_id: str
    entries: list[BlockTableEntry]
    sequence_length: int = 0


@dataclass(frozen=True, slots=True)
class KVBlockReservation:
    """Concrete physical block ids fenced away from the ordinary free list."""

    reservation_id: int
    block_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class AllocatorMetrics:
    total_blocks: int
    allocated_blocks: int
    reserved_blocks: int
    free_blocks: int
    logical_requests: int
    logical_tokens: int
    physical_token_slots_used: int
    physical_token_capacity: int
    internal_fragmentation_slots: int
    shared_block_references: int

    @property
    def utilization(self) -> float:
        if self.physical_token_capacity == 0:
            return 0.0
        return self.physical_token_slots_used / self.physical_token_capacity


class KVBlockAllocator:
    """Reference physical-block allocator for paged KV-cache semantics.

    Besides request ownership/refcounts/COW, the allocator now supports a local
    concrete-block reservation. Reserved ids are removed from the ordinary free
    list before model execution, so another allocation through the same allocator
    cannot steal capacity that a batch has already fenced.

    This remains an in-process reference mechanism. It is not a distributed
    lease, cross-process transaction, or multi-node cache-coherence protocol.
    """

    def __init__(self, total_blocks: int, *, block_size: int = 16) -> None:
        if total_blocks <= 0:
            raise ValueError("total_blocks must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.total_blocks = total_blocks
        self.block_size = block_size
        self.blocks = [PhysicalBlock(block_id=i) for i in range(total_blocks)]
        self._free: list[int] = list(reversed(range(total_blocks)))
        self._reserved: set[int] = set()
        self._reservations: dict[int, list[int]] = {}
        self._next_reservation_id = 1
        self.requests: dict[str, RequestBlockTable] = {}

    def create_request(self, request_id: str) -> None:
        if not request_id:
            raise ValueError("request_id cannot be empty")
        if request_id in self.requests:
            raise ValueError(f"request already exists: {request_id}")
        self.requests[request_id] = RequestBlockTable(request_id, [])

    def _table(self, request_id: str) -> RequestBlockTable:
        try:
            return self.requests[request_id]
        except KeyError as exc:
            raise KeyError(f"unknown request: {request_id}") from exc

    def reserve_blocks(self, count: int) -> KVBlockReservation:
        """Remove concrete block ids from the free list until release/consume."""

        if count < 0:
            raise ValueError("reservation count cannot be negative")
        if count > len(self._free):
            raise MemoryError(
                f"KV block pool exhausted: reservation needs {count} free blocks, "
                f"only {len(self._free)} available"
            )
        reservation_id = self._next_reservation_id
        self._next_reservation_id += 1
        block_ids = [self._free.pop() for _ in range(count)]
        self._reserved.update(block_ids)
        self._reservations[reservation_id] = list(block_ids)
        reservation = KVBlockReservation(reservation_id, tuple(block_ids))
        self.assert_invariants()
        return reservation

    def reservation_remaining(self, reservation: KVBlockReservation) -> int:
        pending = self._reservation_list(reservation)
        return len(pending)

    def _reservation_list(self, reservation: KVBlockReservation) -> list[int]:
        try:
            pending = self._reservations[reservation.reservation_id]
        except KeyError as exc:
            raise RuntimeError("reservation is unknown or already released") from exc
        if set(pending) - set(reservation.block_ids):
            raise RuntimeError("reservation metadata corruption")
        return pending

    def release_reservation(self, reservation: KVBlockReservation) -> None:
        pending = self._reservation_list(reservation)
        for block_id in pending:
            if block_id not in self._reserved:
                raise RuntimeError("reserved block tracking corruption")
            block = self.blocks[block_id]
            if block.refcount != 0 or block.used_tokens != 0:
                raise RuntimeError("unconsumed reserved block has allocated state")
            self._reserved.remove(block_id)
            self._free.append(block_id)
        del self._reservations[reservation.reservation_id]
        self.assert_invariants()

    def _take_reserved_block(self, reservation: KVBlockReservation) -> int:
        pending = self._reservation_list(reservation)
        if not pending:
            raise MemoryError("KV block reservation exhausted")
        block_id = pending.pop()
        if block_id not in self._reserved:
            raise RuntimeError("reserved block tracking corruption")
        self._reserved.remove(block_id)
        return block_id

    def _allocate_block(
        self,
        used_tokens: int = 0,
        *,
        reservation: KVBlockReservation | None = None,
    ) -> int:
        if not 0 <= used_tokens <= self.block_size:
            raise ValueError("used_tokens outside block capacity")
        if reservation is None:
            if not self._free:
                raise MemoryError("KV block pool exhausted")
            block_id = self._free.pop()
        else:
            block_id = self._take_reserved_block(reservation)
        block = self.blocks[block_id]
        if block.refcount != 0 or block.used_tokens != 0:
            raise RuntimeError("allocator selected a block that is not actually free")
        block.used_tokens = used_tokens
        block.refcount = 1
        return block_id

    def _retain(self, block_id: int) -> None:
        block = self.blocks[block_id]
        if block.refcount <= 0:
            raise RuntimeError("cannot retain an unallocated block")
        block.refcount += 1

    def _release_ref(self, block_id: int) -> None:
        block = self.blocks[block_id]
        if block.refcount <= 0:
            raise RuntimeError("block refcount underflow")
        block.refcount -= 1
        if block.refcount == 0:
            block.used_tokens = 0
            self._free.append(block_id)

    def _copy_on_write_last_block(
        self,
        table: RequestBlockTable,
        *,
        reservation: KVBlockReservation | None = None,
    ) -> None:
        if not table.entries:
            return
        last = table.entries[-1]
        block = self.blocks[last.block_id]
        if block.refcount <= 1 or last.logical_tokens >= self.block_size:
            return

        cloned_id = self._allocate_block(
            last.logical_tokens,
            reservation=reservation,
        )
        table.entries[-1] = BlockTableEntry(cloned_id, last.logical_tokens)
        self._release_ref(last.block_id)

    def _required_blocks_for_append(
        self, table: RequestBlockTable, token_count: int
    ) -> int:
        if token_count <= 0:
            return 0
        remaining = token_count
        required = 0
        if table.entries:
            last = table.entries[-1]
            if last.logical_tokens < self.block_size:
                block = self.blocks[last.block_id]
                if block.refcount > 1:
                    required += 1
                free_slots = self.block_size - last.logical_tokens
                consumed = min(free_slots, remaining)
                remaining -= consumed
        if remaining > 0:
            required += (remaining + self.block_size - 1) // self.block_size
        return required

    def append_tokens(
        self,
        request_id: str,
        token_count: int,
        *,
        reservation: KVBlockReservation | None = None,
    ) -> None:
        if token_count < 0:
            raise ValueError("token_count cannot be negative")
        if token_count == 0:
            return
        table = self._table(request_id)

        required = self._required_blocks_for_append(table, token_count)
        available = (
            len(self._free)
            if reservation is None
            else self.reservation_remaining(reservation)
        )
        if required > available:
            source = "free blocks" if reservation is None else "reserved blocks"
            raise MemoryError(
                f"KV block pool exhausted: append needs {required} {source}, "
                f"only {available} available"
            )

        remaining = token_count
        while remaining > 0:
            if table.entries:
                last = table.entries[-1]
                if last.logical_tokens < self.block_size:
                    self._copy_on_write_last_block(
                        table,
                        reservation=reservation,
                    )
                    last = table.entries[-1]
                    block = self.blocks[last.block_id]
                    free_slots = self.block_size - last.logical_tokens
                    take = min(free_slots, remaining)
                    new_used = last.logical_tokens + take
                    block.used_tokens = max(block.used_tokens, new_used)
                    table.entries[-1] = BlockTableEntry(last.block_id, new_used)
                    table.sequence_length += take
                    remaining -= take
                    continue

            take = min(self.block_size, remaining)
            block_id = self._allocate_block(take, reservation=reservation)
            table.entries.append(BlockTableEntry(block_id, take))
            table.sequence_length += take
            remaining -= take

        self.assert_invariants()

    def fork_request(
        self,
        source_request_id: str,
        target_request_id: str,
        *,
        prefix_tokens: int | None = None,
    ) -> None:
        if target_request_id in self.requests:
            raise ValueError(f"request already exists: {target_request_id}")
        source = self._table(source_request_id)
        length = source.sequence_length if prefix_tokens is None else prefix_tokens
        if length < 0 or length > source.sequence_length:
            raise ValueError("prefix_tokens outside source sequence")

        target = RequestBlockTable(target_request_id, [])
        remaining = length
        retained: list[int] = []
        allocated_partial: list[int] = []
        try:
            for entry in source.entries:
                if remaining <= 0:
                    break
                if remaining >= entry.logical_tokens:
                    self._retain(entry.block_id)
                    retained.append(entry.block_id)
                    target.entries.append(entry)
                    target.sequence_length += entry.logical_tokens
                    remaining -= entry.logical_tokens
                    continue

                cloned = self._allocate_block(remaining)
                allocated_partial.append(cloned)
                target.entries.append(BlockTableEntry(cloned, remaining))
                target.sequence_length += remaining
                remaining = 0
                break
        except Exception:
            for block_id in retained:
                self._release_ref(block_id)
            for block_id in allocated_partial:
                self._release_ref(block_id)
            raise

        self.requests[target_request_id] = target
        self.assert_invariants()

    def truncate(self, request_id: str, new_length: int) -> None:
        table = self._table(request_id)
        if new_length < 0 or new_length > table.sequence_length:
            raise ValueError("new_length outside current sequence")
        if new_length == table.sequence_length:
            return

        kept: list[BlockTableEntry] = []
        remaining = new_length
        for entry in table.entries:
            if remaining <= 0:
                self._release_ref(entry.block_id)
                continue
            if remaining >= entry.logical_tokens:
                kept.append(entry)
                remaining -= entry.logical_tokens
                continue

            block = self.blocks[entry.block_id]
            if block.refcount == 1:
                block.used_tokens = remaining
            kept.append(BlockTableEntry(entry.block_id, remaining))
            remaining = 0

        table.entries = kept
        table.sequence_length = new_length
        self.assert_invariants()

    def release_request(self, request_id: str) -> None:
        table = self._table(request_id)
        for entry in table.entries:
            self._release_ref(entry.block_id)
        del self.requests[request_id]
        self.assert_invariants()

    def block_table(self, request_id: str) -> tuple[BlockTableEntry, ...]:
        return tuple(self._table(request_id).entries)

    def block_refcount(self, block_id: int) -> int:
        return self.blocks[block_id].refcount

    def metrics(self) -> AllocatorMetrics:
        allocated = [block for block in self.blocks if block.refcount > 0]
        logical_tokens = sum(table.sequence_length for table in self.requests.values())
        used_slots = sum(block.used_tokens for block in allocated)
        capacity = len(allocated) * self.block_size
        shared_refs = sum(max(0, block.refcount - 1) for block in allocated)
        return AllocatorMetrics(
            total_blocks=self.total_blocks,
            allocated_blocks=len(allocated),
            reserved_blocks=len(self._reserved),
            free_blocks=len(self._free),
            logical_requests=len(self.requests),
            logical_tokens=logical_tokens,
            physical_token_slots_used=used_slots,
            physical_token_capacity=capacity,
            internal_fragmentation_slots=capacity - used_slots,
            shared_block_references=shared_refs,
        )

    def assert_invariants(self) -> None:
        free_set = set(self._free)
        if len(free_set) != len(self._free):
            raise RuntimeError("free list contains duplicate block ids")
        if free_set & self._reserved:
            raise RuntimeError("block cannot be both free and reserved")

        reservation_union: set[int] = set()
        reservation_count = 0
        for pending in self._reservations.values():
            reservation_union.update(pending)
            reservation_count += len(pending)
        if reservation_union != self._reserved or reservation_count != len(self._reserved):
            raise RuntimeError("reservation block tracking is inconsistent")

        expected_refs = [0 for _ in self.blocks]
        for request_id, table in self.requests.items():
            logical = 0
            for index, entry in enumerate(table.entries):
                if not 0 <= entry.block_id < self.total_blocks:
                    raise RuntimeError(f"{request_id}: invalid block id")
                if not 1 <= entry.logical_tokens <= self.block_size:
                    raise RuntimeError(f"{request_id}: invalid logical block length")
                if index < len(table.entries) - 1 and entry.logical_tokens != self.block_size:
                    raise RuntimeError(
                        f"{request_id}: non-final block is not logically full"
                    )
                block = self.blocks[entry.block_id]
                if entry.logical_tokens > block.used_tokens:
                    raise RuntimeError(
                        f"{request_id}: logical view exceeds physical block contents"
                    )
                logical += entry.logical_tokens
                expected_refs[entry.block_id] += 1
            if logical != table.sequence_length:
                raise RuntimeError(
                    f"{request_id}: block table length {logical} != sequence length "
                    f"{table.sequence_length}"
                )

        for block in self.blocks:
            expected = expected_refs[block.block_id]
            if block.refcount != expected:
                raise RuntimeError(
                    f"block {block.block_id}: refcount {block.refcount} != {expected}"
                )
            if expected == 0:
                location_count = int(block.block_id in free_set) + int(
                    block.block_id in self._reserved
                )
                if location_count != 1 or block.used_tokens != 0:
                    raise RuntimeError(
                        f"unallocated block {block.block_id} is neither exclusively free nor reserved"
                    )
            else:
                if block.block_id in free_set or block.block_id in self._reserved:
                    raise RuntimeError(
                        f"allocated block {block.block_id} is free/reserved"
                    )
                if not 1 <= block.used_tokens <= self.block_size:
                    raise RuntimeError(f"allocated block {block.block_id} has invalid use")
