from __future__ import annotations

import pytest

from astra_codex.kv_block_allocator import KVBlockAllocator


def test_allocator_allocates_blocks_and_reports_fragmentation() -> None:
    allocator = KVBlockAllocator(total_blocks=4, block_size=4)
    allocator.create_request("req")
    allocator.append_tokens("req", 6)

    table = allocator.block_table("req")
    assert [(entry.block_id, entry.logical_tokens) for entry in table] == [
        (0, 4),
        (1, 2),
    ]
    metrics = allocator.metrics()
    assert metrics.allocated_blocks == 2
    assert metrics.free_blocks == 2
    assert metrics.logical_tokens == 6
    assert metrics.physical_token_slots_used == 6
    assert metrics.physical_token_capacity == 8
    assert metrics.internal_fragmentation_slots == 2
    assert metrics.utilization == 0.75


def test_exact_fork_shares_blocks_then_partial_append_uses_copy_on_write() -> None:
    allocator = KVBlockAllocator(total_blocks=6, block_size=4)
    allocator.create_request("parent")
    allocator.append_tokens("parent", 6)
    allocator.fork_request("parent", "child")

    parent_before = allocator.block_table("parent")
    child_before = allocator.block_table("child")
    assert parent_before == child_before
    assert allocator.block_refcount(parent_before[0].block_id) == 2
    assert allocator.block_refcount(parent_before[1].block_id) == 2

    allocator.append_tokens("child", 1)
    parent_after = allocator.block_table("parent")
    child_after = allocator.block_table("child")

    assert parent_after[0].block_id == child_after[0].block_id
    assert parent_after[1].block_id != child_after[1].block_id
    assert parent_after[1].logical_tokens == 2
    assert child_after[1].logical_tokens == 3
    assert allocator.block_refcount(parent_after[1].block_id) == 1
    assert allocator.block_refcount(child_after[1].block_id) == 1


def test_partial_prefix_fork_shares_full_blocks_but_clones_partial_tail() -> None:
    allocator = KVBlockAllocator(total_blocks=8, block_size=4)
    allocator.create_request("source")
    allocator.append_tokens("source", 10)
    allocator.fork_request("source", "prefix", prefix_tokens=6)

    source = allocator.block_table("source")
    prefix = allocator.block_table("prefix")
    assert [(entry.logical_tokens) for entry in source] == [4, 4, 2]
    assert [(entry.logical_tokens) for entry in prefix] == [4, 2]
    assert source[0].block_id == prefix[0].block_id
    assert source[1].block_id != prefix[1].block_id
    assert allocator.block_refcount(source[0].block_id) == 2
    assert allocator.block_refcount(source[1].block_id) == 1
    assert allocator.block_refcount(prefix[1].block_id) == 1


def test_truncate_shared_block_shortens_logical_view_without_corrupting_sibling() -> None:
    allocator = KVBlockAllocator(total_blocks=6, block_size=4)
    allocator.create_request("a")
    allocator.append_tokens("a", 3)
    allocator.fork_request("a", "b")

    shared = allocator.block_table("a")[0].block_id
    allocator.truncate("b", 2)
    assert allocator.block_table("a")[0].logical_tokens == 3
    assert allocator.block_table("b")[0].logical_tokens == 2
    assert allocator.block_refcount(shared) == 2

    allocator.append_tokens("b", 1)
    assert allocator.block_table("a")[0].block_id == shared
    assert allocator.block_table("a")[0].logical_tokens == 3
    assert allocator.block_table("b")[0].block_id != shared
    assert allocator.block_table("b")[0].logical_tokens == 3


def test_release_request_reclaims_only_unreferenced_blocks() -> None:
    allocator = KVBlockAllocator(total_blocks=5, block_size=4)
    allocator.create_request("a")
    allocator.append_tokens("a", 8)
    allocator.fork_request("a", "b")

    shared_ids = [entry.block_id for entry in allocator.block_table("a")]
    allocator.release_request("a")
    assert allocator.metrics().allocated_blocks == 2
    assert all(allocator.block_refcount(block_id) == 1 for block_id in shared_ids)

    allocator.release_request("b")
    assert allocator.metrics().allocated_blocks == 0
    assert allocator.metrics().free_blocks == 5


def test_append_oom_is_atomic_even_when_multiple_blocks_would_be_needed() -> None:
    allocator = KVBlockAllocator(total_blocks=2, block_size=4)
    allocator.create_request("req")
    allocator.append_tokens("req", 3)
    before_table = allocator.block_table("req")
    before_metrics = allocator.metrics()

    with pytest.raises(MemoryError, match="append needs 2 free blocks"):
        allocator.append_tokens("req", 6)

    assert allocator.block_table("req") == before_table
    assert allocator.metrics() == before_metrics
    allocator.assert_invariants()


def test_copy_on_write_oom_is_atomic_for_shared_partial_block() -> None:
    allocator = KVBlockAllocator(total_blocks=1, block_size=4)
    allocator.create_request("a")
    allocator.append_tokens("a", 2)
    allocator.fork_request("a", "b")
    before_a = allocator.block_table("a")
    before_b = allocator.block_table("b")

    with pytest.raises(MemoryError, match="append needs 1 free blocks"):
        allocator.append_tokens("b", 1)

    assert allocator.block_table("a") == before_a
    assert allocator.block_table("b") == before_b
    assert allocator.block_refcount(before_a[0].block_id) == 2
    allocator.assert_invariants()
