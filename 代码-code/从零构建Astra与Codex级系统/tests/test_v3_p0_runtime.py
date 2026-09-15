import torch

from astra_codex.config import ModelConfig
from astra_codex.context import ContextStore, FragmentKind
from astra_codex.model import DecoderOnlyTransformer
from astra_codex.paged_cache import ReferencePagedGenerationEngine
from astra_codex.runtime_queue import DurableWorkQueue, WorkStatus


def tiny_model() -> DecoderOnlyTransformer:
    torch.manual_seed(11)
    config = ModelConfig(
        vocab_size=64,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=80,
        max_seq_len=32,
    )
    return DecoderOnlyTransformer(config).eval()


def test_reference_paged_cache_matches_full_forward() -> None:
    model = tiny_model()
    engine = ReferencePagedGenerationEngine(model, page_size=2)
    prefix = torch.tensor([[1, 2, 3, 4, 5]])
    next_token = torch.tensor([[6]])

    prefill_logits, cache = engine.prefill(prefix)
    reference_prefill = model(prefix).logits[:, -1, :]
    torch.testing.assert_close(prefill_logits, reference_prefill)

    assert cache.sequence_length == 5
    assert cache.page_table() == ((2, 2, 1), (2, 2, 1))

    paged_decode = engine.decode_one(next_token, cache)
    full_decode = model(torch.cat((prefix, next_token), dim=1)).logits[:, -1, :]
    torch.testing.assert_close(paged_decode, full_decode, atol=1e-5, rtol=1e-5)
    assert cache.sequence_length == 6
    assert cache.page_table() == ((2, 2, 2), (2, 2, 2))


def test_context_compaction_keeps_raw_provenance(tmp_path) -> None:
    db = tmp_path / "context.sqlite"
    with ContextStore(db) as store:
        raw_a = store.add(
            FragmentKind.RAW_EVENT,
            "pytest failed in test_alpha",
            source="tool:shell",
            fragment_id="ctx_a",
        )
        raw_b = store.add(
            FragmentKind.RAW_EVENT,
            "fixed parser edge case",
            source="tool:filesystem",
            fragment_id="ctx_b",
        )
        summary = store.compact(
            [raw_a, raw_b],
            "Test failed, parser was repaired.",
            policy="educational-summary-v1",
        )

        lineage = store.lineage(summary)
        assert [item.fragment_id for item in lineage] == ["ctx_a", "ctx_b", summary]
        assert store.get(raw_a).content == "pytest failed in test_alpha"

        visible = store.select_for_context(
            max_chars=200, kinds={FragmentKind.SUMMARY}
        )
        assert [item.fragment_id for item in visible] == [summary]


def test_durable_work_queue_lease_expiry_and_reclaim(tmp_path) -> None:
    with DurableWorkQueue(tmp_path / "queue.sqlite") as queue:
        item_id = queue.enqueue(
            "thr_1", "turn", {"goal": "fix bug"}, item_id="work_1", now=100.0
        )

        first = queue.claim("worker_a", lease_seconds=10.0, now=101.0)
        assert first is not None and first.item_id == item_id
        assert first.status is WorkStatus.LEASED
        assert queue.claim("worker_b", lease_seconds=10.0, now=105.0) is None

        reclaimed = queue.claim("worker_b", lease_seconds=10.0, now=112.0)
        assert reclaimed is not None and reclaimed.item_id == item_id
        assert reclaimed.lease_owner == "worker_b"

        queue.ack(item_id, "worker_b", {"ok": True}, now=113.0)
        completed = queue.get(item_id)
        assert completed.status is WorkStatus.COMPLETED
        assert completed.result == {"ok": True}


def test_durable_work_queue_cancel_blocks_claim(tmp_path) -> None:
    with DurableWorkQueue(tmp_path / "queue.sqlite") as queue:
        item_id = queue.enqueue(
            "thr_2", "submission", {"content": "stop"}, item_id="work_cancel", now=1.0
        )
        queue.cancel(item_id, now=2.0)
        assert queue.get(item_id).status is WorkStatus.CANCELLED
        assert queue.claim("worker", now=3.0) is None
