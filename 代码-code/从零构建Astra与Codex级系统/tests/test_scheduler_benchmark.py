from astra_codex.agent import ScriptedBackend
from astra_codex.benchmark import BenchmarkCase, BenchmarkHarness, ExactAnswerGrader
from astra_codex.codex_harness import CodexHarness
from astra_codex.scheduler import BatchKind, ReferenceRequestScheduler, RequestStatus
from astra_codex.tools import ToolRegistry


def test_reference_scheduler_batches_prefill_then_decode() -> None:
    scheduler = ReferenceRequestScheduler(max_batch_size=2, max_prefill_tokens=8)
    scheduler.submit("r1", prompt_tokens=5, max_new_tokens=2, now=0.0)
    scheduler.submit("r2", prompt_tokens=7, max_new_tokens=1, now=1.0)
    scheduler.submit("r3", prompt_tokens=3, max_new_tokens=2, now=2.0)

    batch = scheduler.next_batch()
    assert batch is not None
    assert batch.kind is BatchKind.PREFILL
    assert batch.request_ids == ("r1", "r3")
    assert batch.token_count == 8

    scheduler.mark_prefill_complete(batch.request_ids, now=3.0)
    decode = scheduler.next_batch()
    assert decode is not None
    assert decode.kind is BatchKind.DECODE
    assert decode.request_ids == ("r1", "r3")

    scheduler.mark_token("r1", now=4.0)
    scheduler.mark_token("r3", now=4.0)
    scheduler.mark_token("r1", now=5.0)
    scheduler.mark_token("r3", now=6.0)

    assert scheduler.requests["r1"].status is RequestStatus.FINISHED
    assert scheduler.requests["r3"].status is RequestStatus.FINISHED

    r1 = scheduler.metrics("r1")
    assert r1.ttft_s == 4.0
    assert r1.tpot_s == 1.0
    assert r1.total_latency_s == 5.0

    waiting = scheduler.next_batch()
    assert waiting is not None
    assert waiting.kind is BatchKind.PREFILL
    assert waiting.request_ids == ("r2",)


def test_reference_scheduler_cancel_removes_request_from_batches() -> None:
    scheduler = ReferenceRequestScheduler(max_batch_size=4, max_prefill_tokens=32)
    scheduler.submit("r1", prompt_tokens=3, max_new_tokens=2, now=0.0)
    scheduler.cancel("r1", now=1.0)
    assert scheduler.requests["r1"].status is RequestStatus.CANCELLED
    assert scheduler.next_batch() is None


def test_benchmark_harness_separates_execution_from_grading() -> None:
    def execute(case: BenchmarkCase):  # type: ignore[no-untyped-def]
        backend = ScriptedBackend(["42" if case.case_id == "ok" else "wrong"])
        return CodexHarness(backend, ToolRegistry([])).run_turn(case.goal)

    grader = ExactAnswerGrader({"ok": "42", "bad": "42"})
    harness = BenchmarkHarness(execute, grader)
    records = harness.run_suite(
        [
            BenchmarkCase("ok", "answer the toy task"),
            BenchmarkCase("bad", "answer the other toy task"),
        ]
    )

    assert records[0].grade.passed
    assert records[0].trajectory.success
    assert not records[1].grade.passed
    assert not records[1].trajectory.success

    aggregate = harness.aggregate(records)
    assert aggregate.cases == 2
    assert aggregate.success_rate == 0.5
