from __future__ import annotations

import json

from astra_codex.agent import ScriptedBackend
from astra_codex.rollout_trace import RolloutTraceStore
from astra_codex.rollout_trace_collector import RolloutTraceCollector
from astra_codex.runtime import DurableAgentRuntime
from astra_codex.tools import ToolRegistry
from astra_codex.verification import VerificationResult, Verdict


def test_rollout_trace_hash_chain_detects_payload_tampering(tmp_path) -> None:
    path = tmp_path / "trace.sqlite"
    with RolloutTraceStore(path) as store:
        trace = store.ensure_trace("work_1", "thr_1", created_at=1.0)
        first = store.append(
            "work_1",
            "submission",
            {"goal": "fix bug"},
            thread_id="thr_1",
            created_at=2.0,
        )
        store.append(
            "work_1",
            "verification.completed",
            {"verdict": "pass"},
            thread_id="thr_1",
            parent_event_ids=(first,),
            created_at=3.0,
        )

        assert store.verify_integrity(trace.trace_id)
        exported = store.export_work_item("work_1")
        assert exported["integrityVerified"] is True
        assert exported["events"][1]["parentEventIds"] == [first]

        store.connection.execute(
            "UPDATE rollout_trace_events SET payload_json = ? WHERE event_id = ?",
            (json.dumps({"verdict": "fail"}), first),
        )
        store.connection.commit()
        assert not store.verify_integrity(trace.trace_id)


def test_trace_store_survives_reopen_and_rejects_cross_thread_rebinding(tmp_path) -> None:
    path = tmp_path / "trace.sqlite"
    with RolloutTraceStore(path) as store:
        store.ensure_trace("work_restart", "thr_a", created_at=1.0)
        store.append(
            "work_restart",
            "submission",
            {"content": "continue"},
            thread_id="thr_a",
            created_at=2.0,
        )

    with RolloutTraceStore(path) as reopened:
        trace = reopened.get_by_work_item("work_restart")
        assert trace is not None
        assert trace.thread_id == "thr_a"
        assert reopened.verify_integrity(trace.trace_id)

        try:
            reopened.ensure_trace("work_restart", "thr_b")
        except ValueError as exc:
            assert "already bound" in str(exc)
        else:  # pragma: no cover - makes the invariant explicit
            raise AssertionError("cross-thread trace rebinding should fail")


def test_collector_unifies_runtime_steering_artifact_and_verifier_evidence(tmp_path) -> None:
    state = tmp_path / "state"
    with DurableAgentRuntime(
        state,
        ScriptedBackend(["done after steering"]),
        ToolRegistry([]),
    ) as runtime:
        thread_id = runtime.create_thread("thr_trace")
        work_item_id = runtime.submit(
            thread_id,
            "inspect and fix",
            item_id="work_trace",
            now=1.0,
        )
        steering_id = runtime.steer(
            thread_id,
            "also inspect the documentation",
            steering_id="steer_trace",
            now=1.5,
        )
        record = runtime.run_one("worker_trace", now=2.0)
        assert record is not None and record.status == "completed"
        assert runtime.steering_queue.get(steering_id).status.value == "consumed"

        artifact_path = tmp_path / "patch.diff"
        artifact_path.write_text("diff --git a/a.py b/a.py\n", encoding="utf-8")
        artifact_id = runtime.snapshot_artifact(
            thread_id,
            artifact_path,
            kind="patch",
            metadata={"work_item_id": work_item_id, "producer": "worker_trace"},
            artifact_id="artifact_trace",
            now=3.0,
        )

        with RolloutTraceCollector(runtime) as collector:
            first_sync = collector.sync(work_item_id)
            assert first_sync.appended_events > 0
            assert first_sync.steering_consumptions_added == 1
            assert first_sync.artifact_events_added == 1

            trace_events = collector.store.events(first_sync.trace.trace_id)
            kinds = [event.kind for event in trace_events]
            assert "thread.submitted" in kinds
            assert "thread.steering_submitted" in kinds
            assert "steering.consumed" in kinds
            assert "harness.model_output" in kinds
            assert "harness.turn_completed" in kinds
            assert "work.finished" in kinds
            assert "artifact.snapshot" in kinds

            second_sync = collector.sync(work_item_id)
            assert second_sync.appended_events == 0

            verification_event_id = collector.record_verification(
                work_item_id,
                VerificationResult(
                    Verdict.PASS,
                    "hidden tests passed",
                    {"tests": 12, "source": "independent-grader"},
                ),
                artifact_ids=(artifact_id,),
                turn_id=record.turn_id,
                metadata={"grader": "fixture-v1"},
            )
            verification_event = next(
                event
                for event in collector.store.events(first_sync.trace.trace_id)
                if event.event_id == verification_event_id
            )
            assert verification_event.kind == "verification.completed"
            assert verification_event.parent_event_ids

            bundle = collector.export(work_item_id)
            assert bundle["integrityVerified"] is True
            assert bundle["summary"]["artifactCount"] == 1
            assert bundle["summary"]["verificationCount"] == 1
            assert bundle["artifacts"][0]["artifactId"] == artifact_id
            assert bundle["artifacts"][0]["sha256"]
            assert bundle["headHash"] != "0" * 64
