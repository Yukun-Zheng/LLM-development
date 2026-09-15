from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .event_stream import RuntimeEvent
from .rollout_trace import RolloutTrace, RolloutTraceStore
from .runtime import DurableAgentRuntime
from .steering import SteeringStatus
from .verification import VerificationResult


@dataclass(frozen=True, slots=True)
class TraceSyncResult:
    trace: RolloutTrace
    appended_events: int
    source_events_seen: int
    artifact_events_added: int
    steering_consumptions_added: int


class RolloutTraceCollector:
    """Build a work-item evidence bundle from the runtime's durable stores.

    The collector intentionally treats the existing stores as sources of truth:

    * ``DurableEventStream`` contributes model/tool/control-plane events;
    * ``DurableSteeringQueue`` confirms whether a steering message was consumed;
    * ``ArtifactStore`` contributes immutable work products;
    * verifier results can be attached explicitly after independent grading.

    This design avoids making the control-plane event feed itself the audit
    record. Trace events copy the relevant evidence into an append-only,
    hash-chained bundle and retain ``source_event_id`` back-references.
    """

    def __init__(
        self,
        runtime: DurableAgentRuntime,
        store: RolloutTraceStore | None = None,
    ) -> None:
        self.runtime = runtime
        self.store = store or RolloutTraceStore(runtime.state_dir / "rollout-traces.sqlite")
        self._owns_store = store is None

    @staticmethod
    def _work_item_id(event: RuntimeEvent) -> str | None:
        value = event.payload.get("workItemId")
        return value if isinstance(value, str) and value else None

    def _all_thread_events(self, thread_id: str) -> list[RuntimeEvent]:
        events: list[RuntimeEvent] = []
        cursor = 0
        while True:
            page = self.runtime.event_stream.read(
                after_id=cursor,
                limit=1000,
                thread_id=thread_id,
            )
            if not page:
                break
            events.extend(page)
            cursor = page[-1].event_id
            if len(page) < 1000:
                break
        return events

    def _existing_source_ids(self, trace_id: str) -> set[int]:
        return {
            event.source_event_id
            for event in self.store.events(trace_id)
            if event.source_event_id is not None
        }

    def _existing_kinds_with_key(
        self,
        trace_id: str,
        kind: str,
        payload_key: str,
    ) -> set[str]:
        values: set[str] = set()
        for event in self.store.events(trace_id):
            if event.kind != kind:
                continue
            value = event.payload.get(payload_key)
            if isinstance(value, str):
                values.add(value)
        return values

    def sync(self, work_item_id: str) -> TraceSyncResult:
        item = self.runtime.work_queue.get(work_item_id)
        trace = self.store.ensure_trace(work_item_id, item.thread_id)
        runtime_events = self._all_thread_events(item.thread_id)

        directly_related = [
            event
            for event in runtime_events
            if self._work_item_id(event) == work_item_id
        ]
        turn_ids = {
            event.turn_id for event in directly_related if event.turn_id is not None
        }

        if directly_related:
            lower = min(event.event_id for event in directly_related)
            upper = max(event.event_id for event in directly_related)
        else:
            lower = 0
            upper = 0

        steering_source_events = [
            event
            for event in runtime_events
            if event.topic == "thread.steering_submitted"
            and lower <= event.event_id <= upper
        ]
        relevant_source_events = [
            event
            for event in runtime_events
            if self._work_item_id(event) == work_item_id
            or (event.turn_id is not None and event.turn_id in turn_ids)
            or event in steering_source_events
        ]
        relevant_source_events.sort(key=lambda event: event.event_id)

        existing_source_ids = self._existing_source_ids(trace.trace_id)
        appended = 0
        for event in relevant_source_events:
            if event.event_id in existing_source_ids:
                continue
            self.store.append(
                work_item_id,
                event.topic,
                dict(event.payload),
                thread_id=item.thread_id,
                turn_id=event.turn_id,
                source_event_id=event.event_id,
                created_at=event.created_at,
            )
            existing_source_ids.add(event.event_id)
            appended += 1

        existing_steering = self._existing_kinds_with_key(
            trace.trace_id, "steering.consumed", "steeringId"
        )
        steering_added = 0
        for source in steering_source_events:
            steering_id = source.payload.get("steeringId")
            if not isinstance(steering_id, str) or steering_id in existing_steering:
                continue
            steering = self.runtime.steering_queue.get(steering_id)
            if steering.status is not SteeringStatus.CONSUMED:
                continue
            self.store.append(
                work_item_id,
                "steering.consumed",
                {
                    "steeringId": steering.steering_id,
                    "content": steering.content,
                    "submittedAt": steering.created_at,
                    "consumedAt": steering.consumed_at,
                },
                thread_id=item.thread_id,
                turn_id=next(iter(turn_ids), None),
            )
            existing_steering.add(steering_id)
            steering_added += 1
            appended += 1

        existing_artifacts = {
            event.artifact_id
            for event in self.store.events(trace.trace_id)
            if event.artifact_id is not None
        }
        artifact_added = 0
        for artifact in self.runtime.artifact_store.list_thread(item.thread_id):
            metadata_work_item = artifact.metadata.get("work_item_id")
            if metadata_work_item != work_item_id:
                continue
            if artifact.artifact_id in existing_artifacts:
                continue
            self.store.append(
                work_item_id,
                "artifact.snapshot",
                {
                    "kind": artifact.kind,
                    "sha256": artifact.sha256,
                    "sizeBytes": artifact.size_bytes,
                    "metadata": artifact.metadata,
                },
                thread_id=item.thread_id,
                artifact_id=artifact.artifact_id,
                created_at=artifact.created_at,
            )
            existing_artifacts.add(artifact.artifact_id)
            artifact_added += 1
            appended += 1

        return TraceSyncResult(
            trace=trace,
            appended_events=appended,
            source_events_seen=len(relevant_source_events),
            artifact_events_added=artifact_added,
            steering_consumptions_added=steering_added,
        )

    def record_verification(
        self,
        work_item_id: str,
        result: VerificationResult,
        *,
        artifact_ids: tuple[str, ...] = (),
        turn_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        sync = self.sync(work_item_id)
        parent_ids: list[int] = []
        known_artifacts = set(artifact_ids)
        for event in self.store.events(sync.trace.trace_id):
            if event.artifact_id in known_artifacts:
                parent_ids.append(event.event_id)
        return self.store.append(
            work_item_id,
            "verification.completed",
            {
                "verdict": result.verdict.value,
                "summary": result.summary,
                "evidence": result.evidence,
                "artifactIds": list(artifact_ids),
                "metadata": metadata or {},
            },
            thread_id=sync.trace.thread_id,
            turn_id=turn_id,
            parent_event_ids=tuple(parent_ids),
        )

    def export(self, work_item_id: str) -> dict[str, Any]:
        sync = self.sync(work_item_id)
        bundle = self.store.export_work_item(work_item_id)
        artifacts: list[dict[str, Any]] = []
        for event in self.store.events(sync.trace.trace_id):
            if event.artifact_id is None:
                continue
            record = self.runtime.artifact_store.get(event.artifact_id)
            artifacts.append(
                {
                    "artifactId": record.artifact_id,
                    "kind": record.kind,
                    "sha256": record.sha256,
                    "sizeBytes": record.size_bytes,
                    "metadata": record.metadata,
                }
            )
        bundle["artifacts"] = artifacts
        bundle["summary"] = {
            "eventCount": len(bundle["events"]),
            "artifactCount": len(artifacts),
            "verificationCount": sum(
                1
                for event in bundle["events"]
                if event["kind"] == "verification.completed"
            ),
        }
        return bundle

    def close(self) -> None:
        if self._owns_store:
            self.store.close()

    def __enter__(self) -> "RolloutTraceCollector":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
