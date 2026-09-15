from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .agent_graph import AgentMessage, AgentNode, PersistentAgentGraph
from .artifacts import ArtifactStore
from .parallel_agents import (
    ParallelAgentCoordinator,
    ParallelTask,
    ParallelTaskResult,
)
from .verification import CommandVerifier, VerificationResult, Verdict
from .worktree import Worktree, WorktreeManager


@dataclass(frozen=True, slots=True)
class CodingCandidate:
    task_id: str
    agent_id: str
    branch: str
    worktree_path: str
    patch_artifact_id: str
    verification: VerificationResult
    worker_summary: str
    worker_evidence: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ReviewerDecision:
    selected: CodingCandidate | None
    considered_task_ids: tuple[str, ...]
    rejected: dict[str, str]
    summary: str


@dataclass(frozen=True, slots=True)
class MergeResult:
    applied: bool
    task_id: str | None
    agent_id: str | None
    verification: VerificationResult
    summary: str


CodingWorker = Callable[[AgentNode, AgentMessage, Path], tuple[str, dict[str, Any] | None]]
ReviewerScore = Callable[[CodingCandidate], float]


class WorktreeCodingTeam:
    """Reference parallel coding team with isolated Git worktrees.

    The lifecycle is explicit:

        durable AgentGraph / mailbox
        -> one Git worktree per worker identity
        -> parallel worker callback
        -> binary patch capture
        -> independent CommandVerifier
        -> immutable patch Artifact
        -> reviewer selection
        -> git apply --check
        -> apply to coordinator worktree
        -> post-merge verification (rollback on failure)

    Worker callbacks may use an LLM/Agent, but this class never trusts their
    final prose as correctness evidence. A candidate is reviewer-eligible only
    when an independent verifier passes.

    This remains a local reference implementation: reviewer scoring is supplied
    by the caller, Git worktrees share one repository object database, and merge
    is patch application rather than a distributed transactional merge service.
    """

    def __init__(
        self,
        repository_root: str | Path,
        workspace_root: str | Path,
        artifact_store: ArtifactStore,
        graph: PersistentAgentGraph,
        coordinator_agent_id: str,
        worker_agent_ids: Iterable[str],
        *,
        max_workers: int | None = None,
    ) -> None:
        self.repository_root = Path(repository_root).resolve()
        self.workspace_root = Path(workspace_root).resolve()
        self.artifact_store = artifact_store
        self.graph = graph
        self.coordinator_agent_id = coordinator_agent_id
        self.worker_agent_ids = tuple(worker_agent_ids)
        self.coordinator = ParallelAgentCoordinator(
            graph,
            coordinator_agent_id,
            self.worker_agent_ids,
            max_workers=max_workers,
        )
        self.worktrees = WorktreeManager(self.repository_root, self.workspace_root)
        self._prepared: dict[str, Worktree] = {}
        self.base_commit = self._git(self.repository_root, "rev-parse", "HEAD").strip()

    @staticmethod
    def _git(cwd: Path, *args: str, input_text: str | None = None) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            input=input_text,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
        return completed.stdout

    def prepare_worktrees(self) -> dict[str, Worktree]:
        if self._prepared:
            return dict(self._prepared)
        for index, agent_id in enumerate(self.worker_agent_ids):
            safe_id = "".join(
                character if character.isalnum() or character in "-_" else "-"
                for character in agent_id
            )
            name = f"worker-{index:02d}-{safe_id}"
            self._prepared[agent_id] = self.worktrees.create(
                name,
                base=self.base_commit,
            )
        return dict(self._prepared)

    def _binary_patch(self, worktree: Worktree) -> str:
        return self._git(worktree.path, "diff", "--binary", self.base_commit, "--")

    def run_candidates(
        self,
        tasks: Iterable[ParallelTask],
        worker: CodingWorker,
        *,
        verify_argv: list[str] | tuple[str, ...],
        verify_timeout: float = 60.0,
    ) -> list[CodingCandidate]:
        if not verify_argv:
            raise ValueError("verify_argv cannot be empty")
        prepared = self.prepare_worktrees()

        def execute(agent: AgentNode, message: AgentMessage) -> ParallelTaskResult:
            worktree = prepared[agent.agent_id]
            summary, evidence = worker(agent, message, worktree.path)
            patch = self._binary_patch(worktree)
            verification = CommandVerifier(
                worktree.path,
                verify_argv,
                timeout=verify_timeout,
            ).verify()
            task_id = str(message.payload["taskId"])
            return ParallelTaskResult(
                task_id=task_id,
                agent_id=agent.agent_id,
                ok=verification.passed,
                summary=summary,
                evidence={
                    "worker": evidence or {},
                    "patch": patch,
                    "verification": {
                        "verdict": verification.verdict.value,
                        "summary": verification.summary,
                        "evidence": verification.evidence,
                    },
                    "branch": worktree.branch,
                    "worktreePath": str(worktree.path),
                },
            )

        parallel_results = self.coordinator.run(tasks, execute)
        candidates: list[CodingCandidate] = []
        coordinator_thread = self.graph.get_agent(self.coordinator_agent_id).thread_id
        for result in parallel_results:
            nested_verification = result.evidence.get("verification")
            if not isinstance(nested_verification, dict):
                verification = VerificationResult(
                    Verdict.FAIL,
                    "worker did not return independent verification evidence",
                    {},
                )
            else:
                raw_verdict = str(nested_verification.get("verdict", "fail"))
                try:
                    verdict = Verdict(raw_verdict)
                except ValueError:
                    verdict = Verdict.FAIL
                evidence = nested_verification.get("evidence")
                verification = VerificationResult(
                    verdict,
                    str(nested_verification.get("summary", "")),
                    evidence if isinstance(evidence, dict) else {},
                )

            patch = result.evidence.get("patch")
            if not isinstance(patch, str):
                patch = ""
            artifact_id = self.artifact_store.put_bytes(
                coordinator_thread,
                patch.encode("utf-8"),
                kind="candidate-patch",
                metadata={
                    "task_id": result.task_id,
                    "agent_id": result.agent_id,
                    "base_commit": self.base_commit,
                    "verification_verdict": verification.verdict.value,
                },
            )
            candidates.append(
                CodingCandidate(
                    task_id=result.task_id,
                    agent_id=result.agent_id,
                    branch=str(result.evidence.get("branch", "")),
                    worktree_path=str(result.evidence.get("worktreePath", "")),
                    patch_artifact_id=artifact_id,
                    verification=verification,
                    worker_summary=result.summary,
                    worker_evidence=(
                        result.evidence.get("worker")
                        if isinstance(result.evidence.get("worker"), dict)
                        else {}
                    ),
                )
            )
        return candidates

    @staticmethod
    def review(
        candidates: Iterable[CodingCandidate],
        *,
        score: ReviewerScore | None = None,
    ) -> ReviewerDecision:
        items = tuple(candidates)
        rejected: dict[str, str] = {}
        eligible: list[CodingCandidate] = []
        for candidate in items:
            if not candidate.verification.passed:
                rejected[candidate.task_id] = (
                    f"verification={candidate.verification.verdict.value}: "
                    f"{candidate.verification.summary}"
                )
                continue
            eligible.append(candidate)

        if not eligible:
            return ReviewerDecision(
                selected=None,
                considered_task_ids=tuple(candidate.task_id for candidate in items),
                rejected=rejected,
                summary="no candidate passed independent verification",
            )

        if score is None:
            selected = eligible[0]
        else:
            selected = max(eligible, key=lambda candidate: (score(candidate), candidate.task_id))
        for candidate in eligible:
            if candidate.task_id != selected.task_id:
                rejected[candidate.task_id] = "passed verification but was not selected by reviewer"
        return ReviewerDecision(
            selected=selected,
            considered_task_ids=tuple(candidate.task_id for candidate in items),
            rejected=rejected,
            summary=f"selected {selected.task_id} from {len(items)} candidates",
        )

    def merge_selected(
        self,
        decision: ReviewerDecision,
        *,
        verify_argv: list[str] | tuple[str, ...],
        verify_timeout: float = 60.0,
    ) -> MergeResult:
        candidate = decision.selected
        if candidate is None:
            return MergeResult(
                False,
                None,
                None,
                VerificationResult(Verdict.FAIL, "no reviewer-selected candidate", {}),
                "nothing was applied",
            )
        if not candidate.verification.passed:
            raise RuntimeError("reviewer selected a candidate that did not pass verification")

        dirty = subprocess.run(
            ["git", "diff", "--quiet", self.base_commit, "--"],
            cwd=self.repository_root,
            check=False,
        )
        if dirty.returncode != 0:
            raise RuntimeError("coordinator working tree has tracked changes relative to base commit")

        patch = self.artifact_store.read_bytes(candidate.patch_artifact_id).decode("utf-8")
        if not patch.strip():
            return MergeResult(
                False,
                candidate.task_id,
                candidate.agent_id,
                VerificationResult(Verdict.FAIL, "candidate patch is empty", {}),
                "reviewer-selected candidate had no tracked diff",
            )

        self._git(self.repository_root, "apply", "--check", "-", input_text=patch)
        self._git(self.repository_root, "apply", "-", input_text=patch)
        verification = CommandVerifier(
            self.repository_root,
            verify_argv,
            timeout=verify_timeout,
        ).verify()
        if not verification.passed:
            # The same patch was just applied to the unchanged base. Reverse it
            # before surfacing failure so a rejected merge does not pollute the
            # coordinator worktree.
            self._git(self.repository_root, "apply", "-R", "-", input_text=patch)
            return MergeResult(
                False,
                candidate.task_id,
                candidate.agent_id,
                verification,
                "candidate applied but post-merge verification failed; patch rolled back",
            )
        return MergeResult(
            True,
            candidate.task_id,
            candidate.agent_id,
            verification,
            "reviewer-selected patch applied and post-merge verification passed",
        )

    def cleanup(self, *, delete_branches: bool = True) -> None:
        for worktree in tuple(self._prepared.values()):
            try:
                self.worktrees.remove(
                    worktree,
                    force=True,
                    delete_branch=delete_branches,
                )
            finally:
                self._prepared.pop(worktree.name.split("-", 2)[-1], None)
        self._prepared.clear()

    def __enter__(self) -> "WorktreeCodingTeam":
        self.prepare_worktrees()
        return self

    def __exit__(self, *exc: object) -> None:
        self.cleanup()
