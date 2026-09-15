from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from astra_codex.agent_graph import PersistentAgentGraph
from astra_codex.artifacts import ArtifactStore
from astra_codex.coding_team import WorktreeCodingTeam
from astra_codex.parallel_agents import ParallelTask


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout)
    return completed.stdout.strip()


def _repository(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.email", "tests@example.com")
    _git(root, "config", "user.name", "Astra Codex Tests")
    (root / "answer.txt").write_text("0\n", encoding="utf-8")
    _git(root, "add", "answer.txt")
    _git(root, "commit", "-m", "initial")
    return root


def _graph(path: Path):
    graph = PersistentAgentGraph(path)
    root = graph.create_agent("thr_team", "coordinator", agent_id="coordinator")
    workers = [
        graph.create_agent(
            "thr_team",
            "worker",
            agent_id=f"worker_{index}",
            parent_agent_id=root,
        )
        for index in range(2)
    ]
    return graph, root, workers


def test_worktree_candidates_are_verified_reviewed_and_merged(tmp_path) -> None:
    repo = _repository(tmp_path)
    graph, coordinator, workers = _graph(tmp_path / "agents.sqlite")
    verifier = [
        sys.executable,
        "-c",
        "from pathlib import Path; raise SystemExit(0 if Path('answer.txt').read_text() == '42\\n' else 1)",
    ]

    def worker(agent, message, worktree):  # type: ignore[no-untyped-def]
        del agent
        value = str(message.payload["value"])
        (worktree / "answer.txt").write_text(value + "\n", encoding="utf-8")
        return f"wrote {value}", {"value": value}

    try:
        with ArtifactStore(tmp_path / "artifacts") as artifacts:
            team = WorktreeCodingTeam(
                repo,
                tmp_path / "worktrees",
                artifacts,
                graph,
                coordinator,
                workers,
            )
            try:
                candidates = team.run_candidates(
                    [
                        ParallelTask("candidate_good", {"value": "42"}),
                        ParallelTask("candidate_bad", {"value": "41"}),
                    ],
                    worker,
                    verify_argv=verifier,
                )
                assert len(candidates) == 2
                good = next(item for item in candidates if item.task_id == "candidate_good")
                bad = next(item for item in candidates if item.task_id == "candidate_bad")
                assert good.verification.passed
                assert not bad.verification.passed
                assert b"+42" in artifacts.read_bytes(good.patch_artifact_id)

                decision = team.review(candidates)
                assert decision.selected is not None
                assert decision.selected.task_id == "candidate_good"
                assert "candidate_bad" in decision.rejected

                merged = team.merge_selected(decision, verify_argv=verifier)
                assert merged.applied
                assert merged.verification.passed
                assert (repo / "answer.txt").read_text(encoding="utf-8") == "42\n"
            finally:
                team.cleanup()
    finally:
        graph.close()


def test_post_merge_verifier_failure_rolls_patch_back(tmp_path) -> None:
    repo = _repository(tmp_path)
    graph, coordinator, workers = _graph(tmp_path / "agents.sqlite")
    permissive = [sys.executable, "-c", "raise SystemExit(0)"]
    strict = [
        sys.executable,
        "-c",
        "from pathlib import Path; raise SystemExit(0 if Path('answer.txt').read_text() == '999\\n' else 1)",
    ]

    def worker(agent, message, worktree):  # type: ignore[no-untyped-def]
        del agent, message
        (worktree / "answer.txt").write_text("42\n", encoding="utf-8")
        return "candidate", {}

    try:
        with ArtifactStore(tmp_path / "artifacts") as artifacts:
            team = WorktreeCodingTeam(
                repo,
                tmp_path / "worktrees",
                artifacts,
                graph,
                coordinator,
                workers,
            )
            try:
                candidates = team.run_candidates(
                    [ParallelTask("candidate_a", {}), ParallelTask("candidate_b", {})],
                    worker,
                    verify_argv=permissive,
                )
                decision = team.review(candidates)
                assert decision.selected is not None
                result = team.merge_selected(decision, verify_argv=strict)
                assert not result.applied
                assert not result.verification.passed
                assert "rolled back" in result.summary
                assert (repo / "answer.txt").read_text(encoding="utf-8") == "0\n"
                assert _git(repo, "diff", "--", "answer.txt") == ""
            finally:
                team.cleanup()
    finally:
        graph.close()
