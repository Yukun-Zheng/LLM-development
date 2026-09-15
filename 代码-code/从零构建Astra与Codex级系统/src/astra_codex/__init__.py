"""From-scratch educational runtime for modern LLM and agent systems."""

from .batch_executor import BatchedDecodeState, HomogeneousBatchExecutor
from .benchmark import BenchmarkCase, BenchmarkHarness, BenchmarkRecord, ExactAnswerGrader, Grade
from .config import ModelConfig
from .context import ContextFragment, ContextStore, FragmentKind
from .durable import DurableThreadStore, ThreadProjection, ThreadStatus
from .engine import GenerationEngine
from .evaluation import AggregateMetrics, TrajectoryMetrics, aggregate_metrics, summarize_codex_turn
from .model import DecoderOnlyTransformer, ModelOutput
from .multi_agent import SequentialCoordinator, WorkerResult
from .paged_cache import ReferencePagedGenerationEngine, ReferencePagedKVCache
from .planning import PlanGraph, PlanStep, StepStatus
from .posttraining import (
    DPOResult,
    IGNORE_INDEX,
    SequenceLogProbs,
    causal_lm_loss,
    dpo_batch_loss,
    dpo_from_logprobs,
    dpo_step,
    sequence_logprobs,
    sft_batch_loss,
    sft_step,
)
from .prefix_cache import (
    PrefixCacheEntry,
    PrefixCacheStats,
    PrefixPrefillResult,
    ReferencePrefixCacheEngine,
    ReferencePrefixKVStore,
)
from .repository_eval import RepositoryEvalRecord, RepositoryFixture, RepositoryFixtureHarness, RepositoryGrade
from .rlvr import GroupRelativeResult, grpo_style_objective, group_relative_advantages
from .runtime import DurableAgentRuntime, RuntimeExecutionRecord
from .runtime_queue import DurableWorkQueue, WorkItem, WorkStatus
from .sampling import SamplingConfig
from .scheduler import (
    BatchKind,
    ReferenceRequestScheduler,
    RequestMetrics,
    RequestState,
    RequestStatus,
    ScheduledBatch,
)
from .security import GuardedToolExecutor, PermissionDecision, PermissionProfile
from .tokenizer import ByteBPETokenizer, ByteTokenizer
from .tool_journal import (
    DurableToolJournal,
    ExecutionStatus,
    JournaledToolExecutor,
    ToolExecutionRecord,
    TurnScopedJournaledTools,
)
from .verification import (
    CommandVerifier,
    CompositeVerifier,
    FileExistsVerifier,
    VerificationResult,
    Verdict,
)

__all__ = [
    "AggregateMetrics",
    "BatchKind",
    "BatchedDecodeState",
    "BenchmarkCase",
    "BenchmarkHarness",
    "BenchmarkRecord",
    "ByteBPETokenizer",
    "ByteTokenizer",
    "CommandVerifier",
    "CompositeVerifier",
    "ContextFragment",
    "ContextStore",
    "DPOResult",
    "DecoderOnlyTransformer",
    "DurableAgentRuntime",
    "DurableThreadStore",
    "DurableToolJournal",
    "DurableWorkQueue",
    "ExactAnswerGrader",
    "ExecutionStatus",
    "FileExistsVerifier",
    "FragmentKind",
    "GenerationEngine",
    "Grade",
    "GroupRelativeResult",
    "GuardedToolExecutor",
    "HomogeneousBatchExecutor",
    "IGNORE_INDEX",
    "JournaledToolExecutor",
    "ModelConfig",
    "ModelOutput",
    "PermissionDecision",
    "PermissionProfile",
    "PlanGraph",
    "PlanStep",
    "PrefixCacheEntry",
    "PrefixCacheStats",
    "PrefixPrefillResult",
    "ReferencePagedGenerationEngine",
    "ReferencePagedKVCache",
    "ReferencePrefixCacheEngine",
    "ReferencePrefixKVStore",
    "ReferenceRequestScheduler",
    "RepositoryEvalRecord",
    "RepositoryFixture",
    "RepositoryFixtureHarness",
    "RepositoryGrade",
    "RequestMetrics",
    "RequestState",
    "RequestStatus",
    "RuntimeExecutionRecord",
    "SamplingConfig",
    "ScheduledBatch",
    "SequenceLogProbs",
    "SequentialCoordinator",
    "StepStatus",
    "ThreadProjection",
    "ThreadStatus",
    "ToolExecutionRecord",
    "TrajectoryMetrics",
    "TurnScopedJournaledTools",
    "VerificationResult",
    "Verdict",
    "WorkItem",
    "WorkStatus",
    "WorkerResult",
    "aggregate_metrics",
    "causal_lm_loss",
    "dpo_batch_loss",
    "dpo_from_logprobs",
    "dpo_step",
    "grpo_style_objective",
    "group_relative_advantages",
    "sequence_logprobs",
    "sft_batch_loss",
    "sft_step",
    "summarize_codex_turn",
]
