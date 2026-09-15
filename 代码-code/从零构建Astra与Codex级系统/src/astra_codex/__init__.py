"""From-scratch educational runtime for modern LLM and agent systems."""

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
    "DurableThreadStore",
    "DurableWorkQueue",
    "ExactAnswerGrader",
    "FileExistsVerifier",
    "FragmentKind",
    "GenerationEngine",
    "Grade",
    "GuardedToolExecutor",
    "IGNORE_INDEX",
    "ModelConfig",
    "ModelOutput",
    "PermissionDecision",
    "PermissionProfile",
    "PlanGraph",
    "PlanStep",
    "ReferencePagedGenerationEngine",
    "ReferencePagedKVCache",
    "ReferenceRequestScheduler",
    "RequestMetrics",
    "RequestState",
    "RequestStatus",
    "SamplingConfig",
    "ScheduledBatch",
    "SequenceLogProbs",
    "SequentialCoordinator",
    "StepStatus",
    "ThreadProjection",
    "ThreadStatus",
    "TrajectoryMetrics",
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
    "sequence_logprobs",
    "sft_batch_loss",
    "sft_step",
    "summarize_codex_turn",
]
