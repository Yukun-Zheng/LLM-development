"""From-scratch educational runtime for modern LLM and agent systems."""

from .app_server import AgentAppClient, AgentAppServer, AppServerError, InProcessAppTransport
from .artifacts import ArtifactRecord, ArtifactStore
from .batch_executor import BatchedDecodeState, HomogeneousBatchExecutor
from .benchmark import BenchmarkCase, BenchmarkHarness, BenchmarkRecord, ExactAnswerGrader, Grade
from .bubblewrap_sandbox import BubblewrapExecTool, BubblewrapPolicy, BubblewrapSandbox
from .config import ModelConfig
from .context import ContextFragment, ContextStore, FragmentKind
from .control_auth import (
    AuthenticationError,
    AuthorizationError,
    BearerCredential,
    BearerTokenAuthorizer,
    Principal,
)
from .docker_sandbox import DockerSandbox, DockerSandboxExecTool, DockerSandboxPolicy
from .durable import DurableThreadStore, ThreadProjection, ThreadStatus
from .engine import GenerationEngine
from .evaluation import AggregateMetrics, TrajectoryMetrics, aggregate_metrics, summarize_codex_turn
from .event_stream import DurableEventStream, RuntimeEvent
from .http_app_server import HTTPAppTransport, LocalHTTPAppServer
from .instructions import InstructionSource, ProjectInstructionResolver, ResolvedInstructions
from .kv_block_allocator import (
    AllocatorMetrics,
    BlockTableEntry,
    KVBlockAllocator,
    PhysicalBlock,
    RequestBlockTable,
)
from .kv_tensor_pool import PhysicalBlockGenerationEngine, PhysicalKVTensorPool, TensorPoolStats
from .model import DecoderOnlyTransformer, ModelOutput
from .multi_agent import SequentialCoordinator, WorkerResult
from .page_aware_decode import HeterogeneousPageAwareDecodeReference
from .paged_cache import ReferencePagedGenerationEngine, ReferencePagedKVCache
from .physical_prefix_cache import (
    PhysicalPrefixCacheEngine,
    PhysicalPrefixPrefillResult,
    PrefixMatch,
    TokenPrefixIndex,
)
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
from .runtime_control import BackgroundLeaseHeartbeat, ControlledBackend, LeaseHeartbeat
from .runtime_queue import DurableWorkQueue, WorkItem, WorkStatus
from .sampling import SamplingConfig
from .sandbox import (
    RestrictedSubprocessSandbox,
    SandboxExecTool,
    SandboxExecution,
    SandboxLimits,
    SandboxPolicy,
)
from .scheduler import (
    BatchKind,
    ReferenceRequestScheduler,
    RequestMetrics,
    RequestState,
    RequestStatus,
    ScheduledBatch,
)
from .security import GuardedToolExecutor, PermissionDecision, PermissionProfile
from .sse_events import LocalSSEEventServer, SSEEventClient, SSEMessage
from .steering import DurableSteeringQueue, SteeringMessage, SteeringStatus
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
    "AgentAppClient",
    "AgentAppServer",
    "AggregateMetrics",
    "AllocatorMetrics",
    "AppServerError",
    "ArtifactRecord",
    "ArtifactStore",
    "AuthenticationError",
    "AuthorizationError",
    "BackgroundLeaseHeartbeat",
    "BatchKind",
    "BatchedDecodeState",
    "BearerCredential",
    "BearerTokenAuthorizer",
    "BenchmarkCase",
    "BenchmarkHarness",
    "BenchmarkRecord",
    "BlockTableEntry",
    "BubblewrapExecTool",
    "BubblewrapPolicy",
    "BubblewrapSandbox",
    "ByteBPETokenizer",
    "ByteTokenizer",
    "CommandVerifier",
    "CompositeVerifier",
    "ContextFragment",
    "ContextStore",
    "ControlledBackend",
    "DPOResult",
    "DecoderOnlyTransformer",
    "DockerSandbox",
    "DockerSandboxExecTool",
    "DockerSandboxPolicy",
    "DurableAgentRuntime",
    "DurableEventStream",
    "DurableSteeringQueue",
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
    "HTTPAppTransport",
    "HeterogeneousPageAwareDecodeReference",
    "HomogeneousBatchExecutor",
    "IGNORE_INDEX",
    "InProcessAppTransport",
    "InstructionSource",
    "JournaledToolExecutor",
    "KVBlockAllocator",
    "LeaseHeartbeat",
    "LocalHTTPAppServer",
    "LocalSSEEventServer",
    "ModelConfig",
    "ModelOutput",
    "PermissionDecision",
    "PermissionProfile",
    "PhysicalBlock",
    "PhysicalBlockGenerationEngine",
    "PhysicalKVTensorPool",
    "PhysicalPrefixCacheEngine",
    "PhysicalPrefixPrefillResult",
    "PlanGraph",
    "PlanStep",
    "PrefixCacheEntry",
    "PrefixCacheStats",
    "PrefixMatch",
    "PrefixPrefillResult",
    "Principal",
    "ProjectInstructionResolver",
    "ReferencePagedGenerationEngine",
    "ReferencePagedKVCache",
    "ReferencePrefixCacheEngine",
    "ReferencePrefixKVStore",
    "ReferenceRequestScheduler",
    "RepositoryEvalRecord",
    "RepositoryFixture",
    "RepositoryFixtureHarness",
    "RepositoryGrade",
    "RequestBlockTable",
    "RequestMetrics",
    "RequestState",
    "RequestStatus",
    "ResolvedInstructions",
    "RestrictedSubprocessSandbox",
    "RuntimeEvent",
    "RuntimeExecutionRecord",
    "SSEEventClient",
    "SSEMessage",
    "SamplingConfig",
    "SandboxExecTool",
    "SandboxExecution",
    "SandboxLimits",
    "SandboxPolicy",
    "ScheduledBatch",
    "SequenceLogProbs",
    "SequentialCoordinator",
    "SteeringMessage",
    "SteeringStatus",
    "StepStatus",
    "TensorPoolStats",
    "ThreadProjection",
    "ThreadStatus",
    "TokenPrefixIndex",
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
