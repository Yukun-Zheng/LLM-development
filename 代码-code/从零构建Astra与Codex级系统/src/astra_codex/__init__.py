"""From-scratch educational runtime for modern LLM and agent systems."""

from .a2a import (
    A2AAgentCard,
    A2AAgentInterface,
    A2AAgentSkill,
    A2AArtifact,
    A2AExecutionResult,
    A2AMessage,
    A2APart,
    A2AProtocolError,
    A2ARole,
    A2ASendMessageConfiguration,
    A2ASendMessageRequest,
    A2AService,
    A2ATask,
    A2ATaskState,
    A2ATaskStore,
)
from .a2a_http import (
    A2AHTTPClient,
    A2AHTTPError,
    A2AListTasksPage,
    LocalA2AHTTPServer,
)
from .a2a_runtime_bridge import DurableRuntimeA2AHandler
from .a2a_runtime_http import LocalA2ADurableRuntimeHTTPServer
from .a2a_sse import A2AStreamingHTTPClient, LocalA2AStreamingHTTPServer
from .a2a_streaming import (
    A2AStreamResponse,
    A2ATaskArtifactUpdateEvent,
    A2ATaskStatusUpdateEvent,
)
from .agent_graph import (
    AgentEvent,
    AgentMessage,
    AgentNode,
    AgentStatus,
    MessageStatus,
    PersistentAgentGraph,
)
from .app_server import AgentAppClient, AgentAppServer, AppServerError, InProcessAppTransport
from .artifacts import ArtifactRecord, ArtifactStore
from .batch_executor import BatchedDecodeState, HomogeneousBatchExecutor
from .benchmark import BenchmarkCase, BenchmarkHarness, BenchmarkRecord, ExactAnswerGrader, Grade
from .bubblewrap_sandbox import BubblewrapExecTool, BubblewrapPolicy, BubblewrapSandbox
from .coding_team import CodingCandidate, MergeResult, ReviewerDecision, WorktreeCodingTeam
from .config import ModelConfig
from .context import ContextFragment, ContextStore, FragmentKind
from .continuous_batching import (
    ContinuousBatchingReferenceEngine,
    ContinuousIteration,
    ContinuousRequest,
)
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
    KVBlockReservation,
    PhysicalBlock,
    RequestBlockTable,
)
from .kv_tensor_pool import PhysicalBlockGenerationEngine, PhysicalKVTensorPool, TensorPoolStats
from .model import DecoderOnlyTransformer, ModelOutput
from .multi_agent import SequentialCoordinator, WorkerResult
from .page_aware_decode import HeterogeneousPageAwareDecodeReference
from .paged_cache import ReferencePagedGenerationEngine, ReferencePagedKVCache
from .parallel_agents import ParallelAgentCoordinator, ParallelTask, ParallelTaskResult
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
from .rollout_trace import RolloutTrace, RolloutTraceEvent, RolloutTraceStore
from .rollout_trace_collector import RolloutTraceCollector, TraceSyncResult
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
    "A2AAgentCard",
    "A2AAgentInterface",
    "A2AAgentSkill",
    "A2AArtifact",
    "A2AExecutionResult",
    "A2AHTTPClient",
    "A2AHTTPError",
    "A2AListTasksPage",
    "A2AMessage",
    "A2APart",
    "A2AProtocolError",
    "A2ARole",
    "A2ASendMessageConfiguration",
    "A2ASendMessageRequest",
    "A2AService",
    "A2AStreamResponse",
    "A2AStreamingHTTPClient",
    "A2ATask",
    "A2ATaskArtifactUpdateEvent",
    "A2ATaskState",
    "A2ATaskStatusUpdateEvent",
    "A2ATaskStore",
    "AgentAppClient",
    "AgentAppServer",
    "AgentEvent",
    "AgentMessage",
    "AgentNode",
    "AgentStatus",
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
    "CodingCandidate",
    "CommandVerifier",
    "CompositeVerifier",
    "ContextFragment",
    "ContextStore",
    "ContinuousBatchingReferenceEngine",
    "ContinuousIteration",
    "ContinuousRequest",
    "ControlledBackend",
    "DPOResult",
    "DecoderOnlyTransformer",
    "DockerSandbox",
    "DockerSandboxExecTool",
    "DockerSandboxPolicy",
    "DurableAgentRuntime",
    "DurableEventStream",
    "DurableRuntimeA2AHandler",
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
    "KVBlockReservation",
    "LeaseHeartbeat",
    "LocalA2ADurableRuntimeHTTPServer",
    "LocalA2AHTTPServer",
    "LocalA2AStreamingHTTPServer",
    "LocalHTTPAppServer",
    "LocalSSEEventServer",
    "MergeResult",
    "MessageStatus",
    "ModelConfig",
    "ModelOutput",
    "ParallelAgentCoordinator",
    "ParallelTask",
    "ParallelTaskResult",
    "PermissionDecision",
    "PermissionProfile",
    "PersistentAgentGraph",
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
    "ReviewerDecision",
    "RolloutTrace",
    "RolloutTraceCollector",
    "RolloutTraceEvent",
    "RolloutTraceStore",
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
    "TraceSyncResult",
    "TrajectoryMetrics",
    "TurnScopedJournaledTools",
    "VerificationResult",
    "Verdict",
    "WorkItem",
    "WorkStatus",
    "WorkerResult",
    "WorktreeCodingTeam",
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
