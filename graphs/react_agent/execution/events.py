"""Structured execution events for the react agent runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

ExecutionEventType = Literal[
    # Tool lifecycle
    "tool_limit_notice",
    "tool_started",
    "tool_completed",
    "tool_failed",
    "tool_timeout",
    # Context compaction
    "reactive_compact_attempted",
    "reactive_compact_succeeded",
    "reactive_compact_failed",
    "microcompact_applied",
    "context_collapse_triggered",
    # Retry / fallback
    "retry_exhausted",
    "fallback_mode",
    "fallback_model_activated",
    # Guardrails
    "output_guardrail_blocked",
    "hallucination_detected",
    "grounding_violation_detected",
    # Document processing
    "document_extraction_failed",
    "document_extraction_degraded",
    # Cost / budget
    "cost_threshold_warning",
    # Memory
    "memory_consolidation_completed",
    # Max output recovery
    "max_output_recovery_attempted",
    "max_output_recovery_succeeded",
    "max_output_recovery_failed",
]

ExecutionEventLevel = Literal["info", "warning", "error"]


@dataclass(frozen=True, slots=True)
class ExecutionEvent:
    """Serializable execution event for observability and replay."""

    event_type: ExecutionEventType
    level: ExecutionEventLevel = "info"
    message: str = ""
    tool_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat())


def serialize_event(event: ExecutionEvent) -> dict[str, Any]:
    """Return a JSON-safe dict representation of an execution event."""

    return asdict(event)
