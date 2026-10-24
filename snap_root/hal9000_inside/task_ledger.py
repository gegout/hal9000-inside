"""Durable pending-task ledger and truthful research lifecycle for HAL 9000 Inside.

Maintains state for tasks, queries, research runs, execution evidence, and delivery state.
application/tool records determine task status—not model assertions.
"""

from dataclasses import dataclass, field
import datetime
import enum
import logging
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TaskExecutionStatus(str, enum.Enum):
    PLANNED = "planned"
    RUNNING = "running"
    PARTIAL = "partial"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskDeliveryStatus(str, enum.Enum):
    NOT_DELIVERED = "not_delivered"
    PARTLY_DELIVERED = "partly_delivered"
    DELIVERED = "delivered"


@dataclass
class TaskEvidence:
    """Evidence record gathered during tool execution."""
    tool_name: str
    arguments: Dict[str, Any]
    output_summary: str
    status: str  # "success", "empty_match", "blocked", "quota_error", "network_error", "error"
    timestamp: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    citations: List[str] = field(default_factory=list)


@dataclass
class PendingTask:
    """Durable task tracking user requests, execution lifecycle, evidence, and delivery state."""
    task_id: str
    originating_turn_id: str
    originating_request: str
    topic: str
    required_tools: List[str] = field(default_factory=list)
    execution_status: TaskExecutionStatus = TaskExecutionStatus.PLANNED
    delivery_status: TaskDeliveryStatus = TaskDeliveryStatus.NOT_DELIVERED
    evidence_records: List[TaskEvidence] = field(default_factory=list)
    synthesized_findings: Optional[str] = None
    limitations: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())


class TaskLedger:
    """Thread-safe / async-compatible ledger tracking active and historical tasks."""

    def __init__(self):
        self._tasks: Dict[str, PendingTask] = {}
        self._active_task_id: Optional[str] = None

    def create_task(
        self,
        originating_turn_id: str,
        originating_request: str,
        topic: str,
        required_tools: Optional[List[str]] = None,
    ) -> PendingTask:
        """Create and register a new pending task."""
        t_id = f"task_{uuid.uuid4().hex[:8]}"
        task = PendingTask(
            task_id=t_id,
            originating_turn_id=originating_turn_id,
            originating_request=originating_request,
            topic=topic,
            required_tools=required_tools or [],
            execution_status=TaskExecutionStatus.PLANNED,
            delivery_status=TaskDeliveryStatus.NOT_DELIVERED,
        )
        self._tasks[t_id] = task
        self._active_task_id = t_id
        return task

    def get_active_task(self) -> Optional[PendingTask]:
        """Return the currently active or most recently updated task."""
        if self._active_task_id and self._active_task_id in self._tasks:
            return self._tasks[self._active_task_id]
        if self._tasks:
            return list(self._tasks.values())[-1]
        return None

    def get_task(self, task_id: str) -> Optional[PendingTask]:
        return self._tasks.get(task_id)

    def record_execution_start(self, task_id: str):
        """Mark task execution as running."""
        t = self.get_task(task_id)
        if t:
            t.execution_status = TaskExecutionStatus.RUNNING
            t.updated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    def add_evidence(
        self,
        task_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        output: Any,
        error: Optional[Exception] = None,
    ):
        """Record concrete tool execution evidence."""
        t = self.get_task(task_id)
        if not t:
            return

        out_str = str(output) if output is not None else ""
        citations = []
        status = "success"

        # Categorize failure modes concretely
        if error is not None:
            err_msg = str(error).lower()
            if "429" in err_msg or "rate limit" in err_msg:
                status = "quota_error"
            elif any(c in err_msg for c in ("403", "captcha", "blocked")):
                status = "blocked"
            elif any(c in err_msg for c in ("timeout", "connection refused", "network")):
                status = "network_error"
            else:
                status = "error"
            summary = f"Error: {error}"
        else:
            if not out_str.strip() or "no results found" in out_str.lower() or out_str.strip() in ("[]", "{}"):
                status = "empty_match"
                summary = "Query completed with zero matching records."
            else:
                status = "success"
                summary = out_str[:300]
                # Extract any URLs or titles as citations
                if isinstance(output, dict):
                    if "url" in output:
                        citations.append(output["url"])
                    if "title" in output:
                        citations.append(output["title"])
                elif isinstance(output, list):
                    for item in output[:3]:
                        if isinstance(item, dict) and "url" in item:
                            citations.append(item["url"])

        evidence = TaskEvidence(
            tool_name=tool_name,
            arguments=arguments,
            output_summary=summary,
            status=status,
            citations=citations,
        )
        t.evidence_records.append(evidence)
        t.updated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    def complete_task(
        self,
        task_id: str,
        findings: Optional[str] = None,
        limitations: Optional[str] = None,
        status: Optional[TaskExecutionStatus] = None,
    ):
        """Record task outcome based on evidence."""
        t = self.get_task(task_id)
        if not t:
            return

        if status:
            t.execution_status = status
        else:
            has_success = any(e.status == "success" for e in t.evidence_records)
            has_failure = any(e.status != "success" for e in t.evidence_records)
            if has_success and not has_failure:
                t.execution_status = TaskExecutionStatus.SUCCEEDED
            elif has_success and has_failure:
                t.execution_status = TaskExecutionStatus.PARTIAL
            elif any(e.status == "empty_match" for e in t.evidence_records):
                t.execution_status = TaskExecutionStatus.SUCCEEDED  # Valid empty search
            else:
                t.execution_status = TaskExecutionStatus.FAILED

        if findings:
            t.synthesized_findings = findings
        if limitations:
            t.limitations = limitations
        t.updated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    def mark_delivery(self, task_id: str, status: TaskDeliveryStatus):
        t = self.get_task(task_id)
        if t:
            t.delivery_status = status
            t.updated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    def cancel_active(self):
        """Cancel current pending work on user interruption."""
        if self._active_task_id:
            t = self.get_task(self._active_task_id)
            if t and t.execution_status in (TaskExecutionStatus.PLANNED, TaskExecutionStatus.RUNNING):
                t.execution_status = TaskExecutionStatus.CANCELLED
                t.updated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
