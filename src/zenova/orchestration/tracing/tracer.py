"""Execution tracing and telemetry instrumentation for ZENOVA pipeline."""
import time
import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from zenova.schemas.orchestration import (
    SpanStatus,
    PipelineStatus,
    TraceSpan,
    PipelineTrace
)
from zenova.core.logging import get_logger

logger = get_logger("zenova.orchestration.tracing")


class PipelineTracer:
    """Instruments and aggregates execution spans across all stages of a dialogue turn."""

    def __init__(
        self,
        session_id: str,
        user_id: str,
        turn_id: int = 1,
        trace_id: Optional[str] = None
    ):
        self.trace_id = trace_id or f"trc_{uuid.uuid4().hex[:16]}"
        self.session_id = session_id
        self.user_id = user_id
        self.turn_id = turn_id
        self.status = PipelineStatus.NOMINAL
        self.start_time = time.time()
        self.spans: List[TraceSpan] = []
        self.degraded_modules: List[str] = []
        self.metadata: Dict[str, Any] = {}

    @asynccontextmanager
    async def record_span(
        self,
        span_name: str,
        module_name: Optional[str] = None,
        module_version: Optional[str] = None,
        input_summary: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        swallow_exception: bool = False,
        is_degraded_on_error: bool = True
    ):
        """Asynchronous context manager to time, record, and classify a stage or module span."""
        span_id = f"spn_{uuid.uuid4().hex[:12]}"
        span_start_mono = time.time()
        span_start_iso = datetime.now(timezone.utc).isoformat()
        
        span = TraceSpan(
            span_id=span_id,
            span_name=span_name,
            status=SpanStatus.SUCCESS,
            start_time=span_start_iso,
            module_name=module_name,
            module_version=module_version,
            input_summary=input_summary or {},
            metadata=metadata or {}
        )

        try:
            yield span
        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {str(exc)}"
            span.error_message = err_msg
            if is_degraded_on_error:
                span.status = SpanStatus.DEGRADED
                if module_name and module_name not in self.degraded_modules:
                    self.degraded_modules.append(module_name)
                if self.status == PipelineStatus.NOMINAL:
                    self.status = PipelineStatus.DEGRADED
            else:
                span.status = SpanStatus.ERROR
                self.status = PipelineStatus.ERROR
            
            logger.warning(
                f"Span '{span_name}' encountered error (status={span.status.value}): {err_msg}"
            )
            if not swallow_exception:
                raise
        finally:
            span_end_mono = time.time()
            span.duration_ms = (span_end_mono - span_start_mono) * 1000.0
            self.spans.append(span)

    def mark_degraded(self, module_name: str, reason: Optional[str] = None) -> None:
        """Explicitly flag a module as degraded without raising an exception."""
        if module_name not in self.degraded_modules:
            self.degraded_modules.append(module_name)
        if self.status == PipelineStatus.NOMINAL:
            self.status = PipelineStatus.DEGRADED
        if reason:
            logger.warning(f"Module '{module_name}' marked degraded: {reason}")

    def set_status(self, status: PipelineStatus) -> None:
        """Update overall pipeline execution status."""
        self.status = status

    def finalize(self, status: Optional[PipelineStatus] = None) -> PipelineTrace:
        """Finalize and produce the immutable PipelineTrace."""
        if status is not None:
            self.status = status
        total_duration_ms = (time.time() - self.start_time) * 1000.0

        return PipelineTrace(
            trace_id=self.trace_id,
            session_id=self.session_id,
            turn_id=self.turn_id,
            user_id=self.user_id,
            status=self.status,
            total_duration_ms=total_duration_ms,
            spans=list(self.spans),
            degraded_modules=list(self.degraded_modules),
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata=dict(self.metadata)
        )
