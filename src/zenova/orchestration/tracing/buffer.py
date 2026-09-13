"""In-memory circular ring buffer for execution traces."""
from collections import deque
from typing import Optional, List, Dict
import threading

from zenova.schemas.orchestration import PipelineTrace, PipelineStatus


class InMemoryTraceBuffer:
    """Thread-safe in-memory circular buffer storing recent pipeline traces.
    
    Guarantees trace availability and inspection even during database downtime or lockouts.
    """

    def __init__(self, maxlen: int = 200):
        self._lock = threading.Lock()
        self._buffer: deque[PipelineTrace] = deque(maxlen=maxlen)
        self._index: Dict[str, PipelineTrace] = {}

    def add_trace(self, trace: PipelineTrace) -> None:
        """Add a trace to the buffer, replacing oldest if capacity is exceeded."""
        with self._lock:
            # If about to evict, remove oldest from index
            if len(self._buffer) == self._buffer.maxlen:
                oldest = self._buffer[0]
                self._index.pop(oldest.trace_id, None)
            
            self._buffer.append(trace)
            self._index[trace.trace_id] = trace

    def get_trace(self, trace_id: str) -> Optional[PipelineTrace]:
        """Fetch a trace by its unique trace_id."""
        with self._lock:
            return self._index.get(trace_id)

    def list_traces(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50
    ) -> List[PipelineTrace]:
        """List traces matching optional filters, newest first."""
        with self._lock:
            results = []
            for trace in reversed(self._buffer):
                if session_id and trace.session_id != session_id:
                    continue
                if user_id and trace.user_id != user_id:
                    continue
                if status:
                    curr_status = trace.status.value if hasattr(trace.status, "value") else str(trace.status)
                    if curr_status != status:
                        continue
                results.append(trace)
                if len(results) >= limit:
                    break
            return results

    def clear(self) -> None:
        """Clear all buffered traces."""
        with self._lock:
            self._buffer.clear()
            self._index.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)


# Global singleton instance
_GLOBAL_BUFFER: Optional[InMemoryTraceBuffer] = None
_BUFFER_INIT_LOCK = threading.Lock()


def get_trace_buffer() -> InMemoryTraceBuffer:
    """Retrieve the global singleton trace buffer."""
    global _GLOBAL_BUFFER
    if _GLOBAL_BUFFER is None:
        with _BUFFER_INIT_LOCK:
            if _GLOBAL_BUFFER is None:
                _GLOBAL_BUFFER = InMemoryTraceBuffer(maxlen=200)
    return _GLOBAL_BUFFER
