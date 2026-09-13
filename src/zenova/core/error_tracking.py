"""Pluggable error tracking and diagnostic telemetry abstraction."""
import os
from typing import Optional, Dict, Any
from zenova.core.logging import get_logger

logger = get_logger("zenova.core.error_tracking")


class ErrorTracker:
    """Central error capture facade compatible with Sentry, OpenTelemetry, and local logging."""

    def __init__(self, sentry_dsn: Optional[str] = None):
        self.sentry_dsn = sentry_dsn or os.getenv("SENTRY_DSN")
        self.is_sentry_active = False

        if self.sentry_dsn:
            try:
                import sentry_sdk
                sentry_sdk.init(
                    dsn=self.sentry_dsn,
                    environment=os.getenv("ZENOVA_ENV", "development"),
                    traces_sample_rate=0.2,
                    send_default_pii=False  # Strict healthcare privacy
                )
                self.is_sentry_active = True
                logger.info("Sentry error tracking initialized successfully.")
            except ImportError:
                logger.warning("sentry-sdk not installed; using local structured error logging fallback.")
            except Exception as e:
                logger.error(f"Failed to initialize Sentry: {e}")

    def capture_exception(
        self,
        exc: Exception,
        extra: Optional[Dict[str, Any]] = None,
        fingerprint: Optional[list] = None
    ) -> str:
        """Capture an unhandled or handled exception with context metadata."""
        event_id = f"err_{id(exc)}_{int(os.times().elapsed * 1000)}"

        if self.is_sentry_active:
            try:
                import sentry_sdk
                with sentry_sdk.push_scope() as scope:
                    if extra:
                        for k, v in extra.items():
                            scope.set_extra(k, v)
                    if fingerprint:
                        scope.set_fingerprint(fingerprint)
                    sentry_id = sentry_sdk.capture_exception(exc)
                    if sentry_id:
                        return str(sentry_id)
            except Exception as s_err:
                logger.warning(f"Sentry capture failed: {s_err}")

        # Structured local fallback logging
        logger.error(
            f"Captured Exception [{event_id}]: {exc.__class__.__name__} - {str(exc)} | Context: {extra}",
            exc_info=exc
        )
        return event_id

    def capture_message(self, message: str, level: str = "error", extra: Optional[Dict[str, Any]] = None) -> None:
        """Capture a high-severity diagnostic message."""
        if self.is_sentry_active:
            try:
                import sentry_sdk
                sentry_sdk.capture_message(message, level=level)
            except Exception:
                pass
        log_fn = getattr(logger, level.lower(), logger.error)
        log_fn(f"ErrorTracker message: {message} | Context: {extra}")


# Global singleton
_GLOBAL_ERROR_TRACKER: Optional[ErrorTracker] = None


def get_error_tracker() -> ErrorTracker:
    """Retrieve global ErrorTracker instance."""
    global _GLOBAL_ERROR_TRACKER
    if _GLOBAL_ERROR_TRACKER is None:
        _GLOBAL_ERROR_TRACKER = ErrorTracker()
    return _GLOBAL_ERROR_TRACKER
