"""ZENOVA enterprise logging framework with structured JSON formatting and automated PHI/PII redaction."""
import re
import json
import logging
import sys
import os
from datetime import datetime, timezone
from typing import Optional, Dict, Any


class PIIRedactionFilter(logging.Filter):
    """Automated redaction filter stripping sensitive personal, healthcare, and credential data from log streams."""

    PII_PATTERNS = [
        # Email addresses
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b", "[REDACTED_EMAIL]"),
        # Phone numbers (US/International standard formats)
        (r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[REDACTED_PHONE]"),
        # Social Security Numbers (SSN)
        (r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED_SSN]"),
        # Credit card numbers
        (r"\b(?:\d{4}[-\s]?){3}\d{4}\b", "[REDACTED_CARD]"),
        # IPv4 addresses (excluding localhost)
        (r"\b(?!127\.0\.0\.1)(?:\d{1,3}\.){3}\d{1,3}\b", "[REDACTED_IP]"),
        # Bearer tokens / API keys
        (r"\b(?:bearer\s+[A-Za-z0-9_\-\.]{16,}|(?:api_key|secret_key|password)\s*[:=]\s*['\"]?[A-Za-z0-9_\-\.]{8,}['\"]?)\b", "[REDACTED_CREDENTIAL]")
    ]

    def redact(self, text: str) -> str:
        """Apply all redaction rules to input text."""
        if not text:
            return ""
        redacted = str(text)
        for pattern, replacement in self.PII_PATTERNS:
            redacted = re.sub(pattern, replacement, redacted, flags=re.IGNORECASE)
        return redacted

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter and sanitize log record message in-place."""
        if isinstance(record.msg, str):
            record.msg = self.redact(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: (self.redact(v) if isinstance(v, str) else v) for k, v in record.args.items()}
            elif isinstance(record.args, tuple):
                record.args = tuple((self.redact(v) if isinstance(v, str) else v) for v in record.args)
        return True


class ProductionJsonFormatter(logging.Formatter):
    """Structured JSON log formatter for cloud log forwarders (CloudWatch, Stackdriver, Datadog)."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": "zenova",
            "environment": os.getenv("ZENOVA_ENV", "development"),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        # Include exception trace if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Include extra attributes if passed
        for key in ("trace_id", "session_id", "user_id", "severity", "audit_id"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)

        return json.dumps(log_entry, default=str)


def get_logger(name: str = "zenova", log_level: Optional[str] = None) -> logging.Logger:
    """Obtain a configured logger instance with PII redaction and optional JSON formatting."""
    level_str = log_level or os.getenv("ZENOVA_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        # Attach automated PII redaction filter
        redaction_filter = PIIRedactionFilter()
        handler.addFilter(redaction_filter)

        # Formatter selection based on environment
        structured = os.getenv("ZENOVA_STRUCTURED_LOGGING", "false").lower() in ("true", "1")
        env = os.getenv("ZENOVA_ENV", "development").lower()
        if structured or env == "production":
            handler.setFormatter(ProductionJsonFormatter())
        else:
            formatter = logging.Formatter(
                fmt="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(formatter)

        logger.addHandler(handler)
        logger.propagate = False

    return logger
