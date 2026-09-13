"""Unit tests for production structured logging and automated PII/PHI redaction."""
import json
import logging
from zenova.core.logging import PIIRedactionFilter, ProductionJsonFormatter, get_logger


def test_pii_redaction_email_and_phone():
    """Verify email addresses and phone numbers are redacted from logs."""
    redactor = PIIRedactionFilter()
    msg = "Patient john.doe@example.com called helpline at +1 (555) 123-4567 regarding anxiety."
    redacted = redactor.redact(msg)

    assert "john.doe@example.com" not in redacted
    assert "[REDACTED_EMAIL]" in redacted
    assert "123-4567" not in redacted
    assert "[REDACTED_PHONE]" in redacted


def test_pii_redaction_ssn_and_credit_card():
    """Verify SSNs and credit cards are redacted."""
    redactor = PIIRedactionFilter()
    msg = "SSN 123-45-6789 and Card 4111-2222-3333-4444 should not appear in raw logs."
    redacted = redactor.redact(msg)

    assert "123-45-6789" not in redacted
    assert "[REDACTED_SSN]" in redacted
    assert "4111-2222-3333-4444" not in redacted
    assert "[REDACTED_CARD]" in redacted


def test_pii_redaction_credentials():
    """Verify authorization bearer tokens and API keys are redacted."""
    redactor = PIIRedactionFilter()
    msg = "API request failed with bearer secret_token_abc123456789 and api_key='sk_live_123456789'."
    redacted = redactor.redact(msg)

    assert "secret_token_abc123456789" not in redacted
    assert "[REDACTED_CREDENTIAL]" in redacted


def test_pii_redaction_filter_integration():
    """Verify PIIRedactionFilter mutates log records during logging."""
    redactor = PIIRedactionFilter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="Contact support at help@zenova.ai or dial 800-555-0199.",
        args=(),
        exc_info=None
    )

    redactor.filter(record)
    assert "help@zenova.ai" not in record.msg
    assert "[REDACTED_EMAIL]" in record.msg
    assert "[REDACTED_PHONE]" in record.msg


def test_production_json_formatter():
    """Verify ProductionJsonFormatter outputs compliant structured JSON."""
    formatter = ProductionJsonFormatter()
    record = logging.LogRecord(
        name="zenova.api",
        level=logging.WARNING,
        pathname=__file__,
        lineno=42,
        msg="Sample warning event occurred.",
        args=(),
        exc_info=None
    )
    record.trace_id = "trc_unit_test_123"

    formatted_json = formatter.format(record)
    data = json.loads(formatted_json)

    assert data["service"] == "zenova"
    assert data["level"] == "WARNING"
    assert data["logger"] == "zenova.api"
    assert data["message"] == "Sample warning event occurred."
    assert data["trace_id"] == "trc_unit_test_123"
    assert "timestamp" in data
