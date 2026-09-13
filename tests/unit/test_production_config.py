"""Unit tests for ZENOVA environment configuration, separation, and production validation."""
import pytest
from zenova.core.config import (
    AppEnvironment,
    SystemConfig,
    SecurityConfig,
    DatabaseConfig,
    MonitoringConfig,
    get_system_config
)
from zenova.core.secrets import SecretManager, EnvironmentSecretProvider, get_secret_manager


def test_environment_enumeration():
    """Verify supported environment tiers."""
    assert AppEnvironment.DEVELOPMENT.value == "development"
    assert AppEnvironment.TESTING.value == "testing"
    assert AppEnvironment.STAGING.value == "staging"
    assert AppEnvironment.PRODUCTION.value == "production"


def test_development_config_defaults():
    """Verify sensible development defaults allow local testing without strict production constraints."""
    cfg = SystemConfig(environment=AppEnvironment.DEVELOPMENT)
    assert cfg.environment == AppEnvironment.DEVELOPMENT
    assert cfg.debug is False
    assert "*" in cfg.security.cors_allowed_origins
    assert cfg.monitoring.prometheus_enabled is True


def test_production_validation_debug_prohibited():
    """Verify that enabling debug mode in production raises a critical validation error."""
    with pytest.raises(ValueError, match="debug mode must be False in production"):
        SystemConfig(
            environment=AppEnvironment.PRODUCTION,
            debug=True,
            security=SecurityConfig(
                secret_key="a" * 32,
                cors_allowed_origins=["https://app.zenova.ai"]
            )
        )


def test_production_validation_insecure_secret_rejected():
    """Verify that using default development secret key in production is blocked."""
    with pytest.raises(ValueError, match="production requires a secure, high-entropy secret key"):
        SystemConfig(
            environment=AppEnvironment.PRODUCTION,
            debug=False,
            security=SecurityConfig(
                secret_key="zenova-dev-secret-key-do-not-use-in-production",
                cors_allowed_origins=["https://app.zenova.ai"]
            )
        )


def test_production_validation_wildcard_cors_rejected():
    """Verify that wildcard CORS allowed origins is rejected in production."""
    with pytest.raises(ValueError, match="wildcard CORS allowed origins"):
        SystemConfig(
            environment=AppEnvironment.PRODUCTION,
            debug=False,
            security=SecurityConfig(
                secret_key="a" * 32,
                cors_allowed_origins=["*"]
            )
        )


def test_production_validation_success():
    """Verify that a properly configured production profile passes validation and enforces PII redaction."""
    cfg = SystemConfig(
        environment=AppEnvironment.PRODUCTION,
        debug=False,
        security=SecurityConfig(
            secret_key="f" * 32,
            cors_allowed_origins=["https://app.zenova.ai", "https://clinician.zenova.ai"]
        )
    )
    assert cfg.environment == AppEnvironment.PRODUCTION
    assert cfg.monitoring.pii_redaction is True
    assert cfg.monitoring.structured_logging is True


def test_secret_manager():
    """Verify SecretManager abstraction and masking utilities."""
    manager = get_secret_manager()
    assert manager is not None

    # Masking test
    masked = manager.mask_secret("very_secret_api_key_12345")
    assert masked == "ver...345"
    assert manager.mask_secret("") == "***"

    # Required secret check
    with pytest.raises(ValueError, match="Required secret 'NON_EXISTENT_KEY_123' is missing"):
        manager.get("NON_EXISTENT_KEY_123", required=True)
