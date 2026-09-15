"""ZENOVA configuration management supporting environment separation and production validation."""
import os
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml
from pydantic import BaseModel, Field, model_validator


class AppEnvironment(str, Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class SafetyConfig(BaseModel):
    strict_mode: bool = True
    crisis_override_enabled: bool = True
    disclaimer: str = "ZENOVA is an AI wellbeing support companion, not a medical device or crisis service."
    crisis_resources: Dict[str, str] = Field(default_factory=lambda: {
        "hotline": "988",
        "crisis_text_line": "Text HOME to 741741",
        "international_url": "https://findahelpline.com/"
    })


class PathsConfig(BaseModel):
    raw_data: str = "data/raw"
    processed_data: str = "data/processed"
    models: str = "models"
    metadata: str = "data/metadata"


class DatabaseConfig(BaseModel):
    url_sync: str = Field(default_factory=lambda: os.getenv("DATABASE_URL_SYNC", "sqlite:///data/zenova.db"))
    url_async: str = Field(default_factory=lambda: os.getenv("DATABASE_URL", "sqlite+aiosqlite:///data/zenova.db"))
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 1800


class SecurityConfig(BaseModel):
    secret_key: str = Field(default_factory=lambda: os.getenv("ZENOVA_SECRET_KEY", "zenova-dev-secret-key-do-not-use-in-production"))
    algorithm: str = "HS256"
    token_expire_minutes: int = 30
    refresh_token_expire_days: int = 14
    password_reset_expire_minutes: int = 30
    email_verification_expire_hours: int = 24
    cors_allowed_origins: List[str] = Field(default_factory=lambda: ["*"])
    rate_limit_per_minute: int = 60
    rate_limit_burst: int = 15
    cookie_secure: bool = Field(default_factory=lambda: os.getenv("COOKIE_SECURE", "false").lower() in ("true", "1"))
    cookie_samesite: str = Field(default_factory=lambda: os.getenv("COOKIE_SAMESITE", "lax"))


class EmailConfig(BaseModel):
    provider: str = Field(default_factory=lambda: os.getenv("EMAIL_PROVIDER", "development"))
    smtp_host: Optional[str] = Field(default_factory=lambda: os.getenv("SMTP_HOST", None))
    smtp_port: int = Field(default_factory=lambda: int(os.getenv("SMTP_PORT", "587")))
    smtp_user: Optional[str] = Field(default_factory=lambda: os.getenv("SMTP_USER", None))
    smtp_password: Optional[str] = Field(default_factory=lambda: os.getenv("SMTP_PASSWORD", None))
    smtp_use_tls: bool = Field(default_factory=lambda: os.getenv("SMTP_USE_TLS", "true").lower() in ("true", "1"))
    from_email: str = Field(default_factory=lambda: os.getenv("EMAIL_FROM", "noreply@zenova.ai"))
    frontend_base_url: str = Field(default_factory=lambda: os.getenv("FRONTEND_BASE_URL", "http://localhost:8000"))


class MonitoringConfig(BaseModel):
    prometheus_enabled: bool = True
    sentry_dsn: Optional[str] = Field(default_factory=lambda: os.getenv("SENTRY_DSN", None))
    structured_logging: bool = Field(default_factory=lambda: os.getenv("ZENOVA_STRUCTURED_LOGGING", "false").lower() in ("true", "1"))
    log_level: str = Field(default_factory=lambda: os.getenv("ZENOVA_LOG_LEVEL", "INFO"))
    pii_redaction: bool = True


class SystemConfig(BaseModel):
    name: str = "ZENOVA"
    version: str = "0.1.0"
    environment: AppEnvironment = Field(default_factory=lambda: AppEnvironment(os.getenv("ZENOVA_ENV", "development").lower()))
    random_seed: int = 42
    debug: bool = False
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    email: EmailConfig = Field(default_factory=EmailConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    @model_validator(mode="after")
    def validate_production_readiness(self) -> "SystemConfig":
        """Strict validation ensuring production security standards."""
        if self.environment == AppEnvironment.PRODUCTION:
            if self.debug:
                raise ValueError("CRITICAL: debug mode must be False in production!")
            if "zenova-dev-secret-key" in self.security.secret_key or len(self.security.secret_key) < 32:
                raise ValueError("CRITICAL: production requires a secure, high-entropy secret key (min 32 chars)!")
            if "*" in self.security.cors_allowed_origins:
                raise ValueError("CRITICAL: wildcard CORS allowed origins ('*') is prohibited in production!")
            # Ensure PII redaction is strictly active in production
            self.monitoring.pii_redaction = True
            self.monitoring.structured_logging = True
        return self


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Safely load YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        content = yaml.safe_load(f)
    return content or {}


def get_system_config(config_path: str = "configs/system_config.yaml") -> SystemConfig:
    """Load and validate system configuration."""
    env_str = os.getenv("ZENOVA_ENV", "development").lower()
    
    if os.path.exists(config_path):
        data = load_yaml_config(config_path)
        system_dict = data.get("system", {})
        if "environment" not in system_dict:
            system_dict["environment"] = env_str
        if "safety" in data:
            system_dict["safety"] = data["safety"]
        if "paths" in data:
            system_dict["paths"] = data["paths"]
        if "database" in data:
            system_dict["database"] = data["database"]
        if "security" in data:
            system_dict["security"] = data["security"]
        if "monitoring" in data:
            system_dict["monitoring"] = data["monitoring"]
        return SystemConfig(**system_dict)
    return SystemConfig(environment=AppEnvironment(env_str))
