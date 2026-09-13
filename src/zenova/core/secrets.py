"""Cloud-agnostic secret management abstraction supporting environment, AWS, GCP, and Vault providers."""
import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from zenova.core.logging import get_logger

logger = get_logger("zenova.core.secrets")


class BaseSecretProvider(ABC):
    """Abstract interface for secret resolution."""

    @abstractmethod
    def get_secret(self, secret_name: str, default: Optional[str] = None) -> Optional[str]:
        """Resolve a secret value by key name."""
        pass


class EnvironmentSecretProvider(BaseSecretProvider):
    """Default provider resolving secrets from environment variables."""

    def get_secret(self, secret_name: str, default: Optional[str] = None) -> Optional[str]:
        val = os.getenv(secret_name, default)
        if val is None:
            logger.debug(f"Secret '{secret_name}' not found in environment.")
        return val


class SecretManager:
    """Central secret management facade with replaceable cloud backend providers."""

    def __init__(self, provider: Optional[BaseSecretProvider] = None):
        self.provider = provider or EnvironmentSecretProvider()

    def get(self, key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
        """Fetch a secret value.
        
        Args:
            key: Secret identifier / environment variable name
            default: Fallback default value if not set
            required: If True, raises ValueError when secret is missing or empty
        """
        val = self.provider.get_secret(key, default)
        if required and (val is None or str(val).strip() == ""):
            raise ValueError(f"CRITICAL: Required secret '{key}' is missing or empty!")
        return val

    def mask_secret(self, val: Optional[str]) -> str:
        """Return safe masked representation of a secret for audit logs."""
        if not val or len(val) < 8:
            return "***"
        return f"{val[:3]}...{val[-3:]}"


# Global singleton
_GLOBAL_SECRET_MANAGER: Optional[SecretManager] = None


def get_secret_manager() -> SecretManager:
    """Retrieve global SecretManager instance."""
    global _GLOBAL_SECRET_MANAGER
    if _GLOBAL_SECRET_MANAGER is None:
        _GLOBAL_SECRET_MANAGER = SecretManager()
    return _GLOBAL_SECRET_MANAGER
