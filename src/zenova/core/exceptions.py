"""ZENOVA centralized error handling and domain exceptions."""
from typing import Optional, Dict, Any


class ZenovaException(Exception):
    """Base exception class for all ZENOVA errors."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ValidationException(ZenovaException):
    """Raised when schema validation or input sanitization fails."""
    pass


class ModuleExecutionException(ZenovaException):
    """Raised when an individual analysis or generation module fails during execution."""
    pass


class ModelNotFoundException(ZenovaException):
    """Raised when a requested model ID or provider cannot be resolved by ModelRegistry."""
    pass


class DatabaseException(ZenovaException):
    """Raised when database persistence or querying fails."""
    pass


class SafetyException(ZenovaException):
    """Raised when an unrecoverable safety breach or crisis escalation occurs."""
    pass
