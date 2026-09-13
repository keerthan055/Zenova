"""ZENOVA Model Registry for dynamic provider resolution and model swapping."""
import importlib
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from zenova.core.logging import get_logger
from zenova.core.exceptions import ModelNotFoundException

logger = get_logger("zenova.models.registry")


class ModelRegistration(BaseModel):
    task: str
    provider_name: str
    module_class: str
    version: str
    is_active: bool = False
    device: str = "cpu"
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ModelRegistry:
    """Central registry managing model provider resolution based on configuration."""

    def __init__(self, config_path: str = "configs/models.yaml"):
        self.config_path = config_path
        self._registry: Dict[str, Dict[str, ModelRegistration]] = {}
        self._active_providers: Dict[str, str] = {}
        self._instances: Dict[str, Any] = {}
        self.load_configuration()

    def load_configuration(self) -> None:
        path = Path(self.config_path)
        if not path.exists():
            logger.warning(f"Model config not found at {self.config_path}; using defaults.")
            return

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        self._active_providers = data.get("active_providers", {})
        raw_registry = data.get("registry", {})

        for task, providers in raw_registry.items():
            self._registry[task] = {}
            active_p = self._active_providers.get(task, "placeholder")
            for p_name, p_meta in providers.items():
                reg = ModelRegistration(
                    task=task,
                    provider_name=p_name,
                    module_class=p_meta.get("module_class", ""),
                    version=p_meta.get("version", "0.1.0"),
                    is_active=(p_name == active_p),
                    device=p_meta.get("device", "cpu"),
                    parameters=p_meta
                )
                self._registry[task][p_name] = reg

        logger.info(f"Loaded ModelRegistry with active providers: {self._active_providers}")

    def get_active_provider(self, task: str) -> str:
        return self._active_providers.get(task, "placeholder")

    def set_active_provider(self, task: str, provider_name: str) -> None:
        """Dynamically switch active provider for a task."""
        if task not in self._registry or provider_name not in self._registry[task]:
            raise ModelNotFoundException(f"Provider '{provider_name}' not registered for task '{task}'")
        self._active_providers[task] = provider_name
        # Mark flags
        for p, reg in self._registry[task].items():
            reg.is_active = (p == provider_name)
        # Clear cached instance
        self._instances.pop(task, None)
        logger.info(f"Switched task '{task}' provider to '{provider_name}'")

    def get_module_instance(self, task: str) -> Any:
        """Instantiate or return cached instance of active provider for a task."""
        if task in self._instances:
            return self._instances[task]

        provider_name = self.get_active_provider(task)
        task_providers = self._registry.get(task, {})
        if provider_name not in task_providers:
            raise ModelNotFoundException(f"No provider '{provider_name}' for task '{task}' in registry.")

        reg = task_providers[provider_name]
        module_path, class_name = reg.module_class.rsplit(".", 1)

        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            instance = cls()
            self._instances[task] = instance
            return instance
        except Exception as e:
            raise ModelNotFoundException(
                f"Failed to instantiate {reg.module_class} for task {task}: {str(e)}"
            ) from e

    def register_instance(self, task: str, instance: Any) -> None:
        """Directly register or override an active module instance."""
        self._instances[task] = instance

    def register_module(self, task: str, instance: Any) -> None:
        """Alias for register_instance to facilitate testing and mock injection."""
        self.register_instance(task, instance)

    def list_models(self) -> Dict[str, Any]:
        return {
            "active_providers": self._active_providers,
            "registry": {
                task: {p: r.model_dump() for p, r in providers.items()}
                for task, providers in self._registry.items()
            }
        }
