import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
"""Diagnostic script to verify the ZENOVA runtime environment."""
import sys
import torch
import sklearn
import pydantic
import yaml
from zenova.core.config import get_system_config
from zenova.core.logging import get_logger

logger = get_logger("zenova.scripts.verify")


def verify():
    logger.info(f"Python Version: {sys.version}")
    logger.info(f"PyTorch Version: {torch.__version__} (CUDA Available: {torch.cuda.is_available()})")
    logger.info(f"Scikit-Learn Version: {sklearn.__version__}")
    logger.info(f"Pydantic Version: {pydantic.__version__}")

    cfg = get_system_config()
    logger.info(f"Loaded System Config: Name={cfg.name}, Version={cfg.version}, Strict={cfg.safety.strict_mode}")
    print("\n=== Environment Verification Passed Successfully ===")


if __name__ == "__main__":
    verify()
