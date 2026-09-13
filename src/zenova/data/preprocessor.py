"""Dataset preprocessing interfaces and standard transformation pipelines."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
from zenova.preprocessing.text import clean_text
from zenova.core.logging import get_logger

logger = get_logger("zenova.data.preprocessor")


class BaseDatasetPreprocessor(ABC):
    """Interface for dataset-specific transformations and cleaning."""

    @abstractmethod
    def preprocess_sample(self, raw_sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Transform a single raw record into standard feature-label format."""
        pass

    def preprocess_batch(self, raw_samples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        processed = []
        dropped = 0
        for sample in raw_samples:
            res = self.preprocess_sample(sample)
            if res is not None:
                processed.append(res)
            else:
                dropped += 1
        return processed, {"processed": len(processed), "dropped": dropped}


class TextDialoguePreprocessor(BaseDatasetPreprocessor):
    """Standard dialogue preprocessor cleaning text and validating speaker roles."""

    def __init__(self, text_field: str = "text", target_field: str = "target_strategy"):
        self.text_field = text_field
        self.target_field = target_field

    def preprocess_sample(self, raw_sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        text = raw_sample.get(self.text_field, "")
        cleaned = clean_text(text)
        if not cleaned or len(cleaned) < 2:
            return None

        sample_copy = dict(raw_sample)
        sample_copy[self.text_field] = cleaned
        return sample_copy
