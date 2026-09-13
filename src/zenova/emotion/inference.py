"""Inference engine adhering strictly to ZENOVA output contracts."""
import json
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import torch
import torch.nn.functional as F

from zenova.emotion.transformer import EmotionTransformerModel
from zenova.schemas.standard import EmotionResult, EmotionCategory

# Standard Valence-Arousal-Dominance (VAD) normative coordinates for Ekman classes
EKMAN_VAD_MAPPING = {
    "joy": (0.81, 0.51, 0.46),
    "sadness": (-0.63, -0.27, -0.33),
    "fear": (-0.64, 0.60, -0.43),
    "anger": (-0.51, 0.59, 0.25),
    "disgust": (-0.60, 0.35, 0.11),
    "surprise": (0.40, 0.67, -0.13),
    "neutral": (0.00, 0.00, 0.00)
}


class EmotionInferenceEngine:
    """Production inference engine for real-time text emotion analysis."""

    def __init__(self, model_dir: str = "models/emotion"):
        self.model_dir = Path(model_dir)
        self.model: Optional[EmotionTransformerModel] = None
        self.vocab: Dict[str, int] = {}
        self.labels: List[str] = []
        self.label2id: Dict[str, int] = {}
        self.id2label: Dict[int, str] = {}
        self.max_len = 64
        self.version = "transformer-v1.0.0"
        self.load_model()

    def load_model(self):
        meta_p = self.model_dir / "vocab_and_labels.json"
        ckpt_p = self.model_dir / "transformer_checkpoint.pt"

        if not meta_p.exists() or not ckpt_p.exists():
            return

        with open(meta_p, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.labels = meta["labels"]
        self.vocab = meta["vocab"]
        self.label2id = meta["label2id"]
        self.id2label = {int(k): v for k, v in meta["id2label"].items()}
        self.max_len = meta.get("max_len", 64)

        model = EmotionTransformerModel(
            vocab_size=len(self.vocab),
            num_classes=len(self.labels),
            d_model=meta.get("d_model", 128)
        )
        model.load_state_dict(torch.load(ckpt_p, map_location="cpu"))
        model.eval()
        self.model = model

    def tokenize(self, text: str) -> List[int]:
        words = re.findall(r"\w+", text.lower())
        return [self.vocab.get(w, 1) for w in words]

    def predict(self, text: str) -> Dict[str, Any]:
        """Predict emotions adhering to requested output contract:
        {
          'emotions': [{'label': '...', 'confidence': 0.0}, ...],
          'primary_emotion': '...',
          'model_version': '...',
          'timestamp': '...'
        }
        """
        if self.model is None:
            raise RuntimeError(f"Emotion model not loaded from {self.model_dir}")

        ids = self.tokenize(text)[:self.max_len]
        length = len(ids)
        padded = ids + [0] * (self.max_len - length)
        mask = [False] * length + [True] * (self.max_len - length)

        x_tensor = torch.tensor([padded], dtype=torch.long)
        mask_tensor = torch.tensor([mask], dtype=torch.bool)

        with torch.no_grad():
            logits = self.model(x_tensor, src_key_padding_mask=mask_tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).tolist()

        ranked = sorted(
            [{"label": self.id2label[i], "confidence": round(p, 4)} for i, p in enumerate(probs)],
            key=lambda item: item["confidence"],
            reverse=True
        )

        primary = ranked[0]["label"]
        primary_conf = ranked[0]["confidence"]

        v, a, d = EKMAN_VAD_MAPPING.get(primary, (0.0, 0.0, 0.0))

        prob_dict = {item["label"]: item["confidence"] for item in ranked}

        return {
            "emotions": ranked,
            "primary_emotion": primary,
            "confidence": primary_conf,
            "probabilities": prob_dict,
            "valence": v,
            "arousal": a,
            "dominance": d,
            "is_placeholder": False,
            "model_version": self.version,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
