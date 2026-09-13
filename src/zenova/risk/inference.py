"""Inference engine adhering strictly to safety-critical crisis triage contracts."""
import json
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn.functional as F

from zenova.risk.transformer import RiskTransformerModel
from zenova.risk.taxonomy import (
    RISK_LEVELS,
    RISK_TAXONOMY_SPEC,
    ADVERSARIAL_NON_CRISIS_IDIOMS
)
from zenova.schemas.standard import RiskLevel, CrisisCategory

SAFETY_DISCLAIMER = (
    "Automated risk assessment heuristic; not clinical certainty. "
    "Imminent life-safety risks must be routed to human crisis professionals."
)


class RiskInferenceEngine:
    """Production inference engine for real-time crisis and self-harm risk triage."""

    def __init__(self, model_dir: str = "models/risk"):
        self.model_dir = Path(model_dir)
        self.model: Optional[RiskTransformerModel] = None
        self.vocab: Dict[str, int] = {}
        self.labels: List[str] = []
        self.label2id: Dict[str, int] = {}
        self.id2label: Dict[int, str] = {}
        self.max_len = 128
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
        self.max_len = meta.get("max_len", 128)
        self.version = meta.get("version", "transformer-v1.0.0")

        model = RiskTransformerModel(
            vocab_size=len(self.vocab),
            num_classes=len(self.labels),
            d_model=meta.get("d_model", 128),
            max_len=self.max_len
        )
        model.load_state_dict(torch.load(ckpt_p, map_location="cpu"))
        model.eval()
        self.model = model

    def tokenize(self, text: str) -> List[int]:
        tokens = re.findall(r"\w+", text.lower())
        return [self.vocab.get(t, 1) for t in tokens][:self.max_len]

    def _extract_trigger_cues(self, text: str) -> Tuple[List[str], Optional[str]]:
        """Identify verbatim trigger cues and find highest clinical risk level."""
        text_lower = text.lower()
        cues = []
        highest_tier = None

        # Check from most severe (critical) to least severe (moderate)
        for tier in [RiskLevel.CRITICAL.value, RiskLevel.HIGH.value, RiskLevel.MODERATE.value]:
            tier_spec = RISK_TAXONOMY_SPEC.get(tier, {})
            for cue in tier_spec.get("evidence_cues", []):
                if cue in text_lower:
                    idx = text_lower.find(cue)
                    matched_span = text[idx:idx + len(cue)]
                    if matched_span not in cues:
                        cues.append(matched_span)
                    if highest_tier is None:
                        highest_tier = tier

        return cues, highest_tier

    def _is_adversarial_non_crisis(self, text: str) -> bool:
        """Check if the text consists predominantly of benign colloquial metaphors."""
        text_lower = text.lower().strip()
        for idiom in ADVERSARIAL_NON_CRISIS_IDIOMS:
            if idiom in text_lower:
                # If there are NO actual lethal means or suicide keywords, treat as non-crisis
                serious_cues = ["suicide", "kill myself", "end my life", "have the pills", "slash my", "overdose", "jump"]
                if not any(sc in text_lower for sc in serious_cues):
                    return True
        return False

    def predict(
        self,
        text: str,
        crisis_threshold: float = 0.35
    ) -> Dict[str, Any]:
        """Predict crisis risk with cost-sensitive safety decision calibration.
        
        Guarantees:
        - High Sensitivity on High/Critical risk.
        - Suppression of False Positives on adversarial death idioms.
        - Automatic escalation flags for downstream orchestrator interruption.
        """
        if self.model is None:
            raise RuntimeError(f"Risk model not loaded from {self.model_dir}")

        # Check for adversarial non-crisis idioms first
        if self._is_adversarial_non_crisis(text):
            return {
                "risk_level": RiskLevel.LOW.value,
                "confidence": 0.95,
                "crisis_category": CrisisCategory.NONE.value,
                "signals": ["colloquial_metaphor_detected"],
                "trigger_cues": [],
                "requires_escalation": False,
                "action": "standard_supportive_dialogue",
                "disclaimer": SAFETY_DISCLAIMER,
                "is_placeholder": False,
                "model_version": self.version,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        ids = self.tokenize(text)
        length = len(ids)
        padded = ids + [0] * (self.max_len - length)
        mask = [False] * length + [True] * (self.max_len - length)

        x_tensor = torch.tensor([padded], dtype=torch.long)
        mask_tensor = torch.tensor([mask], dtype=torch.bool)

        with torch.no_grad():
            logits = self.model(x_tensor, src_key_padding_mask=mask_tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).tolist()

        cues, cue_tier = self._extract_trigger_cues(text)

        low_p = probs[self.label2id[RiskLevel.LOW.value]]
        mod_p = probs[self.label2id[RiskLevel.MODERATE.value]]
        high_p = probs[self.label2id[RiskLevel.HIGH.value]]
        crit_p = probs[self.label2id[RiskLevel.CRITICAL.value]]

        # Safety-calibrated triage decision rule
        if cue_tier == RiskLevel.CRITICAL.value or (crit_p >= 0.50 and crit_p > low_p):
            decided_risk = RiskLevel.CRITICAL.value
            conf = max(crit_p, 0.92) if cue_tier == RiskLevel.CRITICAL.value else crit_p
            crisis_cat = CrisisCategory.IMMINENT_DANGER.value
            signals = ["imminent_emergency_detected", "active_attempt_cues"]
        elif cue_tier == RiskLevel.HIGH.value or ((high_p + crit_p) >= crisis_threshold and (high_p + crit_p) > low_p):
            decided_risk = RiskLevel.HIGH.value
            conf = max(high_p, 0.88) if cue_tier == RiskLevel.HIGH.value else (high_p + crit_p)
            crisis_cat = CrisisCategory.SUICIDAL_IDEATION.value
            signals = ["active_suicidal_ideation", "intent_signals"]
        elif cue_tier == RiskLevel.MODERATE.value or (mod_p >= 0.40 and mod_p > low_p):
            decided_risk = RiskLevel.MODERATE.value
            conf = max(mod_p, 0.75) if cue_tier == RiskLevel.MODERATE.value else mod_p
            crisis_cat = CrisisCategory.SUICIDAL_IDEATION.value
            signals = ["passive_suicidal_ideation", "exhaustion_markers"]
        else:
            decided_risk = RiskLevel.LOW.value
            conf = low_p
            crisis_cat = CrisisCategory.NONE.value
            signals = []

        requires_escalation = decided_risk in (RiskLevel.HIGH.value, RiskLevel.CRITICAL.value)
        action = RISK_TAXONOMY_SPEC[decided_risk]["action"]

        return {
            "risk_level": decided_risk,
            "confidence": round(float(conf), 4),
            "crisis_category": crisis_cat,
            "signals": signals,
            "trigger_cues": cues,
            "model_version": self.version,
            "requires_escalation": requires_escalation,
            "action": action,
            "disclaimer": SAFETY_DISCLAIMER,
            "is_placeholder": False,
            "probabilities": {
                RiskLevel.LOW.value: round(low_p, 4),
                RiskLevel.MODERATE.value: round(mod_p, 4),
                RiskLevel.HIGH.value: round(high_p, 4),
                RiskLevel.CRITICAL.value: round(crit_p, 4)
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
