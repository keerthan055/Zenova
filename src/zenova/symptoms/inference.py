"""Inference engine adhering strictly to non-diagnostic ZENOVA symptom output contracts."""
import json
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import torch

from zenova.symptoms.transformer import SymptomTransformerModel
from zenova.symptoms.taxonomy import TAXONOMY_SPEC, SYMPTOM_SIGNALS
from zenova.schemas.standard import SymptomSeverity

CLINICAL_DISCLAIMER = (
    "Observational marker only; does not constitute clinical or psychiatric diagnosis."
)


class SymptomInferenceEngine:
    """Production inference engine for real-time symptom and signal identification."""

    def __init__(self, model_dir: str = "models/symptoms"):
        self.model_dir = Path(model_dir)
        self.model: Optional[SymptomTransformerModel] = None
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

        model = SymptomTransformerModel(
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

    def _extract_evidence_spans(self, text: str, signal_name: str) -> List[str]:
        """Extract verbatim spans in the user utterance that correspond to the signal."""
        spans = []
        spec = TAXONOMY_SPEC.get(signal_name, {})
        cues = spec.get("evidence_cues", [])
        text_lower = text.lower()

        for cue in cues:
            idx = text_lower.find(cue.lower())
            if idx != -1:
                # Extract verbatim span preserving original casing
                matched = text[idx:idx + len(cue)]
                if matched not in spans:
                    spans.append(matched)

        # Fallback: if model has high confidence but no literal dictionary cue, extract clause
        if not spans:
            clauses = re.split(r"[,;.\n]+", text)
            for c in clauses:
                c_strip = c.strip()
                if len(c_strip) > 5 and any(w in c_strip.lower() for w in signal_name.split()):
                    spans.append(c_strip)
                    break

        return spans[:3]

    def _estimate_severity(self, signal_name: str, confidence: float, evidence: List[str]) -> SymptomSeverity:
        """Heuristic severity calibration based on confidence and intensity markers."""
        spec = TAXONOMY_SPEC.get(signal_name, {})
        if spec.get("is_crisis_signal", False):
            return SymptomSeverity.SEVERE

        if confidence >= 0.75:
            return SymptomSeverity.MODERATE
        elif confidence >= 0.40:
            return SymptomSeverity.MILD
        else:
            return SymptomSeverity.SUBCLINICAL

    def predict(
        self,
        text: str,
        threshold: float = 0.40,
        subclinical_threshold: float = 0.35
    ) -> Dict[str, Any]:
        """Predict multi-label symptom signals adhering strictly to non-diagnostic contracts.
        
        Guarantees:
        - Never outputs clinical diagnostic statements.
        - Provides ranked observational signals with confidence and evidence spans.
        - Handles zero-symptom, subclinical, multi-symptom, and crisis inputs.
        """
        if self.model is None:
            raise RuntimeError(f"Symptom model not loaded from {self.model_dir}")

        ids = self.tokenize(text)
        length = len(ids)
        padded = ids + [0] * (self.max_len - length)
        mask = [False] * length + [True] * (self.max_len - length)

        x_tensor = torch.tensor([padded], dtype=torch.long)
        mask_tensor = torch.tensor([mask], dtype=torch.bool)

        with torch.no_grad():
            logits = self.model(x_tensor, src_key_padding_mask=mask_tensor)
            probs = torch.sigmoid(logits).squeeze(0).tolist()

        detected_signals = []
        is_crisis_flagged = False

        for col_idx, conf in enumerate(probs):
            label_name = self.id2label[col_idx]
            spec = TAXONOMY_SPEC.get(label_name, {})
            is_crisis = spec.get("is_crisis_signal", False)

            evidence = self._extract_evidence_spans(text, label_name)

            # Crisis signals have lower threshold for safety; evidence-grounded signals have calibrated threshold
            if is_crisis:
                active_thresh = 0.35
            elif evidence:
                active_thresh = min(threshold, 0.40)
            else:
                active_thresh = threshold

            if conf >= active_thresh:
                severity = self._estimate_severity(label_name, conf, evidence)

                signal_item = {
                    "label": label_name,
                    "marker_name": label_name,
                    "confidence": round(conf, 4),
                    "severity": severity.value,
                    "evidence_spans": evidence,
                    "clinical_disclaimer": CLINICAL_DISCLAIMER
                }
                detected_signals.append(signal_item)

                if is_crisis:
                    is_crisis_flagged = True

        # Sort signals in descending order of confidence
        detected_signals.sort(key=lambda s: s["confidence"], reverse=True)

        # Compute aggregate severity
        if not detected_signals:
            aggregate_sev = SymptomSeverity.NONE
            notes = "No observable symptom signals identified above detection threshold."
        elif is_crisis_flagged:
            aggregate_sev = SymptomSeverity.SEVERE
            notes = "Crisis / self-harm signal identified; safety layer notification recommended."
        else:
            severities = [s["severity"] for s in detected_signals]
            if SymptomSeverity.SEVERE.value in severities:
                aggregate_sev = SymptomSeverity.SEVERE
            elif SymptomSeverity.MODERATE.value in severities:
                aggregate_sev = SymptomSeverity.MODERATE
            elif SymptomSeverity.MILD.value in severities:
                aggregate_sev = SymptomSeverity.MILD
            else:
                aggregate_sev = SymptomSeverity.SUBCLINICAL
            notes = f"{len(detected_signals)} observational symptom signals identified."

        return {
            "signals": detected_signals,
            "aggregate_severity": aggregate_sev.value,
            "is_crisis_flagged": is_crisis_flagged,
            "disclaimer": CLINICAL_DISCLAIMER,
            "notes": notes,
            "is_placeholder": False,
            "model_version": self.version,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
