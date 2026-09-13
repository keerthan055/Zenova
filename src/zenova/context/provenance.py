"""Provenance Tracker for ZENOVA Step 9.

Tracks the analytical lineage, model versions, timestamps, and execution
latencies of all modules contributing to the multimodal context.
"""
from typing import Dict, Any, Optional
from datetime import datetime, timezone


class ProvenanceTracker:
    """Records and structures provenance metadata across analytical modules."""

    @staticmethod
    def build_module_record(
        module_name: str,
        version: str,
        is_available: bool,
        source_modality: str,
        latency_ms: Optional[float] = None,
        is_derived: bool = False,
        extra_meta: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        record = {
            "module_name": module_name,
            "version": version,
            "is_available": is_available,
            "source_modality": source_modality,
            "is_derived": is_derived,
            "recorded_at": datetime.now(timezone.utc).isoformat()
        }
        if latency_ms is not None:
            record["latency_ms"] = round(latency_ms, 2)
        if extra_meta:
            record["meta"] = extra_meta
        return record

    @classmethod
    def assemble_provenance(
        cls,
        emotion_meta: Optional[Dict[str, Any]] = None,
        symptom_meta: Optional[Dict[str, Any]] = None,
        risk_meta: Optional[Dict[str, Any]] = None,
        baseline_meta: Optional[Dict[str, Any]] = None,
        behavior_meta: Optional[Dict[str, Any]] = None,
        voice_meta: Optional[Dict[str, Any]] = None,
        history_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Assemble all module records into the master provenance map."""
        provenance = {}
        if emotion_meta:
            provenance["emotion"] = emotion_meta
        if symptom_meta:
            provenance["symptoms"] = symptom_meta
        if risk_meta:
            provenance["risk"] = risk_meta
        if baseline_meta:
            provenance["baseline"] = baseline_meta
        if behavior_meta:
            provenance["behavior"] = behavior_meta
        if voice_meta:
            provenance["voice"] = voice_meta
        if history_meta:
            provenance["history"] = history_meta
        return provenance
