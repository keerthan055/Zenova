"""Central Multimodal Context Engine for ZENOVA Step 9.

Aggregates heterogeneous analytical results into the normalized 9-block
MultimodalContext object, enforcing deterministic versioning, provenance,
confidence propagation, strict missing-data handling, and privacy safeguards.
"""
import json
import uuid
import hashlib
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

from zenova.core.interfaces import BaseMultimodalContextEngine
from zenova.schemas.standard import (
    UserInput,
    EmotionResult,
    SymptomResult,
    RiskResult,
    BaselineResult,
    BehavioralResult,
    VoiceResult,
    StrategyResult,
    ConversationTurn,
    ConversationContext,
    DialogStage
)
from zenova.schemas.context import (
    MultimodalContext,
    ConversationContextBlock,
    EmotionContextBlock,
    SymptomsContextBlock,
    RiskContextBlock,
    BaselineContextBlock,
    BehaviorContextBlock,
    VoiceContextBlock,
    HistoryContextBlock,
    MetadataContextBlock,
    PrivacyLevel
)
from zenova.context.normalizers import ContextNormalizers
from zenova.context.confidence import ConfidencePropagator
from zenova.context.provenance import ProvenanceTracker
from zenova.context.privacy import PrivacyEngine
from zenova.core.logging import get_logger

logger = get_logger("zenova.context.engine")


class MultimodalContextEngine(BaseMultimodalContextEngine):
    """Production implementation of the ZENOVA Multimodal Context Engine."""

    SCHEMA_VERSION = "1.0.0"
    ENGINE_VERSION = "zenova-context-v1.0.0"

    def __init__(
        self,
        domain_weights: Optional[Dict[str, float]] = None,
        privacy_salt: Optional[str] = None
    ):
        self.confidence_propagator = ConfidencePropagator(domain_weights)
        self.privacy_engine = PrivacyEngine(privacy_salt or "zenova_privacy_salt_2026")
        self.normalizers = ContextNormalizers()

    def build_context(
        self,
        user_input: UserInput,
        history: Optional[List[Any]] = None,
        emotion: Optional[EmotionResult] = None,
        symptoms: Optional[SymptomResult] = None,
        risk: Optional[RiskResult] = None,
        baseline: Optional[BaselineResult] = None,
        behavior: Optional[BehavioralResult] = None,
        voice: Optional[VoiceResult] = None,
        previous_strategies: Optional[List[StrategyResult]] = None,
        previous_outcomes: Optional[List[Dict[str, Any]]] = None,
        user_feedback: Optional[List[Dict[str, Any]]] = None,
        escalation_events: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        dialog_stage: DialogStage = DialogStage.EXPLORATION,
        privacy_level: PrivacyLevel = PrivacyLevel.STANDARD,
        is_crisis_bypass: bool = False
    ) -> MultimodalContext:
        """Combine heterogeneous module outputs into the normalized 9-block context."""
        meta_in = metadata or {}
        turn_id = meta_in.get("turn_id", len(history or []) + 1)
        turn_count = len(history or []) + 1

        # Detect active input modalities
        input_mods = ["text"]
        if voice and voice.is_available:
            input_mods.append("voice")
        if behavior and behavior.is_available:
            input_mods.append("behavior")

        # 1. Normalize Conversation Block
        conv_block: ConversationContextBlock = self.normalizers.normalize_conversation(
            user_input=user_input,
            turn_id=turn_id,
            turn_count=turn_count,
            dialog_stage=dialog_stage,
            input_modalities=input_mods,
            is_crisis_bypass=is_crisis_bypass
        )

        # 2. Normalize Emotion Block
        emotion_block: EmotionContextBlock = self.normalizers.normalize_emotion(
            emotion=emotion,
            voice=voice
        )

        # 3. Normalize Symptoms Block
        symptoms_block: SymptomsContextBlock = self.normalizers.normalize_symptoms(
            symptom=symptoms
        )

        # 4. Normalize Risk Block
        risk_block: RiskContextBlock = self.normalizers.normalize_risk(
            risk=risk
        )

        # 5. Normalize Baseline Block
        baseline_block: BaselineContextBlock = self.normalizers.normalize_baseline(
            baseline=baseline
        )

        # 6. Normalize Behavior Block (Strict Non-Fabrication)
        behavior_block: BehaviorContextBlock = self.normalizers.normalize_behavior(
            behavior=behavior
        )

        # 7. Normalize Voice Block (Strict Non-Fabrication)
        voice_block: VoiceContextBlock = self.normalizers.normalize_voice(
            voice=voice
        )

        # 8. Normalize History Block
        history_block: HistoryContextBlock = self.normalizers.normalize_history(
            turns=history,
            current_valence=emotion_block.valence if emotion_block.is_available else None,
            current_arousal=emotion_block.arousal if emotion_block.is_available else None,
            current_emotion=emotion_block.primary_emotion if emotion_block.is_available else None,
            user_feedback=user_feedback,
            escalation_events=escalation_events
        )

        # 9. Track Available and Missing Modalities
        available_modalities = ["conversation"]
        missing_modalities = []

        active_confidences: Dict[str, float] = {}

        if emotion_block.is_available:
            available_modalities.append("emotion")
            active_confidences["emotion"] = emotion_block.confidence
        else:
            missing_modalities.append("emotion")

        if symptoms_block.is_available:
            available_modalities.append("symptoms")
            active_confidences["symptoms"] = symptoms_block.max_confidence
        else:
            missing_modalities.append("symptoms")

        if risk_block.is_available:
            available_modalities.append("risk")
            active_confidences["risk"] = risk_block.confidence
        else:
            missing_modalities.append("risk")

        if baseline_block.is_available:
            available_modalities.append("baseline")
            active_confidences["baseline"] = baseline_block.confidence
        else:
            missing_modalities.append("baseline")

        if behavior_block.is_available:
            available_modalities.append("behavior")
            active_confidences["behavior"] = behavior_block.confidence
        else:
            missing_modalities.append("behavior")

        if voice_block.is_available:
            available_modalities.append("voice")
            active_confidences["voice"] = voice_block.confidence
        else:
            missing_modalities.append("voice")

        # 10. Confidence Propagation
        conf_prop = self.confidence_propagator.compute_propagation(active_confidences)

        # 11. Provenance Records
        provenance = ProvenanceTracker.assemble_provenance(
            emotion_meta=ProvenanceTracker.build_module_record(
                module_name="EmotionAnalyzer",
                version="1.0.0",
                is_available=emotion_block.is_available,
                source_modality="multimodal_fusion" if (voice and voice.is_available) else "text",
                is_derived=(voice is not None and voice.is_available)
            ),
            symptom_meta=ProvenanceTracker.build_module_record(
                module_name="SymptomAnalyzer",
                version=symptoms_block.model_version or "1.0.0",
                is_available=symptoms_block.is_available,
                source_modality="text"
            ),
            risk_meta=ProvenanceTracker.build_module_record(
                module_name="RiskAnalyzer",
                version=risk_block.model_version or "1.0.0",
                is_available=risk_block.is_available,
                source_modality="text"
            ),
            baseline_meta=ProvenanceTracker.build_module_record(
                module_name="PersonalBaselineEngine",
                version="1.0.0",
                is_available=baseline_block.is_available,
                source_modality="longitudinal_profile"
            ),
            behavior_meta=ProvenanceTracker.build_module_record(
                module_name="BehavioralAnalyzer",
                version="1.0.0",
                is_available=behavior_block.is_available,
                source_modality=behavior_block.source_device or "passive_sensor"
            ),
            voice_meta=ProvenanceTracker.build_module_record(
                module_name="VoiceEmotionAnalyzer",
                version="1.0.0",
                is_available=voice_block.is_available,
                source_modality="acoustic_audio"
            ),
            history_meta=ProvenanceTracker.build_module_record(
                module_name="HistoryTracker",
                version="1.0.0",
                is_available=(len(history or []) > 0),
                source_modality="session_database"
            )
        )

        # 12. Deterministic Context Hashing
        # Serializes content payload without timestamps/ephemeral IDs
        canonical_content = {
            "conversation": {
                "session_id": conv_block.session_id,
                "user_id": conv_block.user_id,
                "turn_id": conv_block.turn_id,
                "current_text": conv_block.current_text,
                "dialog_stage": conv_block.dialog_stage.value,
                "input_modalities": sorted(conv_block.input_modalities),
                "is_crisis_bypass": conv_block.is_crisis_bypass
            },
            "emotion": {
                "is_available": emotion_block.is_available,
                "primary_emotion": emotion_block.primary_emotion,
                "confidence": emotion_block.confidence,
                "valence": emotion_block.valence,
                "arousal": emotion_block.arousal,
                "is_discrepancy_detected": emotion_block.is_discrepancy_detected
            },
            "symptoms": {
                "is_available": symptoms_block.is_available,
                "primary_signals": sorted(symptoms_block.primary_signals),
                "signal_count": symptoms_block.signal_count,
                "max_confidence": symptoms_block.max_confidence
            },
            "risk": {
                "is_available": risk_block.is_available,
                "risk_level": risk_block.risk_level,
                "confidence": risk_block.confidence,
                "crisis_category": risk_block.crisis_category,
                "is_high_risk": risk_block.is_high_risk,
                "requires_escalation": risk_block.requires_escalation
            },
            "baseline": {
                "is_available": baseline_block.is_available,
                "status": baseline_block.status,
                "confidence": baseline_block.confidence,
                "anomalous_features": sorted(baseline_block.anomalous_features)
            },
            "behavior": {
                "is_available": behavior_block.is_available,
                "bdi": behavior_block.bdi,
                "is_anomalous": behavior_block.is_anomalous,
                "affected_domains": sorted(behavior_block.affected_domains)
            },
            "voice": {
                "is_available": voice_block.is_available,
                "predicted_emotion": voice_block.predicted_emotion,
                "valence": voice_block.valence,
                "confidence": voice_block.confidence
            },
            "history": {
                "turn_count": history_block.turn_count,
                "valence_trend": history_block.valence_trend.value,
                "dominant_themes": sorted(history_block.dominant_themes)
            }
        }
        canonical_bytes = json.dumps(canonical_content, sort_keys=True).encode("utf-8")
        deterministic_hash = hashlib.sha256(canonical_bytes).hexdigest()
        context_id = f"ctx_{deterministic_hash[:16]}"

        # 13. Assemble Metadata Block
        meta_block = MetadataContextBlock(
            schema_version=self.SCHEMA_VERSION,
            engine_version=self.ENGINE_VERSION,
            context_id=context_id,
            context_hash=deterministic_hash,
            created_at=datetime.now(timezone.utc),
            provenance=provenance,
            confidence_propagation=conf_prop,
            missing_modalities=missing_modalities,
            available_modalities=available_modalities,
            fused_multimodal_state=meta_in.get("fused_multimodal_state"),
            privacy={
                "privacy_level": privacy_level.value,
                "is_redacted": False,
                "redacted_fields_count": 0,
                "retention_policy": "standard_retention" if privacy_level == PrivacyLevel.STANDARD else (
                    "anonymized_research" if privacy_level == PrivacyLevel.ANONYMIZED else "zero_retention"
                )
            }
        )

        # Assemble Master Context
        raw_ctx = MultimodalContext(
            conversation=conv_block,
            emotion=emotion_block,
            symptoms=symptoms_block,
            risk=risk_block,
            baseline=baseline_block,
            behavior=behavior_block,
            voice=voice_block,
            history=history_block,
            metadata=meta_block
        )

        # 14. Apply Privacy Safeguards (Scrubbing / Pseudonymization)
        if privacy_level != PrivacyLevel.STANDARD:
            sanitized_dict, _ = self.privacy_engine.apply_privacy(
                raw_ctx.model_dump(),
                privacy_level=privacy_level
            )
            return MultimodalContext.model_validate(sanitized_dict)

        return raw_ctx

    @staticmethod
    def to_conversation_context(multimodal_ctx: MultimodalContext) -> ConversationContext:
        """Convert MultimodalContext to legacy ConversationContext for backwards compatibility."""
        return ConversationContext(
            session_id=multimodal_ctx.conversation.session_id,
            user_id=multimodal_ctx.conversation.user_id,
            turns=[],
            active_stage=multimodal_ctx.conversation.dialog_stage,
            metadata={
                "context_id": multimodal_ctx.metadata.context_id,
                "context_hash": multimodal_ctx.metadata.context_hash,
                "engine_version": multimodal_ctx.metadata.engine_version,
                "confidence": multimodal_ctx.metadata.confidence_propagation.get("overall_context_confidence", 0.0)
            },
            updated_at=multimodal_ctx.metadata.created_at
        )
