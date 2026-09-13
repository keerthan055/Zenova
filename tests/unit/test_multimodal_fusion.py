"""Unit tests for ZENOVA Step 16: Multimodal Fusion Subsystem."""
import pytest
import torch
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any

from zenova.schemas.standard import (
    UserInput,
    EmotionResult,
    EmotionCategory,
    SymptomResult,
    SymptomSignal,
    SymptomSeverity,
    RiskResult,
    RiskLevel,
    CrisisCategory,
    VoiceResult,
    BehavioralResult,
    BaselineResult,
)
from zenova.schemas.baseline import BaselineStatus
from zenova.schemas.fusion import (
    ModalityName,
    ModalityMask,
    CrossModalDiscrepancy,
    FusedMultimodalState,
    ComparativeEvaluationReport,
)
from zenova.fusion.masking import ModalityMaskExtractor
from zenova.fusion.vectorizer import MultimodalFeatureVectorizer
from zenova.fusion.baseline_rule import WeightedRuleFusion
from zenova.fusion.learnable_gmu import (
    GatedMultimodalUnitNetwork,
    LearnableGMUFusionEngine
)
from zenova.fusion.evaluator import MultimodalFusionEvaluator
from zenova.fusion.engine import UnifiedMultimodalFusionEngine


@pytest.fixture
def user_input_text_only():
    return UserInput(text="I've been feeling somewhat tired lately.", session_id="s_test_1", user_id="u_test_1")


@pytest.fixture
def emotion_neutral():
    return EmotionResult(
        primary_emotion=EmotionCategory.NEUTRAL,
        confidence=0.85,
        valence=0.05,
        arousal=0.20,
        dominance=0.0,
        probabilities={EmotionCategory.NEUTRAL.value: 0.85, EmotionCategory.SADNESS.value: 0.10}
    )


@pytest.fixture
def symptom_mild():
    return SymptomResult(
        is_available=True,
        signals=[
            SymptomSignal(marker_name="insomnia", severity=SymptomSeverity.MILD, confidence=0.75)
        ],
        primary_signals=["sleep_disturbance"],
        distress_frequency=0.20
    )


@pytest.fixture
def risk_low():
    return RiskResult(
        risk_level=RiskLevel.LOW,
        confidence=0.90,
        crisis_category=CrisisCategory.NONE,
        is_high_risk=False,
        trigger_cues=[]
    )


@pytest.fixture
def voice_distressed():
    return VoiceResult(
        is_available=True,
        acoustic_features={
            "f0_mean": 290.0,
            "jitter": 0.045,
            "shimmer": 0.082,
            "duration": 4.5,
            "pitch_std": 45.0
        },
        primary_emotion=EmotionCategory.FEAR,
        valence=-0.75,
        arousal=0.88,
        confidence=0.92
    )


@pytest.fixture
def behavior_anomalous():
    return BehavioralResult(
        is_available=True,
        behavioral_anomaly_detected=True,
        metrics={
            "anomaly_score": 0.82,
            "step_count": 850,
            "screen_time_mins": 580,
            "sleep_duration_hours": 3.2
        }
    )


@pytest.fixture
def baseline_established():
    return BaselineResult(
        user_id="u_test_1",
        status=BaselineStatus.ESTABLISHED_BASELINE.value,
        total_observations=20,
        is_significant_deviation=True,
        metric_deviations={
            "valence": -2.8,
            "arousal": 2.2,
            "distress_score": 3.1
        },
        confidence=0.85
    )


# --- 1. Masking Tests ---

def test_mask_extractor_text_only(user_input_text_only, emotion_neutral, symptom_mild, risk_low):
    mask = ModalityMaskExtractor.extract_mask(
        user_input=user_input_text_only,
        emotion=emotion_neutral,
        symptoms=symptom_mild,
        risk=risk_low,
        voice=None,
        behavior=None,
        baseline=None,
        history=None
    )
    assert mask.mask[ModalityName.TEXT.value] is True
    assert mask.mask[ModalityName.EMOTION.value] is True
    assert mask.mask[ModalityName.SYMPTOMS.value] is True
    assert mask.mask[ModalityName.RISK.value] is True
    assert mask.mask[ModalityName.VOICE.value] is False
    assert mask.mask[ModalityName.BEHAVIOR.value] is False
    assert mask.mask[ModalityName.BASELINE.value] is False
    assert mask.mask[ModalityName.HISTORY.value] is False
    assert mask.available_count == 4


def test_mask_extractor_full(
    user_input_text_only, emotion_neutral, symptom_mild, risk_low,
    voice_distressed, behavior_anomalous, baseline_established
):
    mask = ModalityMaskExtractor.extract_mask(
        user_input=user_input_text_only,
        emotion=emotion_neutral,
        symptoms=symptom_mild,
        risk=risk_low,
        voice=voice_distressed,
        behavior=behavior_anomalous,
        baseline=baseline_established,
        history=[{"turn_id": 1, "speaker": "user", "content": "Hi"}]
    )
    assert mask.available_count == 8
    for mod in ModalityName:
        assert mask.mask[mod.value] is True


# --- 2. Feature Vectorizer Tests ---

def test_vectorizer_dimension_and_imputation(
    user_input_text_only, emotion_neutral, symptom_mild, risk_low
):
    vec, mask_arr, conf_arr = MultimodalFeatureVectorizer.vectorize(
        user_input=user_input_text_only,
        emotion=emotion_neutral,
        symptoms=symptom_mild,
        risk=risk_low,
        voice=None,
        behavior=None,
        baseline=None,
        history=None
    )
    assert vec.shape == (MultimodalFeatureVectorizer.FEATURE_DIM,)
    assert mask_arr.shape == (8,)
    assert conf_arr.shape == (8,)
    # Voice indices (22 to 27) should be 0.0
    voice_slice = vec[22:28]
    assert np.all(voice_slice == 0.0)
    # Emotion valence should match (index 5)
    assert abs(vec[5] - 0.05) < 1e-4


# --- 3. Weighted Rule Fusion Tests ---

def test_weighted_rule_fusion_text_only(
    user_input_text_only, emotion_neutral, symptom_mild, risk_low
):
    fuser = WeightedRuleFusion()
    result = fuser.fuse(
        user_input=user_input_text_only,
        emotion=emotion_neutral,
        symptoms=symptom_mild,
        risk=risk_low
    )
    assert isinstance(result, FusedMultimodalState)
    assert result.fusion_method == "weighted_rule"
    assert result.primary_affect == EmotionCategory.NEUTRAL
    assert -1.0 <= result.fused_valence <= 1.0
    assert 0.0 <= result.fused_arousal <= 1.0
    assert 0.0 <= result.fused_distress_score <= 1.0
    assert result.fused_risk_level == RiskLevel.LOW
    assert not result.discrepancy.detected
    # Missing modalities should have weight 0.0
    assert result.modality_weights.get("voice", 0.0) == 0.0
    assert result.modality_weights.get("behavior", 0.0) == 0.0
    assert sum(result.modality_weights.values()) == pytest.approx(1.0, abs=1e-3)


def test_weighted_rule_fusion_acoustic_masking(
    user_input_text_only, emotion_neutral, symptom_mild, risk_low, voice_distressed
):
    fuser = WeightedRuleFusion()
    result = fuser.fuse(
        user_input=user_input_text_only,
        emotion=emotion_neutral,  # Calm words: valence 0.05
        symptoms=symptom_mild,
        risk=risk_low,
        voice=voice_distressed    # Agitated voice: valence -0.75, arousal 0.88
    )
    assert result.discrepancy.detected is True
    assert "acoustic_semantic_masking" in result.discrepancy.discrepancy_types
    # Fused valence should be dragged negative by acoustic evidence
    assert result.fused_valence < 0.0
    assert result.modality_weights["voice"] > 0.0


def test_weighted_rule_fusion_behavioral_masking(
    user_input_text_only, emotion_neutral, symptom_mild, risk_low, behavior_anomalous
):
    fuser = WeightedRuleFusion()
    result = fuser.fuse(
        user_input=user_input_text_only,
        emotion=emotion_neutral,
        symptoms=symptom_mild,
        risk=risk_low,
        behavior=behavior_anomalous
    )
    assert result.discrepancy.detected is True
    assert "behavioral_verbal_masking" in result.discrepancy.discrepancy_types


# --- 4. Gated Multimodal Unit (GMU) Neural Network Tests ---

def test_gmu_network_forward_pass():
    model = GatedMultimodalUnitNetwork(hidden_dim=32)
    model.eval()

    batch_size = 4
    dims = MultimodalFeatureVectorizer.MODALITY_DIMS
    features = {m: torch.randn(batch_size, dims[m]) for m in model.modalities}
    masks = torch.ones(batch_size, 8)
    # Simulate missing voice and behavior for half the batch
    masks[2:, 4] = 0.0
    masks[2:, 5] = 0.0
    confs = torch.ones(batch_size, 8) * 0.85

    distress, risk_logits, valence, arousal, weights, discrepancy = model(features, masks, confs)
    assert distress.shape == (batch_size, 1)
    assert risk_logits.shape == (batch_size, 4)
    assert valence.shape == (batch_size, 1)
    assert arousal.shape == (batch_size, 1)
    assert weights.shape == (batch_size, 8)
    assert discrepancy.shape == (batch_size, 1)

    # Inactive modalities must receive near-zero softmax gating weights
    assert float(weights[2, 4].detach()) < 1e-4
    assert float(weights[3, 5].detach()) < 1e-4


def test_learnable_gmu_engine_fuse(
    user_input_text_only, emotion_neutral, symptom_mild, risk_low, voice_distressed
):
    engine = LearnableGMUFusionEngine()
    result = engine.fuse(
        user_input=user_input_text_only,
        emotion=emotion_neutral,
        symptoms=symptom_mild,
        risk=risk_low,
        voice=voice_distressed
    )
    assert isinstance(result, FusedMultimodalState)
    assert result.fusion_method == "learnable_gmu"
    assert -1.0 <= result.fused_valence <= 1.0
    assert 0.0 <= result.fused_distress_score <= 1.0
    assert result.modality_mask["voice"] is True
    assert result.modality_mask["behavior"] is False


def test_learnable_gmu_checkpoint_save_and_load(tmp_path):
    ckpt_path = tmp_path / "gmu_test.pt"
    engine1 = LearnableGMUFusionEngine(checkpoint_path=str(ckpt_path))
    engine1.save_checkpoint()
    assert ckpt_path.exists()

    engine2 = LearnableGMUFusionEngine(checkpoint_path=str(ckpt_path))
    assert engine2.network is not None


# --- 5. Comparative Evaluator Tests ---

def test_multimodal_evaluator_benchmark():
    evaluator = MultimodalFusionEvaluator()
    report = evaluator.run_benchmark(num_samples_per_regime=15)
    assert isinstance(report, ComparativeEvaluationReport)
    assert report.total_eval_samples == 60  # 4 regimes * 15 samples
    assert "text_only_baseline" in report.models
    assert "weighted_rule_fusion" in report.models
    assert "learnable_gmu" in report.models

    # Multimodal fusion should outperform text-only baseline
    text_model = report.models["text_only_baseline"]
    rule_model = report.models["weighted_rule_fusion"]
    gmu_model = report.models["learnable_gmu"]

    assert rule_model.average_macro_f1 >= text_model.average_macro_f1
    assert rule_model.average_distress_mae <= text_model.average_distress_mae
    assert report.multimodal_improves_performance is True
    assert report.best_overall_model in ["weighted_rule_fusion", "learnable_gmu"]


# --- 6. Unified Multimodal Fusion Engine Provider Switching ---

def test_unified_fusion_engine_switching(
    user_input_text_only, emotion_neutral, symptom_mild, risk_low
):
    # Rule provider
    rule_engine = UnifiedMultimodalFusionEngine(provider="weighted_rule")
    res_rule = rule_engine.fuse(
        user_input=user_input_text_only,
        emotion=emotion_neutral,
        symptoms=symptom_mild,
        risk=risk_low
    )
    assert res_rule.fusion_method == "weighted_rule"

    # GMU provider
    gmu_engine = UnifiedMultimodalFusionEngine(provider="learnable_gmu")
    res_gmu = gmu_engine.fuse(
        user_input=user_input_text_only,
        emotion=emotion_neutral,
        symptoms=symptom_mild,
        risk=risk_low
    )
    assert res_gmu.fusion_method == "learnable_gmu"
