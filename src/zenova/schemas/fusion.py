"""Pydantic schemas for the ZENOVA Multimodal Fusion Subsystem."""
from enum import Enum
from datetime import datetime, timezone
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from zenova.schemas.standard import EmotionCategory, RiskLevel


class ModalityName(str, Enum):
    """Enumeration of all supported modalities in the ZENOVA pipeline."""
    TEXT = "text"
    EMOTION = "emotion"
    SYMPTOMS = "symptoms"
    RISK = "risk"
    VOICE = "voice"
    BEHAVIOR = "behavior"
    BASELINE = "baseline"
    HISTORY = "history"


class ModalityMask(BaseModel):
    """Availability mask and confidence scores across all pipeline modalities."""
    mask: Dict[str, bool] = Field(..., description="Boolean flag for each modality indicating presence")
    confidences: Dict[str, float] = Field(..., description="Raw confidence score (0.0 to 1.0) for each modality")
    available_count: int = Field(..., description="Total count of available modalities")
    active_modalities: List[str] = Field(default_factory=list, description="Names of available modalities")


class CrossModalDiscrepancy(BaseModel):
    """Detailed report of cross-modal incongruities and affective masking."""
    detected: bool = False
    discrepancy_types: List[str] = Field(default_factory=list, description="Types of discrepancy identified")
    reasons: List[str] = Field(default_factory=list, description="Human-readable explanation of incongruity")
    confidence: float = 0.0


class FusedMultimodalState(BaseModel):
    """Unified multimodal affective, clinical, and risk assessment."""
    primary_affect: EmotionCategory
    fused_valence: float = Field(..., ge=-1.0, le=1.0, description="Calibrated valence (-1.0 negative to +1.0 positive)")
    fused_arousal: float = Field(..., ge=0.0, le=1.0, description="Calibrated arousal (0.0 calm to 1.0 excited/agitated)")
    fused_dominance: float = Field(default=0.0, ge=-1.0, le=1.0, description="Calibrated dominance")
    fused_distress_score: float = Field(..., ge=0.0, le=1.0, description="Unified distress severity index (0.0 to 1.0)")
    fused_risk_level: RiskLevel = Field(..., description="Unified crisis and triage risk tier")
    urgency_score: float = Field(..., ge=0.0, le=1.0, description="Overall intervention urgency (0.0 to 1.0)")
    modality_weights: Dict[str, float] = Field(..., description="Normalized contribution weights of each modality")
    modality_mask: Dict[str, bool] = Field(..., description="Availability mask at time of fusion")
    discrepancy: CrossModalDiscrepancy = Field(default_factory=CrossModalDiscrepancy)
    fusion_method: str = Field(..., description="'weighted_rule' or 'learnable_gmu'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence of the fused assessment")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EvaluationRegimeMetrics(BaseModel):
    """Evaluation metrics for a specific modality availability regime."""
    regime_name: str
    sample_count: int
    distress_mae: float
    distress_rmse: float
    risk_macro_f1: float
    risk_accuracy: float
    discrepancy_f1: float
    calibration_brier_score: float


class ModelComparisonMetrics(BaseModel):
    """Performance metrics across regimes for a specific model."""
    model_name: str
    regimes: Dict[str, EvaluationRegimeMetrics]
    average_macro_f1: float
    average_distress_mae: float


class ComparativeEvaluationReport(BaseModel):
    """Comparative evaluation benchmark report comparing Text-Only, Rule Fusion, and Learnable GMU."""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_eval_samples: int
    models: Dict[str, ModelComparisonMetrics]
    summary_findings: str
    multimodal_improves_performance: bool
    best_overall_model: str
