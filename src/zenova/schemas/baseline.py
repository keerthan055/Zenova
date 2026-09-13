"""Personal baseline tracking and longitudinal deviation schemas for ZENOVA.

Clinical Disclaimer:
All baseline models, rolling statistics, and deviations are observational heuristics.
Statistical deviations from an individual baseline do NOT constitute a clinical diagnosis.
"""
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class BaselineStatus(str, Enum):
    INSUFFICIENT_DATA = "insufficient_data"
    PROVISIONAL_BASELINE = "provisional_baseline"
    ESTABLISHED_BASELINE = "established_baseline"


class DeviationInterpretation(str, Enum):
    WITHIN_NORMAL = "within_normal_personal_variation"
    MODERATELY_ABOVE = "moderately_above_personal_baseline"
    NOTABLY_ABOVE = "notably_above_personal_baseline"
    SUBSTANTIALLY_ABOVE = "substantially_above_personal_baseline"
    MODERATELY_BELOW = "moderately_below_personal_baseline"
    NOTABLY_BELOW = "notably_below_personal_baseline"
    SUBSTANTIALLY_BELOW = "substantially_below_personal_baseline"


class FeatureDeviation(BaseModel):
    """Detailed statistical comparison of a specific feature against personal baseline."""
    feature: str
    current_value: float
    baseline_mean: float
    baseline_std: float
    deviation: float = Field(description="Z-score deviation relative to personal baseline")
    interpretation: str = Field(description="Standardized qualitative deviation categorization")


class MultimodalObservation(BaseModel):
    """Snapshot observation for an individual user turn or out-of-band event.
    
    Gracefully supports missing modalities. None of the modality-specific fields
    are strictly required.
    """
    user_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: Optional[str] = None
    turn_id: Optional[int] = None

    # Emotion features (optional)
    valence: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    arousal: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    dominant_emotion: Optional[str] = None
    emotion_confidences: Dict[str, float] = Field(default_factory=dict)

    # Symptom / signal features (optional)
    symptom_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Confidence or intensity per identified symptom marker"
    )
    active_symptom_count: Optional[int] = None

    # Risk signals (optional)
    risk_level: Optional[str] = None
    risk_confidence: Optional[float] = None
    risk_severity_score: Optional[float] = Field(default=None, ge=0.0, le=3.0)

    # Behavioral features (optional)
    word_count: Optional[int] = None
    sentiment_polarity: Optional[float] = None
    interaction_interval_hours: Optional[float] = None

    # Voice features (optional)
    pitch_mean: Optional[float] = None
    pitch_std: Optional[float] = None
    speaking_rate: Optional[float] = None
    jitter: Optional[float] = None

    # User-reported wellbeing (optional)
    ema_mood: Optional[float] = Field(default=None, ge=1.0, le=5.0)
    phq9_score: Optional[float] = Field(default=None, ge=0.0, le=27.0)
    gad7_score: Optional[float] = Field(default=None, ge=0.0, le=21.0)
    sleep_hours: Optional[float] = None

    # Additional / domain-specific approved features
    extra_features: Dict[str, float] = Field(default_factory=dict)


class FeatureBaselineStats(BaseModel):
    """Statistical summary for an individual tracked feature."""
    feature: str
    mean: float = 0.0
    std: float = 0.0
    variance: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    count: int = 0
    last_value: float = 0.0
    ewma_mean: Optional[float] = None
    ewma_variance: Optional[float] = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class UserBaselineProfile(BaseModel):
    """Legacy and high-level personal baseline profile representation."""
    user_id: str
    status: str = BaselineStatus.INSUFFICIENT_DATA.value
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    total_interactions_recorded: int = Field(default=0, ge=0)
    baseline_valence_mean: float = Field(default=0.0, ge=-1.0, le=1.0)
    baseline_valence_std: float = Field(default=0.2, ge=0.0)
    typical_turn_length_words: float = Field(default=20.0, ge=0.0)
    typical_sentiment_positivity: float = Field(default=0.5, ge=0.0, le=1.0)
    established_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    feature_baselines: Dict[str, FeatureBaselineStats] = Field(default_factory=dict)


class BaselineDeviationReport(BaseModel):
    """Evaluation report comparing current observation against personal baseline."""
    user_id: str
    status: str = BaselineStatus.INSUFFICIENT_DATA.value
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    total_observations: int = 0
    z_score_valence: float = Field(default=0.0, description="Deviation of current valence from personal baseline")
    is_significant_deviation: bool = False
    deviation_notes: Optional[str] = None
    deviations: List[FeatureDeviation] = Field(default_factory=list)
    metric_deviations: Dict[str, float] = Field(default_factory=dict)
    deviating_features: List[str] = Field(default_factory=list)
    disclaimer: str = (
        "Statistical deviation from personal baseline; observational and non-diagnostic."
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
