"""Behavioral and passive-sensing data contracts for ZENOVA.

Clinical Boundary:
Passive behavioral signals are observational proxies.
Do NOT infer sensitive clinical conclusions (e.g., psychiatric diagnoses)
directly from any isolated behavioral feature.
"""
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class DeviceType(str, Enum):
    SMARTPHONE_PASSIVE = "smartphone_passive"
    APPLE_WATCH = "apple_watch"
    FITBIT = "fitbit"
    GOOGLE_FIT = "google_fit"
    GARMIN = "garmin"
    GENERIC_WEARABLE = "generic_wearable"


class ActivityLevel(str, Enum):
    SEDENTARY = "sedentary"
    LOW = "low"
    MODERATE = "moderate"
    ACTIVE = "active"


class BehavioralObservation(BaseModel):
    """Raw or preprocessed observation of passive behavioral signals for a user."""
    user_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_device: DeviceType = DeviceType.SMARTPHONE_PASSIVE

    # 1. Physical Activity Proxies (minutes per day)
    walking_minutes: Optional[float] = Field(default=None, ge=0.0)
    running_minutes: Optional[float] = Field(default=None, ge=0.0)
    sedentary_minutes: Optional[float] = Field(default=None, ge=0.0)
    step_count: Optional[int] = Field(default=None, ge=0)

    # 2. Sleep-Related Proxies
    sleep_duration_hours: Optional[float] = Field(default=None, ge=0.0, le=24.0)
    bedtime: Optional[datetime] = None
    waketime: Optional[datetime] = None
    sleep_disturbances_count: Optional[int] = Field(default=None, ge=0)

    # 3. Mobility & Location-Derived Patterns
    mobility_radius_km: Optional[float] = Field(default=None, ge=0.0)
    location_entropy: Optional[float] = Field(default=None, ge=0.0, description="Normalized location diversity")
    time_at_home_hours: Optional[float] = Field(default=None, ge=0.0, le=24.0)

    # 4. Social Communication Patterns
    conversation_duration_minutes: Optional[float] = Field(default=None, ge=0.0)
    conversation_count: Optional[int] = Field(default=None, ge=0)

    # 5. Phone / Device Usage
    screen_unlock_count: Optional[int] = Field(default=None, ge=0)
    screen_time_minutes: Optional[float] = Field(default=None, ge=0.0)

    # 6. Interaction Frequency (with companion app)
    companion_interactions_count: Optional[int] = Field(default=None, ge=0)

    # Extensible metrics for future devices
    extra_metrics: Dict[str, float] = Field(default_factory=dict)


class BehavioralAnomalyReport(BaseModel):
    """Report evaluating potential multi-feature behavioral disruptions against personal baseline."""
    user_id: str
    is_anomaly: bool = False
    disruption_index: float = Field(default=0.0, ge=0.0, description="Composite Behavioral Disruption Index (BDI)")
    anomalous_features: List[str] = Field(default_factory=list)
    feature_z_scores: Dict[str, float] = Field(default_factory=dict)
    observation_count: int = 0
    clinical_notes: str = "Behavioral patterns within normal individual variance."
    disclaimer: str = (
        "Passive behavioral signals are observational proxies; "
        "do not interpret as clinical diagnoses."
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
