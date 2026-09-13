"""Extensible Dashboard Module Registry for ZENOVA."""
from typing import Dict, List, Optional
from zenova.schemas.dashboard import DashboardModuleDescriptor


class DashboardModuleRegistry:
    """Registry allowing pluggable dashboard modules, analytical cards, and sensor widgets."""

    def __init__(self):
        self._modules: Dict[str, DashboardModuleDescriptor] = {}
        self._register_default_modules()

    def _register_default_modules(self) -> None:
        defaults = [
            DashboardModuleDescriptor(
                module_id="overview_alerts",
                name="Active Clinical Alerts",
                category="overview",
                description="Live triage queue of active crisis and safety escalation incidents."
            ),
            DashboardModuleDescriptor(
                module_id="overview_risk",
                name="Risk Distribution Trends",
                category="overview",
                description="Population-level breakdown of user risk trajectories."
            ),
            DashboardModuleDescriptor(
                module_id="overview_system",
                name="System Component Health",
                category="overview",
                description="Real-time model registry states, latencies, and device status."
            ),
            DashboardModuleDescriptor(
                module_id="user_emotion_trajectory",
                name="Emotion Trajectory Inspector",
                category="user_trends",
                description="Turn-by-turn primary emotion, valence, and arousal tracking."
            ),
            DashboardModuleDescriptor(
                module_id="user_symptom_radar",
                name="Symptom Signal Tracker",
                category="user_trends",
                description="Longitudinal monitoring of depressive, anxious, and sleep disturbance signals."
            ),
            DashboardModuleDescriptor(
                module_id="user_baseline_deviation",
                name="Baseline Deviation Gauge",
                category="user_trends",
                description="Z-score shifts compared to personalized historical baseline."
            ),
            DashboardModuleDescriptor(
                module_id="user_strategy_history",
                name="Support Strategy Distribution",
                category="user_trends",
                description="Taxonomy breakdown of ESConv strategies utilized during dialogue."
            ),
            DashboardModuleDescriptor(
                module_id="longitudinal_timeline",
                name="Unified Longitudinal Timeline",
                category="timeline",
                description="Chronological event stream uniting conversations, sensor observations, and alerts."
            ),
            DashboardModuleDescriptor(
                module_id="explainability_inspector",
                name="ML Transparency & Explainability Inspector",
                category="explainability",
                description="Granular model version, confidence, cues, and clinical limitations."
            ),
            DashboardModuleDescriptor(
                module_id="behavioral_sensor_card",
                name="Passive Behavioral Sensor Insights",
                category="sensor",
                description="Mobility, step count, and sleep proxy deviation tracker."
            ),
            DashboardModuleDescriptor(
                module_id="voice_acoustic_card",
                name="Voice Acoustic Emotion Card",
                category="sensor",
                description="Acoustic prosody, pitch, jitter, and vocal affect markers."
            )
        ]
        for m in defaults:
            self._modules[m.module_id] = m

    def register(self, descriptor: DashboardModuleDescriptor) -> None:
        """Register a new or custom dashboard module."""
        self._modules[descriptor.module_id] = descriptor

    def unregister(self, module_id: str) -> Optional[DashboardModuleDescriptor]:
        """Unregister an existing dashboard module."""
        return self._modules.pop(module_id, None)

    def get_module(self, module_id: str) -> Optional[DashboardModuleDescriptor]:
        """Retrieve a registered module by ID."""
        return self._modules.get(module_id)

    def list_modules(self, category: Optional[str] = None) -> List[DashboardModuleDescriptor]:
        """Enumerate registered modules with optional category filtering."""
        mods = list(self._modules.values())
        if category:
            mods = [m for m in mods if m.category == category]
        return mods
