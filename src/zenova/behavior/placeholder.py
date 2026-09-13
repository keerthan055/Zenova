"""Explicit placeholder implementation for Behavioral/Passive-Sensing Analyzer."""
from typing import Optional
from datetime import datetime, timezone
from zenova.core.interfaces import BaseBehaviorAnalyzer
from zenova.schemas.standard import UserInput, BehavioralResult, ConversationContext


class BehaviorPlaceholderAnalyzer(BaseBehaviorAnalyzer):
    MODULE_NAME = "BehaviorPlaceholderAnalyzer"
    VERSION = "placeholder-v0.1.0"

    def analyze(
        self,
        user_input: UserInput,
        context: Optional[ConversationContext] = None,
    ) -> BehavioralResult:
        """Return neutral placeholder behavioral result."""
        # If user metadata explicitly passed raw behavior, mark available but placeholder
        has_metadata = bool(user_input.metadata and "behavior" in user_input.metadata)

        return BehavioralResult(
            is_placeholder=True,
            module_version=self.VERSION,
            is_available=has_metadata,
            activity_level=None,
            sleep_duration_hours=None,
            social_conversation_minutes=None,
            phone_screen_unlocks=None,
            mobility_radius_km=None,
            behavioral_anomaly_detected=False,
            anomaly_notes="Placeholder behavioral analyzer: passive sensing evaluation bypassed.",
            metrics={},
            disclaimer="Passive behavioral signals are observational proxies; do not interpret as clinical diagnoses.",
            timestamp=datetime.now(timezone.utc),
        )
