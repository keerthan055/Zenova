"""Storage manager for longitudinal baseline profiles and observation series.

Provides thread-safe in-memory caching and optional asynchronous database persistence.
"""
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from zenova.schemas.baseline import UserBaselineProfile, FeatureBaselineStats, BaselineStatus
from zenova.core.logging import get_logger

logger = get_logger("zenova.baseline.storage")


class BaselineStorageManager:
    """Manages baseline profiles and historical time-series observations."""

    def __init__(self, persist_to_db: bool = True):
        self.persist_to_db = persist_to_db
        # In-memory storage: user_id -> UserBaselineProfile
        self._profiles: Dict[str, UserBaselineProfile] = {}
        # In-memory history: user_id -> List of feature dicts
        self._observations: Dict[str, List[Dict[str, Any]]] = {}

    def get_profile(self, user_id: str) -> Optional[UserBaselineProfile]:
        """Retrieve existing baseline profile from memory cache."""
        return self._profiles.get(user_id)

    def get_or_create_profile(self, user_id: str) -> UserBaselineProfile:
        """Retrieve or initialize a clean baseline profile."""
        if user_id not in self._profiles:
            self._profiles[user_id] = UserBaselineProfile(
                user_id=user_id,
                status=BaselineStatus.INSUFFICIENT_DATA.value,
                confidence=0.0,
                total_interactions_recorded=0,
                established_at=datetime.now(timezone.utc),
                last_updated_at=datetime.now(timezone.utc),
                feature_baselines={}
            )
        return self._profiles[user_id]

    def get_observations(self, user_id: str) -> List[Dict[str, Any]]:
        """Retrieve historical observation dictionaries for a user."""
        return self._observations.get(user_id, [])

    def add_observation(
        self,
        user_id: str,
        features: Dict[str, float],
        session_id: Optional[str] = None,
        turn_id: Optional[int] = None,
        timestamp: Optional[datetime] = None
    ):
        """Append an observation to the user's longitudinal time series."""
        if user_id not in self._observations:
            self._observations[user_id] = []

        obs_record = {
            "timestamp": (timestamp or datetime.now(timezone.utc)).isoformat(),
            "session_id": session_id,
            "turn_id": turn_id,
            "features": features
        }
        self._observations[user_id].append(obs_record)

    def save_profile(self, profile: UserBaselineProfile):
        """Save baseline profile in-memory."""
        profile.last_updated_at = datetime.now(timezone.utc)
        self._profiles[profile.user_id] = profile

    def reset_user(self, user_id: str):
        """Clear all historical observations and baseline for a user."""
        self._profiles.pop(user_id, None)
        self._observations.pop(user_id, None)
        logger.info(f"Reset baseline and observation history for user '{user_id}'")

    async def sync_to_db(self, user_id: str):
        """Asynchronously sync profile and observations to persistent database."""
        if not self.persist_to_db:
            return

        try:
            from zenova.db.session import get_db_session
            from zenova.db.repositories import BaselineRepository

            profile = self.get_profile(user_id)
            if not profile:
                return

            async with get_db_session() as db:
                repo = BaselineRepository(db)
                await repo.save_baseline(
                    user_id=user_id,
                    status=profile.status,
                    confidence=profile.confidence,
                    total_observations=profile.total_interactions_recorded,
                    profile_dict=profile.model_dump(mode="json")
                )
        except Exception as e:
            logger.warning(f"Could not sync baseline to database for {user_id}: {str(e)}")
