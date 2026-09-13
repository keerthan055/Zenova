"""Extensible wearable and passive-sensing device plugin interface for ZENOVA.

Allows heterogeneous device data (Android passive sensing, Apple HealthKit, Fitbit,
Google Fit, Garmin, and custom wearables) to be ingested and mapped into standardized
BehavioralObservation schemas without modifying downstream models.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Type, Optional
from datetime import datetime, timezone

from zenova.schemas.behavior import BehavioralObservation, DeviceType
from zenova.core.logging import get_logger

logger = get_logger("zenova.behavior.wearable")


class BaseWearableDeviceAdapter(ABC):
    """Abstract device adapter plugin."""

    @property
    @abstractmethod
    def device_type(self) -> DeviceType:
        """The specific device or vendor format handled by this adapter."""
        pass

    @abstractmethod
    def validate_payload(self, raw_payload: Dict[str, Any]) -> bool:
        """Verify that the raw payload conforms to expected vendor structure."""
        pass

    @abstractmethod
    def parse_payload(self, raw_payload: Dict[str, Any]) -> BehavioralObservation:
        """Transform raw vendor telemetry into a standardized BehavioralObservation."""
        pass


class SmartphonePassiveAdapter(BaseWearableDeviceAdapter):
    """Adapter for StudentLife / Android background passive sensing telemetry."""

    @property
    def device_type(self) -> DeviceType:
        return DeviceType.SMARTPHONE_PASSIVE

    def validate_payload(self, raw_payload: Dict[str, Any]) -> bool:
        return isinstance(raw_payload, dict) and "user_id" in raw_payload

    def parse_payload(self, raw_payload: Dict[str, Any]) -> BehavioralObservation:
        user_id = str(raw_payload["user_id"])
        ts_str = raw_payload.get("timestamp")
        timestamp = (
            datetime.fromisoformat(ts_str) if ts_str else datetime.now(timezone.utc)
        )

        return BehavioralObservation(
            user_id=user_id,
            timestamp=timestamp,
            source_device=DeviceType.SMARTPHONE_PASSIVE,
            walking_minutes=raw_payload.get("walking_minutes"),
            running_minutes=raw_payload.get("running_minutes"),
            sedentary_minutes=raw_payload.get("sedentary_minutes"),
            step_count=raw_payload.get("step_count"),
            sleep_duration_hours=raw_payload.get("sleep_duration_hours"),
            sleep_disturbances_count=raw_payload.get("sleep_disturbances_count"),
            mobility_radius_km=raw_payload.get("mobility_radius_km"),
            location_entropy=raw_payload.get("location_entropy"),
            time_at_home_hours=raw_payload.get("time_at_home_hours"),
            conversation_duration_minutes=raw_payload.get("conversation_duration_minutes"),
            conversation_count=raw_payload.get("conversation_count"),
            screen_unlock_count=raw_payload.get("screen_unlock_count"),
            screen_time_minutes=raw_payload.get("screen_time_minutes"),
            extra_metrics=raw_payload.get("extra_metrics", {})
        )


class AppleHealthKitAdapter(BaseWearableDeviceAdapter):
    """Adapter for Apple HealthKit export formats (Apple Watch / iPhone)."""

    @property
    def device_type(self) -> DeviceType:
        return DeviceType.APPLE_WATCH

    def validate_payload(self, raw_payload: Dict[str, Any]) -> bool:
        return isinstance(raw_payload, dict) and "user_id" in raw_payload

    def parse_payload(self, raw_payload: Dict[str, Any]) -> BehavioralObservation:
        user_id = str(raw_payload["user_id"])
        ts_str = raw_payload.get("timestamp")
        timestamp = datetime.fromisoformat(ts_str) if ts_str else datetime.now(timezone.utc)

        hk_data = raw_payload.get("healthKitData", raw_payload)
        steps = hk_data.get("HKQuantityTypeIdentifierStepCount", raw_payload.get("step_count"))
        sleep_hrs = hk_data.get("sleepDurationHours", raw_payload.get("sleep_duration_hours"))
        active_mins = hk_data.get("HKQuantityTypeIdentifierAppleExerciseTime", raw_payload.get("walking_minutes"))

        return BehavioralObservation(
            user_id=user_id,
            timestamp=timestamp,
            source_device=DeviceType.APPLE_WATCH,
            step_count=int(steps) if steps is not None else None,
            walking_minutes=float(active_mins) if active_mins is not None else None,
            sleep_duration_hours=float(sleep_hrs) if sleep_hrs is not None else None,
            extra_metrics=raw_payload.get("extra_metrics", {})
        )


class FitbitGoogleFitAdapter(BaseWearableDeviceAdapter):
    """Adapter for Fitbit and Google Fit REST telemetry payloads."""

    @property
    def device_type(self) -> DeviceType:
        return DeviceType.FITBIT

    def validate_payload(self, raw_payload: Dict[str, Any]) -> bool:
        return isinstance(raw_payload, dict) and "user_id" in raw_payload

    def parse_payload(self, raw_payload: Dict[str, Any]) -> BehavioralObservation:
        user_id = str(raw_payload["user_id"])
        ts_str = raw_payload.get("timestamp")
        timestamp = datetime.fromisoformat(ts_str) if ts_str else datetime.now(timezone.utc)

        steps = raw_payload.get("summary", {}).get("steps", raw_payload.get("step_count"))
        sedentary = raw_payload.get("summary", {}).get("sedentaryMinutes", raw_payload.get("sedentary_minutes"))
        sleep_summary = raw_payload.get("sleep", {})
        sleep_minutes = sleep_summary.get("totalMinutesAsleep")
        sleep_hrs = (sleep_minutes / 60.0) if sleep_minutes else raw_payload.get("sleep_duration_hours")

        return BehavioralObservation(
            user_id=user_id,
            timestamp=timestamp,
            source_device=DeviceType.FITBIT,
            step_count=int(steps) if steps is not None else None,
            sedentary_minutes=float(sedentary) if sedentary is not None else None,
            sleep_duration_hours=round(float(sleep_hrs), 2) if sleep_hrs is not None else None,
            extra_metrics=raw_payload.get("extra_metrics", {})
        )


class GenericWearableAdapter(BaseWearableDeviceAdapter):
    """Standard generic fallback adapter accepting flat key-value metrics."""

    @property
    def device_type(self) -> DeviceType:
        return DeviceType.GENERIC_WEARABLE

    def validate_payload(self, raw_payload: Dict[str, Any]) -> bool:
        return isinstance(raw_payload, dict) and "user_id" in raw_payload

    def parse_payload(self, raw_payload: Dict[str, Any]) -> BehavioralObservation:
        user_id = str(raw_payload["user_id"])
        ts_str = raw_payload.get("timestamp")
        timestamp = datetime.fromisoformat(ts_str) if ts_str else datetime.now(timezone.utc)

        return BehavioralObservation(
            user_id=user_id,
            timestamp=timestamp,
            source_device=DeviceType.GENERIC_WEARABLE,
            walking_minutes=raw_payload.get("walking_minutes"),
            running_minutes=raw_payload.get("running_minutes"),
            sedentary_minutes=raw_payload.get("sedentary_minutes"),
            step_count=raw_payload.get("step_count"),
            sleep_duration_hours=raw_payload.get("sleep_duration_hours"),
            sleep_disturbances_count=raw_payload.get("sleep_disturbances_count"),
            mobility_radius_km=raw_payload.get("mobility_radius_km"),
            location_entropy=raw_payload.get("location_entropy"),
            time_at_home_hours=raw_payload.get("time_at_home_hours"),
            conversation_duration_minutes=raw_payload.get("conversation_duration_minutes"),
            conversation_count=raw_payload.get("conversation_count"),
            screen_unlock_count=raw_payload.get("screen_unlock_count"),
            screen_time_minutes=raw_payload.get("screen_time_minutes"),
            companion_interactions_count=raw_payload.get("companion_interactions_count"),
            extra_metrics=raw_payload.get("extra_metrics", {})
        )


class WearableAdapterRegistry:
    """Registry allowing pluggable third-party device integrations."""

    _ADAPTERS: Dict[str, BaseWearableDeviceAdapter] = {
        DeviceType.SMARTPHONE_PASSIVE.value: SmartphonePassiveAdapter(),
        DeviceType.APPLE_WATCH.value: AppleHealthKitAdapter(),
        DeviceType.FITBIT.value: FitbitGoogleFitAdapter(),
        DeviceType.GOOGLE_FIT.value: FitbitGoogleFitAdapter(),
        DeviceType.GENERIC_WEARABLE.value: GenericWearableAdapter(),
        "default": GenericWearableAdapter()
    }

    @classmethod
    def register(cls, device_key: str, adapter: BaseWearableDeviceAdapter):
        """Register a new wearable device adapter plugin at runtime."""
        cls._ADAPTERS[device_key.lower()] = adapter
        logger.info(f"Registered wearable adapter for device type '{device_key}'")

    @classmethod
    def get_adapter(cls, device_key: Optional[str] = None) -> BaseWearableDeviceAdapter:
        """Resolve adapter for device key, falling back to GenericWearableAdapter."""
        if not device_key:
            return cls._ADAPTERS["default"]
        return cls._ADAPTERS.get(device_key.lower(), cls._ADAPTERS["default"])
