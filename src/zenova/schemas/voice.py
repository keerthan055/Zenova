"""Voice and acoustic emotion analysis data contracts for ZENOVA.

Clinical Boundary:
Acoustic emotion signals are observational vocal affect proxies.
Do NOT infer psychiatric diagnoses directly from vocal acoustics alone.
"""
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field, field_validator
from zenova.schemas.standard import EmotionCategory


class AudioFormat(str, Enum):
    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"
    FLAC = "flac"
    RAW_PCM = "raw_pcm"


class AudioPayload(BaseModel):
    """Inbound audio payload supporting base64 encoded audio or precomputed acoustic telemetry."""
    user_id: str
    session_id: Optional[str] = None
    audio_base64: Optional[str] = Field(default=None, description="Base64-encoded audio waveform")
    sampling_rate: int = Field(default=16000, ge=4000, le=96000)
    format: AudioFormat = Field(default=AudioFormat.WAV)
    channels: int = Field(default=1, ge=1, le=2)
    precomputed_features: Optional[Dict[str, float]] = Field(
        default=None, description="Direct acoustic features if audio already extracted client-side"
    )
    transcription_hint: Optional[str] = Field(
        default=None, description="Optional client-provided ASR transcription hint"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AcousticFeatures(BaseModel):
    """Container for standardized speech acoustic and prosodic features."""
    duration_seconds: float = Field(default=0.0, ge=0.0)
    
    # 1. Fundamental Frequency (F0 / Pitch) in Hz
    pitch_mean_hz: Optional[float] = None
    pitch_std_hz: Optional[float] = None
    pitch_min_hz: Optional[float] = None
    pitch_max_hz: Optional[float] = None

    # 2. Energy & Intensity
    rms_energy_mean: Optional[float] = None
    rms_energy_std: Optional[float] = None
    intensity_db: Optional[float] = None

    # 3. Spectral Properties
    spectral_centroid_mean: Optional[float] = None
    spectral_rolloff_mean: Optional[float] = None
    zero_crossing_rate: Optional[float] = None

    # 4. Mel-Frequency Cepstral Coefficients (MFCCs 1-13)
    mfcc_means: List[float] = Field(default_factory=list)

    # 5. Vocal Perturbation & Voice Quality (Jitter / Shimmer / HNR)
    jitter_local: Optional[float] = None
    shimmer_local: Optional[float] = None
    hnr_db: Optional[float] = None

    def to_flat_dict(self) -> Dict[str, float]:
        """Flatten features into a dictionary of scalar floats."""
        d = {
            "duration_seconds": self.duration_seconds,
            "pitch_mean_hz": self.pitch_mean_hz or 0.0,
            "pitch_std_hz": self.pitch_std_hz or 0.0,
            "pitch_min_hz": self.pitch_min_hz or 0.0,
            "pitch_max_hz": self.pitch_max_hz or 0.0,
            "rms_energy_mean": self.rms_energy_mean or 0.0,
            "rms_energy_std": self.rms_energy_std or 0.0,
            "intensity_db": self.intensity_db or 0.0,
            "spectral_centroid_mean": self.spectral_centroid_mean or 0.0,
            "spectral_rolloff_mean": self.spectral_rolloff_mean or 0.0,
            "zero_crossing_rate": self.zero_crossing_rate or 0.0,
            "jitter_local": self.jitter_local or 0.0,
            "shimmer_local": self.shimmer_local or 0.0,
            "hnr_db": self.hnr_db or 0.0,
        }
        for idx, mfcc_val in enumerate(self.mfcc_means[:13]):
            d[f"mfcc_{idx+1}"] = mfcc_val
        return d


class VoiceEmotionResult(BaseModel):
    """Structured voice/acoustic emotion analysis result."""
    is_placeholder: bool = Field(default=False)
    module_version: str = Field(default="1.0.0")
    is_available: bool = Field(default=True)
    primary_emotion: EmotionCategory = EmotionCategory.NEUTRAL
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    probabilities: Dict[str, float] = Field(default_factory=dict)
    valence: float = Field(default=0.0, ge=-1.0, le=1.0, description="Pleasantness: -1.0 to 1.0")
    arousal: float = Field(default=0.0, ge=-1.0, le=1.0, description="Activation energy: -1.0 to 1.0")
    dominance: float = Field(default=0.0, ge=-1.0, le=1.0, description="Perceived control: -1.0 to 1.0")
    transcription: Optional[str] = Field(default=None, description="ASR speech-to-text transcription")
    transcription_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    acoustic_features: Optional[Dict[str, float]] = Field(default_factory=dict)
    duration_seconds: Optional[float] = None
    disclaimer: str = Field(
        default="Acoustic emotion signals are observational vocal affect proxies; do not interpret as psychiatric diagnoses."
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("probabilities")
    @classmethod
    def validate_probabilities(cls, v: Dict[str, float]) -> Dict[str, float]:
        for k, p in v.items():
            if not (0.0 <= p <= 1.0):
                raise ValueError(f"Probability for {k} must be in [0, 1], got {p}")
        return v


# Standard alias
VoiceResult = VoiceEmotionResult

