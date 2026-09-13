"""Audio validation and preprocessing for ZENOVA voice processing pipeline.

Handles:
- Inbound audio format decoding (Base64, raw bytes, SoundFile).
- Waveform normalization (mono conversion, float32 scaling).
- Duration and signal length checks.
- Silence and energy sanity verification.
"""
import io
import base64
from typing import Tuple, Optional
import numpy as np
import soundfile as sf
from zenova.core.logging import get_logger

logger = get_logger("zenova.voice.validator")


class AudioValidationError(ValueError):
    """Raised when inbound audio does not meet acoustic format or quality standards."""
    pass


class AudioValidator:
    """Validates and prepares audio waveforms for acoustic feature extraction."""

    def __init__(
        self,
        min_duration_sec: float = 0.25,
        max_duration_sec: float = 60.0,
        min_amplitude: float = 1e-4,
        target_sample_rate: int = 16000,
    ):
        self.min_duration_sec = min_duration_sec
        self.max_duration_sec = max_duration_sec
        self.min_amplitude = min_amplitude
        self.target_sample_rate = target_sample_rate

    def decode_base64_audio(self, b64_str: str) -> bytes:
        """Decode base64 string to raw audio bytes, handling data URI headers if present."""
        if "," in b64_str:
            # Strip data URI header e.g. "data:audio/wav;base64,"
            b64_str = b64_str.split(",", 1)[1]
        try:
            return base64.b64decode(b64_str.strip())
        except Exception as e:
            raise AudioValidationError(f"Invalid base64 audio encoding: {str(e)}") from e

    def load_audio_from_bytes(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        """Read audio bytes into a 1D float32 numpy array and sample rate using SoundFile."""
        if not audio_bytes:
            raise AudioValidationError("Audio bytes payload is empty.")

        try:
            buffer = io.BytesIO(audio_bytes)
            data, sr = sf.read(buffer, dtype="float32")
        except Exception as e:
            # Fallback: attempt to interpret as raw 16-bit PCM if header is missing
            try:
                pcm16 = np.frombuffer(audio_bytes, dtype=np.int16)
                data = pcm16.astype(np.float32) / 32768.0
                sr = self.target_sample_rate
            except Exception:
                raise AudioValidationError(f"Failed to decode audio bytes: {str(e)}") from e

        # Convert multi-channel to mono by averaging across channels
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        return data.astype(np.float32), sr

    def validate_and_normalize(
        self,
        data: np.ndarray,
        sr: int,
    ) -> Tuple[np.ndarray, int]:
        """Validate waveform length, amplitude, and normalize amplitude range."""
        if data.size == 0:
            raise AudioValidationError("Decoded audio waveform is empty.")

        duration = len(data) / float(sr)
        if duration < self.min_duration_sec:
            raise AudioValidationError(
                f"Audio duration ({duration:.2f}s) is shorter than minimum required ({self.min_duration_sec}s)."
            )
        if duration > self.max_duration_sec:
            raise AudioValidationError(
                f"Audio duration ({duration:.2f}s) exceeds maximum allowed limit ({self.max_duration_sec}s)."
            )

        # Silence / minimum amplitude check
        peak_amp = float(np.max(np.abs(data)))
        if peak_amp < self.min_amplitude:
            raise AudioValidationError(
                f"Audio signal is near-silent or completely empty (peak amplitude={peak_amp:.6f})."
            )

        # Peak normalization to prevent clipping while preserving dynamics
        if peak_amp > 1.0:
            data = data / peak_amp

        # Resample if needed (simple linear interpolation for robust pure numpy execution)
        if sr != self.target_sample_rate:
            num_target_samples = int(len(data) * self.target_sample_rate / sr)
            if num_target_samples > 0:
                indices = np.linspace(0, len(data) - 1, num_target_samples)
                data = np.interp(indices, np.arange(len(data)), data).astype(np.float32)
                sr = self.target_sample_rate

        return data, sr

    def process_raw_or_base64(
        self,
        audio_bytes: Optional[bytes] = None,
        audio_base64: Optional[str] = None,
        precomputed_waveform: Optional[np.ndarray] = None,
        sample_rate: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """Process any inbound audio representation into a normalized waveform."""
        if precomputed_waveform is not None:
            sr = sample_rate or self.target_sample_rate
            return self.validate_and_normalize(precomputed_waveform, sr)

        if audio_base64:
            audio_bytes = self.decode_base64_audio(audio_base64)

        if not audio_bytes:
            raise AudioValidationError("No audio data supplied (bytes or base64 string required).")

        raw_data, sr = self.load_audio_from_bytes(audio_bytes)
        return self.validate_and_normalize(raw_data, sr)
