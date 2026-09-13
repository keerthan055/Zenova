"""Pure SciPy/NumPy acoustic feature extraction for speech emotion analysis.

Extracts prosodic, spectral, energy, and voice quality features:
1. Fundamental Frequency (F0 / Pitch) via autocorrelation
2. Energy & Intensity (RMS, dB)
3. Spectral Properties (Centroid, Rolloff, Zero-Crossing Rate)
4. Mel-Frequency Cepstral Coefficients (MFCCs 1-13)
5. Vocal Perturbation (Jitter, Shimmer, Harmonics-to-Noise Ratio)
"""
import math
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import signal
from scipy import fft as sfft

from zenova.schemas.voice import AcousticFeatures
from zenova.core.logging import get_logger

logger = get_logger("zenova.voice.extractor")


def hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def create_mel_filterbank(
    num_filters: int = 26,
    fft_size: int = 512,
    sample_rate: int = 16000,
    low_freq: float = 0.0,
    high_freq: Optional[float] = None,
) -> np.ndarray:
    """Construct triangular Mel filterbank matrix."""
    high_freq = high_freq or (sample_rate / 2.0)
    low_mel = hz_to_mel(low_freq)
    high_mel = hz_to_mel(high_freq)
    mel_points = np.linspace(low_mel, high_mel, num_filters + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((fft_size + 1) * hz_points / sample_rate).astype(int)

    filterbank = np.zeros((num_filters, fft_size // 2 + 1))
    for m in range(1, num_filters + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]

        for k in range(f_m_minus, f_m):
            if f_m > f_m_minus:
                filterbank[m - 1, k] = (k - bin_points[m - 1]) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus > f_m:
                filterbank[m - 1, k] = (bin_points[m + 1] - k) / (f_m_plus - f_m)

    return filterbank


class AcousticFeatureExtractor:
    """Extracts standardized acoustic and prosodic features from speech waveforms."""

    def __init__(
        self,
        frame_length_ms: float = 25.0,
        frame_shift_ms: float = 10.0,
        min_pitch_hz: float = 65.0,
        max_pitch_hz: float = 500.0,
        num_mfcc: int = 13,
        num_mel_filters: int = 26,
        fft_size: int = 512,
    ):
        self.frame_length_ms = frame_length_ms
        self.frame_shift_ms = frame_shift_ms
        self.min_pitch_hz = min_pitch_hz
        self.max_pitch_hz = max_pitch_hz
        self.num_mfcc = num_mfcc
        self.num_mel_filters = num_mel_filters
        self.fft_size = fft_size

    def frame_signal(self, data: np.ndarray, sr: int) -> np.ndarray:
        """Partition signal into overlapping windowed frames."""
        frame_len = int(self.frame_length_ms * sr / 1000.0)
        frame_shift = int(self.frame_shift_ms * sr / 1000.0)
        if len(data) < frame_len:
            # Pad with zeros if shorter than one frame
            data = np.pad(data, (0, frame_len - len(data)), mode="constant")

        num_frames = 1 + (len(data) - frame_len) // frame_shift
        frames = np.lib.stride_tricks.as_strided(
            data,
            shape=(num_frames, frame_len),
            strides=(data.strides[0] * frame_shift, data.strides[0]),
            writeable=False
        )
        # Apply Hamming window
        window = np.hamming(frame_len)
        return frames * window

    def extract_f0_pitch(self, frames: np.ndarray, sr: int) -> Tuple[List[float], List[float]]:
        """Estimate fundamental frequency (F0) per frame via normalized autocorrelation.
        
        Returns:
            f0_estimates: List of voiced pitch values in Hz.
            peak_amplitudes: Cycle peak amplitudes for shimmer calculation.
        """
        frame_len = frames.shape[1]
        min_lag = max(1, int(sr / self.max_pitch_hz))
        max_lag = min(frame_len - 1, int(sr / self.min_pitch_hz))

        f0_estimates: List[float] = []
        peak_amplitudes: List[float] = []

        for frame in frames:
            energy = np.sum(frame ** 2)
            if energy < 1e-5:
                continue

            # Normalized autocorrelation
            corr = np.correlate(frame, frame, mode="full")
            corr = corr[len(frame) - 1:]  # Keep non-negative lags
            r0 = corr[0]
            if r0 <= 0:
                continue

            norm_corr = corr / r0
            if max_lag > min_lag:
                search_region = norm_corr[min_lag:max_lag]
                peak_idx = int(np.argmax(search_region)) + min_lag
                peak_val = norm_corr[peak_idx]

                # Voicing threshold
                if peak_val > 0.35:
                    f0 = sr / float(peak_idx)
                    f0_estimates.append(f0)
                    peak_amplitudes.append(float(np.max(np.abs(frame))))

        return f0_estimates, peak_amplitudes

    def extract_mfccs(self, frames: np.ndarray, sr: int) -> List[float]:
        """Compute average Mel-Frequency Cepstral Coefficients (MFCCs 1-13)."""
        # Magnitude spectrum
        mag_spec = np.abs(np.fft.rfft(frames, n=self.fft_size, axis=-1))
        power_spec = (mag_spec ** 2) / float(self.fft_size)

        # Apply Mel filterbank
        fb = create_mel_filterbank(
            num_filters=self.num_mel_filters,
            fft_size=self.fft_size,
            sample_rate=sr
        )
        mel_energies = np.dot(power_spec, fb.T)
        log_mel_energies = np.log(np.maximum(mel_energies, 1e-8))

        # Discrete Cosine Transform (DCT-II)
        dct_matrix = sfft.dct(log_mel_energies, type=2, axis=-1, norm="ortho")
        # Take first num_mfcc coefficients and average across time
        mfcc_means = np.mean(dct_matrix[:, :self.num_mfcc], axis=0)
        return [round(float(v), 4) for v in mfcc_means]

    def extract_spectral_properties(
        self, frames: np.ndarray, sr: int
    ) -> Tuple[float, float, float]:
        """Calculate mean spectral centroid, spectral rolloff, and zero-crossing rate."""
        # STFT magnitude
        mag = np.abs(np.fft.rfft(frames, n=self.fft_size, axis=-1))
        freqs = np.fft.rfftfreq(self.fft_size, d=1.0 / sr)

        # 1. Spectral Centroid
        mag_sum = np.sum(mag, axis=-1, keepdims=True)
        mag_sum_safe = np.where(mag_sum == 0, 1e-8, mag_sum)
        centroids = np.sum(mag * freqs, axis=-1, keepdims=True) / mag_sum_safe
        mean_centroid = float(np.mean(centroids))

        # 2. Spectral Rolloff (85% energy point)
        cum_energy = np.cumsum(mag, axis=-1)
        total_energy = cum_energy[:, -1:]
        rolloff_threshold = 0.85 * total_energy
        rolloff_bins = np.argmax(cum_energy >= rolloff_threshold, axis=-1)
        rolloff_freqs = freqs[rolloff_bins]
        mean_rolloff = float(np.mean(rolloff_freqs))

        # 3. Zero Crossing Rate (on unwindowed raw frames)
        frame_len = frames.shape[1]
        zcr_per_frame = np.mean(np.abs(np.diff(np.sign(frames), axis=-1)) > 0, axis=-1)
        mean_zcr = float(np.mean(zcr_per_frame))

        return round(mean_centroid, 2), round(mean_rolloff, 2), round(mean_zcr, 4)

    def extract_energy_and_intensity(self, frames: np.ndarray) -> Tuple[float, float, float]:
        """Calculate RMS energy mean, std, and intensity in dB."""
        rms_per_frame = np.sqrt(np.mean(frames ** 2, axis=-1))
        mean_rms = float(np.mean(rms_per_frame))
        std_rms = float(np.std(rms_per_frame))
        intensity_db = float(20.0 * np.log10(max(mean_rms, 1e-6)))
        return round(mean_rms, 6), round(std_rms, 6), round(intensity_db, 2)

    def compute_jitter_shimmer_hnr(
        self, f0_list: List[float], peak_amps: List[float], frames: np.ndarray
    ) -> Tuple[float, float, float]:
        """Calculate vocal perturbation measures: local jitter, local shimmer, and HNR."""
        # 1. Local Jitter (relative period variation)
        if len(f0_list) >= 2:
            periods = [1.0 / f for f in f0_list if f > 0]
            if len(periods) >= 2:
                period_diffs = np.abs(np.diff(periods))
                mean_p = np.mean(periods)
                jitter = float(np.mean(period_diffs) / max(mean_p, 1e-6))
            else:
                jitter = 0.0
        else:
            jitter = 0.0

        # 2. Local Shimmer (relative peak amplitude variation)
        if len(peak_amps) >= 2:
            amp_diffs = np.abs(np.diff(peak_amps))
            mean_a = np.mean(peak_amps)
            shimmer = float(np.mean(amp_diffs) / max(mean_a, 1e-6))
        else:
            shimmer = 0.0

        # 3. Harmonics-to-Noise Ratio (HNR)
        if len(f0_list) > 0 and len(frames) > 0:
            avg_frame = np.mean(frames, axis=0)
            signal_power = np.sum(avg_frame ** 2)
            noise_power = np.mean(np.var(frames, axis=0))
            if noise_power > 1e-8:
                hnr = float(10.0 * np.log10(max(signal_power, 1e-8) / noise_power))
            else:
                hnr = 30.0  # Max ceiling
        else:
            hnr = 15.0

        return round(min(1.0, jitter), 4), round(min(1.0, shimmer), 4), round(hnr, 2)

    def extract(self, waveform: np.ndarray, sr: int = 16000) -> AcousticFeatures:
        """Perform end-to-end acoustic feature extraction on waveform."""
        duration_sec = round(len(waveform) / float(sr), 3)
        frames = self.frame_signal(waveform, sr)

        # 1. Pitch
        f0_list, peak_amps = self.extract_f0_pitch(frames, sr)
        if f0_list:
            pitch_mean = round(float(np.mean(f0_list)), 2)
            pitch_std = round(float(np.std(f0_list)), 2)
            pitch_min = round(float(np.min(f0_list)), 2)
            pitch_max = round(float(np.max(f0_list)), 2)
        else:
            pitch_mean = None
            pitch_std = None
            pitch_min = None
            pitch_max = None

        # 2. Energy
        rms_mean, rms_std, intensity_db = self.extract_energy_and_intensity(frames)

        # 3. Spectral
        centroid, rolloff, zcr = self.extract_spectral_properties(frames, sr)

        # 4. MFCCs
        mfccs = self.extract_mfccs(frames, sr)

        # 5. Jitter / Shimmer / HNR
        jitter, shimmer, hnr = self.compute_jitter_shimmer_hnr(f0_list, peak_amps, frames)

        return AcousticFeatures(
            duration_seconds=duration_sec,
            pitch_mean_hz=pitch_mean,
            pitch_std_hz=pitch_std,
            pitch_min_hz=pitch_min,
            pitch_max_hz=pitch_max,
            rms_energy_mean=rms_mean,
            rms_energy_std=rms_std,
            intensity_db=intensity_db,
            spectral_centroid_mean=centroid,
            spectral_rolloff_mean=rolloff,
            zero_crossing_rate=zcr,
            mfcc_means=mfccs,
            jitter_local=jitter,
            shimmer_local=shimmer,
            hnr_db=hnr,
        )
