"""PyTorch Acoustic Neural Network and Emotion Classifier for Voice.

Implements a calibrated Multi-Layer Perceptron (MLP) with Batch Normalization
over 25-dimensional acoustic prosodic, spectral, and cepstral features.
Maps RAVDESS 8-emotion taxonomy into ZENOVA canonical EmotionCategory
and computes continuous Valence, Arousal, and Dominance (VAD) vectors.
"""
from typing import Dict, Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn

from zenova.schemas.standard import EmotionCategory
from zenova.schemas.voice import AcousticFeatures
from zenova.core.logging import get_logger

logger = get_logger("zenova.voice.model")

RAVDESS_EMOTIONS = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised"
]

RAVDESS_TO_ZENOVA = {
    "neutral": EmotionCategory.NEUTRAL,
    "calm": EmotionCategory.NEUTRAL,
    "happy": EmotionCategory.JOY,
    "sad": EmotionCategory.SADNESS,
    "angry": EmotionCategory.ANGER,
    "fearful": EmotionCategory.FEAR,
    "disgust": EmotionCategory.FRUSTRATION,
    "surprised": EmotionCategory.NEUTRAL,
}

# Empirical VAD coordinates for RAVDESS acoustic emotions
EMOTION_VAD = {
    "happy": (0.80, 0.60, 0.60),
    "calm": (0.50, -0.60, 0.30),
    "neutral": (0.00, -0.10, 0.00),
    "surprised": (0.20, 0.70, 0.10),
    "sad": (-0.65, -0.50, -0.60),
    "disgust": (-0.60, 0.40, 0.20),
    "fearful": (-0.75, 0.80, -0.80),
    "angry": (-0.70, 0.85, 0.75),
}

# Feature normalization reference parameters (mean, std) for 25 dimensions
FEATURE_MEANS = [
    180.0, 35.0, 110.0, 260.0,  # pitch: mean, std, min, max
    0.04, 0.02, -28.0,          # rms: mean, std, intensity_db
    1800.0, 3200.0, 0.08,       # spectral: centroid, rolloff, zcr
    0.02, 0.05, 14.0,           # jitter, shimmer, hnr
    -12.0, 8.0, -3.0, 4.0, -2.0, 2.0, -1.0, 1.0, 0.0, 1.0, 0.0, 0.5 # 12 MFCCs
]
FEATURE_STDS = [
    60.0, 25.0, 40.0, 90.0,
    0.03, 0.02, 10.0,
    600.0, 900.0, 0.05,
    0.02, 0.04, 6.0,
    8.0, 6.0, 4.0, 3.0, 3.0, 2.5, 2.0, 2.0, 1.5, 1.5, 1.0, 1.0
]


class VoiceEmotionClassifier(nn.Module):
    """Deep neural acoustic classifier for speech emotion recognition."""

    def __init__(self, input_dim: int = 25, num_classes: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw class logits."""
        return self.net(x)

    @classmethod
    def extract_feature_vector(cls, features: AcousticFeatures) -> np.ndarray:
        """Convert AcousticFeatures object into standardized 25-dimensional vector."""
        vec = [
            features.pitch_mean_hz or 180.0,
            features.pitch_std_hz or 25.0,
            features.pitch_min_hz or 120.0,
            features.pitch_max_hz or 240.0,
            features.rms_energy_mean or 0.04,
            features.rms_energy_std or 0.02,
            features.intensity_db or -28.0,
            features.spectral_centroid_mean or 1800.0,
            features.spectral_rolloff_mean or 3200.0,
            features.zero_crossing_rate or 0.08,
            features.jitter_local or 0.02,
            features.shimmer_local or 0.05,
            features.hnr_db or 14.0,
        ]
        # Append MFCCs (up to 12 coefficients)
        mfccs = features.mfcc_means[:12]
        if len(mfccs) < 12:
            mfccs = mfccs + [0.0] * (12 - len(mfccs))
        vec.extend(mfccs)

        # Normalize with reference statistics
        norm_vec = [
            (val - m) / max(s, 1e-4)
            for val, m, s in zip(vec, FEATURE_MEANS, FEATURE_STDS)
        ]
        return np.array(norm_vec, dtype=np.float32)

    def predict(
        self, features: AcousticFeatures
    ) -> Tuple[EmotionCategory, float, Dict[str, float], float, float, float]:
        """Classify acoustic emotion and compute VAD continuous vectors.
        
        Returns:
            primary_emotion: Canonical Ekman EmotionCategory
            confidence: Peak prediction confidence in [0, 1]
            probabilities: Probability distribution over ZENOVA emotion categories
            valence: [-1.0, 1.0]
            arousal: [-1.0, 1.0]
            dominance: [-1.0, 1.0]
        """
        self.eval()
        feat_vec = self.extract_feature_vector(features)
        x_tensor = torch.tensor(feat_vec, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits = self.forward(x_tensor)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        # Build raw RAVDESS class probabilities
        raw_probs = {
            RAVDESS_EMOTIONS[i]: float(probs[i]) for i in range(len(RAVDESS_EMOTIONS))
        }

        # Map to canonical ZENOVA emotion categories (summing probabilities)
        zenova_probs: Dict[str, float] = {
            cat.value: 0.0 for cat in EmotionCategory
        }
        for rav_lbl, p in raw_probs.items():
            z_cat = RAVDESS_TO_ZENOVA[rav_lbl]
            zenova_probs[z_cat.value] += p

        # Normalize so probabilities sum to 1.0
        tot = sum(zenova_probs.values())
        if tot > 0:
            zenova_probs = {k: round(v / tot, 4) for k, v in zenova_probs.items()}

        # Pick primary emotion
        best_cat_str = max(zenova_probs, key=zenova_probs.get)
        confidence = float(zenova_probs[best_cat_str])
        primary_emotion = EmotionCategory(best_cat_str)

        # Compute Valence, Arousal, Dominance from distribution
        val = sum(raw_probs[e] * EMOTION_VAD[e][0] for e in RAVDESS_EMOTIONS)
        aro = sum(raw_probs[e] * EMOTION_VAD[e][1] for e in RAVDESS_EMOTIONS)
        dom = sum(raw_probs[e] * EMOTION_VAD[e][2] for e in RAVDESS_EMOTIONS)

        # Acoustic modulation on Arousal: elevated energy & pitch boost arousal
        pitch_std = features.pitch_std_hz or 25.0
        intensity = features.intensity_db or -30.0
        if pitch_std > 50.0 or intensity > -20.0:
            aro = min(1.0, aro + 0.15)
        elif pitch_std < 15.0 and intensity < -35.0:
            aro = max(-1.0, aro - 0.15)

        return (
            primary_emotion,
            round(confidence, 3),
            zenova_probs,
            round(float(np.clip(val, -1.0, 1.0)), 3),
            round(float(np.clip(aro, -1.0, 1.0)), 3),
            round(float(np.clip(dom, -1.0, 1.0)), 3),
        )
