"""Acoustic heuristic and rule-based baseline for speech emotion comparison."""
from typing import Dict, Tuple
from zenova.schemas.standard import EmotionCategory
from zenova.schemas.voice import AcousticFeatures
from zenova.core.logging import get_logger

logger = get_logger("zenova.voice.baseline")


class VoiceAcousticBaseline:
    """Interpretable rule-based acoustic baseline classifier."""

    def predict(
        self, features: AcousticFeatures
    ) -> Tuple[EmotionCategory, float, Dict[str, float]]:
        pitch_mean = features.pitch_mean_hz or 170.0
        pitch_std = features.pitch_std_hz or 25.0
        intensity = features.intensity_db or -28.0
        jitter = features.jitter_local or 0.02

        # Probabilities dictionary initialized to uniform baseline
        probs = {cat.value: 0.10 for cat in EmotionCategory}

        if intensity > -22.0 and pitch_std > 50.0:
            if jitter > 0.04:
                primary = EmotionCategory.ANGER
                probs[EmotionCategory.ANGER.value] = 0.55
                probs[EmotionCategory.FEAR.value] = 0.25
            else:
                primary = EmotionCategory.JOY
                probs[EmotionCategory.JOY.value] = 0.50
                probs[EmotionCategory.HOPE.value] = 0.20
        elif intensity < -35.0 and pitch_mean < 140.0:
            primary = EmotionCategory.SADNESS
            probs[EmotionCategory.SADNESS.value] = 0.60
            probs[EmotionCategory.NEUTRAL.value] = 0.25
        elif pitch_std > 60.0 and jitter > 0.05:
            primary = EmotionCategory.FEAR
            probs[EmotionCategory.FEAR.value] = 0.50
            probs[EmotionCategory.ANGER.value] = 0.25
        else:
            primary = EmotionCategory.NEUTRAL
            probs[EmotionCategory.NEUTRAL.value] = 0.60

        # Normalize
        tot = sum(probs.values())
        probs = {k: round(v / tot, 4) for k, v in probs.items()}
        conf = probs[primary.value]

        return primary, conf, probs
