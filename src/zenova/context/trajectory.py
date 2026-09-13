"""Affective Trajectory and Outcome Analyzer for ZENOVA Step 9.

Tracks emotional shifts across turns, evaluates the impact of past support
strategies, and computes longitudinal valence trends.
"""
from typing import List, Dict, Any, Optional
from zenova.schemas.context import ValenceTrend, TurnOutcome


class TrajectoryAnalyzer:
    """Analyzes affective trajectory, sentiment deltas, and strategy effectiveness."""

    @staticmethod
    def analyze_trajectory(
        recent_turns: List[Dict[str, Any]],
        current_valence: Optional[float] = None,
        current_arousal: Optional[float] = None,
        current_emotion: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compute trajectory metrics, valence trend, and sentiment deltas.

        Returns:
            {
                "affective_trajectory": [...],
                "valence_trend": ValenceTrend,
                "previous_outcomes": [...],
                "dominant_themes": [...]
            }
        """
        trajectory = []
        outcomes = []

        # Extract affective points from historical user turns
        valences = []
        for t in recent_turns:
            speaker = t.get("speaker", "user")
            turn_id = t.get("turn_id", 0)
            emo = t.get("emotion") or {}
            val = emo.get("valence")
            aro = emo.get("arousal", 0.0)
            primary = emo.get("primary_emotion")

            if speaker == "user" and val is not None:
                valences.append(float(val))
                trajectory.append({
                    "turn_id": turn_id,
                    "valence": round(float(val), 3),
                    "arousal": round(float(aro), 3),
                    "primary_emotion": primary
                })

        # Append current turn affective point if provided
        if current_valence is not None:
            valences.append(float(current_valence))
            trajectory.append({
                "turn_id": (recent_turns[-1].get("turn_id", 0) + 1) if recent_turns else 1,
                "valence": round(float(current_valence), 3),
                "arousal": round(float(current_arousal or 0.0), 3),
                "primary_emotion": current_emotion
            })

        # Evaluate Sentiment Deltas and Outcomes for past strategies
        # Pair assistant strategies with subsequent user turn valences
        for i in range(len(recent_turns) - 1):
            curr_t = recent_turns[i]
            next_t = recent_turns[i + 1]

            if curr_t.get("speaker") == "assistant" and curr_t.get("strategy"):
                strat_info = curr_t.get("strategy")
                # Find valence before strategy (turn i-1) and valence after strategy (turn i+1)
                val_before = 0.0
                if i > 0 and recent_turns[i - 1].get("emotion"):
                    val_before = float(recent_turns[i - 1]["emotion"].get("valence", 0.0))

                val_after = 0.0
                if next_t.get("emotion"):
                    val_after = float(next_t["emotion"].get("valence", 0.0))

                delta = round(val_after - val_before, 3)
                outcome_score = max(-1.0, min(1.0, delta))
                outcomes.append({
                    "turn_id": curr_t.get("turn_id"),
                    "strategy": strat_info.get("selected_strategy") if isinstance(strat_info, dict) else str(strat_info),
                    "outcome_type": "affective_shift",
                    "sentiment_delta": delta,
                    "outcome_score": outcome_score,
                    "notes": "positive_shift" if delta > 0.15 else ("negative_shift" if delta < -0.15 else "neutral")
                })

        # Determine Valence Trend
        if len(valences) < 2:
            trend = ValenceTrend.INSUFFICIENT_DATA
        else:
            diffs = [valences[j] - valences[j - 1] for j in range(1, len(valences))]
            avg_diff = sum(diffs) / len(diffs)

            # Check for fluctuation (alternating signs)
            sign_changes = sum(
                1 for j in range(1, len(diffs))
                if (diffs[j] > 0.05 and diffs[j - 1] < -0.05) or (diffs[j] < -0.05 and diffs[j - 1] > 0.05)
            )

            if sign_changes >= 2:
                trend = ValenceTrend.FLUCTUATING
            elif avg_diff > 0.08:
                trend = ValenceTrend.IMPROVING
            elif avg_diff < -0.08:
                trend = ValenceTrend.DETERIORATING
            else:
                trend = ValenceTrend.STABLE

        # Extract naive dominant themes/keywords from recent turns
        dominant_themes = []
        for t in recent_turns:
            symptoms = t.get("symptoms") or {}
            signals = symptoms.get("signals", [])
            for sig in signals:
                lbl = sig.get("label") if isinstance(sig, dict) else str(sig)
                if lbl and lbl not in dominant_themes:
                    dominant_themes.append(lbl)

        return {
            "affective_trajectory": trajectory,
            "valence_trend": trend,
            "previous_outcomes": outcomes,
            "dominant_themes": dominant_themes[:5]
        }
