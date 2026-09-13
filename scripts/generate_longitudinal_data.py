"""Generate realistic synthetic longitudinal data for ZENOVA Personal Baseline Engine testing.

Simulates 5 clinical & behavioral user archetypes over multi-week trajectories:
1. user_stable: Healthy baseline with natural day-to-day fluctuations (40 observations).
2. user_gradual_depression: Progressive slide into depressive symptoms and sleep disturbance over 35 days.
3. user_acute_panic_spike: 24 stable days followed by an acute panic spike (z > 3.0).
4. user_cold_start: Sparse data (2 observations) testing minimum-data requirements.
5. user_multimodal: Comprehensive trajectory including voice features and daily EMA self-reports.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json
import random
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any

from zenova.baseline.engine import PersonalBaselineEngine
from zenova.baseline.storage import BaselineStorageManager
from zenova.schemas.baseline import MultimodalObservation, BaselineStatus
from zenova.core.logging import get_logger

logger = get_logger("zenova.scripts.longitudinal_generator")


def generate_longitudinal_benchmark(seed: int = 42) -> Dict[str, List[Dict[str, Any]]]:
    """Generate longitudinal observation timelines across 5 user archetypes."""
    random.seed(seed)
    base_time = datetime(2026, 8, 1, 10, 0, 0, tzinfo=timezone.utc)
    dataset: Dict[str, List[Dict[str, Any]]] = {}

    # =========================================================================
    # 1. User Stable: Healthy baseline with mild natural variance (40 days)
    # =========================================================================
    user_id = "user_stable_baseline"
    records = []
    for day in range(40):
        t = base_time + timedelta(days=day, hours=random.uniform(0, 4))
        records.append({
            "user_id": user_id,
            "day": day + 1,
            "timestamp": t.isoformat(),
            "valence": round(random.gauss(0.35, 0.10), 4),
            "arousal": round(random.gauss(0.35, 0.08), 4),
            "word_count": int(random.gauss(25, 5)),
            "symptom_scores": {
                "anxiety": round(max(0.0, random.gauss(0.10, 0.04)), 4),
                "sleep_disturbance": round(max(0.0, random.gauss(0.12, 0.05)), 4)
            },
            "risk_severity_score": 0.0,
            "sleep_hours": round(random.gauss(7.5, 0.5), 1),
            "ema_mood": round(min(5.0, max(1.0, random.gauss(4.1, 0.3))), 1)
        })
    dataset[user_id] = records

    # =========================================================================
    # 2. User Gradual Depression: Steady 35-day downward slide
    # =========================================================================
    user_id = "user_gradual_depression"
    records = []
    for day in range(35):
        t = base_time + timedelta(days=day, hours=random.uniform(0, 4))
        if day < 15:
            # Baseline period (healthy)
            val = random.gauss(0.25, 0.10)
            sleep_dist = max(0.0, random.gauss(0.10, 0.05))
            depr_score = max(0.0, random.gauss(0.08, 0.04))
            ema = min(5.0, max(1.0, random.gauss(3.9, 0.3)))
        else:
            # Progressive depressive slide
            severity_factor = (day - 14) / 20.0  # 0.0 to 1.0
            val = random.gauss(0.25 - 0.85 * severity_factor, 0.10)
            sleep_dist = min(1.0, random.gauss(0.10 + 0.75 * severity_factor, 0.06))
            depr_score = min(1.0, random.gauss(0.08 + 0.80 * severity_factor, 0.06))
            ema = max(1.0, random.gauss(3.9 - 2.5 * severity_factor, 0.3))

        records.append({
            "user_id": user_id,
            "day": day + 1,
            "timestamp": t.isoformat(),
            "valence": round(val, 4),
            "arousal": round(random.gauss(0.25, 0.05), 4),
            "word_count": max(4, int(random.gauss(24 - (day / 3.0), 4))),
            "symptom_scores": {
                "sleep_disturbance": round(sleep_dist, 4),
                "depressive_mood": round(depr_score, 4)
            },
            "risk_severity_score": 1.0 if day >= 28 else 0.0,
            "sleep_hours": round(max(3.0, 7.5 - (day * 0.1)), 1),
            "ema_mood": round(ema, 1)
        })
    dataset[user_id] = records

    # =========================================================================
    # 3. User Acute Panic Spike: Stable for 24 days, then acute spike on day 25
    # =========================================================================
    user_id = "user_acute_panic_spike"
    records = []
    for day in range(25):
        t = base_time + timedelta(days=day, hours=random.uniform(0, 4))
        if day < 24:
            anx = max(0.0, random.gauss(0.08, 0.03))
            aro = random.gauss(0.30, 0.05)
            val = random.gauss(0.20, 0.08)
        else:
            # Acute panic attack spike
            anx = 0.88
            aro = 0.90
            val = -0.70

        records.append({
            "user_id": user_id,
            "day": day + 1,
            "timestamp": t.isoformat(),
            "valence": round(val, 4),
            "arousal": round(aro, 4),
            "word_count": 45 if day == 24 else int(random.gauss(20, 4)),
            "symptom_scores": {
                "anxiety": round(anx, 4)
            },
            "risk_severity_score": 1.0 if day == 24 else 0.0
        })
    dataset[user_id] = records

    # =========================================================================
    # 4. User Cold Start: Sparse interaction (2 days only)
    # =========================================================================
    user_id = "user_cold_start"
    dataset[user_id] = [
        {
            "user_id": user_id,
            "day": 1,
            "timestamp": base_time.isoformat(),
            "valence": 0.15,
            "arousal": 0.30,
            "word_count": 18,
            "symptom_scores": {"anxiety": 0.12},
            "risk_severity_score": 0.0
        },
        {
            "user_id": user_id,
            "day": 2,
            "timestamp": (base_time + timedelta(days=1)).isoformat(),
            "valence": 0.20,
            "arousal": 0.28,
            "word_count": 22,
            "symptom_scores": {"anxiety": 0.10},
            "risk_severity_score": 0.0
        }
    ]

    # =========================================================================
    # 5. User Multimodal: Interspersed voice features and EMA surveys (20 days)
    # =========================================================================
    user_id = "user_multimodal"
    records = []
    for day in range(20):
        t = base_time + timedelta(days=day, hours=random.uniform(0, 4))
        has_voice = (day % 2 == 0)
        rec = {
            "user_id": user_id,
            "day": day + 1,
            "timestamp": t.isoformat(),
            "valence": round(random.gauss(0.15, 0.12), 4),
            "arousal": round(random.gauss(0.40, 0.08), 4),
            "word_count": int(random.gauss(30, 6)),
            "symptom_scores": {
                "anxiety": round(max(0.0, random.gauss(0.20, 0.06)), 4)
            },
            "risk_severity_score": 0.0,
            "ema_mood": round(random.gauss(3.5, 0.4), 1)
        }
        if has_voice:
            rec["pitch_mean"] = round(random.gauss(185.0, 15.0), 1)
            rec["jitter"] = round(random.gauss(0.015, 0.003), 4)
        records.append(rec)
    dataset[user_id] = records

    return dataset


def main():
    print("=== Generating Synthetic Longitudinal Wellbeing Benchmark ===")
    dataset = generate_longitudinal_benchmark(seed=42)

    output_dir = Path("data/synthetic_longitudinal")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "longitudinal_benchmark.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    total_records = sum(len(records) for records in dataset.values())
    print(f"Generated {total_records} observations across {len(dataset)} distinct user profiles.")
    print(f"Saved dataset to: {out_file}\n")

    # Simulate personal baseline engine across these timelines
    print("--- Simulating Personal Baseline Engine Across Trajectories ---")
    engine = PersonalBaselineEngine(algorithm="rolling_window", window_size=30)

    for user_id, records in dataset.items():
        print(f"\nEvaluating profile: '{user_id}' ({len(records)} observations)...")
        last_report = None
        for r in records:
            obs = MultimodalObservation(
                user_id=user_id,
                valence=r.get("valence"),
                arousal=r.get("arousal"),
                word_count=r.get("word_count"),
                risk_severity_score=r.get("risk_severity_score"),
                ema_mood=r.get("ema_mood"),
                sleep_hours=r.get("sleep_hours"),
                pitch_mean=r.get("pitch_mean"),
                jitter=r.get("jitter"),
                symptom_scores=r.get("symptom_scores", {})
            )
            last_report = engine.evaluate_observation(obs)

        print(f"  Final Status: {last_report.status}")
        print(f"  Confidence: {last_report.confidence}")
        print(f"  Total Tracked: {last_report.total_observations + 1}")
        print(f"  Significant Deviation Detected: {last_report.is_significant_deviation}")
        if last_report.deviations:
            print("  Active Deviations on Final Day:")
            for d in last_report.deviations:
                if abs(d.deviation) >= 1.0:
                    print(f"    • {d.feature:24s}: z={d.deviation:+.2f} ({d.interpretation})")


if __name__ == "__main__":
    main()
