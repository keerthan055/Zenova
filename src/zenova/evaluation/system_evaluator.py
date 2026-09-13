"""System performance, latency quantiles, reliability, throughput, and modality stability benchmark."""
import time
import uuid
import numpy as np
from typing import List, Dict, Any, Optional

from zenova.orchestration.engine import ZenovaOrchestrationEngine
from zenova.schemas.standard import UserInput, ModalityType, SafetyAction
from zenova.evaluation.schemas import SystemPerformanceReport
from zenova.core.logging import get_logger

logger = get_logger("zenova.evaluation.system")


class SystemPerformanceEvaluator:
    """Measures pipeline latency quantiles, reliability, throughput, and modality robustness."""

    def __init__(self, orchestrator_engine: Optional[ZenovaOrchestrationEngine] = None):
        self.engine = orchestrator_engine or ZenovaOrchestrationEngine()

    async def run_benchmark(self, num_turns: int = 15) -> SystemPerformanceReport:
        """Execute empirical benchmark measuring end-to-end system metrics."""
        logger.info(f"Running system performance benchmark across {num_turns} conversational turns...")
        latencies: List[float] = []
        failures = 0
        safety_actions: Dict[str, int] = {"allow": 0, "revise": 0, "block_and_escalate": 0}
        total_tokens = 0

        start_benchmark_mono = time.time()

        benchmark_inputs = [
            "I had a busy day at work and feeling a little drained.",
            "Could you explain some simple breathing techniques for stress?",
            "I am feeling completely overwhelmed by my college exams.",
            "My friend said something hurtful and I can't stop thinking about it.",
            "I felt peaceful after taking a walk in the park earlier today.",
            "I cannot focus on anything and my sleep has been terrible lately.",
            "I am trying to stay hopeful but it's hard when everything piles up.",
            "Thank you for being here, it helps to talk through this.",
            "I am dealing with a lot of pressure from my family right now.",
            "Sometimes I just want to stay in bed all day and shut the world out."
        ]

        # 1. Measure Latencies and Reliability
        for i in range(num_turns):
            text_prompt = benchmark_inputs[i % len(benchmark_inputs)]
            sess_id = f"bench_sess_{uuid.uuid4().hex[:8]}"
            user_in = UserInput(
                session_id=sess_id,
                user_id=f"bench_user_{i}",
                text=text_prompt
            )

            turn_start = time.time()
            try:
                res = await self.engine.process_turn(user_in)
                turn_dur = (time.time() - turn_start) * 1000.0
                latencies.append(turn_dur)

                # Track safety action
                saf_dict = res.get("safety")
                if saf_dict and "action" in saf_dict:
                    act = str(saf_dict["action"]).lower()
                    if "block" in act:
                        safety_actions["block_and_escalate"] += 1
                    elif "revise" in act:
                        safety_actions["revise"] += 1
                    else:
                        safety_actions["allow"] += 1
                else:
                    safety_actions["allow"] += 1

                # Token count estimate (approx 1.3 tokens per word)
                words = len(text_prompt.split()) + len(res.get("response", "").split())
                total_tokens += int(words * 1.3)

            except Exception as e:
                logger.error(f"Benchmark turn {i} failed: {e}")
                failures += 1

        total_benchmark_time = max(0.001, time.time() - start_benchmark_mono)
        throughput = round(len(latencies) / total_benchmark_time, 2)
        failure_rate = round(failures / max(1, num_turns), 4)
        api_reliability = round((1.0 - failure_rate) * 100.0, 2)

        # Latency statistics
        arr = np.array(latencies) if latencies else np.array([50.0])
        p50 = round(float(np.percentile(arr, 50)), 2)
        p90 = round(float(np.percentile(arr, 90)), 2)
        p95 = round(float(np.percentile(arr, 95)), 2)
        p99 = round(float(np.percentile(arr, 99)), 2)
        mean_l = round(float(np.mean(arr)), 2)
        min_l = round(float(np.min(arr)), 2)
        max_l = round(float(np.max(arr)), 2)

        # Cost estimation: $0.002 per 1K input tokens, $0.005 per 1K output tokens (~$0.0035/1k avg)
        cost_per_1k_turns = round((total_tokens / max(1, len(latencies))) * 1000 * 0.0035 / 1000.0, 4)

        # 2. Missing Modality Handling Robustness
        modality_regimes = {
            "text_only": UserInput(session_id=f"mod_t_{uuid.uuid4().hex[:6]}", user_id="u_mod", text="Just text input"),
            "text_voice": UserInput(
                session_id=f"mod_tv_{uuid.uuid4().hex[:6]}",
                user_id="u_mod",
                text="Text with acoustic features",
                audio_features={"f0_mean": 210.5, "jitter": 0.02}
            ),
            "text_behavior": UserInput(
                session_id=f"mod_tb_{uuid.uuid4().hex[:6]}",
                user_id="u_mod",
                text="Text with behavioral signals",
                metadata={"behavior_metrics": {"daily_steps_delta": -1200.0, "screen_time_delta_mins": 85.0}}
            ),
            "text_voice_behavior": UserInput(
                session_id=f"mod_all_{uuid.uuid4().hex[:6]}",
                user_id="u_mod",
                text="Full multimodal signals",
                audio_features={"f0_mean": 195.0, "jitter": 0.015},
                metadata={"behavior_metrics": {"daily_steps_delta": -950.0}}
            )
        }

        modality_stability: Dict[str, float] = {}
        for regime, test_input in modality_regimes.items():
            try:
                mod_res = await self.engine.process_turn(test_input)
                modality_stability[regime] = 1.0 if mod_res.get("response") else 0.0
            except Exception as mod_err:
                logger.warning(f"Modality regime '{regime}' error: {mod_err}")
                modality_stability[regime] = 0.0

        # Safety-gate statistics
        total_actions = sum(safety_actions.values()) or 1
        safety_rates = {
            "allow_rate": round(safety_actions["allow"] / total_actions, 4),
            "revise_rate": round(safety_actions["revise"] / total_actions, 4),
            "block_and_escalate_rate": round(safety_actions["block_and_escalate"] / total_actions, 4)
        }

        return SystemPerformanceReport(
            total_turns_measured=len(latencies),
            latency_p50_ms=p50,
            latency_p90_ms=p90,
            latency_p95_ms=p95,
            latency_p99_ms=p99,
            latency_mean_ms=mean_l,
            latency_min_ms=min_l,
            latency_max_ms=max_l,
            throughput_qps=throughput,
            failure_rate=failure_rate,
            api_reliability_pct=api_reliability,
            estimated_cost_per_1k_turns_usd=cost_per_1k_turns,
            missing_modality_stability=modality_stability,
            safety_interception_rate=safety_rates
        )
