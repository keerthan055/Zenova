"""Performance, latency, and concurrency tests for ZENOVA complete pipeline."""
import time
import uuid
import asyncio
import numpy as np
import pytest

from zenova.core.orchestrator import ZenovaOrchestrator
from zenova.schemas.standard import UserInput
from zenova.db.session import init_db


@pytest.mark.asyncio
async def test_end_to_end_latency_quantiles():
    """Benchmark end-to-end turn processing latency distribution (p50, p90, p99)."""
    await init_db()
    orchestrator = ZenovaOrchestrator()
    latencies = []

    test_inputs = [
        "I feel worried about my future career prospects.",
        "Can we talk about feeling lonely when moving to a new city?",
        "I need some guidance on managing social anxiety.",
        "Today was a productive day and I want to celebrate small wins.",
        "I'm feeling mentally exhausted from staring at screens all day."
    ]

    for text in test_inputs:
        uid = f"perf_user_{uuid.uuid4().hex[:6]}"
        sid = f"perf_sess_{uuid.uuid4().hex[:6]}"
        u_in = UserInput(session_id=sid, user_id=uid, turn_id=1, text=text)

        t0 = time.perf_counter()
        result = await orchestrator.process_turn(u_in)
        duration_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(duration_ms)
        assert len(result["response"]) > 0

    p50 = float(np.percentile(latencies, 50))
    p90 = float(np.percentile(latencies, 90))
    p99 = float(np.percentile(latencies, 99))
    mean_lat = float(np.mean(latencies))

    # Assert performance targets
    assert p50 < 400.0, f"p50 latency too high: {p50:.1f}ms"
    assert mean_lat < 500.0, f"Mean latency too high: {mean_lat:.1f}ms"
    assert len(latencies) == len(test_inputs)


@pytest.mark.asyncio
async def test_concurrent_sessions_throughput():
    """Verify concurrent multi-session turn handling without race conditions."""
    await init_db()
    orchestrator = ZenovaOrchestrator()
    num_concurrent = 6

    async def run_single_turn(idx: int):
        uid = f"concurrent_user_{idx}_{uuid.uuid4().hex[:6]}"
        sid = f"concurrent_sess_{idx}_{uuid.uuid4().hex[:6]}"
        u_in = UserInput(
            session_id=sid,
            user_id=uid,
            turn_id=1,
            text=f"This is concurrent message {idx} asking for emotional validation."
        )
        t_start = time.perf_counter()
        res = await orchestrator.process_turn(u_in)
        dur = (time.perf_counter() - t_start) * 1000.0
        return res, dur

    tasks = [run_single_turn(i) for i in range(num_concurrent)]
    results = await asyncio.gather(*tasks)

    assert len(results) == num_concurrent
    for res, dur in results:
        assert res["session_id"] is not None
        assert len(res["response"]) > 0
        assert res["safety"]["is_safe"] is True


@pytest.mark.asyncio
async def test_rapid_multi_turn_session_stability():
    """Verify turn counter increments and stability under rapid sequential turns."""
    await init_db()
    orchestrator = ZenovaOrchestrator()
    uid = f"rapid_user_{uuid.uuid4().hex[:6]}"
    sid = f"rapid_sess_{uuid.uuid4().hex[:6]}"

    for turn_num in range(1, 4):
        u_in = UserInput(
            session_id=sid,
            user_id=uid,
            turn_id=turn_num,
            text=f"Turn {turn_num}: I am sharing another thought with you."
        )
        res = await orchestrator.process_turn(u_in)
        assert res["turn_id"] >= turn_num
        assert len(res["response"]) > 0
