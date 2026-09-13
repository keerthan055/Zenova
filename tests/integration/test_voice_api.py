"""Integration tests for Voice API and end-to-end multimodal conversation orchestration."""
import io
import base64
import pytest
import numpy as np
import soundfile as sf
from httpx import AsyncClient, ASGITransport
from zenova.api.app import app


def create_synthetic_wav_base64(
    duration_sec: float = 1.0,
    freq_hz: float = 240.0,
    sr: int = 16000,
    amplitude: float = 0.5,
) -> str:
    """Generate base64 encoded string of synthetic WAV audio."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    waveform = (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, waveform, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@pytest.mark.asyncio
async def test_voice_analyze_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        b64_audio = create_synthetic_wav_base64(duration_sec=1.5, freq_hz=210.0)
        payload = {
            "user_id": "usr_voice_test_01",
            "session_id": "sess_voice_01",
            "audio_base64": b64_audio,
            "transcription_hint": "I am feeling somewhat stressed today.",
        }
        resp = await ac.post("/api/v1/voice/analyze", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data["is_available"] is True
        assert data["primary_emotion"] is not None
        assert 0.0 <= data["confidence"] <= 1.0
        assert data["transcription"] == "I am feeling somewhat stressed today."
        assert "Acoustic emotion signals are observational vocal affect proxies" in data["disclaimer"]
        assert "pitch_mean_hz" in data["acoustic_features"]


@pytest.mark.asyncio
async def test_voice_extract_features_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        b64_audio = create_synthetic_wav_base64(duration_sec=1.0, freq_hz=180.0)
        payload = {
            "user_id": "usr_feat_test",
            "audio_base64": b64_audio,
        }
        resp = await ac.post("/api/v1/voice/extract-features", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        assert "pitch_mean_hz" in data
        assert "rms_energy_mean" in data
        assert "spectral_centroid_mean" in data
        assert len(data["mfcc_means"]) == 13


@pytest.mark.asyncio
async def test_voice_transcribe_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        b64_audio = create_synthetic_wav_base64(duration_sec=1.2, freq_hz=250.0)
        payload = {
            "user_id": "usr_transcribe_test",
            "audio_base64": b64_audio,
            "transcription_hint": "Testing the speech to text transcription.",
        }
        resp = await ac.post("/api/v1/voice/transcribe", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data["transcription"] == "Testing the speech to text transcription."
        assert data["confidence"] >= 0.90
        assert data["duration_seconds"] == pytest.approx(1.2, abs=0.1)


@pytest.mark.asyncio
async def test_end_to_end_voice_turn_endpoint():
    """Verify POST /api/v1/voice/turn processes audio end-to-end through ASR and multimodal orchestrator."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        b64_audio = create_synthetic_wav_base64(duration_sec=1.8, freq_hz=230.0)
        payload = {
            "user_id": "usr_turn_voice_01",
            "session_id": "sess_turn_voice_01",
            "audio_base64": b64_audio,
            "transcription_hint": "I have been feeling really exhausted and sad lately.",
        }
        resp = await ac.post("/api/v1/voice/turn", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data["session_id"] == "sess_turn_voice_01"
        assert data["response"] is not None
        assert data["user_input"] == "I have been feeling really exhausted and sad lately."
        assert "voice" in data
        assert data["voice"] is not None
        assert data["voice"]["is_available"] is True
        assert data["voice"]["transcription"] == "I have been feeling really exhausted and sad lately."


@pytest.mark.asyncio
async def test_conversation_turn_with_audio_metadata():
    """Verify POST /api/v1/conversation/turn with audio_base64 metadata blends text and voice."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        b64_audio = create_synthetic_wav_base64(duration_sec=1.5, freq_hz=200.0)
        turn_payload = {
            "session_id": "sess_multi_01",
            "user_id": "usr_multi_01",
            "text": "...",  # Empty placeholder text, should be populated by ASR
            "modality": "voice",
            "metadata": {
                "audio_base64": b64_audio,
                "transcription_hint": "I am working through some difficult feelings today.",
            }
        }
        resp = await ac.post("/api/v1/conversation/turn", json=turn_payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data["user_input"] == "I am working through some difficult feelings today."
        assert data["voice"] is not None
        assert data["voice"]["is_available"] is True
        assert data["emotion"] is not None


@pytest.mark.asyncio
async def test_voice_history_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        user_id = "usr_hist_test"
        b64_audio = create_synthetic_wav_base64(duration_sec=1.0)
        payload = {
            "user_id": user_id,
            "audio_base64": b64_audio,
            "transcription_hint": "History test utterance",
        }
        await ac.post("/api/v1/voice/analyze", json=payload)

        # Retrieve history
        resp = await ac.get(f"/api/v1/voice/{user_id}/history")
        assert resp.status_code == 200
        hist = resp.json()
        assert hist["user_id"] == user_id
        assert hist["observations_count"] >= 1
