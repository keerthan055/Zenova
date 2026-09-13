"""Voice/acoustic emotion analysis, ASR transcription, and speech turn processing routes."""
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from zenova.schemas.standard import UserInput, VoiceResult, ModalityType
from zenova.schemas.voice import AudioPayload, AcousticFeatures, VoiceEmotionResult
from zenova.voice.analyzer import VoiceEmotionAnalyzer
from zenova.core.orchestrator import ZenovaOrchestrator
from zenova.models.registry import ModelRegistry
from zenova.db.session import get_db_session
from zenova.db.repositories import VoiceRepository
from zenova.core.logging import get_logger

logger = get_logger("zenova.api.routes.voice")
router = APIRouter(prefix="/api/v1/voice", tags=["Voice & Acoustic Emotion"])

# Singleton analyzer instance
_voice_analyzer: Optional[VoiceEmotionAnalyzer] = None


def get_voice_analyzer() -> VoiceEmotionAnalyzer:
    global _voice_analyzer
    if _voice_analyzer is None:
        try:
            registry = ModelRegistry()
            analyzer = registry.get_module_instance("voice")
            if isinstance(analyzer, VoiceEmotionAnalyzer):
                _voice_analyzer = analyzer
            else:
                _voice_analyzer = VoiceEmotionAnalyzer()
        except Exception:
            _voice_analyzer = VoiceEmotionAnalyzer()
    return _voice_analyzer


@router.post("/analyze", response_model=VoiceEmotionResult)
async def analyze_voice(
    payload: AudioPayload,
    analyzer: VoiceEmotionAnalyzer = Depends(get_voice_analyzer),
):
    """Analyze inbound audio payload for acoustic prosody and vocal affect.
    
    Observational disclaimer attached. Enforces non-diagnostic vocal affect boundary.
    """
    try:
        user_input = UserInput(
            session_id=payload.session_id or f"voice_sess_{payload.user_id}",
            user_id=payload.user_id,
            text=payload.transcription_hint or "...",
            modality=ModalityType.VOICE,
            metadata={
                "audio_base64": payload.audio_base64,
                "sampling_rate": payload.sampling_rate,
                "transcription_hint": payload.transcription_hint,
                "acoustic_features": payload.precomputed_features,
            }
        )
        res = analyzer.analyze(user_input)

        # Persist observation in background / DB
        try:
            async with get_db_session() as db:
                repo = VoiceRepository(db)
                await repo.record_observation(
                    user_id=payload.user_id,
                    session_id=payload.session_id,
                    duration_seconds=res.duration_seconds or 0.0,
                    transcription=res.transcription,
                    emotion=res.primary_emotion.value if res.primary_emotion else None,
                    confidence=res.confidence,
                    features_dict=res.acoustic_features,
                )
        except Exception as db_err:
            logger.warning(f"Voice DB persistence skipped: {db_err}")

        return res
    except Exception as e:
        logger.error(f"Voice analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Voice analysis failed: {str(e)}")


@router.post("/extract-features", response_model=AcousticFeatures)
async def extract_acoustic_features(
    payload: AudioPayload,
    analyzer: VoiceEmotionAnalyzer = Depends(get_voice_analyzer),
):
    """Extract 25 standardized prosodic, spectral, and cepstral features from audio without classification."""
    try:
        if payload.precomputed_features:
            return AcousticFeatures(**payload.precomputed_features)

        if not payload.audio_base64:
            raise HTTPException(status_code=400, detail="audio_base64 string is required for feature extraction.")

        waveform, sr = analyzer.validator.process_raw_or_base64(audio_base64=payload.audio_base64)
        return analyzer.extractor.extract(waveform, sr)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Acoustic feature extraction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")


@router.post("/transcribe")
async def transcribe_audio(
    payload: AudioPayload,
    analyzer: VoiceEmotionAnalyzer = Depends(get_voice_analyzer),
):
    """Transcribe speech audio waveform into text using the ASR pipeline."""
    try:
        if not payload.audio_base64:
            raise HTTPException(status_code=400, detail="audio_base64 string is required for transcription.")

        waveform, sr = analyzer.validator.process_raw_or_base64(audio_base64=payload.audio_base64)
        text, conf = analyzer.asr_engine.transcribe(
            waveform, sr, transcription_hint=payload.transcription_hint
        )
        return {
            "user_id": payload.user_id,
            "transcription": text,
            "confidence": conf,
            "duration_seconds": round(len(waveform) / float(sr), 2),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ASR transcription failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@router.post("/turn", response_model=Dict[str, Any])
async def process_voice_turn(payload: AudioPayload):
    """Process an end-to-end voice conversational turn:
    
    Audio -> ASR -> Text Analysis (Emotion, Symptoms, Risk) ->
    Acoustic Emotion -> Multimodal Fusion -> Strategy -> Generation -> Safety Gate.
    """
    try:
        orchestrator = ZenovaOrchestrator()
        user_input = UserInput(
            session_id=payload.session_id or f"voice_turn_{payload.user_id}",
            user_id=payload.user_id,
            text=payload.transcription_hint or "...",
            modality=ModalityType.VOICE,
            metadata={
                "audio_base64": payload.audio_base64,
                "sampling_rate": payload.sampling_rate,
                "transcription_hint": payload.transcription_hint,
                "acoustic_features": payload.precomputed_features,
            }
        )
        turn_result = await orchestrator.process_turn(user_input)
        return turn_result
    except Exception as e:
        logger.error(f"Voice turn processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Voice turn processing failed: {str(e)}")


@router.get("/{user_id}/history", response_model=Dict[str, Any])
async def get_voice_history(user_id: str):
    """Retrieve stored voice interaction observations for a user."""
    try:
        async with get_db_session() as db:
            repo = VoiceRepository(db)
            obs_list = await repo.get_recent_observations(user_id)
            return {
                "user_id": user_id,
                "observations_count": len(obs_list),
                "observations": [
                    {
                        "id": o.id,
                        "session_id": o.session_id,
                        "duration_seconds": o.duration_seconds,
                        "transcription": o.transcription,
                        "emotion": o.emotion,
                        "confidence": o.confidence,
                        "created_at": o.created_at.isoformat() if o.created_at else None,
                    }
                    for o in obs_list
                ]
            }
    except Exception as e:
        logger.error(f"Failed to fetch voice history for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch voice history: {str(e)}")
