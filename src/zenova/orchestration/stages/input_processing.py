"""Input Preprocessing and Voice Transcription Stage."""
import uuid
from typing import Optional

from zenova.orchestration.stages.base import BasePipelineStage, PipelineStageContext
from zenova.schemas.standard import VoiceResult
from zenova.safety.adversarial import PromptInjectionDetector
from zenova.core.logging import get_logger

logger = get_logger("zenova.orchestration.stages.input_processing")


class InputProcessingStage(BasePipelineStage):
    """Sanitizes user input, validates identifiers, defends against prompt injection, and handles acoustic transcription."""

    def __init__(self):
        super().__init__()
        self.injection_detector = PromptInjectionDetector()

    @property
    def stage_name(self) -> str:
        return "input_processing"

    async def execute(self, ctx: PipelineStageContext) -> None:
        async with ctx.tracer.record_span(
            span_name=self.stage_name,
            input_summary={
                "session_id": ctx.user_input.session_id,
                "user_id": ctx.user_input.user_id,
                "text_length": len(ctx.user_input.text) if ctx.user_input.text else 0,
                "has_audio": bool(getattr(ctx.user_input, "audio_base64", None) or getattr(ctx.user_input, "audio_features", None) or (ctx.user_input.metadata and "audio" in ctx.user_input.metadata))
            }
        ) as span:
            # 1. Validate identifiers
            if not ctx.user_input.session_id:
                ctx.user_input.session_id = f"sess_{uuid.uuid4().hex[:12]}"
            if not ctx.user_input.user_id:
                ctx.user_input.user_id = f"usr_{uuid.uuid4().hex[:12]}"

            # 2. Voice Branch Analysis (if voice analyzer is present)
            voice_analyzer = None
            try:
                voice_analyzer = ctx.registry.get_module_instance("voice")
            except Exception as e:
                logger.debug(f"Voice analyzer not registered or unavailable: {e}")

            if voice_analyzer:
                try:
                    ctx.voice_res = voice_analyzer.analyze(ctx.user_input, ctx.context)
                    span.metadata["voice_analyzed"] = True
                except Exception as v_err:
                    logger.warning(f"Voice analysis failed or skipped: {v_err}")
                    ctx.tracer.mark_degraded("voice", reason=str(v_err))

            # 3. Audio Transcription Fallback
            # If user input text was empty/placeholder and voice transcription was produced, use transcribed text!
            raw_text = ctx.user_input.text.strip() if ctx.user_input.text else ""
            if (not raw_text or raw_text in ("", "...")) and ctx.voice_res and ctx.voice_res.transcription:
                ctx.user_input.text = ctx.voice_res.transcription
                logger.info(f"Populated user_input.text from voice transcription: '{ctx.user_input.text}'")
                span.metadata["transcription_applied"] = True

            # Ensure text is not None
            if not ctx.user_input.text:
                ctx.user_input.text = ""

            # 4. Adversarial Prompt Injection Detection & Sanitization
            is_inj, threat_codes, explanation = self.injection_detector.detect(ctx.user_input.text)
            if is_inj:
                span.metadata["prompt_injection_detected"] = True
                span.metadata["threat_codes"] = threat_codes
                logger.warning(f"Adversarial prompt injection sanitized for session {ctx.user_input.session_id}: {threat_codes}")
                ctx.user_input.text = self.injection_detector.sanitize(ctx.user_input.text)

            span.output_summary = {
                "sanitized_text_length": len(ctx.user_input.text),
                "voice_available": ctx.voice_res.is_available if ctx.voice_res else False,
                "prompt_injection_detected": is_inj
            }
