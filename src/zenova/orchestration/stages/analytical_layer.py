"""Analytical Extraction and Multimodal Fusion Layer Stage."""
from typing import Optional, Dict, Any

from zenova.orchestration.stages.base import BasePipelineStage, PipelineStageContext
from zenova.orchestration.fallbacks import (
    get_degraded_emotion_result,
    get_degraded_symptom_result,
    get_safe_fallback_risk_result
)
from zenova.schemas.standard import (
    EmotionResult,
    SymptomResult,
    RiskResult,
    BaselineResult,
    BehavioralResult
)
from zenova.schemas.fusion import FusedMultimodalState
from zenova.core.logging import get_logger

logger = get_logger("zenova.orchestration.stages.analytical_layer")


class AnalyticalLayerStage(BasePipelineStage):
    """Executes parallel analytical extraction across emotion, symptoms, risk, baseline, and fusion."""

    @property
    def stage_name(self) -> str:
        return "analytical_layer"

    async def execute(self, ctx: PipelineStageContext) -> None:
        async with ctx.tracer.record_span(span_name=self.stage_name) as span:
            # 1. Fetch module instances safely
            emotion_analyzer = self._safe_get_module(ctx, "emotion")
            symptom_analyzer = self._safe_get_module(ctx, "symptom")
            risk_analyzer = self._safe_get_module(ctx, "risk")
            baseline_engine = self._safe_get_module(ctx, "baseline")
            behavior_analyzer = self._safe_get_module(ctx, "behavior")
            fusion_engine = self._safe_get_module(ctx, "fusion")

            # 2. Emotion Analysis
            async with ctx.tracer.record_span(
                span_name="emotion_analysis",
                module_name="emotion",
                swallow_exception=True,
                is_degraded_on_error=True
            ) as em_span:
                if emotion_analyzer:
                    try:
                        ctx.emotion_res = emotion_analyzer.analyze(ctx.user_input, ctx.context)
                        em_span.output_summary = {
                            "primary_emotion": ctx.emotion_res.primary_emotion.value,
                            "confidence": ctx.emotion_res.confidence
                        }
                    except Exception as e:
                        logger.warning(f"Emotion analysis exception: {e}")
                        ctx.emotion_res = get_degraded_emotion_result(ctx.user_input, str(e))
                        ctx.tracer.mark_degraded("emotion", reason=str(e))
                else:
                    ctx.emotion_res = get_degraded_emotion_result(ctx.user_input, "Module not registered")
                    ctx.tracer.mark_degraded("emotion", reason="Module not registered")

            # 3. Symptom Analysis
            async with ctx.tracer.record_span(
                span_name="symptom_analysis",
                module_name="symptom",
                swallow_exception=True,
                is_degraded_on_error=True
            ) as sy_span:
                if symptom_analyzer:
                    try:
                        ctx.symptom_res = symptom_analyzer.analyze(ctx.user_input, ctx.context)
                        sy_span.output_summary = {
                            "aggregate_severity": ctx.symptom_res.aggregate_severity.value if hasattr(ctx.symptom_res.aggregate_severity, "value") else str(ctx.symptom_res.aggregate_severity),
                            "signals_count": len(ctx.symptom_res.signals)
                        }
                    except Exception as e:
                        logger.warning(f"Symptom analysis exception: {e}")
                        ctx.symptom_res = get_degraded_symptom_result(ctx.user_input, str(e))
                        ctx.tracer.mark_degraded("symptom", reason=str(e))
                else:
                    ctx.symptom_res = get_degraded_symptom_result(ctx.user_input, "Module not registered")
                    ctx.tracer.mark_degraded("symptom", reason="Module not registered")

            # 4. Risk Analysis (Safety-Critical Conservative Posture)
            async with ctx.tracer.record_span(
                span_name="risk_analysis",
                module_name="risk",
                swallow_exception=True,
                is_degraded_on_error=True
            ) as rk_span:
                if risk_analyzer:
                    try:
                        ctx.risk_res = risk_analyzer.analyze(ctx.user_input, ctx.context)
                        rk_span.output_summary = {
                            "risk_level": ctx.risk_res.risk_level.value,
                            "is_high_risk": ctx.risk_res.is_high_risk
                        }
                    except Exception as e:
                        logger.warning(f"Risk analysis exception: {e}")
                        ctx.risk_res = get_safe_fallback_risk_result(ctx.user_input, str(e))
                        ctx.tracer.mark_degraded("risk", reason=str(e))
                else:
                    ctx.risk_res = get_safe_fallback_risk_result(ctx.user_input, "Module not registered")
                    ctx.tracer.mark_degraded("risk", reason="Module not registered")

            # 5. Behavioral Analysis
            if behavior_analyzer:
                try:
                    ctx.behavior_res = behavior_analyzer.analyze(ctx.user_input, ctx.context)
                except Exception as b_err:
                    logger.warning(f"Behavioral analysis failed or skipped: {b_err}")
                    ctx.tracer.mark_degraded("behavior", reason=str(b_err))

            # 6. Multimodal Features for Baseline Evaluation
            extra_features: Dict[str, Any] = {}
            if ctx.behavior_res and ctx.behavior_res.is_available and ctx.behavior_res.metrics:
                extra_features.update({f"behavior_{k}": v for k, v in ctx.behavior_res.metrics.items()})
            if ctx.voice_res and ctx.voice_res.is_available and ctx.voice_res.acoustic_features:
                extra_features.update({f"voice_{k}": v for k, v in ctx.voice_res.acoustic_features.items()})
            extra_dict = extra_features if extra_features else None

            # Baseline Evaluation
            if baseline_engine:
                try:
                    ctx.baseline_res = baseline_engine.evaluate(
                        user_input=ctx.user_input,
                        emotion=ctx.emotion_res,
                        context=ctx.context,
                        symptom=ctx.symptom_res,
                        risk=ctx.risk_res,
                        extra_features=extra_dict
                    )
                except Exception as base_err:
                    logger.warning(f"Baseline evaluation failed: {base_err}")
                    ctx.tracer.mark_degraded("baseline", reason=str(base_err))

            # 7. Fetch turn history to determine turn_idx
            if ctx.turn_repo:
                try:
                    ctx.history = await ctx.turn_repo.get_history(ctx.user_input.session_id)
                    ctx.turn_idx = len(ctx.history) + 1
                except Exception as h_err:
                    logger.warning(f"Could not load turn history from database: {h_err}")
                    ctx.turn_idx = 1
            else:
                ctx.turn_idx = 1

            # 8. Multimodal Fusion Engine
            if fusion_engine:
                try:
                    ctx.fused_state = fusion_engine.fuse(
                        user_input=ctx.user_input,
                        emotion=ctx.emotion_res,
                        symptoms=ctx.symptom_res,
                        risk=ctx.risk_res,
                        voice=ctx.voice_res,
                        behavior=ctx.behavior_res,
                        baseline=ctx.baseline_res,
                        history=ctx.history
                    )
                    # Calibrate affect with fused values
                    if ctx.emotion_res and ctx.fused_state:
                        ctx.emotion_res.primary_emotion = ctx.fused_state.primary_affect
                        ctx.emotion_res.valence = ctx.fused_state.fused_valence
                        ctx.emotion_res.arousal = ctx.fused_state.fused_arousal
                        ctx.emotion_res.dominance = ctx.fused_state.fused_dominance
                        if ctx.fused_state.discrepancy.detected:
                            logger.info(
                                f"Multimodal cross-modal discrepancy detected: "
                                f"{ctx.fused_state.discrepancy.discrepancy_types}"
                            )
                except Exception as f_err:
                    logger.warning(f"Multimodal fusion skipped or failed: {f_err}")
                    ctx.tracer.mark_degraded("fusion", reason=str(f_err))

            span.output_summary = {
                "emotion": ctx.emotion_res.primary_emotion.value if ctx.emotion_res else None,
                "risk_level": ctx.risk_res.risk_level.value if ctx.risk_res else None,
                "is_high_risk": ctx.risk_res.is_high_risk if ctx.risk_res else False,
                "turn_id": ctx.turn_idx
            }

    def _safe_get_module(self, ctx: PipelineStageContext, module_name: str) -> Optional[Any]:
        try:
            return ctx.registry.get_module_instance(module_name)
        except Exception:
            return None
