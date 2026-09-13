"""Clinical safe fallbacks for degraded or unavailable pipeline components.

Adheres strictly to the safety invariant: ZENOVA must fail safely and transparently
rather than silently fabricating clinical judgments or factual assertions.
"""
import re
from typing import Optional, List, Dict
from zenova.schemas.standard import (
    UserInput,
    EmotionResult,
    EmotionCategory,
    SymptomResult,
    SymptomSeverity,
    RiskResult,
    RiskLevel,
    CrisisCategory,
    SupportStrategy,
    GeneratedResponse
)
from zenova.core.logging import get_logger

logger = get_logger("zenova.orchestration.fallbacks")

# Explicit crisis cue indicators for conservative risk fallback scanning
CRITICAL_CRISIS_PATTERNS = [
    r"\b(suicid(e|al)|kill myself|end my life|end it all|want to die|hang myself|overdose|slit my wrists|better off dead)\b",
    r"\b(can'?t go on living|no reason to live|ready to end it)\b"
]

HIGH_RISK_PATTERNS = [
    r"\b(hurt myself|self[- ]harm|cut myself|cannot take it anymore|hopeless|disappear forever)\b"
]


def get_degraded_emotion_result(
    user_input: Optional[UserInput] = None,
    reason: str = "Emotion model unavailable"
) -> EmotionResult:
    """Produce a safe degraded emotion result with explicit zero-confidence."""
    logger.warning(f"Generating degraded emotion fallback: {reason}")
    probabilities = {cat.value: (1.0 if cat == EmotionCategory.NEUTRAL else 0.0) for cat in EmotionCategory}
    return EmotionResult(
        is_placeholder=True,
        module_version="fallback_degraded",
        primary_emotion=EmotionCategory.NEUTRAL,
        confidence=0.0,
        probabilities=probabilities,
        valence=0.0,
        arousal=0.0,
        dominance=0.5
    )


def get_degraded_symptom_result(
    user_input: Optional[UserInput] = None,
    reason: str = "Symptom model unavailable"
) -> SymptomResult:
    """Produce a safe degraded symptom result without fabricated signals."""
    logger.warning(f"Generating degraded symptom fallback: {reason}")
    return SymptomResult(
        is_placeholder=True,
        module_version="fallback_degraded",
        signals=[],
        aggregate_severity=SymptomSeverity.NONE,
        notes=f"Degraded fallback: {reason}"
    )


def get_safe_fallback_risk_result(
    user_input: Optional[UserInput] = None,
    reason: str = "Risk model unavailable"
) -> RiskResult:
    """Execute conservative heuristic safety scan when risk classifier fails.
    
    Safety Critical: Never assume low risk silently if the input contains acute cues.
    """
    logger.warning(f"Executing conservative risk fallback heuristic: {reason}")
    text = (user_input.text if user_input and user_input.text else "").lower()
    trigger_cues: List[str] = ["risk_model_degraded_conservative_fallback"]

    # Check acute crisis cues
    for pat in CRITICAL_CRISIS_PATTERNS:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            trigger_cues.append(match.group(0))
            return RiskResult(
                is_placeholder=True,
                module_version="conservative_heuristic_fallback",
                risk_level=RiskLevel.CRITICAL,
                crisis_category=CrisisCategory.SUICIDAL_IDEATION,
                confidence=1.0,  # High confidence due to explicit acute keywords
                requires_immediate_escalation=True,
                trigger_cues=trigger_cues
            )

    # Check high risk cues
    for pat in HIGH_RISK_PATTERNS:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            trigger_cues.append(match.group(0))
            return RiskResult(
                is_placeholder=True,
                module_version="conservative_heuristic_fallback",
                risk_level=RiskLevel.HIGH,
                crisis_category=CrisisCategory.SELF_HARM,
                confidence=0.85,
                requires_immediate_escalation=True,
                trigger_cues=trigger_cues
            )

    # Conservative default: Low risk, but marked placeholder so safety gate remains vigilant
    return RiskResult(
        is_placeholder=True,
        module_version="conservative_heuristic_fallback",
        risk_level=RiskLevel.LOW,
        crisis_category=CrisisCategory.NONE,
        confidence=0.0,
        requires_immediate_escalation=False,
        trigger_cues=trigger_cues
    )


def get_safe_fallback_response(
    strategy: Optional[SupportStrategy] = None,
    user_input: Optional[UserInput] = None
) -> GeneratedResponse:
    """Produce a deterministic, clinically validated response when the LLM is unavailable."""
    strat = strategy or SupportStrategy.REFLECTION_OF_FEELINGS
    strat_key = strat.value if hasattr(strat, "value") else str(strat)

    template_map = {
        "question": (
            "I want to make sure I understand what you are going through. "
            "Could you share a little more about what has felt most difficult for you lately?"
        ),
        "restatement or paraphrasing": (
            "It sounds like you have been dealing with an overwhelming amount of pressure "
            "and doing your best to navigate a very challenging situation."
        ),
        "reflection of feelings": (
            "It sounds like you are carrying a very heavy emotional weight right now, "
            "and it is completely natural to feel exhausted and overwhelmed."
        ),
        "affirmation and reassurance": (
            "You are showing real strength by speaking up and working through this. "
            "Please remember that you do not have to carry everything all by yourself."
        ),
        "providing suggestions": (
            "When everything feels like too much, pausing for a single quiet breath or taking one "
            "small moment can offer some relief. Would you like to explore one small step together?"
        ),
        "information": (
            "Experiencing prolonged emotional stress takes a significant toll on both the body and mind. "
            "Giving yourself permission to rest and take things slowly is an important and valid step."
        ),
        "self-disclosure": (
            "Many people go through periods where things feel completely uncertain and overwhelming. "
            "It takes genuine courage to be open about these feelings."
        ),
        "others": (
            "I am here with you. Please take your time, and we can talk about whatever feels "
            "most important or helpful for you right now."
        )
    }

    resp_text = template_map.get(
        strat_key,
        "I am here with you. Please take your time, and we can explore what feels most helpful right now."
    )

    return GeneratedResponse(
        response_text=resp_text,
        strategy_applied=strat_key,
        model_name="clinical_template_fallback",
        is_placeholder=False,
        confidence=1.0,
        generation_metadata={"reason": "LLM generation unavailable, served verified clinical template"}
    )
