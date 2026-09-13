"""Structured prompt builder with explicit emotional-support strategy directives."""
from typing import List, Dict, Any, Optional
from zenova.schemas.standard import (
    UserInput,
    StrategyResult,
    SupportStrategy,
    DialogStage,
    ConversationContext,
    MultimodalContext
)
from zenova.strategy.taxonomy import StrategyTaxonomy


# Detailed behavioural directives for the 8 canonical ESConv strategies
STRATEGY_DIRECTIVES: Dict[str, Dict[str, str]] = {
    SupportStrategy.QUESTION.value: {
        "objective": "Ask an open, empathetic, and exploratory question that encourages the user to clarify, elaborate, or reflect deeper on their feelings and situation.",
        "dos": [
            "Ask at most 1 or 2 focused, gentle, open-ended questions.",
            "Show that your question builds directly on what they just expressed.",
            "Keep the inquiry collaborative and non-interrogative."
        ],
        "donts": [
            "DO NOT offer advice, solutions, or action items.",
            "DO NOT lecture, explain, or give psychoeducation.",
            "DO NOT ask closed rapid-fire yes/no questions."
        ]
    },
    SupportStrategy.RESTATEMENT_OR_PARAPHRASING.value: {
        "objective": "Restate or rephrase the core content and meaning of what the user just expressed in your own words, demonstrating active listening.",
        "dos": [
            "Capture the essence of their situation and thoughts accurately.",
            "Use warm, neutral phrasing (e.g., 'What I am hearing is...', 'It sounds like...').",
            "Leave room for the user to confirm or clarify your understanding."
        ],
        "donts": [
            "DO NOT offer unsolicited advice or problem-solving.",
            "DO NOT psychoanalyze or invent motives not stated by the user.",
            "DO NOT judge or evaluate their actions."
        ]
    },
    SupportStrategy.REFLECTION_OF_FEELINGS.value: {
        "objective": "Focus exclusively on identifying, naming, and validating the user's emotional state, feelings, and underlying affective distress.",
        "dos": [
            "Articulate the specific emotions they are experiencing (e.g., overwhelmed, exhausted, anxious, isolated).",
            "Validate that their emotional response is completely understandable given their circumstances.",
            "Maintain an empathetic, holding, non-judgmental presence."
        ],
        "donts": [
            "DO NOT offer coping advice, solutions, or actionable steps.",
            "DO NOT rush to fix, dismiss, or brighten the emotion.",
            "DO NOT ask analytical questions about why they feel that way."
        ]
    },
    SupportStrategy.AFFIRMATION_AND_REASSURANCE.value: {
        "objective": "Provide heartfelt emotional affirmation, validate their personal strengths, normalize their struggle, and offer gentle reassurance.",
        "dos": [
            "Acknowledge the courage, effort, or resilience it takes to navigate their situation.",
            "Reassure them that they are doing the best they can and are not broken.",
            "Normalize that many people struggle with similar difficulties."
        ],
        "donts": [
            "DO NOT use toxic positivity (e.g., 'Everything happens for a reason', 'Look on the bright side').",
            "DO NOT dismiss the gravity or pain of their experience.",
            "DO NOT promise unrealistic clinical outcomes."
        ]
    },
    SupportStrategy.SELF_DISCLOSURE.value: {
        "objective": "Share a relatable, universal human perspective or empathetic understanding to build therapeutic connection and alleviate feelings of isolation.",
        "dos": [
            "Communicate relatable common human experiences (e.g., 'It is so common to feel alone when facing this...').",
            "Keep the spotlight and emotional gravity firmly on the user.",
            "Use disclosure solely to foster rapport and mutual understanding."
        ],
        "donts": [
            "DO NOT make the conversation about yourself or fabricate a personal life history.",
            "DO NOT derail the user's narrative to discuss your own feelings.",
            "DO NOT compare your struggles to theirs in a minimizing manner."
        ]
    },
    SupportStrategy.PROVIDING_SUGGESTIONS.value: {
        "objective": "Collaboratively propose constructive, gentle, low-pressure coping strategies, self-care ideas, or actionable problem-solving steps.",
        "dos": [
            "Frame suggestions as gentle invitations (e.g., 'If you feel up to it, something that might help is...', 'Would it be possible to...').",
            "Offer small, practical, manageable micro-actions.",
            "Check in on whether the suggestion feels feasible or appealing to them."
        ],
        "donts": [
            "DO NOT issue mandatory orders or medical prescriptions.",
            "DO NOT overwhelm the user with a lengthy checklist of demands.",
            "DO NOT dismiss their emotional need for validation before suggesting action."
        ]
    },
    SupportStrategy.INFORMATION.value: {
        "objective": "Provide clear, objective, factual psychoeducational or wellbeing information relevant to their situation.",
        "dos": [
            "Explain common physiological or psychological patterns in plain, de-stigmatizing language.",
            "Reference general wellbeing principles (e.g., sleep cycles, stress response, grounding mechanisms).",
            "Cite supportive resources or helpline contact information when relevant."
        ],
        "donts": [
            "DO NOT diagnose psychiatric or medical disorders.",
            "DO NOT prescribe medical drugs or clinical interventions.",
            "DO NOT overwhelm with clinical jargon."
        ]
    },
    SupportStrategy.OTHERS.value: {
        "objective": "Provide warm conversational check-ins, empathetic transitions, supportive greetings, or respectful closings.",
        "dos": [
            "Express genuine presence, warmth, and readiness to listen whenever they are ready.",
            "Facilitate a smooth, natural conversational transition.",
            "Keep the tone gentle, welcoming, and supportive."
        ],
        "donts": [
            "DO NOT force deep therapeutic exploration if the user is making casual conversation.",
            "DO NOT abandon empathy or become robotic."
        ]
    }
}


SYSTEM_PROMPT = """You are ZENOVA, an empathetic, evidence-informed AI wellbeing support companion based on Hill's Helping Skills model.
Your purpose is to provide compassionate, non-judgmental conversational support that helps users explore their feelings, feel heard, and discover constructive ways forward.

CRITICAL CLINICAL & ETHICAL BOUNDARIES (STRICTLY ENFORCED):
1. NON-DIAGNOSTIC MANDATE: NEVER diagnose any medical, psychiatric, or psychological illness (e.g. never say "You have major depressive disorder" or "This is anxiety disorder").
2. NON-THERAPIST MANDATE: NEVER claim, imply, or suggest that you are a licensed therapist, clinical psychologist, psychiatrist, or medical doctor.
3. NON-PRESCRIPTIVE MANDATE: NEVER prescribe medication, clinical therapies, or tell the user to stop or alter existing medical treatments.
4. STRATEGY FIDELITY: You MUST strictly execute the assigned emotional-support strategy provided in the prompt. Do NOT switch to an unassigned strategy (e.g., do NOT give advice when assigned to reflect feelings).
5. CONCISE & EMPATHETIC: Keep your responses warm, conversational, and focused (typically 2 to 4 sentences). Avoid overly verbose or essay-like responses.
"""


class StrategyPromptBuilder:
    """Constructs structured, strategy-conditioned prompts for LLM response generation."""

    def __init__(self, taxonomy: Optional[StrategyTaxonomy] = None):
        self.taxonomy = taxonomy or StrategyTaxonomy.default()

    def build_prompt(
        self,
        user_input: UserInput,
        strategy: StrategyResult,
        context: Optional[ConversationContext] = None,
        rag_context: Optional[List[str]] = None,
        multimodal_context: Optional[MultimodalContext] = None
    ) -> Dict[str, str]:
        """Generate system prompt and structured user prompt containing all contextual signals."""
        strategy_label = (
            strategy.selected_strategy.value
            if hasattr(strategy.selected_strategy, "value")
            else str(strategy.selected_strategy)
        )
        directive_data = STRATEGY_DIRECTIVES.get(
            strategy_label,
            STRATEGY_DIRECTIVES[SupportStrategy.OTHERS.value]
        )

        lines = []

        # 1. Assigned Strategy Directives
        lines.append("=== ASSIGNED SUPPORT STRATEGY (MANDATORY) ===")
        lines.append(f"STRATEGY: {strategy_label}")
        lines.append(f"DIALOGUE STAGE: {strategy.stage.value if hasattr(strategy.stage, 'value') else strategy.stage}")
        lines.append(f"OBJECTIVE: {directive_data['objective']}")
        lines.append("DO:")
        for do_item in directive_data["dos"]:
            lines.append(f"  • {do_item}")
        lines.append("DO NOT:")
        for dont_item in directive_data["donts"]:
            lines.append(f"  • {dont_item}")
        if strategy.rationale:
            lines.append(f"CLINICAL RATIONALE: {strategy.rationale}")
        lines.append("")

        # 2. Multimodal Context Signals
        lines.append("=== CONTEXTUAL SIGNALS ===")
        if multimodal_context:
            if multimodal_context.emotion.is_available:
                lines.append(
                    f"• Inferred Emotion: {multimodal_context.emotion.primary_emotion} "
                    f"(Valence: {multimodal_context.emotion.valence:.2f}, Arousal: {multimodal_context.emotion.arousal:.2f})"
                )
                if multimodal_context.emotion.is_discrepancy_detected:
                    lines.append("• ACOUSTIC DISCREPANCY: User's voice tone reveals distress not apparent in words. Address deeper distress.")
            if multimodal_context.symptoms.is_available and multimodal_context.symptoms.signals:
                sigs = [s.get("label", "") for s in multimodal_context.symptoms.signals]
                lines.append(f"• Observed Behavioral Signals: {', '.join(sigs)} (Observational only; non-diagnostic)")
            if multimodal_context.baseline.is_available and multimodal_context.baseline.status != "insufficient_data":
                lines.append(f"• Personal Baseline: {multimodal_context.baseline.interpretation_summary}")
        lines.append("")

        # 3. Optional RAG Grounding
        if rag_context:
            lines.append("=== RELEVANT WELLBEING RESOURCES (RAG GROUNDING) ===")
            for item in rag_context:
                lines.append(f"• {item.strip()}")
            lines.append("")

        # 4. Conversation History
        lines.append("=== RECENT CONVERSATION HISTORY ===")
        if context and context.turns:
            for t in context.turns[-6:]:
                spk = "Seeker" if t.speaker in ("user", "seeker") else "Supporter (ZENOVA)"
                lines.append(f"{spk}: {t.content.strip()}")
        elif multimodal_context and multimodal_context.history.recent_turns:
            for t in multimodal_context.history.recent_turns[-6:]:
                spk = "Seeker" if t.get("speaker") in ("user", "seeker") else "Supporter (ZENOVA)"
                lines.append(f"{spk}: {t.get('content', '').strip()}")
        else:
            lines.append("(New conversation session)")
        lines.append("")

        # 5. Current User Message
        lines.append("=== CURRENT USER MESSAGE ===")
        lines.append(f"Seeker: {user_input.text.strip()}")
        lines.append("")
        lines.append(f"Respond strictly as Supporter using the '{strategy_label}' strategy:")

        user_prompt = "\n".join(lines)

        return {
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": user_prompt
        }
