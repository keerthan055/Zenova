"""C-SSRS clinical risk taxonomy and safety classifications for ZENOVA.

Grounded in:
- Columbia-Suicide Severity Rating Scale (C-SSRS)
- Gaur et al., WWW 2019: 'Knowledge-aware Assessment of Severity of Suicide Risk for Early Intervention'
- Zirikly et al., NAACL-HLT CLPsych 2019: 'Predicting the Degree of Suicide Risk in Reddit Posts'

CRITICAL SAFETY BOUNDARY:
Automated risk assessment heuristic; does not represent clinical certainty.
Any indication of imminent crisis must be routed to trained human emergency professionals.
"""
from typing import Dict, List, Any
from zenova.schemas.standard import RiskLevel, CrisisCategory

# 4-tier C-SSRS Risk Level Taxonomy
RISK_LEVELS = [
    RiskLevel.LOW.value,
    RiskLevel.MODERATE.value,
    RiskLevel.HIGH.value,
    RiskLevel.CRITICAL.value
]

# Crisis Categories
CRISIS_CATEGORIES = [
    CrisisCategory.NONE.value,
    CrisisCategory.SELF_HARM.value,
    CrisisCategory.SUICIDAL_IDEATION.value,
    CrisisCategory.IMMINENT_DANGER.value,
    CrisisCategory.VIOLENCE.value
]

# Taxonomy specifications for C-SSRS Risk Levels
RISK_TAXONOMY_SPEC = {
    RiskLevel.LOW.value: {
        "cssrs_level": "Indicator / No Ideation",
        "description": "General distress, sadness, academic/work stress, or everyday frustration without suicidal ideation or self-harm intent.",
        "requires_escalation": False,
        "action": "standard_supportive_dialogue",
        "evidence_cues": []
    },
    RiskLevel.MODERATE.value: {
        "cssrs_level": "Passive Ideation",
        "description": "Passive suicidal thoughts, vague death wishes, or feeling that others would be better off without them, without explicit lethal plan or intent.",
        "requires_escalation": False,
        "action": "heightened_monitoring_and_resource_sharing",
        "evidence_cues": [
            "wish i was never born", "don't want to wake up", "dont want to wake up",
            "wish i could disappear", "tired of existing", "better off without me",
            "can't carry on like this", "life feels too heavy", "no reason to keep living",
            "hate waking up in the morning"
        ]
    },
    RiskLevel.HIGH.value: {
        "cssrs_level": "Active Ideation with Plan / Intent & Self-Harm",
        "description": "Active thoughts of suicide, explicit desire to end life, formulated suicide plan, or active self-harm urges.",
        "requires_escalation": True,
        "action": "safety_workflow_and_human_escalation",
        "evidence_cues": [
            "want to end my life", "going to kill myself", "want to die", "commit suicide",
            "plan to end it all", "ready to die", "urge to hurt myself", "cutting myself",
            "decided to end my life", "thinking about suicide constantly", "planning my suicide"
        ]
    },
    RiskLevel.CRITICAL.value: {
        "cssrs_level": "Imminent Danger / Attempt in Progress",
        "description": "Immediate lethal emergency, suicide attempt in progress, access to lethal means, or explicit final goodbye countdown.",
        "requires_escalation": True,
        "action": "immediate_crisis_lifeline_override",
        "evidence_cues": [
            "i have the pills", "swallowed a bottle of", "standing on the roof", "standing on the ledge",
            "gun to my head", "goodbye everyone", "this is my final goodbye", "about to jump",
            "bleeding out", "no one can stop me now", "written my suicide note and ready"
        ]
    }
}

# Adversarial non-crisis idioms (colloquial metaphors that must NOT trigger false crisis lockouts)
ADVERSARIAL_NON_CRISIS_IDIOMS = [
    "this traffic is killing me",
    "i'm dying of laughter",
    "im dying of laughter",
    "i'm dead tired",
    "im dead tired",
    "killing time until my appointment",
    "this workout completely killed my legs",
    "to die for",
    "over my dead body",
    "bored to death",
    "scared to death of the roller coaster",
    "i bombed that exam",
    "my feet are killing me",
    "i would kill for a cup of coffee"
]
