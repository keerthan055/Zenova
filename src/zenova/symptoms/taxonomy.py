"""PsySym DSM-5 symptom taxonomy and conversational signal definitions for ZENOVA.

Grounded in:
- Zhang et al., EMNLP 2022: 'Symptom Identification for Interpretable Detection
  of Multiple Mental Disorders on Social Media'
- DSM-5 (Diagnostic and Statistical Manual of Mental Disorders, 5th Edition)

CRITICAL CLINICAL BOUNDARY:
These classes represent observational textual signals, NOT psychiatric diagnoses.
"""
from typing import Dict, List, Any


# 7 Psychiatric Disorders modeled in the PsySym Knowledge Graph
PSYSYM_DISORDERS = {
    "MDD": "Major Depressive Disorder",
    "GAD": "Generalized Anxiety Disorder",
    "BIPOLAR": "Bipolar Disorder",
    "PTSD": "Post-Traumatic Stress Disorder",
    "OCD": "Obsessive-Compulsive Disorder",
    "ADHD": "Attention-Deficit/Hyperactivity Disorder",
    "ED": "Eating Disorder"
}

# Core observational symptom signals for conversational wellbeing support
SYMPTOM_SIGNALS = [
    "sleep disturbance",
    "loss of interest",
    "depressed mood",
    "fatigue / low energy",
    "anxiety / panic",
    "cognitive difficulty",
    "feelings of worthlessness",
    "appetite / eating change",
    "social withdrawal / isolation",
    "suicidal ideation / crisis thoughts"
]

# Clinical definitions and evidence cues for each observational signal
TAXONOMY_SPEC = {
    "sleep disturbance": {
        "psysym_classes": ["Insomnia", "Hypersomnia", "Disrupted Sleep"],
        "dsm5_criteria": "MDD Criterion A4 / GAD Criterion C6: Insomnia or hypersomnia nearly every day.",
        "description": "Difficulty falling asleep, frequent nighttime awakenings, unrefreshing sleep, or excessive sleeping.",
        "evidence_cues": [
            "haven't been sleeping", "cant sleep", "can't sleep", "insomnia", "tossing and turning",
            "waking up at 3am", "sleeping all day", "barely slept", "sleep schedule is broken",
            "exhausted waking up", "trouble falling asleep", "restless nights"
        ]
    },
    "loss of interest": {
        "psysym_classes": ["Anhedonia", "Loss of Interest", "Apathy"],
        "dsm5_criteria": "MDD Criterion A2: Markedly diminished interest or pleasure in all, or almost all, activities.",
        "description": "Reduced capacity to experience pleasure, apathy towards hobbies, friends, or activities previously enjoyed.",
        "evidence_cues": [
            "don't enjoy things anymore", "dont enjoy", "lost interest", "nothing brings me joy",
            "used to love", "feel numb", "nothing feels fun", "apathetic", "can't feel pleasure",
            "no motivation for hobbies", "activities feel pointless", "passion is gone"
        ]
    },
    "depressed mood": {
        "psysym_classes": ["Depressed Mood", "Sadness", "Despair", "Tearfulness"],
        "dsm5_criteria": "MDD Criterion A1: Depressed mood most of the day, nearly every day.",
        "description": "Persistent feelings of sadness, emotional emptiness, hopelessness, or frequent tearfulness.",
        "evidence_cues": [
            "feel so sad", "crying all the time", "hopeless", "feeling down", "empty inside",
            "deep sadness", "miserable", "crying uncontrollably", "dark cloud", "feel sorrow",
            "gloom", "heart hurts", "everything feels bleak"
        ]
    },
    "fatigue / low energy": {
        "psysym_classes": ["Fatigue", "Loss of Energy", "Physical Exhaustion"],
        "dsm5_criteria": "MDD Criterion A6 / GAD Criterion C2: Fatigue or loss of energy nearly every day.",
        "description": "Profound physical or mental exhaustion, feeling drained even after minimal effort.",
        "evidence_cues": [
            "completely exhausted", "no energy", "drained", "can barely get out of bed",
            "constant fatigue", "wiped out", "zero physical energy", "tired all the time",
            "burnout", "body feels heavy", "lethargic"
        ]
    },
    "anxiety / panic": {
        "psysym_classes": ["Excessive Anxiety", "Panic", "Nervousness", "Restlessness"],
        "dsm5_criteria": "GAD Criterion A & C1 / Panic Disorder: Excessive anxiety and worry, feeling keyed up or on edge.",
        "description": "Persistent worry, racing heartbeat, hyperventilation, trembling, dread, or panicky sensations.",
        "evidence_cues": [
            "having a panic attack", "anxious", "can't stop worrying", "heart racing", "chest tight",
            "shaking with anxiety", "constant dread", "paralyzed with fear", "freaking out",
            "hyperventilating", "terrified for no reason", "nervous wreck"
        ]
    },
    "cognitive difficulty": {
        "psysym_classes": ["Diminished Concentration", "Indecisiveness", "Brain Fog"],
        "dsm5_criteria": "MDD Criterion A8 / GAD Criterion C3: Diminished ability to think or concentrate, or indecisiveness.",
        "description": "Difficulty focusing on tasks, memory lapses, slowed thinking, or extreme indecisiveness.",
        "evidence_cues": [
            "can't concentrate", "cant focus", "brain fog", "mind goes blank", "unable to make decisions",
            "forgetting everything", "can't get any work done", "scatterbrained", "mind racing",
            "struggling to think clearly"
        ]
    },
    "feelings of worthlessness": {
        "psysym_classes": ["Worthlessness", "Excessive Guilt", "Self-Blame"],
        "dsm5_criteria": "MDD Criterion A7: Feelings of worthlessness or excessive or inappropriate guilt.",
        "description": "Intense self-loathing, feeling like a burden to others, pervasive guilt, or feelings of inadequacy.",
        "evidence_cues": [
            "i am a burden", "hate myself", "worthless", "failure", "everyone would be better without me",
            "i mess up everything", "hate who i am", "good for nothing", "blame myself",
            "ashamed of who i am", "never do anything right"
        ]
    },
    "appetite / eating change": {
        "psysym_classes": ["Appetite Loss", "Hyperphagia", "Eating Disturbances"],
        "dsm5_criteria": "MDD Criterion A3 / ED criteria: Significant weight loss when not dieting or weight gain, or decrease or increase in appetite.",
        "description": "Noticeable changes in eating habits, such as food aversion, forgetting to eat, or bingeing.",
        "evidence_cues": [
            "no appetite", "haven't eaten in days", "food makes me nauseous", "bingeing food",
            "lost weight suddenly", "forgetting to eat", "forcing myself to swallow food",
            "compulsive eating", "restricting food"
        ]
    },
    "social withdrawal / isolation": {
        "psysym_classes": ["Social Withdrawal", "Alienation", "Avoidance"],
        "dsm5_criteria": "Interpersonal functional impairment in MDD, GAD, PTSD: Social avoidance and detachment from others.",
        "description": "Isolating from loved ones, avoiding social encounters, ghosting friends, and withdrawing into solitude.",
        "evidence_cues": [
            "isolating myself", "ignoring my friends", "withdrawn from everyone", "locked in my room",
            "avoiding people", "ghosting everyone", "don't want to see anyone", "distancing myself",
            "pushing people away"
        ]
    },
    "suicidal ideation / crisis thoughts": {
        "psysym_classes": ["Suicidal Ideation", "Self-Harm Impulses", "Death Wishes"],
        "dsm5_criteria": "MDD Criterion A9: Recurrent thoughts of death, suicidal ideation without a specific plan, or a suicide attempt.",
        "description": "Active or passive death wishes, thoughts of self-harm, or severe crisis expressions.",
        "evidence_cues": [
            "want to end my life", "kill myself", "suicide", "want to die", "harm myself",
            "cut myself", "better off dead", "end it all", "don't want to wake up",
            "wish i was never born", "ready to give up completely"
        ],
        "is_crisis_signal": True
    }
}
