# ZENOVA System Architecture

## Overview

ZENOVA is a research-grade, safety-first mental health conversational support and longitudinal wellbeing tracking system.

## Modular Component Pipeline

```
                              [ USER INPUT ]
                                    │
                                    ▼
                        [ INPUT / PREPROCESSING ]
                                    │
    ┌───────────────────────────────┴──────────────────────────────┐
    │                               │                              │
    ▼                               ▼                              ▼
[ EMOTION ANALYSIS ]       [ SYMPTOM EXTRACTION ]         [ CRISIS / RISK DETECTOR ]
(Valence, Arousal,        (Signals, Sleep, Low Energy,   (Suicidal ideation,
 Categorical Empathy)      Subclinical markers)           Delusion, Imminent danger)
    │                               │                              │
    └───────────────────────┬───────┴──────────────────────────────┘
                            │
                            ▼
              [ PERSONAL BASELINE ENGINE ]
              (Individual historical stats,
               z-score anomaly detection)
                            │
                            ▼
                    [ CONTEXT ENGINE ]
                            │
               ┌────────────┴────────────┐
               │                         │
     [Is Risk High/Critical?]            │ (Normal Support Pathway)
               │ YES                     │
               ▼                         ▼
   [ CRISIS BYPASS GATE ]      [ SUPPORT STRATEGY PLANNER ]
   (Disallow LLM Free-form;    (ESConv Hill's Helping Skills:
    Inject Crisis Protocols &   Exploration -> Comforting -> Action)
    Escalate to Human/Help)              │
               │                         ▼
               │               [ LLM + RAG GENERATION ]
               │               (Grounded evidence,
               │                Empathetic alignment)
               │                         │
               └────────────┬────────────┘
                            │
                            ▼
                     [ SAFETY GATE ]
                     (Toxicity, Over-promising,
                      Medical claim filtration)
                            │
                            ▼
                  [ FINAL SAFE RESPONSE ]
```

## Module Contracts

Each module communicates across explicit boundaries defined by Pydantic v2 data models:

1. **Emotion Analysis**: Produces `EmotionAnalysisResult` with valence/arousal coordinates and multi-class emotion probabilities.
2. **Symptom Analysis**: Produces `SymptomAnalysisResult` tracking non-diagnostic behavioural signals and observational evidence spans.
3. **Risk Analysis**: Produces `CrisisRiskAssessment`. High/Critical risk triggers immediate override of conversational generation.
4. **Baseline Engine**: Evaluates individual variance via `BaselineDeviationReport`.
5. **Strategy Planner**: Selects one of the 8 Hill's Helping Skills codified in ESConv (`Question`, `Restatement`, `Reflection of Feelings`, `Affirmation`, `Self-disclosure`, `Providing Suggestions`, `Information`, `Others`).
6. **Safety Gate**: Evaluates `SafetyGateResult`, ensuring responses never simulate medical diagnosis or encourage self-harm.
