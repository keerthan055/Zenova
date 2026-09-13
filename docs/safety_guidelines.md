# ZENOVA Safety and Clinical Governance Guidelines

## 1. Non-Diagnostic Boundary
ZENOVA is explicitly designed as an supportive wellbeing platform, **never** an autonomous diagnostic or emergency medical device.
- Models must not emit diagnostic claims (e.g. "You have major depressive disorder").
- Models must provide observational framing and emotional reflection.
- Standard clinical disclaimers must be enforced across all API responses.

## 2. High-Risk Crisis Protocols
Whenever input exhibits signals of:
- Suicidal ideation
- Self-harm intentions
- Imminent violence to self or others
- Acute delusional breaks

The standard conversational generation pipeline is **bypassed immediately**:
1. Flag `requires_immediate_escalation` is raised.
2. The user is provided standardized crisis helpline resources (e.g., 988 Suicide & Crisis Lifeline, Crisis Text Line).
3. The event is logged securely for clinician escalation review.

## 3. ESConv Scope of Use
The ESConv dataset is used strictly for **support-strategy planning** (Hill's helping skills model).
- It must **not** be used or represented as a psychiatric diagnostic dataset or suicide risk training set.
