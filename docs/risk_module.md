# ZENOVA Crisis and Suicide Risk Detection Module

## 1. Overview & Clinical Safety Mission

The **Crisis and Risk Detection Module** is the core life-safety triage engine of **ZENOVA**.
Its mission is to continuously monitor conversational interactions for explicit or implicit cues of:
- **Suicidal ideation** (passive wishes to disappear or active intent)
- **Self-harm impulses and non-suicidal self-injury (NSSI)**
- **Imminent acute crisis** (active suicide attempts in progress or access to lethal means)
- **Violence or physical endangerment**

### Asymmetric Risk Policy: Zero Tolerance for False Negatives
In mental health conversational AI, **a False Negative is potentially fatal**.
- Missing an active crisis turn (False Negative) deprives the user of urgent life-saving crisis intervention.
- A False Positive causes an empathetic safety intervention and hotline referral.
Therefore, the module's loss functions, decision boundaries, and evaluation criteria explicitly maximize **Sensitivity (Recall) on High and Critical Risk**.

### Deterministic Interruption: Zero LLM Override
When `high` or `critical` risk is identified:
1. Normal dialogue response generation is **immediately halted**.
2. An immutable `EscalationEvent` ticket is generated in the clinical database.
3. The LLM is **never permitted to override** or suppress the crisis detection.
4. A deterministic, clinically vetted safety response is delivered, providing immediate access to trained emergency personnel (e.g., 988 Suicide & Crisis Lifeline, 741741 Crisis Text Line).

---

## 2. Clinical Grounding & Dataset Provenance

### Columbia-Suicide Severity Rating Scale (C-SSRS)
The risk taxonomy is grounded in the **Columbia-Suicide Severity Rating Scale (C-SSRS)**, an evidence-based clinical gold standard endorsed by the FDA, CDC, and WHO:

- **Primary Literature**:
  - *Knowledge-aware Assessment of Severity of Suicide Risk for Early Intervention* (Manas Gaur et al., WWW 2019, pp. 514–525. DOI: [10.1145/3308558.3313698](https://dl.acm.org/doi/10.1145/3308558.3313698))
  - *CLPsych 2019 Shared Task: Predicting the Degree of Suicide Risk in Reddit Posts* (Ayah Zirikly et al., NAACL-HLT 2019, pp. 24–33. [ACL Anthology W19-3019](https://aclanthology.org/W19-3019/))
- **Annotation Methodology**:
  - Reddit submissions from mental health communities (`r/SuicideWatch`, `r/depression`, `r/CasualConversation`) annotated by psychiatric researchers and clinicians according to C-SSRS behavioral definitions.
- **Ethics & Privacy Protocol**:
  - Full de-identification of usernames, URLs, and personal identifiers.
  - Strict non-commercial academic research data sharing agreement; user re-identification is explicitly prohibited.

---

## 3. 4-Tier Risk Taxonomy & Action Protocol

| Risk Level | C-SSRS Clinical Equivalent | Behavioral Description | System Action | Escalation SLA |
| :--- | :--- | :--- | :--- | :--- |
| **`low`** | Indicator / No Ideation | General emotional distress, sadness, academic/work stress, or everyday frustration without self-harm or suicidal thoughts. | Standard supportive conversation with Hill's helping skills. | None |
| **`moderate`** | Passive Ideation | Passive death wishes (e.g., *"I wish I didn't wake up"*, *"life feels too heavy"*), feeling like a burden, without plan or lethal intent. | Supportive dialogue, watchful monitoring, proactive wellbeing resource sharing. | Logged for longitudinal baseline monitoring |
| **`high`** | Active Ideation with Intent / Self-Harm | Active desire to die, formulated suicide plans, explicit self-harm threats (e.g., *"I want to end my life"*, *"planning my suicide"*). | **IMMEDIATE GENERATION INTERRUPT**. Route to safety workflow, log `EscalationEvent`, provide crisis hotlines. | Immediate (0 ms delay) |
| **`critical`** | Imminent Danger / Attempt in Progress | Acute emergency in progress, lethal means at hand, explicit final goodbyes (e.g., *"I have the pills right now and I'm swallowing them"*). | **HARD LOCKOUT**. Bypass all conversational generation, deliver emergency 988 intervention, notify safety logs. | Immediate (0 ms delay) |

---

## 4. Modeling Architecture

### 1. Cost-Sensitive TF-IDF Baseline
- **Features**: Sublinear word and character n-grams $(1, 2)$ over a 5,000-term vocabulary.
- **Cost-Sensitive Weighting**: `class_weight={"low": 1.0, "moderate": 1.5, "high": 3.5, "critical": 5.0}`.
- **Decision Thresholding**: Classifies as `HIGH` if $P(\text{high}) + P(\text{critical}) \ge 0.35$, prioritizing sensitivity.

### 2. PyTorch Cost-Sensitive Risk Transformer
- **Architecture**:
  - Learnable token embedding ($d_{\text{model}} = 128$)
  - Sinusoidal positional encodings
  - 2-layer `TransformerEncoder` with 4 attention heads and GELU feedforward ($d_{\text{ff}} = 256$)
  - Masked mean pooling across non-padded token positions
  - Linear classification head to 4 risk logits
- **Loss Function**: Class-weighted `nn.CrossEntropyLoss(weight=[1.0, 1.5, 3.5, 5.0])`.
  - Asymmetric penalty ensures that misclassifying a `high` or `critical` turn as `low` incurs up to $5\times$ higher gradient loss than a false positive.

---

## 5. Adversarial Testing & Negative Control

A persistent failure mode of keyword-based suicide detectors is false-positive triggers on **colloquial metaphors** (e.g., *"I'm dying of laughter"*, *"this traffic is killing me"*, *"I'm dead tired"*, *"killing time"*).

ZENOVA implements an **adversarial non-crisis idiom filter**:
1. Checks for idiomatic phrases in the input text.
2. Verifies the absence of actual lethal means or active self-harm intent.
3. Suppresses false escalation, preventing disruptive crisis lockouts during harmless banter.

---

## 6. Output Contract (`POST /api/v1/risk/detect`)

```json
{
  "risk_level": "high",
  "confidence": 0.9421,
  "crisis_category": "suicidal_ideation",
  "signals": ["active_suicidal_ideation", "intent_signals"],
  "trigger_cues": ["want to end my life"],
  "requires_escalation": true,
  "action": "safety_workflow_and_human_escalation",
  "disclaimer": "Automated risk assessment heuristic; not clinical certainty. Imminent life-safety risks must be routed to human crisis professionals.",
  "model_version": "transformer-v1.0.0",
  "probabilities": {
    "low": 0.0124,
    "moderate": 0.0455,
    "high": 0.8921,
    "critical": 0.0500
  },
  "timestamp": "2026-09-10T02:00:00Z"
}
```

---

## 7. Clinical Disclaimers & Non-Certainty Notice

> [!WARNING]
> **Model predictions do NOT constitute clinical certainty**:
> - Natural language processing models evaluate textual artifacts and cannot observe physical vital signs, non-verbal distress cues, or offline behavioral actions.
> - An automated prediction of `low` risk does not guarantee the absence of suicidal ideation.
> - An automated prediction of `high` risk does not replace emergency clinical triage.
> - When in doubt, ZENOVA errs on the side of safety by providing immediate, unhindered access to human crisis professionals.
