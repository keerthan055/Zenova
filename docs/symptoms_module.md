# ZENOVA Symptom and Observational Signal Identification Module

## 1. Overview & Clinical Mission

The **ZENOVA Mental-Health Symptom and Signal Identification Module** identifies observable, textual mental-health signals from conversational user turns.

### Primary Principle: Observational Signals vs. Psychiatric Diagnosis
Under no circumstances does ZENOVA issue clinical or psychiatric diagnoses.
- **Prohibited Output**: *"You have depression."*, *"Diagnostic Assessment: Major Depressive Disorder."*
- **Permitted Output**: Observational markers with confidence scores, verbatim evidence spans, and explicit clinical disclaimers (e.g. `sleep disturbance` [0.86], `loss of interest` [0.79]).

---

## 2. Dataset Provenance & Literature Foundation: PsySym (EMNLP 2022)

The conceptual taxonomy and symptom definitions are grounded in **PsySym**:

- **Paper Title**: *Symptom Identification for Interpretable Detection of Multiple Mental Disorders on Social Media*
- **Authors**: Zhiling Zhang, Siyuan Chen, Mengyue Wu, Kenny Q. Zhu (Shanghai Jiao Tong University)
- **Conference**: Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP 2022), pages 9970–9985
- **Official URL**: [https://aclanthology.org/2022.emnlp-main.677/](https://aclanthology.org/2022.emnlp-main.677/)
- **DOI**: `10.18653/v1/2022.emnlp-main.677`
- **Code Repository**: [https://github.com/blmoistawinde/EMNLP22-PsySym](https://github.com/blmoistawinde/EMNLP22-PsySym)
- **Licensing & Access Protocol**:
  - Code and Knowledge Graph files (`symptom_kg.owl`, `parsed_kg_info.json`) are licensed under MIT.
  - Raw Reddit post text is subject to restricted researcher agreements to protect user privacy and comply with psychiatric NLP ethical guidelines.

---

## 3. Scope: 7 Disorders & 38 Symptom Classes

PsySym formalizes diagnostic criteria compiled from established psychiatric manuals (primarily DSM-5) across **7 mental disorders**:

1. **Major Depressive Disorder (MDD)**
2. **Generalized Anxiety Disorder (GAD)**
3. **Bipolar Disorder (BD)**
4. **Post-Traumatic Stress Disorder (PTSD)**
5. **Obsessive-Compulsive Disorder (OCD)**
6. **Attention-Deficit/Hyperactivity Disorder (ADHD)**
7. **Eating Disorders (ED)**

### Core Conversational Observational Signals
From the 38 clinical symptom classes, ZENOVA maps conversational turns into 10 high-utility observational signal categories:

| Signal Name | DSM-5 Source Criteria | Clinical Description |
| :--- | :--- | :--- |
| `sleep disturbance` | MDD A4 / GAD C6 | Insomnia, delayed sleep onset, frequent nocturnal awakenings, or hypersomnia. |
| `loss of interest` | MDD A2 (Anhedonia) | Markedly diminished interest or pleasure in activities once enjoyed. |
| `depressed mood` | MDD A1 | Persistent sadness, feelings of emptiness, tearfulness, or gloom. |
| `fatigue / low energy` | MDD A6 / GAD C2 | Physical and mental exhaustion, lack of energy for basic activities. |
| `anxiety / panic` | GAD A, C1 / Panic Disorder | Excessive worry, racing heart, trembling, shortness of breath, terror. |
| `cognitive difficulty` | MDD A8 / GAD C3 | Brain fog, diminished concentration, indecisiveness, executive dysfunction. |
| `feelings of worthlessness` | MDD A7 | Excessive guilt, feeling like a burden, severe self-blame and inadequacy. |
| `appetite / eating change` | MDD A3 / ED criteria | Appetite loss, food aversion, compulsive eating, sudden weight shifts. |
| `social withdrawal / isolation` | Functional impairment | Detachment from loved ones, avoiding social interaction, self-isolation. |
| `suicidal ideation / crisis thoughts` | MDD A9 | Passive/active thoughts of death, self-harm urges (*Critical Escalation Trigger*). |

---

## 4. Modeling Architecture

Symptom identification is formulated as a **Multi-Label Sequence Classification** task, where an utterance may express zero, one, or multiple co-occurring symptom signals simultaneously.

### 1. Multi-Label TF-IDF Baseline
- **Vectorizer**: Sublinear term-frequency with character & word n-grams (1, 2).
- **Estimator**: `MultiOutputClassifier(LogisticRegression(class_weight="balanced"))` implementing Binary Relevance.
- **Functionality**: Establishes a fast, transparent baseline with per-label calibrated probabilities.

### 2. PyTorch Multi-Label Transformer Sequence Classifier
- **Architecture**:
  - Learnable token embedding ($d_{\text{model}} = 128$)
  - Sinusoidal positional encodings
  - 2-layer `TransformerEncoder` with 4 attention heads and GELU feedforward ($d_{\text{ff}} = 256$)
  - Masked mean pooling across non-padded token positions
  - Linear classification head yielding unnormalized logits $\mathbf{z} \in \mathbb{R}^{K}$
- **Loss Function**: `nn.BCEWithLogitsLoss()`
  $$\mathcal{L} = - \frac{1}{K} \sum_{k=1}^{K} \left[ y_k \log \sigma(z_k) + (1 - y_k) \log (1 - \sigma(z_k)) \right]$$
- **Inference**: Sigmoid activation computes independent probabilities $\hat{p}_k = \sigma(z_k) \in [0, 1]$ for each symptom class $k$.

---

## 5. Output Contract & Clinical Safety Guarantees

### Standalone API Contract (`POST /api/v1/symptoms/identify`)

```json
{
  "signals": [
    {
      "label": "sleep disturbance",
      "confidence": 0.86,
      "severity": "mild",
      "evidence_spans": ["haven't been sleeping properly"]
    },
    {
      "label": "loss of interest",
      "confidence": 0.79,
      "severity": "moderate",
      "evidence_spans": ["don't enjoy things anymore"]
    }
  ],
  "aggregate_severity": "moderate",
  "is_crisis_flagged": false,
  "disclaimer": "Observational marker only; does not constitute clinical or psychiatric diagnosis.",
  "notes": "2 observational symptom signals identified.",
  "model_version": "transformer-v1.0.0",
  "timestamp": "2026-09-10T01:50:00Z"
}
```

---

## 6. Edge Cases & Safety Policies

1. **Zero Symptoms Detected (Benign Inputs)**:
   - Queries like *"What is the weather today?"* or *"Can you recommend a good book?"* return an empty list `signals: []` with `aggregate_severity: "none"`.
2. **Ambiguous or Subclinical Mentions**:
   - Mentions that fall below detection thresholds are filtered out or tagged as `subclinical` rather than producing false positive clinical alerts.
3. **Overlapping / Co-occurring Symptoms**:
   - The multi-label formulation natively captures comorbidity patterns (e.g. insomnia + fatigue + depressed mood) without artificial single-class constraints.
4. **Crisis & Self-Harm Mentions**:
   - Texts containing death wishes or self-harm triggers are flagged (`is_crisis_flagged = True`, `severity = "severe"`).
   - In the central orchestrator, this immediately bypasses LLM generation and routes to the emergency 988 lifeline safety intervention protocol.

---

## 7. Limitations & Non-Goals

- **Non-Goals**:
  - ZENOVA will **never** attempt to diagnose psychiatric illnesses, replace DSM-5 clinical interviews, or recommend pharmacological prescriptions.
- **Domain Shift**:
  - Text patterns in social media posts often exhibit greater informality and colloquial slang compared to real-time clinical dialogue.
- **Temporality**:
  - DSM-5 diagnoses require persistent symptoms over specified time windows (e.g. $\ge 2$ weeks for MDD). A single turn observation is only a snapshot indicator, not longitudinal proof of chronicity.
