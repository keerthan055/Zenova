# ZENOVA Step 3: Text Emotion Analysis Module

## 1. Module Overview

The **ZENOVA Emotion Detection Module** provides real-time, non-diagnostic emotion analysis and affective state extraction from conversational text utterances. It maps utterances into 7 standard Ekman emotion categories (`anger`, `disgust`, `fear`, `joy`, `neutral`, `sadness`, `surprise`) and produces corresponding normative Valence-Arousal-Dominance (VAD) affective coordinates.

---

## 2. Dataset Provenance: GoEmotions (Google Research, ACL 2020)

- **Source**: Demszky et al., Google Research, Association for Computational Linguistics (ACL 2020)
- **Official URL**: [https://github.com/google-research/google-research/tree/master/goemotions](https://github.com/google-research/google-research/tree/master/goemotions)
- **License**: Apache-2.0
- **Size**: 58,000 Reddit comments annotated by trained raters.
- **Ekman Taxonomy Mapping**:
  - `anger`: anger, annoyance, disapproval
  - `disgust`: disgust
  - `fear`: fear, nervousness (capturing anxiety)
  - `joy`: joy, amusement, approval, excitement, gratitude, love, optimism, pride, relief
  - `sadness`: sadness, disappointment, grief, embarrassment, remorse
  - `surprise`: surprise, realization, confusion, curiosity
  - `neutral`: neutral
- **Why not ESConv for Emotion Detection?**:
  ESConv (Liu et al., 2021) is designed specifically for **support strategy planning** (Hill's helping skills). Its emotion annotations are coarse, single-label dialogue metadata chosen by the seeker at the start of the session. In contrast, GoEmotions provides utterance-level granularity with high statistical confidence.

---

## 3. Comparative Modeling & Evaluation Results

Both a **TF-IDF + LogisticRegression Baseline** and a **PyTorch Transformer Sequence Classifier** were trained and evaluated on the held-out test split (883 samples):

| Metric | TF-IDF Baseline | PyTorch Transformer |
| :--- | :--- | :--- |
| **Accuracy** | **51.53%** | 44.39% |
| **Macro Precision** | **51.33%** | 45.56% |
| **Macro Recall** | **52.76%** | 45.80% |
| **Macro F1** | **51.84%** | 45.43% |
| **Fear (Anxiety) F1** | **62.89%** | 60.53% |
| **Joy F1** | **63.05%** | 51.70% |
| **Sadness F1** | **52.11%** | 49.50% |
| **Surprise F1** | **54.31%** | 40.28% |

*(Full per-class metrics and confusion matrices are serialized in `models/emotion/evaluation_report.json`)*.

---

## 4. Output Contract

```json
{
  "emotions": [
    {
      "label": "sadness",
      "confidence": 0.8523
    },
    {
      "label": "fear",
      "confidence": 0.0812
    }
  ],
  "primary_emotion": "sadness",
  "valence": -0.63,
  "arousal": -0.27,
  "dominance": -0.33,
  "is_placeholder": false,
  "model_version": "transformer-v1.0.0",
  "timestamp": "2026-09-10T01:30:00Z"
}
```

---

## 5. Domain Shift & Clinical Governance Limitations

> [!WARNING]
> **Domain Shift between Social Media and Mental Health Conversations**:
> 1. **Informal Internet Vernacular**: GoEmotions derives from Reddit comments, featuring internet slang, sarcasm, and situational memes. In contrast, mental health seekers express vulnerability through extended disclosures and narrative descriptions.
> 2. **Emotional Concealment & Anhedonia**: In clinical contexts, users often describe severe emotional distress (e.g. chronic fatigue, flat affect) using structurally neutral phrasing. Utterance-level models may classify these as `neutral`.
> 3. **Mitigation in ZENOVA**:
>    - Single-utterance predictions are never used in isolation for clinical inferences.
>    - The **Personal Baseline Engine** monitors z-score shifts over longitudinal interactions.
>    - The **Crisis & Risk Engine** provides an independent deterministic and neural safety bypass for high-risk signals.
