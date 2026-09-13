# Dataset Datasheet: Emotional Support Conversation Dataset (ESConv)

**Version**: 1.0.0  
**Intended Task**: `support_strategy_planning`  
**License**: `Apache-2.0`  
**Download Date**: 2026-09-09  

---

## 1. Provenance & Citation

- **Source / Authors**: Tsinghua University Conversational AI (CoAI) Group
- **Official URL**: [https://github.com/thu-coai/Emotional-Support-Conversation](https://github.com/thu-coai/Emotional-Support-Conversation)
- **Academic Citation**:
> Liu, S., Zheng, C., Demasi, O., Sabour, S., Li, Y., Yu, Z., Jiang, Y., & Huang, M. (2021). Towards Emotional Support Dialog Systems. ACL 2021, pp. 3469-3483.

---

## 2. Dataset Scope & Non-Goals

### Features
`dialog_history`, `situation`, `emotion_type`, `problem_type`, `turn_index`

### Target Labels
`Question`, `Restatement or Paraphrasing`, `Reflection of feelings`, `Affirmation and Reassurance`, `Self-disclosure`, `Providing Suggestions`, `Information`, `Others`

### Explicit Non-Goals & Boundaries
- **PROHIBITED**: Clinical diagnostic classification
- **PROHIBITED**: Suicide risk assessment
- **PROHIBITED**: Psychiatric disorder diagnosis

---

## 3. Dataset Volume & Splits

- **Total Processed Samples**: 18376
- **Train Split**: 14700 samples (80.0%)
- **Validation Split**: 1838 samples (10.0%)
- **Test Split**: 1838 samples (10.0%)
- **Leakage Check Passed**: `PASSED`

---

## 4. Class Balance

| Class Label | Count | Percentage |
| :--- | :--- | :--- |
| `Question` | 3801 | 20.68% |
| `Others` | 3341 | 18.18% |
| `Providing Suggestions` | 2954 | 16.08% |
| `Affirmation and Reassurance` | 2827 | 15.38% |
| `Self-disclosure` | 1713 | 9.32% |
| `Reflection of feelings` | 1436 | 7.81% |
| `Information` | 1215 | 6.61% |
| `Restatement or Paraphrasing` | 1089 | 5.93% |

---

## 5. Linguistic & Length Profile

- **Mean Word Length**: 18.11 ± 13.91
- **Median Word Length**: 15.0
- **95th Percentile**: 44.0 words
- **Range**: [1 - 166] words
