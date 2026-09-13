# Dataset Datasheet: Synthetic Wellbeing Benchmark

**Version**: 1.0.0  
**Intended Task**: `support_strategy_planning`  
**License**: `MIT`  
**Download Date**: 2026-09-09  

---

## 1. Provenance & Citation

- **Source / Authors**: ZENOVA Synthetic Benchmark Generator
- **Official URL**: [https://zenova.internal/datasets/synthetic_wellbeing](https://zenova.internal/datasets/synthetic_wellbeing)
- **Academic Citation**:
> Zenova Research Team. (2026). Synthetic Wellbeing Benchmark Dataset for Controlled Pipeline Validation.

---

## 2. Dataset Scope & Non-Goals

### Features
`text`, `situation`, `emotion`, `dialog_id`, `turn_id`

### Target Labels
`Question`, `Restatement or Paraphrasing`, `Reflection of feelings`, `Affirmation and Reassurance`, `Self-disclosure`, `Providing Suggestions`, `Information`, `Others`

### Explicit Non-Goals & Boundaries
- **PROHIBITED**: Clinical diagnostic classification
- **PROHIBITED**: Automated medication prescribing

---

## 3. Dataset Volume & Splits

- **Total Processed Samples**: 240
- **Train Split**: 192 samples (80.0%)
- **Validation Split**: 24 samples (10.0%)
- **Test Split**: 24 samples (10.0%)
- **Leakage Check Passed**: `PASSED`

---

## 4. Class Balance

| Class Label | Count | Percentage |
| :--- | :--- | :--- |
| `Affirmation and Reassurance` | 42 | 17.5% |
| `Information` | 33 | 13.75% |
| `Restatement or Paraphrasing` | 31 | 12.92% |
| `Self-disclosure` | 28 | 11.67% |
| `Reflection of feelings` | 28 | 11.67% |
| `Providing Suggestions` | 28 | 11.67% |
| `Others` | 26 | 10.83% |
| `Question` | 24 | 10.0% |

---

## 5. Linguistic & Length Profile

- **Mean Word Length**: 22.11 ± 1.94
- **Median Word Length**: 22.0
- **95th Percentile**: 25.0 words
- **Range**: [17 - 28] words
