# Dataset Datasheet: GoEmotions Ekman

**Version**: 1.0.0  
**Intended Task**: `emotion_analysis`  
**License**: `Apache-2.0`  
**Download Date**: 2026-09-09  

---

## 1. Provenance & Citation

- **Source / Authors**: Google Research (Demszky et al., ACL 2020)
- **Official URL**: [https://github.com/google-research/google-research/tree/master/goemotions](https://github.com/google-research/google-research/tree/master/goemotions)
- **Academic Citation**:
> Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., & Ravi, S. (2020). GoEmotions: A Dataset of Fine-Grained Emotions. In ACL 2020, pp. 4040-4054.

---

## 2. Dataset Scope & Non-Goals

### Features
`text`, `fine_emotions`

### Target Labels
`anger`, `disgust`, `fear`, `joy`, `sadness`, `surprise`, `neutral`

### Explicit Non-Goals & Boundaries
- **PROHIBITED**: Psychiatric disorder diagnosis
- **PROHIBITED**: Depression / Bipolar diagnostic classification
- **PROHIBITED**: Suicide risk assessment

---

## 3. Dataset Volume & Splits

- **Total Processed Samples**: 8822
- **Train Split**: 7057 samples (80.0%)
- **Validation Split**: 882 samples (10.0%)
- **Test Split**: 883 samples (10.0%)
- **Leakage Check Passed**: `PASSED`

---

## 4. Class Balance

| Class Label | Count | Percentage |
| :--- | :--- | :--- |
| `anger` | 1500 | 17.0% |
| `surprise` | 1500 | 17.0% |
| `neutral` | 1500 | 17.0% |
| `joy` | 1500 | 17.0% |
| `sadness` | 1500 | 17.0% |
| `fear` | 690 | 7.82% |
| `disgust` | 632 | 7.16% |

---

## 5. Linguistic & Length Profile

- **Mean Word Length**: 12.98 ± 6.63
- **Median Word Length**: 13.0
- **95th Percentile**: 24.0 words
- **Range**: [1 - 31] words
