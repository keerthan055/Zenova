# Dataset Datasheet: C-SSRS Suicide and Crisis Risk Benchmark

**Version**: 1.0.0  
**Intended Task**: `crisis_risk_detection`  
**License**: `Academic Research Use (De-identified C-SSRS Benchmark)`  
**Download Date**: 2026-09-09  

---

## 1. Provenance & Citation

- **Source / Authors**: C-SSRS Social Media Corpus (Gaur et al., WWW 2019 / CLPsych 2019)
- **Official URL**: [https://dl.acm.org/doi/10.1145/3308558.3313698](https://dl.acm.org/doi/10.1145/3308558.3313698)
- **Academic Citation**:
> Gaur, M., Alambo, A., Sain, J., Kursuncu, U., Thirunarayan, K., Kavuluru, R., ... & Pathak, J. (2019). Knowledge-aware Assessment of Severity of Suicide Risk for Early Intervention. In The World Wide Web Conference (WWW '19), pp. 514-525.

---

## 2. Dataset Scope & Non-Goals

### Features
`text`, `crisis_category`, `requires_escalation`, `sample_type`

### Target Labels
`low`, `moderate`, `high`, `critical`

### Explicit Non-Goals & Boundaries
- **PROHIBITED**: Clinical psychiatric diagnosis
- **PROHIBITED**: Emergency dispatch replacement
- **PROHIBITED**: Autonomous involuntary commitment

---

## 3. Dataset Volume & Splits

- **Total Processed Samples**: 1218
- **Train Split**: 974 samples (80.0%)
- **Validation Split**: 122 samples (10.0%)
- **Test Split**: 122 samples (10.0%)
- **Leakage Check Passed**: `PASSED`

---

## 4. Class Balance

| Class Label | Count | Percentage |
| :--- | :--- | :--- |
| `low` | 654 | 53.69% |
| `moderate` | 226 | 18.56% |
| `high` | 198 | 16.26% |
| `critical` | 140 | 11.49% |

---

## 5. Linguistic & Length Profile

- **Mean Word Length**: 16.22 ± 3.81
- **Median Word Length**: 16.0
- **95th Percentile**: 22.0 words
- **Range**: [3 - 27] words
