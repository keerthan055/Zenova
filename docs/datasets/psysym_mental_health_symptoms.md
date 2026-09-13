# Dataset Datasheet: PsySym Mental Health Symptoms

**Version**: 1.0.0  
**Intended Task**: `symptom_signal_identification`  
**License**: `MIT (Code) / Author-Agreement Research Use (PsySym Corpus)`  
**Download Date**: 2026-09-09  

---

## 1. Provenance & Citation

- **Source / Authors**: Zhang et al. (EMNLP 2022) / DSM-5 PsySym Knowledge Graph
- **Official URL**: [https://aclanthology.org/2022.emnlp-main.677/](https://aclanthology.org/2022.emnlp-main.677/)
- **Academic Citation**:
> Zhang, Z., Chen, S., Wu, M., & Zhu, K. (2022). Symptom Identification for Interpretable Detection of Multiple Mental Disorders on Social Media. In Proceedings of the 2022 EMNLP, pp. 9970-9985.

---

## 2. Dataset Scope & Non-Goals

### Features
`text`, `num_signals`, `sample_type`

### Target Labels
`sleep disturbance`, `loss of interest`, `depressed mood`, `fatigue / low energy`, `anxiety / panic`, `cognitive difficulty`, `feelings of worthlessness`, `appetite / eating change`, `social withdrawal / isolation`, `suicidal ideation / crisis thoughts`

### Explicit Non-Goals & Boundaries
- **PROHIBITED**: Psychiatric disorder diagnosis
- **PROHIBITED**: Depression / Bipolar / Anxiety diagnostic labeling
- **PROHIBITED**: Clinical prescription recommendation
- **PROHIBITED**: Replacing psychiatric evaluation

---

## 3. Dataset Volume & Splits

- **Total Processed Samples**: 1673
- **Train Split**: 1338 samples (80.0%)
- **Validation Split**: 167 samples (10.0%)
- **Test Split**: 168 samples (10.0%)
- **Leakage Check Passed**: `PASSED`

---

## 4. Class Balance

| Class Label | Count | Percentage |
| :--- | :--- | :--- |
| `depressed mood` | 450 | 26.9% |
| `sleep disturbance` | 323 | 19.31% |
| `feelings of worthlessness` | 287 | 17.15% |
| `anxiety / panic` | 287 | 17.15% |
| `fatigue / low energy` | 258 | 15.42% |
| `loss of interest` | 224 | 13.39% |
| `social withdrawal / isolation` | 223 | 13.33% |
| `cognitive difficulty` | 223 | 13.33% |
| `appetite / eating change` | 223 | 13.33% |
| `suicidal ideation / crisis thoughts` | 188 | 11.24% |
| `__NONE__` | 100 | 5.98% |

---

## 5. Linguistic & Length Profile

- **Mean Word Length**: 20.72 ± 7.47
- **Median Word Length**: 21.0
- **95th Percentile**: 34.0 words
- **Range**: [6 - 45] words
