# Dataset Datasheet: StudentLife Passive Sensing Benchmark

**Version**: 1.0.0  
**Intended Task**: `behavioral_sensing_benchmark`  
**License**: `Academic Research Use (Dartmouth Open Data)`  
**Download Date**: 2026-09-10  

---

## 1. Provenance & Citation

- **Source / Authors**: Dartmouth College (Wang et al., UbiComp 2014)
- **Official URL**: [https://studentlife.cs.dartmouth.edu/](https://studentlife.cs.dartmouth.edu/)
- **Academic Citation**:
> Wang, R., Chen, F., Chen, Z., Li, T., Harari, G., Tignor, S., Zhou, X., Ben-Zeev, D., & Campbell, A. T. (2014). StudentLife: assessing mental health, cognitive ability and academic performance of college students using smartphones. In UbiComp '14, pp. 93-104.

---

## 2. Dataset Scope & Non-Goals

### Features
`step_count`, `walking_minutes`, `sedentary_minutes`, `sleep_duration_hours`, `sleep_disturbances_count`, `mobility_radius_km`, `location_entropy`, `conversation_duration_minutes`, `screen_unlock_count`

### Target Labels
`baseline_normal`, `mild_variation`, `multi_domain_disruption`

### Explicit Non-Goals & Boundaries
- **PROHIBITED**: Psychiatric diagnosis (e.g., Major Depressive Disorder or Bipolar Disorder)
- **PROHIBITED**: Crisis or suicidal intent detection from passive sensors alone
- **PROHIBITED**: Unilateral clinician notification without conversational context
- **PROHIBITED**: Single-feature inference of mental distress

---

## 3. Dataset Volume & Splits

- **Total Processed Samples**: 600
- **Train Split**: 480 samples (80.0%)
- **Validation Split**: 60 samples (10.0%)
- **Test Split**: 60 samples (10.0%)
- **Leakage Check Passed**: `PASSED`

---

## 4. Class Balance

| Class Label | Count | Percentage |
| :--- | :--- | :--- |
| `baseline_normal` | 426 | 71.0% |
| `mild_variation` | 102 | 17.0% |
| `multi_domain_disruption` | 72 | 12.0% |

---

## 5. Linguistic & Length Profile

- **Mean Word Length**: 12.0 ± 0.0
- **Median Word Length**: 12.0
- **95th Percentile**: 12.0 words
- **Range**: [12 - 12] words
