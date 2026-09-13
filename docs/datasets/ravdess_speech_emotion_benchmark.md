# Dataset Datasheet: RAVDESS Speech Emotion Benchmark

**Version**: 1.0.0  
**Intended Task**: `speech_emotion_analysis`  
**License**: `Creative Commons Attribution 4.0 International (CC BY 4.0)`  
**Download Date**: 2026-09-10  

---

## 1. Provenance & Citation

- **Source / Authors**: Ryerson University (Livingstone & Russo, PLoS ONE 2018)
- **Official URL**: [https://zenodo.org/record/1188976](https://zenodo.org/record/1188976)
- **Academic Citation**:
> Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. PLoS ONE, 13(5), e0196391.

---

## 2. Dataset Scope & Non-Goals

### Features
`pitch_mean_hz`, `pitch_std_hz`, `rms_energy_mean`, `intensity_db`, `spectral_centroid_mean`, `spectral_rolloff_mean`, `zero_crossing_rate`, `jitter_local`, `shimmer_local`, `hnr_db`, `duration_seconds`

### Target Labels
`neutral`, `calm`, `happy`, `sad`, `angry`, `fearful`, `disgust`, `surprised`

### Explicit Non-Goals & Boundaries
- **PROHIBITED**: Psychiatric or psychiatric disorder diagnosis (e.g., Major Depressive Disorder)
- **PROHIBITED**: Crisis, self-harm, or suicide risk determination from acoustic audio alone
- **PROHIBITED**: Inferring clinical pathology from isolated vocal tone without consent
- **PROHIBITED**: Single-modality diagnostic inference

---

## 3. Dataset Volume & Splits

- **Total Processed Samples**: 576
- **Train Split**: 460 samples (80.0%)
- **Validation Split**: 58 samples (10.0%)
- **Test Split**: 58 samples (10.0%)
- **Leakage Check Passed**: `PASSED`

---

## 4. Class Balance

| Class Label | Count | Percentage |
| :--- | :--- | :--- |
| `neutral` | 72 | 12.5% |
| `calm` | 72 | 12.5% |
| `happy` | 72 | 12.5% |
| `sad` | 72 | 12.5% |
| `angry` | 72 | 12.5% |
| `fearful` | 72 | 12.5% |
| `disgust` | 72 | 12.5% |
| `surprised` | 72 | 12.5% |

---

## 5. Linguistic & Length Profile

- **Mean Word Length**: 14.0 ± 0.0
- **Median Word Length**: 14.0
- **95th Percentile**: 14.0 words
- **Range**: [14 - 14] words
