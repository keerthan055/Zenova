# Dataset Datasheet: Test Pipeline Run

**Version**: 1.0.0  
**Intended Task**: `test_task`  
**License**: `MIT`  
**Download Date**: 2026-09-13  

---

## 1. Provenance & Citation

- **Source / Authors**: Integration Test
- **Official URL**: [https://zenova.test](https://zenova.test)
- **Academic Citation**:
> Test Citation (2026)

---

## 2. Dataset Scope & Non-Goals

### Features
`text`, `dialog_id`

### Target Labels
`strat_0`, `strat_1`, `strat_2`

### Explicit Non-Goals & Boundaries
- **PROHIBITED**: Clinical diagnostic classification
- **PROHIBITED**: Automated medication prescribing

---

## 3. Dataset Volume & Splits

- **Total Processed Samples**: 30
- **Train Split**: 24 samples (80.0%)
- **Validation Split**: 2 samples (10.0%)
- **Test Split**: 4 samples (10.0%)
- **Leakage Check Passed**: `PASSED`

---

## 4. Class Balance

| Class Label | Count | Percentage |
| :--- | :--- | :--- |
| `strat_0` | 10 | 33.33% |
| `strat_1` | 10 | 33.33% |
| `strat_2` | 10 | 33.33% |

---

## 5. Linguistic & Length Profile

- **Mean Word Length**: 6.0 ± 0.0
- **Median Word Length**: 6.0
- **95th Percentile**: 6.0 words
- **Range**: [6 - 6] words
