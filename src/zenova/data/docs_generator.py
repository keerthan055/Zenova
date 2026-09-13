"""Automated Dataset Documentation and Datasheet generator."""
from pathlib import Path
from zenova.data.metadata import DatasetMetadata


class DatasetDocumentationGenerator:
    """Produces research-grade Markdown datasheets for registered datasets."""

    @staticmethod
    def generate_markdown_datasheet(metadata: DatasetMetadata, output_path: str) -> str:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        stats = metadata.statistics
        class_dist = stats.get("class_distribution", {})
        length_stats = stats.get("text_length_statistics", {})

        md = f"""# Dataset Datasheet: {metadata.dataset_name}

**Version**: {metadata.version}  
**Intended Task**: `{metadata.intended_task}`  
**License**: `{metadata.license}`  
**Download Date**: {metadata.download_date}  

---

## 1. Provenance & Citation

- **Source / Authors**: {metadata.source}
- **Official URL**: [{metadata.official_url}]({metadata.official_url})
- **Academic Citation**:
> {metadata.citation}

---

## 2. Dataset Scope & Non-Goals

### Features
{', '.join(f'`{f}`' for f in metadata.features)}

### Target Labels
{', '.join(f'`{l}`' for l in metadata.labels)}

### Explicit Non-Goals & Boundaries
"""
        for ng in metadata.explicit_non_goals:
            md += f"- **PROHIBITED**: {ng}\n"

        md += f"""
---

## 3. Dataset Volume & Splits

- **Total Processed Samples**: {metadata.number_of_samples.get('processed', 0)}
- **Train Split**: {metadata.train_validation_test_split.train_count} samples ({round(metadata.train_validation_test_split.train_ratio * 100, 1)}%)
- **Validation Split**: {metadata.train_validation_test_split.val_count} samples ({round(metadata.train_validation_test_split.val_ratio * 100, 1)}%)
- **Test Split**: {metadata.train_validation_test_split.test_count} samples ({round(metadata.train_validation_test_split.test_ratio * 100, 1)}%)
- **Leakage Check Passed**: `{'PASSED' if metadata.leakage_check_passed else 'FAILED'}`

---

## 4. Class Balance

| Class Label | Count | Percentage |
| :--- | :--- | :--- |
"""
        for lbl, info in class_dist.items():
            md += f"| `{lbl}` | {info.get('count', 0)} | {info.get('percentage', 0.0)}% |\n"

        if length_stats:
            md += f"""
---

## 5. Linguistic & Length Profile

- **Mean Word Length**: {length_stats.get('mean_words')} ± {length_stats.get('std_words')}
- **Median Word Length**: {length_stats.get('median_words')}
- **95th Percentile**: {length_stats.get('p95_words')} words
- **Range**: [{length_stats.get('min_words')} - {length_stats.get('max_words')}] words
"""

        with open(p, "w", encoding="utf-8") as f:
            f.write(md)

        return md
