"""Dataset statistical profiling and distribution computation."""
from typing import List, Dict, Any
from collections import Counter
import numpy as np


class DatasetStatisticsGenerator:
    """Generates class frequencies, length distributions, and missingness metrics."""

    @staticmethod
    def compute_statistics(
        samples: List[Dict[str, Any]],
        label_field: str = "label",
        text_field: str = "text"
    ) -> Dict[str, Any]:
        if not samples:
            return {"total_samples": 0}

        # Label counts (supporting both single-class and multi-label lists)
        flat_labels = []
        for s in samples:
            val = s.get(label_field, "unknown")
            if isinstance(val, (list, set, tuple)):
                if not val:
                    flat_labels.append("__NONE__")
                else:
                    flat_labels.extend([str(item) for item in val])
            else:
                flat_labels.append(str(val))
        label_counts = Counter(flat_labels)
        total = len(samples)

        class_dist = {
            k: {"count": count, "percentage": round((count / total) * 100, 2)}
            for k, count in label_counts.most_common()
        }

        # Text length metrics (words and chars)
        word_lens = []
        char_lens = []
        missing_text = 0

        for s in samples:
            t = s.get(text_field)
            if t and isinstance(t, str):
                words = t.split()
                word_lens.append(len(words))
                char_lens.append(len(t))
            else:
                missing_text += 1

        word_stats = {}
        if word_lens:
            word_stats = {
                "mean_words": round(float(np.mean(word_lens)), 2),
                "std_words": round(float(np.std(word_lens)), 2),
                "min_words": int(np.min(word_lens)),
                "max_words": int(np.max(word_lens)),
                "median_words": float(np.median(word_lens)),
                "p95_words": round(float(np.percentile(word_lens, 95)), 2)
            }

        return {
            "total_samples": total,
            "class_distribution": class_dist,
            "text_length_statistics": word_stats,
            "missing_text_count": missing_text,
            "missing_text_rate": round(missing_text / total, 4)
        }
