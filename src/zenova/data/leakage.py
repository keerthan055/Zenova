"""Dataset leakage detection and cross-split audit engine."""
from typing import List, Dict, Any, Set, Tuple


class DatasetLeakageDetector:
    """Audits splits for target, text, and group leakage."""

    @staticmethod
    def check_group_leakage(
        train_samples: List[Dict[str, Any]],
        val_samples: List[Dict[str, Any]],
        test_samples: List[Dict[str, Any]],
        group_field: str = "dialog_id"
    ) -> Tuple[bool, Dict[str, Any]]:
        """Ensure no group ID (dialog_id, user_id) appears across multiple splits."""
        train_groups: Set[Any] = {s[group_field] for s in train_samples if group_field in s and s[group_field] is not None}
        val_groups: Set[Any] = {s[group_field] for s in val_samples if group_field in s and s[group_field] is not None}
        test_groups: Set[Any] = {s[group_field] for s in test_samples if group_field in s and s[group_field] is not None}

        train_val_overlap = list(train_groups.intersection(val_groups))
        train_test_overlap = list(train_groups.intersection(test_groups))
        val_test_overlap = list(val_groups.intersection(test_groups))

        has_leakage = bool(train_val_overlap or train_test_overlap or val_test_overlap)
        report = {
            "group_field": group_field,
            "has_leakage": has_leakage,
            "train_val_overlap_count": len(train_val_overlap),
            "train_test_overlap_count": len(train_test_overlap),
            "val_test_overlap_count": len(val_test_overlap),
            "train_val_overlap_sample": train_val_overlap[:5],
            "train_test_overlap_sample": train_test_overlap[:5]
        }
        return not has_leakage, report

    @staticmethod
    def check_exact_text_leakage(
        train_samples: List[Dict[str, Any]],
        test_samples: List[Dict[str, Any]],
        text_field: str = "text"
    ) -> Tuple[bool, Dict[str, Any]]:
        """Ensure no exact input texts from test set are present in train set."""
        train_texts = {s[text_field].strip().lower() for s in train_samples if text_field in s and s[text_field]}
        test_texts = {s[text_field].strip().lower() for s in test_samples if text_field in s and s[text_field]}

        overlap = train_texts.intersection(test_texts)
        has_leakage = len(overlap) > 0

        report = {
            "text_field": text_field,
            "has_leakage": has_leakage,
            "overlap_count": len(overlap),
            "overlap_sample": list(overlap)[:5]
        }
        return not has_leakage, report
