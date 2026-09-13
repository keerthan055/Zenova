"""Reproducible dataset splitting with stratification and group isolation."""
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split

from zenova.core.logging import get_logger

logger = get_logger("zenova.data.splitter")


class DatasetSplitter:
    """Generates reproducible train/validation/test splits."""

    @staticmethod
    def split_stratified(
        samples: List[Dict[str, Any]],
        label_field: str = "label",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Standard stratified split based on target label."""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"
        # Convert label values (including lists/sets for multi-label tasks) to 1D hashable signatures
        strat_labels = []
        for s in samples:
            v = s.get(label_field, "unknown")
            if isinstance(v, (list, set, tuple)):
                strat_labels.append("+".join(sorted(str(x) for x in v)) if v else "__EMPTY__")
            else:
                strat_labels.append(str(v))

        # Check if stratification is feasible (all classes need >= 2 samples)
        counts = defaultdict(int)
        for lbl in strat_labels:
            counts[lbl] += 1

        can_stratify = all(c >= 2 for c in counts.values())
        strat_y = strat_labels if can_stratify else None

        temp_ratio = val_ratio + test_ratio
        train_s, temp_s, _, temp_l = train_test_split(
            samples, strat_labels, test_size=temp_ratio, random_state=seed, stratify=strat_y
        )

        test_sub_ratio = test_ratio / temp_ratio
        temp_counts = defaultdict(int)
        for lbl in temp_l:
            temp_counts[lbl] += 1
        can_stratify_temp = all(c >= 2 for c in temp_counts.values())
        strat_temp_y = temp_l if can_stratify_temp else None

        val_s, test_s = train_test_split(
            temp_s, test_size=test_sub_ratio, random_state=seed, stratify=strat_temp_y
        )

        logger.info(f"Stratified split: Train={len(train_s)}, Val={len(val_s)}, Test={len(test_s)} (seed={seed}, stratified={can_stratify})")
        return train_s, val_s, test_s

    @staticmethod
    def split_by_group(
        samples: List[Dict[str, Any]],
        group_field: str = "dialog_id",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Grouped split preventing samples from the same entity/dialogue from leaking across splits."""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"
        groups = list(set(s.get(group_field) for s in samples if s.get(group_field) is not None))
        np.random.seed(seed)
        np.random.shuffle(groups)

        n_groups = len(groups)
        n_train = int(n_groups * train_ratio)
        n_val = int(n_groups * val_ratio)

        train_groups = set(groups[:n_train])
        val_groups = set(groups[n_train:n_train + n_val])
        test_groups = set(groups[n_train + n_val:])

        train_s = [s for s in samples if s.get(group_field) in train_groups]
        val_s = [s for s in samples if s.get(group_field) in val_groups]
        test_s = [s for s in samples if s.get(group_field) in test_groups]

        logger.info(f"Grouped split ({group_field}): Train={len(train_s)}, Val={len(val_s)}, Test={len(test_s)}")
        return train_s, val_s, test_s
