"""Unit tests for stratified and grouped dataset splitting."""
import pytest
from zenova.data.splitter import DatasetSplitter


def test_stratified_split():
    samples = [{"id": i, "label": f"class_{i % 3}", "text": f"sample_{i}"} for i in range(120)]
    train, val, test = DatasetSplitter.split_stratified(
        samples, label_field="label", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42
    )
    assert len(train) == 96
    assert len(val) == 12
    assert len(test) == 12


def test_grouped_split_prevents_overlap():
    samples = []
    # 20 dialogues with 5 turns each
    for d in range(20):
        for t in range(5):
            samples.append({"dialog_id": f"dlg_{d}", "turn": t, "text": f"text_{d}_{t}"})

    train, val, test = DatasetSplitter.split_by_group(
        samples, group_field="dialog_id", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42
    )
    train_dlgs = {s["dialog_id"] for s in train}
    val_dlgs = {s["dialog_id"] for s in val}
    test_dlgs = {s["dialog_id"] for s in test}

    assert len(train_dlgs.intersection(val_dlgs)) == 0
    assert len(train_dlgs.intersection(test_dlgs)) == 0
    assert len(val_dlgs.intersection(test_dlgs)) == 0
