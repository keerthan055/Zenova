"""Unit tests for dataset cross-split leakage detection."""
from zenova.data.leakage import DatasetLeakageDetector


def test_group_leakage_detection():
    train_s = [{"dialog_id": "dlg_1", "text": "a"}, {"dialog_id": "dlg_2", "text": "b"}]
    val_s = [{"dialog_id": "dlg_3", "text": "c"}]
    test_clean = [{"dialog_id": "dlg_4", "text": "d"}]
    test_leaky = [{"dialog_id": "dlg_1", "text": "e"}]  # dlg_1 leaked from train!

    passed, report = DatasetLeakageDetector.check_group_leakage(train_s, val_s, test_clean, "dialog_id")
    assert passed is True
    assert report["has_leakage"] is False

    failed, leak_report = DatasetLeakageDetector.check_group_leakage(train_s, val_s, test_leaky, "dialog_id")
    assert failed is False
    assert leak_report["has_leakage"] is True
    assert "dlg_1" in leak_report["train_test_overlap_sample"]


def test_exact_text_leakage_detection():
    train_s = [{"text": "Feeling anxious today"}, {"text": "Good morning"}]
    test_clean = [{"text": "Completely new thought"}]
    test_leaky = [{"text": "Feeling anxious today"}]

    passed, _ = DatasetLeakageDetector.check_exact_text_leakage(train_s, test_clean, "text")
    assert passed is True

    failed, rep = DatasetLeakageDetector.check_exact_text_leakage(train_s, test_leaky, "text")
    assert failed is False
    assert rep["has_leakage"] is True
