"""Unit tests for ESConv data ingestion and parsing."""
import pytest
from zenova.strategy.dataset import ESConvDatasetLoader, infer_dialog_stage, STRATEGY_MAP
from zenova.schemas.strategy import SupportStrategy, DialogStage


def test_strategy_map_completeness():
    assert len(STRATEGY_MAP) == 8
    for label, strat in STRATEGY_MAP.items():
        assert isinstance(strat, SupportStrategy)


def test_infer_dialog_stage():
    assert infer_dialog_stage(1, 10) == DialogStage.EXPLORATION
    assert infer_dialog_stage(5, 10) == DialogStage.COMFORTING
    assert infer_dialog_stage(9, 10) == DialogStage.ACTION


def test_esconv_sample_extraction_with_mock_data():
    mock_dialogue = [{
        "situation": "Stressed with exams",
        "emotion_type": "anxiety",
        "problem_type": "academic",
        "dialog": [
            {"speaker": "seeker", "content": "Hello, I feel overwhelmed."},
            {"speaker": "supporter", "annotation": {"strategy": "Question"}, "content": "What is causing the stress?"},
            {"speaker": "seeker", "content": "Upcoming midterms."},
            {"speaker": "supporter", "annotation": {"strategy": "Affirmation and Reassurance"}, "content": "You are capable."}
        ]
    }]

    loader = ESConvDatasetLoader(raw_data_path="dummy_path.json")
    samples = loader.extract_strategy_samples(mock_dialogue)

    assert len(samples) == 2
    assert samples[0].target_strategy == SupportStrategy.QUESTION
    assert samples[0].situation == "Stressed with exams"
    assert len(samples[0].dialog_history) == 1
    assert samples[1].target_strategy == SupportStrategy.AFFIRMATION_AND_REASSURANCE
    assert len(samples[1].dialog_history) == 3


def test_reproducible_splits():
    # Generate dummy samples across strategies
    from zenova.strategy.dataset import ESConvTurnSample
    strategies = list(SupportStrategy)
    dummy_samples = []
    for i in range(100):
        dummy_samples.append(
            ESConvTurnSample(
                dialog_id=i,
                turn_index=1,
                situation="test",
                emotion_type="sad",
                problem_type="work",
                dialog_history=[],
                target_strategy=strategies[i % len(strategies)],
                supporter_response="test response",
                stage=DialogStage.COMFORTING
            )
        )

    loader = ESConvDatasetLoader()
    train1, val1, test1 = loader.create_splits(dummy_samples, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
    train2, val2, test2 = loader.create_splits(dummy_samples, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)

    assert len(train1) == 80
    assert len(val1) == 10
    assert len(test1) == 10

    # Ensure deterministic identical splits
    assert [s.dialog_id for s in train1] == [s.dialog_id for s in train2]
    assert [s.dialog_id for s in val1] == [s.dialog_id for s in val2]
