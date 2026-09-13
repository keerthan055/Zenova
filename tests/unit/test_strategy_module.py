"""Comprehensive unit tests for the support strategy module."""
import pytest
import torch
import shutil
from pathlib import Path

from zenova.schemas.strategy import SupportStrategy, DialogStage
from zenova.schemas.standard import (
    UserInput,
    EmotionResult,
    SymptomResult,
    EmotionCategory,
    SymptomSeverity,
    StrategyResult
)
from zenova.strategy.taxonomy import StrategyTaxonomy, DEFAULT_ESCONV_STRATEGIES
from zenova.strategy.dataset import ESConvTurnSample, ESConvDatasetLoader
from zenova.strategy.baseline import StrategyTfidfBaseline
from zenova.strategy.transformer import (
    StrategyTransformerModel,
    StrategyTokenizer,
    StrategyDataset
)
from zenova.strategy.evaluator import StrategyEvaluator
from zenova.strategy.planner import ESConvStrategyPlanner


def test_strategy_taxonomy_canonical():
    tax = StrategyTaxonomy.default()
    assert tax.num_classes == 8
    assert len(tax.strategies) == 8
    assert "Question" in tax.strategies
    assert "Reflection of feelings" in tax.strategies

    # Test normalization
    assert tax.normalize_label("question") == "Question"
    assert tax.normalize_label("Providing Suggestions") == "Providing Suggestions"
    assert tax.normalize_label("unknown_label") == "Others"

    # Descriptions
    desc = tax.get_description("Question")
    assert "clarification" in desc.lower() or "exploratory" in desc.lower()


def test_zero_dialogue_leakage_assertion():
    """Verify that create_splits strictly partitions dialogues without leaking turns."""
    loader = ESConvDatasetLoader()
    # Create 30 dialogues, each having 4 turns
    samples = []
    for d_id in range(30):
        for t_idx in range(4):
            samples.append(
                ESConvTurnSample(
                    dialog_id=d_id,
                    turn_index=t_idx,
                    situation="mock situation",
                    emotion_type="anxiety",
                    problem_type="academic",
                    dialog_history=[{"speaker": "seeker", "content": "I feel lost."}],
                    target_strategy=SupportStrategy.QUESTION if t_idx % 2 == 0 else SupportStrategy.COMFORTING if hasattr(SupportStrategy, "COMFORTING") else SupportStrategy.OTHERS,
                    supporter_response="How can I help?",
                    stage=DialogStage.EXPLORATION,
                    current_user_message="I feel lost."
                )
            )

    train, val, test = loader.create_splits(samples, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=123)

    train_dialogs = set(s.dialog_id for s in train)
    val_dialogs = set(s.dialog_id for s in val)
    test_dialogs = set(s.dialog_id for s in test)

    # Absolute zero intersection assertions
    assert len(train_dialogs & val_dialogs) == 0
    assert len(train_dialogs & test_dialogs) == 0
    assert len(val_dialogs & test_dialogs) == 0
    assert len(train_dialogs) + len(val_dialogs) + len(test_dialogs) == 30


def test_format_input_ablation_conditions():
    sample = ESConvTurnSample(
        dialog_id=1,
        turn_index=2,
        situation="Fighting with roommate",
        emotion_type="anger",
        problem_type="relationship",
        dialog_history=[
            {"speaker": "seeker", "content": "I am furious."},
            {"speaker": "supporter", "content": "Tell me what happened."}
        ],
        target_strategy=SupportStrategy.REFLECTION_OF_FEELINGS,
        supporter_response="It sounds like you feel unheard.",
        stage=DialogStage.COMFORTING,
        current_user_message="He left the kitchen dirty again."
    )

    cond_a = sample.format_input("A")
    assert "[EMOTION:" not in cond_a
    assert "Seeker: I am furious." in cond_a

    cond_b = sample.format_input("B")
    assert "[EMOTION: anger]" in cond_b
    assert "[SITUATION:" not in cond_b

    cond_c = sample.format_input("C")
    assert "[EMOTION: anger]" in cond_c
    assert "[SITUATION: Fighting with roommate]" in cond_c
    assert "[PROBLEM: relationship]" in cond_c


def test_baseline_tfidf_train_predict_save_load(tmp_path):
    texts = [
        "What seems to be troubling you the most today?",
        "I hear how painful this loss is for you.",
        "You have overcome challenges before, and you can do this.",
        "Have you considered trying a relaxation technique?",
        "I felt the exact same way during college exams.",
        "Here are the crisis helpline numbers and counseling hours.",
        "Hello, good morning! How are you doing?",
        "So what you mean is that work is demanding too much time."
    ]
    labels = [
        SupportStrategy.QUESTION.value,
        SupportStrategy.REFLECTION_OF_FEELINGS.value,
        SupportStrategy.AFFIRMATION_AND_REASSURANCE.value,
        SupportStrategy.PROVIDING_SUGGESTIONS.value,
        SupportStrategy.SELF_DISCLOSURE.value,
        SupportStrategy.INFORMATION.value,
        SupportStrategy.OTHERS.value,
        SupportStrategy.RESTATEMENT_OR_PARAPHRASING.value,
    ]

    baseline = StrategyTfidfBaseline()
    baseline.train(texts, labels)

    preds = baseline.predict(["Can you tell me more about it?"])
    assert len(preds) == 1
    assert preds[0] in baseline.labels

    probs = baseline.predict_proba(["I understand completely."])
    assert len(probs) == 1
    assert len(probs[0]) == 8
    assert abs(sum(probs[0].values()) - 1.0) < 0.05

    # Save and Load
    save_dir = tmp_path / "baseline_model"
    baseline.save(str(save_dir))
    loaded = StrategyTfidfBaseline.load(str(save_dir))
    assert loaded.is_trained is True
    assert loaded.labels == baseline.labels


def test_transformer_model_forward():
    vocab_size = 50
    num_classes = 8
    model = StrategyTransformerModel(
        vocab_size=vocab_size,
        num_classes=num_classes,
        d_model=32,
        nhead=2,
        num_layers=1,
        dim_feedforward=64
    )
    input_ids = torch.randint(0, vocab_size, (4, 16))
    mask = torch.zeros((4, 16), dtype=torch.bool)
    mask[:, 12:] = True

    logits = model(input_ids, src_key_padding_mask=mask)
    assert logits.shape == (4, num_classes)


def test_tokenizer_and_dataset():
    tok = StrategyTokenizer(max_vocab=100, max_len=20)
    texts = [
        "[EMOTION: sadness] Seeker: I feel down. Supporter: Why is that?",
        "Seeker: Exams are coming up soon."
    ]
    tok.build_vocab(texts)
    assert len(tok.vocab) > 5

    input_ids, mask = tok.encode("Seeker: I feel down.")
    assert input_ids.shape == (1, 20)
    assert mask.shape == (1, 20)


def test_evaluator_metrics_and_ece():
    evaluator = StrategyEvaluator()
    y_true = [SupportStrategy.QUESTION.value, SupportStrategy.REFLECTION_OF_FEELINGS.value]
    y_pred = [SupportStrategy.QUESTION.value, SupportStrategy.OTHERS.value]
    y_prob = [
        {SupportStrategy.QUESTION.value: 0.9, SupportStrategy.REFLECTION_OF_FEELINGS.value: 0.1},
        {SupportStrategy.OTHERS.value: 0.6, SupportStrategy.REFLECTION_OF_FEELINGS.value: 0.4}
    ]

    metrics = evaluator.evaluate(y_true, y_pred, y_prob)
    assert metrics["accuracy"] == 0.5
    assert "macro_f1" in metrics
    assert "confusion_matrix" in metrics
    assert "ece" in metrics
    assert 0.0 <= metrics["ece"] <= 1.0


def test_strategy_planner_with_mock_signals():
    planner = ESConvStrategyPlanner(model_dir="models/strategy", use_transformer=False)
    user_in = UserInput(
        session_id="test-session",
        user_id="test-user",
        text="I am not sure what to do next with my studies.",
        metadata={"situation": "Exam failure", "problem_type": "academic"}
    )
    emo_res = EmotionResult(
        primary_emotion=EmotionCategory.ANXIETY,
        confidence=0.88,
        valence=-0.6,
        arousal=0.7,
        dominance=-0.4
    )
    sym_res = SymptomResult(
        signals=[],
        clinical_disclaimer="Non-diagnostic."
    )

    result = planner.predict_strategy(user_in, emo_res, sym_res)
    assert isinstance(result, StrategyResult)
    assert result.is_placeholder is False
    assert result.selected_strategy in SupportStrategy
    assert 0.0 <= result.confidence <= 1.0
    assert result.stage in DialogStage
    assert len(result.alternatives) > 0
    assert result.rationale is not None
