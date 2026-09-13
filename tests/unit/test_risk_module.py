"""Unit tests for ZENOVA Crisis and Risk Detection Module."""
import pytest
import numpy as np
import torch
from pathlib import Path

from zenova.risk.taxonomy import (
    RISK_LEVELS,
    CRISIS_CATEGORIES,
    RISK_TAXONOMY_SPEC,
    ADVERSARIAL_NON_CRISIS_IDIOMS
)
from zenova.risk.baseline import RiskTfidfBaseline
from zenova.risk.transformer import RiskTransformerModel
from zenova.risk.evaluator import RiskModelEvaluator
from zenova.risk.inference import RiskInferenceEngine
from zenova.risk.analyzer import RiskTransformerAnalyzer
from zenova.schemas.standard import UserInput, RiskLevel, CrisisCategory


class TestRiskTaxonomy:
    """Test C-SSRS risk taxonomy specifications."""

    def test_taxonomy_completeness(self):
        assert len(RISK_LEVELS) == 4
        assert len(CRISIS_CATEGORIES) == 5
        for lvl in RISK_LEVELS:
            assert lvl in RISK_TAXONOMY_SPEC
            spec = RISK_TAXONOMY_SPEC[lvl]
            assert "cssrs_level" in spec
            assert "description" in spec
            assert "requires_escalation" in spec
            assert "action" in spec

        assert len(ADVERSARIAL_NON_CRISIS_IDIOMS) >= 10


class TestRiskBaselineModel:
    """Test Cost-Sensitive TF-IDF Baseline."""

    def test_train_and_predict(self, tmp_path):
        texts = [
            "Just having a great time walking in the park today.",
            "I feel so down, I wish I could just disappear.",
            "I want to end my life, planning to commit suicide.",
            "I have the pills right now and I am taking them, goodbye."
        ]
        labels = [
            RiskLevel.LOW.value,
            RiskLevel.MODERATE.value,
            RiskLevel.HIGH.value,
            RiskLevel.CRITICAL.value
        ]

        baseline = RiskTfidfBaseline(labels=labels)
        baseline.train(texts, labels)

        probs = baseline.predict_proba(texts)
        assert probs.shape == (4, 4)
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(4), atol=1e-4)

        preds = baseline.predict(texts)
        assert len(preds) == 4
        assert all(p in labels for p in preds)

        # Save and load round-trip
        baseline.save(str(tmp_path))
        loaded = RiskTfidfBaseline.load(str(tmp_path))
        l_probs = loaded.predict_proba(texts)
        np.testing.assert_allclose(probs, l_probs)


class TestRiskTransformerArchitecture:
    """Test PyTorch Transformer architecture for risk classification."""

    def test_forward_pass(self):
        model = RiskTransformerModel(
            vocab_size=100,
            num_classes=4,
            d_model=64,
            nhead=2,
            num_layers=1,
            dim_feedforward=128,
            max_len=32
        )
        x = torch.randint(0, 100, (4, 32))
        mask = torch.zeros((4, 32), dtype=torch.bool)
        logits = model(x, src_key_padding_mask=mask)

        assert logits.shape == (4, 4)


class TestRiskEvaluator:
    """Test evaluator metrics, specificity, sensitivity, and False-Negative analysis."""

    def test_evaluator_metrics_and_fn_detection(self):
        y_true = ["low", "moderate", "high", "critical"]
        y_pred = ["low", "moderate", "low", "critical"]  # Deliberate False Negative on 'high'
        texts = ["Text 1", "Text 2", "Critical text missed", "Emergency text"]

        eval_res = RiskModelEvaluator.evaluate(y_true, y_pred, texts=texts, labels=RISK_LEVELS)

        assert eval_res["accuracy"] == 0.75
        assert eval_res["safety_sensitivity_high_critical"] == 0.5  # 1 of 2 detected
        assert eval_res["total_false_negatives_count"] == 1
        fn = eval_res["false_negative_cases"][0]
        assert fn["true_risk"] == "high"
        assert fn["predicted_risk"] == "low"
        assert fn["text"] == "Critical text missed"


class TestRiskInferenceAndSafety:
    """Test inference engine, adversarial idiom filtering, and safety triage."""

    @pytest.fixture
    def inference_engine(self):
        engine = RiskInferenceEngine(model_dir="models/risk")
        return engine

    def test_adversarial_non_crisis_idioms(self, inference_engine):
        """Verify colloquial death idioms do NOT trigger false crisis escalations."""
        if inference_engine.model is None:
            pytest.skip("Models not yet trained.")

        idioms = [
            "This traffic is killing me right now!",
            "I'm dying of laughter, that was hilarious!",
            "I'm dead tired after my morning workout.",
            "I would kill for a cup of coffee."
        ]

        for text in idioms:
            res = inference_engine.predict(text)
            assert res["risk_level"] == RiskLevel.LOW.value
            assert res["requires_escalation"] is False

    def test_passive_suicidal_ideation(self, inference_engine):
        """Verify passive death wishes trigger MODERATE without immediate lockout."""
        if inference_engine.model is None:
            pytest.skip("Models not yet trained.")

        text = "I don't want to wake up anymore, everything feels so heavy."
        res = inference_engine.predict(text)

        assert res["risk_level"] == RiskLevel.MODERATE.value
        assert res["requires_escalation"] is False

    def test_active_crisis_trigger(self, inference_engine):
        """Verify active suicidal ideation triggers HIGH risk and escalation."""
        if inference_engine.model is None:
            pytest.skip("Models not yet trained.")

        text = "I want to end my life, I am planning my suicide tonight."
        res = inference_engine.predict(text)

        assert res["risk_level"] == RiskLevel.HIGH.value
        assert res["requires_escalation"] is True
        assert len(res["trigger_cues"]) > 0

    def test_imminent_attempt_emergency(self, inference_engine):
        """Verify acute attempt triggers CRITICAL and emergency override."""
        if inference_engine.model is None:
            pytest.skip("Models not yet trained.")

        text = "I have the pills right next to me and I am taking them now. Goodbye."
        res = inference_engine.predict(text)

        assert res["risk_level"] == RiskLevel.CRITICAL.value
        assert res["requires_escalation"] is True
        assert res["action"] == "immediate_crisis_lifeline_override"

    def test_clinical_non_certainty_disclaimer(self, inference_engine):
        """Verify disclaimer explicitly communicates automated heuristic nature."""
        if inference_engine.model is None:
            pytest.skip("Models not yet trained.")

        res = inference_engine.predict("I'm feeling fine today.")
        assert "not clinical certainty" in res["disclaimer"].lower()


class TestRiskAnalyzerAdapter:
    """Test integration adapter RiskTransformerAnalyzer."""

    def test_analyze_user_input(self):
        analyzer = RiskTransformerAnalyzer(model_dir="models/risk")
        if analyzer.engine.model is None:
            pytest.skip("Models not yet trained.")

        u_input = UserInput(
            session_id="sess_risk_test",
            user_id="user_risk_test",
            text="I want to end my life, I cannot take this anymore."
        )
        res = analyzer.analyze(u_input)

        assert res.is_placeholder is False
        assert res.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
        assert res.requires_immediate_escalation is True
        assert res.is_high_risk is True
