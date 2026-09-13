"""Unit tests for ZENOVA Mental-Health Symptom and Observational Signal Identification."""
import pytest
import numpy as np
import torch
from pathlib import Path

from zenova.symptoms.taxonomy import SYMPTOM_SIGNALS, TAXONOMY_SPEC, PSYSYM_DISORDERS
from zenova.symptoms.baseline import SymptomTfidfBaseline
from zenova.symptoms.transformer import SymptomTransformerModel
from zenova.symptoms.evaluator import SymptomModelEvaluator
from zenova.symptoms.inference import SymptomInferenceEngine
from zenova.symptoms.analyzer import SymptomTransformerAnalyzer
from zenova.schemas.standard import UserInput, SymptomSeverity


class TestSymptomTaxonomy:
    """Test DSM-5 and PsySym taxonomy specifications."""

    def test_taxonomy_integrity(self):
        assert len(SYMPTOM_SIGNALS) == 10
        assert len(PSYSYM_DISORDERS) == 7
        for sig in SYMPTOM_SIGNALS:
            assert sig in TAXONOMY_SPEC
            spec = TAXONOMY_SPEC[sig]
            assert "psysym_classes" in spec
            assert "dsm5_criteria" in spec
            assert "evidence_cues" in spec
            assert len(spec["evidence_cues"]) > 0

    def test_crisis_signal_flag(self):
        assert TAXONOMY_SPEC["suicidal ideation / crisis thoughts"].get("is_crisis_signal") is True
        assert TAXONOMY_SPEC["sleep disturbance"].get("is_crisis_signal", False) is False


class TestSymptomBaselineModel:
    """Test TF-IDF Multi-Label Baseline Classifier."""

    def test_train_and_predict(self, tmp_path):
        texts = [
            "I haven't been sleeping properly at all",
            "I don't enjoy anything anymore, everything feels empty",
            "Having a severe panic attack and my heart is racing",
            "Just checking in to say hello, nice weather today"
        ]
        labels = [
            ["sleep disturbance"],
            ["loss of interest"],
            ["anxiety / panic"],
            []
        ]

        baseline = SymptomTfidfBaseline(labels=SYMPTOM_SIGNALS)
        baseline.train(texts, labels)

        # Predict proba
        probs = baseline.predict_proba(texts)
        assert probs.shape == (4, len(SYMPTOM_SIGNALS))
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

        # Predict binary
        preds = baseline.predict(texts, threshold=0.3)
        assert preds.shape == (4, len(SYMPTOM_SIGNALS))

        # Save and load round-trip
        baseline.save(str(tmp_path))
        loaded = SymptomTfidfBaseline.load(str(tmp_path))
        assert loaded.is_trained is True
        l_probs = loaded.predict_proba(texts)
        np.testing.assert_allclose(probs, l_probs)


class TestSymptomTransformerArchitecture:
    """Test PyTorch Transformer sequence classification model."""

    def test_forward_pass(self):
        model = SymptomTransformerModel(
            vocab_size=100,
            num_classes=10,
            d_model=64,
            nhead=2,
            num_layers=1,
            dim_feedforward=128,
            max_len=32
        )
        x = torch.randint(0, 100, (4, 32))
        mask = torch.zeros((4, 32), dtype=torch.bool)
        logits = model(x, src_key_padding_mask=mask)

        assert logits.shape == (4, 10)
        probs = torch.sigmoid(logits)
        assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)


class TestSymptomEvaluator:
    """Test Multi-Label Evaluation Metrics."""

    def test_evaluator_metrics(self):
        y_true = np.array([
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1]
        ])
        y_pred = np.array([
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1]
        ])
        eval_res = SymptomModelEvaluator.evaluate(y_true, y_pred, labels=["sig1", "sig2", "sig3"])

        assert eval_res["micro_f1"] == 1.0
        assert eval_res["macro_f1"] == 1.0
        assert eval_res["subset_accuracy"] == 1.0
        assert eval_res["hamming_loss"] == 0.0
        assert "sig1" in eval_res["per_label"]
        assert eval_res["per_label"]["sig1"]["support"] == 2


class TestSymptomInferenceAndSafety:
    """Test inference engine, non-diagnostic constraints, and safety edge cases."""

    @pytest.fixture
    def inference_engine(self):
        engine = SymptomInferenceEngine(model_dir="models/symptoms")
        return engine

    def test_canonical_prompt_benchmark(self, inference_engine):
        """Input: 'I haven't been sleeping properly and I don't enjoy things anymore.'"""
        if inference_engine.model is None:
            pytest.skip("Models not yet trained.")

        text = "I haven't been sleeping properly and I don't enjoy things anymore."
        res = inference_engine.predict(text)

        labels = [s["marker_name"] for s in res["signals"]]
        assert "sleep disturbance" in labels
        assert "loss of interest" in labels
        assert res["is_placeholder"] is False
        assert len(res["signals"]) >= 2

        # Verify evidence spans were extracted
        for s in res["signals"]:
            if s["marker_name"] in ("sleep disturbance", "loss of interest"):
                assert len(s["evidence_spans"]) > 0

    def test_benign_zero_symptom_input(self, inference_engine):
        """Verify benign input produces empty signals list without false positives."""
        if inference_engine.model is None:
            pytest.skip("Models not yet trained.")

        text = "Can you recommend a good Italian restaurant for dinner tonight?"
        res = inference_engine.predict(text)

        assert res["signals"] == []
        assert res["aggregate_severity"] == SymptomSeverity.NONE.value
        assert res["is_crisis_flagged"] is False

    def test_crisis_safety_trigger(self, inference_engine):
        """Verify crisis text flags suicidal ideation with SEVERE rating."""
        if inference_engine.model is None:
            pytest.skip("Models not yet trained.")

        text = "I just want to end my life, I want to hurt myself and die."
        res = inference_engine.predict(text)

        assert res["is_crisis_flagged"] is True
        assert res["aggregate_severity"] == SymptomSeverity.SEVERE.value
        signal_names = [s["marker_name"] for s in res["signals"]]
        assert "suicidal ideation / crisis thoughts" in signal_names

    def test_strict_non_diagnostic_boundary(self, inference_engine):
        """Verify output payload NEVER asserts psychiatric diagnosis."""
        if inference_engine.model is None:
            pytest.skip("Models not yet trained.")

        texts = [
            "I haven't been sleeping properly and I don't enjoy things anymore.",
            "I feel so sad and depressed all the time.",
            "I am completely exhausted and having panic attacks."
        ]

        prohibited_phrases = [
            "you have depression",
            "diagnosed with depression",
            "you are bipolar",
            "psychiatric diagnosis:",
            "clinical diagnosis:"
        ]

        for text in texts:
            res = inference_engine.predict(text)
            res_str = str(res).lower()
            for phrase in prohibited_phrases:
                assert phrase not in res_str, f"Violated non-diagnostic boundary with phrase '{phrase}'"

            # Verify presence of non-diagnostic disclaimer
            assert "not constitute" in res["disclaimer"].lower() or "informational" in res["disclaimer"].lower()


class TestSymptomAnalyzerAdapter:
    """Test integration adapter SymptomTransformerAnalyzer."""

    def test_analyze_user_input(self):
        analyzer = SymptomTransformerAnalyzer(model_dir="models/symptoms")
        if analyzer.engine.model is None:
            pytest.skip("Models not yet trained.")

        u_input = UserInput(
            session_id="sess_symptom_test",
            user_id="user_symptom_test",
            text="I haven't been sleeping properly and I don't enjoy things anymore."
        )
        res = analyzer.analyze(u_input)

        assert res.is_placeholder is False
        assert len(res.signals) >= 2
        assert any(s.marker_name == "sleep disturbance" for s in res.signals)
        assert any(s.marker_name == "loss of interest" for s in res.signals)
