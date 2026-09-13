"""Unit and integration tests for Step 3: Text Emotion Detection."""
import pytest
from fastapi.testclient import TestClient
from zenova.api.app import app
from zenova.emotion.inference import EmotionInferenceEngine
from zenova.emotion.analyzer import EmotionTransformerAnalyzer
from zenova.emotion.baseline import EmotionTfidfBaseline
from zenova.emotion.evaluator import EmotionModelEvaluator
from zenova.schemas.standard import UserInput, EmotionCategory

client = TestClient(app)


def test_emotion_inference_engine_happy():
    engine = EmotionInferenceEngine(model_dir="models/emotion")
    res = engine.predict("I am so grateful and happy with this amazing result!")

    assert "emotions" in res
    assert len(res["emotions"]) == 7
    assert res["is_placeholder"] is False
    assert res["model_version"] == "transformer-v1.0.0"
    assert res["primary_emotion"] == "joy"
    assert res["valence"] > 0.0


def test_emotion_inference_engine_fear_anxiety():
    engine = EmotionInferenceEngine(model_dir="models/emotion")
    res = engine.predict("I am terrified, scared, and in fear.")

    assert res["is_placeholder"] is False
    assert res["primary_emotion"] == "fear"
    assert res["valence"] < 0.0


def test_emotion_transformer_analyzer_adapter():
    analyzer = EmotionTransformerAnalyzer(model_dir="models/emotion")
    inp = UserInput(
        session_id="test_sess_emo",
        user_id="test_user_emo",
        text="I feel completely heartbroken and miserable."
    )
    result = analyzer.analyze(inp)

    assert result.is_placeholder is False
    assert result.module_version == "transformer-v1.0.0"
    assert result.primary_emotion in [EmotionCategory.SADNESS, EmotionCategory.ANXIETY]
    assert result.confidence > 0.0
    assert len(result.probabilities) == 7


def test_emotion_evaluator():
    y_true = ["joy", "sadness", "anger"]
    y_pred = ["joy", "sadness", "joy"]
    labels = ["joy", "sadness", "anger"]

    metrics = EmotionModelEvaluator.evaluate(y_true, y_pred, labels=labels)
    assert metrics["accuracy"] == round(2 / 3, 4)
    assert "macro_f1" in metrics
    assert "confusion_matrix" in metrics
    assert "per_class" in metrics


def test_api_emotion_analyze_endpoint():
    payload = {"text": "I feel really stressed and anxious about my deadlines."}
    resp = client.post("/api/v1/emotion/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert "emotions" in data
    assert "primary_emotion" in data
    assert "model_version" in data
    assert len(data["emotions"]) > 0
    assert "confidence" in data["emotions"][0]


def test_api_conversation_uses_real_emotion_model():
    payload = {
        "session_id": "api_emo_live_sess",
        "user_id": "api_emo_live_user",
        "text": "I am so happy and grateful today!"
    }
    resp = client.post("/api/v1/conversation/turn", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["emotion"]["is_placeholder"] is False
    assert data["emotion"]["primary_emotion"] == "joy"
    assert data["emotion"]["module_version"] == "transformer-v1.0.0"
