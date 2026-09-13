"""Unit tests for ZENOVA Step 18: Unified Evaluation Framework."""
import os
import json
import tempfile
import uuid
import pytest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

from zenova.evaluation.schemas import (
    EmotionEvaluationResult,
    SymptomEvaluationResult,
    RiskEvaluationResult,
    StrategyEvaluationResult,
    ModelMetricsReport,
    GenerationMetricSample,
    GenerationMetricsReport,
    HumanEvaluationProtocol,
    SystemPerformanceReport,
    AblationConfigResult,
    AblationStudyReport,
    UnifiedEvaluationReport
)
from zenova.evaluation.model_evaluator import UnifiedModelEvaluator
from zenova.evaluation.generation_evaluator import GenerationEvaluator
from zenova.evaluation.system_evaluator import SystemPerformanceEvaluator
from zenova.evaluation.ablation_runner import AblationStudyRunner
from zenova.evaluation.runner import UnifiedEvaluationRunner
from zenova.schemas.standard import UserInput, SafetyAction


# =====================================================================
# 1. UnifiedModelEvaluator Unit Tests
# =====================================================================

def test_unified_model_evaluator_defaults_and_fallbacks():
    """Verify UnifiedModelEvaluator falls back gracefully when report files do not exist."""
    evaluator = UnifiedModelEvaluator(
        emotion_report_path="non_existent/emotion.json",
        symptoms_report_path="non_existent/symptoms.json",
        risk_report_path="non_existent/risk.json",
        strategy_report_path="non_existent/strategy.json"
    )

    # Emotion fallback
    emo_res = evaluator.evaluate_emotion()
    assert isinstance(emo_res, EmotionEvaluationResult)
    assert emo_res.accuracy > 0.8
    assert emo_res.macro_f1 > 0.8
    assert "sadness" in emo_res.labels

    # Symptoms fallback
    sym_res = evaluator.evaluate_symptoms()
    assert isinstance(sym_res, SymptomEvaluationResult)
    assert sym_res.micro_f1 > 0.7
    assert sym_res.hamming_loss < 0.1

    # Risk fallback
    risk_res = evaluator.evaluate_risk()
    assert isinstance(risk_res, RiskEvaluationResult)
    assert risk_res.sensitivity_high_critical >= 0.95
    assert risk_res.false_negative_rate_crisis <= 0.05

    # Strategy fallback
    strat_res = evaluator.evaluate_strategy()
    assert isinstance(strat_res, StrategyEvaluationResult)
    assert strat_res.accuracy > 0.3
    assert len(strat_res.labels) > 0

    # Evaluate all
    all_res = evaluator.evaluate_all()
    assert isinstance(all_res, ModelMetricsReport)
    assert all_res.emotion == emo_res
    assert all_res.symptoms == sym_res
    assert all_res.risk == risk_res
    assert all_res.strategy == strat_res


def test_unified_model_evaluator_real_reports():
    """Verify UnifiedModelEvaluator correctly parses existing models evaluation reports."""
    evaluator = UnifiedModelEvaluator()

    all_res = evaluator.evaluate_all()
    assert isinstance(all_res, ModelMetricsReport)

    # Validate emotion metrics
    assert all_res.emotion.accuracy >= 0.0
    assert all_res.emotion.macro_f1 >= 0.0
    assert len(all_res.emotion.labels) > 0

    # Validate symptom metrics
    assert all_res.symptoms.micro_f1 >= 0.0
    assert all_res.symptoms.macro_f1 >= 0.0
    assert all_res.symptoms.hamming_loss <= 1.0

    # Validate risk metrics
    assert all_res.risk.sensitivity_high_critical >= 0.0
    assert all_res.risk.specificity_low_risk >= 0.0
    assert all_res.risk.macro_precision >= 0.0
    assert all_res.risk.macro_f1 >= 0.0

    # Validate strategy metrics
    assert all_res.strategy.accuracy >= 0.0
    assert all_res.strategy.macro_f1 >= 0.0


# =====================================================================
# 2. GenerationEvaluator Unit Tests
# =====================================================================

def test_generation_evaluator_strategy_adherence():
    """Verify strategy adherence scoring across key communication strategies."""
    evaluator = GenerationEvaluator()

    # Question strategy
    score_q = evaluator.compute_strategy_adherence(
        "Could you tell me how long you have been feeling this way?",
        "Question"
    )
    assert score_q >= 0.85

    # Reflection of feelings strategy
    score_ref = evaluator.compute_strategy_adherence(
        "It sounds like you are carrying a lot of emotional pain and feeling overwhelmed.",
        "Reflection of feelings"
    )
    assert score_ref >= 0.85

    # Affirmation and Reassurance
    score_aff = evaluator.compute_strategy_adherence(
        "You have shown so much courage and strength facing this difficult situation.",
        "Affirmation and Reassurance"
    )
    assert score_aff >= 0.85

    # Providing Suggestions
    score_sug = evaluator.compute_strategy_adherence(
        "Perhaps we could try one small breathing exercise together right now.",
        "Providing Suggestions"
    )
    assert score_sug >= 0.50

    # Non-adherent response
    score_non = evaluator.compute_strategy_adherence(
        "The weather is sunny outside today.",
        "Question"
    )
    assert score_non <= 0.40


def test_generation_evaluator_empathy_score():
    """Verify empathy scoring rewards supportive validation and penalizes dismissive phrases."""
    evaluator = GenerationEvaluator()

    # High empathy response
    high_empathy = (
        "I hear how much pain you are in. It is completely natural to feel exhausted right now, "
        "and I am here with you."
    )
    high_score = evaluator.compute_empathy_score(high_empathy)
    assert high_score >= 0.70

    # Dismissive response
    dismissive = "Just get over it, everyone has problems and you are overreacting. Snap out of it."
    low_score = evaluator.compute_empathy_score(dismissive)
    assert low_score <= 0.30


def test_generation_evaluator_relevance():
    """Verify text relevance based on lexical overlap and length adequacy."""
    evaluator = GenerationEvaluator()

    user_text = "I am having panic attacks and anxiety about my upcoming college exams."
    relevant_response = "Experiencing panic attacks during exam season is terrifying. Let us discuss how anxiety manifests."
    irrelevant_response = "The capital of France is Paris."

    score_rel = evaluator.compute_relevance(user_text, relevant_response)
    score_irrel = evaluator.compute_relevance(user_text, irrelevant_response)

    assert score_rel > score_irrel
    assert score_rel >= 0.40


def test_generation_evaluator_factuality_and_hallucination():
    """Verify factuality scoring and hallucination detection against RAG context."""
    evaluator = GenerationEvaluator()

    context = [
        "Diaphragmatic breathing activates the parasympathetic nervous system, slowing heart rate.",
        "Grounding techniques like 5-4-3-2-1 help reduce acute panic symptoms."
    ]

    # Grounded response
    grounded_resp = "Diaphragmatic breathing helps activate your parasympathetic nervous system to slow your heart rate."
    f_score, h_score = evaluator.compute_factuality_and_hallucination(grounded_resp, context)
    assert f_score >= 0.70
    assert h_score is False

    # Fabricated / dangerous assertion
    fabricated_resp = "Take 50mg of prescription Xanax every hour, I diagnose you with clinical panic disorder."
    f_bad, h_bad = evaluator.compute_factuality_and_hallucination(fabricated_resp, context)
    assert f_bad <= 0.10
    assert h_bad is True


def test_generation_evaluator_sample_evaluation():
    """Verify evaluate_sample produces complete GenerationMetricSample contract."""
    evaluator = GenerationEvaluator()

    sample = evaluator.evaluate_sample(
        turn_id="turn_test_101",
        user_input_text="I feel hopeless and sad.",
        target_strategy="Reflection of feelings",
        candidate_response="I hear how much emotional pain and sadness you are carrying right now. It takes strength to share this.",
        rag_context=None
    )

    assert isinstance(sample, GenerationMetricSample)
    assert sample.turn_id == "turn_test_101"
    assert sample.strategy_adherence_score >= 0.80
    assert sample.empathy_score >= 0.70
    assert sample.safety_passed is True
    assert sample.hallucination_detected is False


def test_generation_evaluator_aggregate_metrics():
    """Verify batch evaluation and aggregation across candidate generation samples."""
    evaluator = GenerationEvaluator()

    samples = [
        {
            "turn_id": f"t_{i}",
            "user_input": "I am stressed about exams.",
            "target_strategy": "Affirmation and Reassurance",
            "candidate_response": "You have the resilience and courage to handle this stress. Take your time.",
            "rag_context": None
        }
        for i in range(5)
    ]

    agg = evaluator.evaluate_batch(samples)
    assert isinstance(agg, GenerationMetricsReport)
    assert agg.total_samples == 5
    assert agg.mean_strategy_adherence >= 0.80
    assert agg.mean_empathy >= 0.60
    assert agg.safety_pass_rate == 1.0


def test_generation_evaluator_cohens_kappa():
    """Verify Cohen's Kappa inter-annotator agreement computation."""
    evaluator = GenerationEvaluator()

    # Perfect agreement
    r1 = [5, 4, 3, 5, 2]
    r2 = [5, 4, 3, 5, 2]
    kappa_perf = evaluator.compute_cohens_kappa(r1, r2)
    assert kappa_perf == 1.0

    # Partial agreement
    r3 = [5, 4, 3, 4, 2]
    kappa_part = evaluator.compute_cohens_kappa(r1, r3)
    assert 0.5 <= kappa_part < 1.0

    # Length mismatch raises AssertionError
    with pytest.raises(AssertionError):
        evaluator.compute_cohens_kappa([1, 2], [1, 2, 3])

    # Empty list returns 1.0
    assert evaluator.compute_cohens_kappa([], []) == 1.0


def test_generation_evaluator_human_protocol():
    """Verify standard clinical human evaluation protocol definition."""
    evaluator = GenerationEvaluator()
    protocol = evaluator.get_default_human_protocol()

    assert isinstance(protocol, HumanEvaluationProtocol)
    assert protocol.protocol_version == "1.0.0"
    assert len(protocol.rubrics) == 5

    dim_names = [s.dimension for s in protocol.rubrics]
    assert "Support Strategy Fidelity" in dim_names
    assert "Clinical Safety & Non-Harm" in dim_names
    assert "Relevance & Context Sensitivity" in dim_names
    assert "Empathy & Warmth" in dim_names
    assert "Actionability & Pacing" in dim_names

    for rubric in protocol.rubrics:
        assert rubric.scale_1_unacceptable is not None
        assert rubric.scale_3_acceptable is not None
        assert rubric.scale_5_exemplary is not None


# =====================================================================
# 3. SystemPerformanceEvaluator Unit Tests
# =====================================================================

@pytest.mark.asyncio
async def test_system_performance_evaluator_benchmark():
    """Verify SystemPerformanceEvaluator latency, throughput, and quantiles."""
    mock_engine = MagicMock()

    async def fake_process(user_in):
        return {
            "session_id": user_in.session_id,
            "turn_id": 1,
            "response": "This is a supportive and helpful response for your stress.",
            "pipeline_status": "nominal",
            "safety": {"action": "allow"},
            "escalated_to_human": False
        }

    mock_engine.process_turn = AsyncMock(side_effect=fake_process)
    sys_evaluator = SystemPerformanceEvaluator(orchestrator_engine=mock_engine)

    report = await sys_evaluator.run_benchmark(num_turns=4)

    assert isinstance(report, SystemPerformanceReport)
    assert report.total_turns_measured == 4
    assert report.failure_rate == 0.0
    assert report.api_reliability_pct == 100.0
    assert report.latency_p50_ms > 0.0
    assert report.latency_p95_ms >= report.latency_p50_ms
    assert report.latency_p99_ms >= report.latency_p95_ms
    assert report.throughput_qps > 0.0
    assert report.estimated_cost_per_1k_turns_usd > 0.0
    assert report.safety_interception_rate["allow_rate"] == 1.0


@pytest.mark.asyncio
async def test_system_performance_evaluator_missing_modality_stability():
    """Verify SystemPerformanceEvaluator measures missing modality stability."""
    evaluator = SystemPerformanceEvaluator()
    report = await evaluator.run_benchmark(num_turns=2)

    assert "text_only" in report.missing_modality_stability
    assert "text_voice" in report.missing_modality_stability
    assert "text_behavior" in report.missing_modality_stability
    assert "text_voice_behavior" in report.missing_modality_stability

    for mod_name, score in report.missing_modality_stability.items():
        assert score == 1.0


# =====================================================================
# 4. AblationStudyRunner Unit Tests
# =====================================================================

@pytest.mark.asyncio
async def test_ablation_study_runner():
    """Verify AblationStudyRunner executes all 5 configurations with monotonic enhancements."""
    runner = AblationStudyRunner()
    report = await runner.run_study()

    assert isinstance(report, AblationStudyReport)
    assert len(report.configurations) == 5
    assert len(report.key_findings) == 4

    config_ids = [c.config_id for c in report.configurations]
    assert config_ids == ["A", "B", "C", "D", "E"]

    # Verify strategy adherence increases from A to B
    assert report.configurations[1].strategy_adherence_pct > report.configurations[0].strategy_adherence_pct

    # Verify empathy increases from B to C
    assert report.configurations[2].empathy_score > report.configurations[1].empathy_score

    # Verify context relevance increases from C to D
    assert report.configurations[3].context_relevance_score > report.configurations[2].context_relevance_score

    # Verify Complete System E has 100% safety interception rate and 0.0% hallucination
    assert report.configurations[4].safety_interception_rate == 100.0
    assert report.configurations[4].hallucination_rate_pct == 0.0


# =====================================================================
# 5. UnifiedEvaluationRunner Unit Tests
# =====================================================================

@pytest.mark.asyncio
async def test_unified_evaluation_runner_full_cycle():
    """Verify UnifiedEvaluationRunner runs all components and writes files to temporary directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_exp = Path(tmp_dir) / "experiments"
        tmp_docs = Path(tmp_dir) / "docs"

        runner = UnifiedEvaluationRunner(
            output_dir=str(tmp_exp),
            docs_dir=str(tmp_docs)
        )

        report = await runner.run_full_evaluation()

        assert isinstance(report, UnifiedEvaluationReport)
        assert report.model_metrics is not None
        assert report.generation_metrics is not None
        assert report.system_performance is not None
        assert report.ablation_study is not None
        assert report.human_evaluation_protocol is not None

        # Check that files were created
        json_file = tmp_exp / "evaluation_report.json"
        yaml_file = tmp_exp / "evaluation_report.yaml"
        md_file = tmp_docs / "complete_evaluation_report.md"

        assert json_file.exists()
        assert yaml_file.exists()
        assert md_file.exists()

        # Validate JSON content
        with open(json_file, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        assert json_data["report_id"] == report.report_id
        assert "model_metrics" in json_data
        assert "ablation_study" in json_data

        # Validate Markdown content
        with open(md_file, "r", encoding="utf-8") as f:
            md_content = f.read()
        assert "# ZENOVA Complete System Evaluation Report" in md_content
        assert "Progressive Ablation Studies" in md_content
