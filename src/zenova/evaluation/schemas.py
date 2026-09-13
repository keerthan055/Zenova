"""Typed data contracts and schemas for ZENOVA Complete Evaluation Framework."""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime, timezone


class PerClassMetric(BaseModel):
    """Precision, recall, F1, and support for a single class or label."""
    precision: float
    recall: float
    f1: float
    support: int


class EmotionEvaluationResult(BaseModel):
    """Evaluation metrics for emotion classification."""
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class: Dict[str, PerClassMetric] = Field(default_factory=dict)
    confusion_matrix: List[List[int]] = Field(default_factory=list)
    labels: List[str] = Field(default_factory=list)


class SymptomEvaluationResult(BaseModel):
    """Multi-label evaluation metrics for symptom signal identification."""
    micro_f1: float
    micro_precision: float
    micro_recall: float
    macro_f1: float
    macro_precision: float
    macro_recall: float
    subset_accuracy: float
    hamming_loss: float
    per_label: Dict[str, PerClassMetric] = Field(default_factory=dict)
    labels: List[str] = Field(default_factory=list)


class RiskEvaluationResult(BaseModel):
    """Clinical safety evaluation metrics for crisis risk prediction."""
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    sensitivity_high_critical: float
    specificity_low_risk: float
    false_negative_rate_crisis: float
    total_false_negatives_count: int
    false_negative_cases: List[Dict[str, Any]] = Field(default_factory=list)
    confusion_matrix: Dict[str, Any] = Field(default_factory=dict)
    per_class: Dict[str, PerClassMetric] = Field(default_factory=dict)


class StrategyEvaluationResult(BaseModel):
    """Evaluation metrics for support strategy planning."""
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class: Dict[str, PerClassMetric] = Field(default_factory=dict)
    confusion_matrix: List[List[int]] = Field(default_factory=list)
    labels: List[str] = Field(default_factory=list)
    ece: Optional[float] = None


class ModelMetricsReport(BaseModel):
    """Aggregated offline model performance across all specialized classifiers."""
    emotion: EmotionEvaluationResult
    symptoms: SymptomEvaluationResult
    risk: RiskEvaluationResult
    strategy: StrategyEvaluationResult


class GenerationMetricSample(BaseModel):
    """Per-turn evaluation sample for conversational generation."""
    turn_id: str
    user_input: str
    target_strategy: str
    candidate_response: str
    strategy_adherence_score: float
    factuality_score: float
    relevance_score: float
    empathy_score: float
    safety_passed: bool
    hallucination_detected: bool
    violations: List[str] = Field(default_factory=list)


class GenerationMetricsReport(BaseModel):
    """Automated evaluation metrics across generated conversational responses."""
    total_samples: int
    mean_strategy_adherence: float
    mean_factuality: float
    mean_relevance: float
    mean_empathy: float
    safety_pass_rate: float
    hallucination_rate: float
    samples: List[GenerationMetricSample] = Field(default_factory=list)


class HumanEvaluationScale(BaseModel):
    """Definition of a 5-point Likert scale dimension for clinical human evaluation."""
    dimension: str
    definition: str
    scale_1_unacceptable: str
    scale_3_acceptable: str
    scale_5_exemplary: str


class HumanEvaluationProtocol(BaseModel):
    """Standardized clinical human-evaluation protocol with guidelines and agreement metrics."""
    protocol_version: str = "1.0.0"
    target_evaluator_role: str = "Licensed Clinician or Clinical Supervisor"
    sample_selection_strategy: str = "Stratified sampling across risk tiers, strategies, and discrepancy flags"
    rubrics: List[HumanEvaluationScale] = Field(default_factory=list)
    agreement_metric: str = "Cohen's Weighted Kappa (2 raters) / Fleiss' Kappa (>2 raters)"
    quality_threshold: str = "Kappa >= 0.70 (substantial agreement), minimum 3 independent clinician raters"


class SystemPerformanceReport(BaseModel):
    """Live system latency, throughput, reliability, and modality robustness metrics."""
    total_turns_measured: int
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_mean_ms: float
    latency_min_ms: float
    latency_max_ms: float
    throughput_qps: float
    failure_rate: float
    api_reliability_pct: float
    estimated_cost_per_1k_turns_usd: float
    missing_modality_stability: Dict[str, float] = Field(default_factory=dict)
    safety_interception_rate: Dict[str, float] = Field(default_factory=dict)


class AblationConfigResult(BaseModel):
    """Empirical performance metrics for a specific system ablation configuration."""
    config_id: str
    config_name: str
    description: str
    strategy_adherence_pct: float
    empathy_score: float
    safety_interception_rate: float
    context_relevance_score: float
    hallucination_rate_pct: float
    avg_latency_ms: float


class AblationStudyReport(BaseModel):
    """Comparative analysis across the 5 ablation regimes (A through E)."""
    configurations: List[AblationConfigResult] = Field(default_factory=list)
    key_findings: List[str] = Field(default_factory=list)


class UnifiedEvaluationReport(BaseModel):
    """Comprehensive master evaluation report combining all evaluation dimensions."""
    report_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    platform_version: str
    status: str = "completed"
    model_metrics: ModelMetricsReport
    generation_metrics: GenerationMetricsReport
    human_evaluation_protocol: HumanEvaluationProtocol
    system_performance: SystemPerformanceReport
    ablation_study: AblationStudyReport
