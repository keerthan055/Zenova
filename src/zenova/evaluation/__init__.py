"""ZENOVA Unified Evaluation Framework with lazy loading to prevent circular import chains."""
from zenova.evaluation.schemas import (
    PerClassMetric,
    EmotionEvaluationResult,
    SymptomEvaluationResult,
    RiskEvaluationResult,
    StrategyEvaluationResult,
    ModelMetricsReport,
    GenerationMetricSample,
    GenerationMetricsReport,
    HumanEvaluationScale,
    HumanEvaluationProtocol,
    SystemPerformanceReport,
    AblationConfigResult,
    AblationStudyReport,
    UnifiedEvaluationReport
)

__all__ = [
    "PerClassMetric",
    "EmotionEvaluationResult",
    "SymptomEvaluationResult",
    "RiskEvaluationResult",
    "StrategyEvaluationResult",
    "ModelMetricsReport",
    "GenerationMetricSample",
    "GenerationMetricsReport",
    "HumanEvaluationScale",
    "HumanEvaluationProtocol",
    "SystemPerformanceReport",
    "AblationConfigResult",
    "AblationStudyReport",
    "UnifiedEvaluationReport",
    "UnifiedModelEvaluator",
    "GenerationEvaluator",
    "SystemPerformanceEvaluator",
    "AblationStudyRunner",
    "UnifiedEvaluationRunner",
    "get_evaluation_runner"
]


def __getattr__(name: str):
    if name == "UnifiedModelEvaluator":
        from zenova.evaluation.model_evaluator import UnifiedModelEvaluator
        return UnifiedModelEvaluator
    elif name == "GenerationEvaluator":
        from zenova.evaluation.generation_evaluator import GenerationEvaluator
        return GenerationEvaluator
    elif name == "SystemPerformanceEvaluator":
        from zenova.evaluation.system_evaluator import SystemPerformanceEvaluator
        return SystemPerformanceEvaluator
    elif name == "AblationStudyRunner":
        from zenova.evaluation.ablation_runner import AblationStudyRunner
        return AblationStudyRunner
    elif name in ("UnifiedEvaluationRunner", "get_evaluation_runner"):
        import zenova.evaluation.runner as r
        return getattr(r, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
