"""REST API routes for ZENOVA Unified Evaluation Framework."""
import json
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query

from zenova.evaluation.schemas import (
    UnifiedEvaluationReport,
    AblationStudyReport,
    HumanEvaluationProtocol,
    ModelMetricsReport,
    SystemPerformanceReport
)
from zenova.evaluation.runner import get_evaluation_runner
from zenova.core.logging import get_logger

logger = get_logger("zenova.api.routes.evaluation")

router = APIRouter(prefix="/api/v1/evaluation", tags=["Evaluation"])


@router.post("/run", response_model=UnifiedEvaluationReport)
async def trigger_evaluation_run() -> UnifiedEvaluationReport:
    """Trigger full empirical evaluation suite across models, generation, system, and ablations."""
    try:
        runner = get_evaluation_runner()
        report = await runner.run_full_evaluation()
        return report
    except Exception as e:
        logger.error(f"Evaluation execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@router.get("/report", response_model=UnifiedEvaluationReport)
async def get_latest_evaluation_report() -> UnifiedEvaluationReport:
    """Retrieve the latest comprehensive evaluation report."""
    runner = get_evaluation_runner()
    json_path = runner.output_dir / "evaluation_report.json"

    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return UnifiedEvaluationReport(**data)
        except Exception as e:
            logger.warning(f"Failed to read cached evaluation report: {e}")

    # If no report on disk yet, run evaluation
    return await runner.run_full_evaluation()


@router.get("/ablation", response_model=AblationStudyReport)
async def get_ablation_study() -> AblationStudyReport:
    """Retrieve the 5-configuration ablation study comparison."""
    runner = get_evaluation_runner()
    return await runner.ablation_runner.run_study()


@router.get("/protocol", response_model=HumanEvaluationProtocol)
def get_human_evaluation_protocol() -> HumanEvaluationProtocol:
    """Retrieve the clinical human-evaluation protocol and 5-point Likert rubrics."""
    runner = get_evaluation_runner()
    return runner.generation_evaluator.get_default_human_protocol()
