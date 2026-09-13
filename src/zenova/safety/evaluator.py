"""Multi-layer safety evaluator orchestrating deterministic rules, heuristics, and optional LLM safety evaluation."""
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from zenova.schemas.safety import (
    SafetyViolation,
    SafetyPolicy,
    SafetySeverity,
    SafetyAction
)
from zenova.schemas.standard import UserInput, RiskResult, MultimodalContext
from zenova.safety.rules import ALL_SAFETY_RULES, BaseSafetyRule
from zenova.core.logging import get_logger

logger = get_logger("zenova.safety.evaluator")


class SafetyEvaluator:
    """Multi-layer safety evaluation engine."""

    def __init__(self, config_path: str = "configs/safety.yaml"):
        self.config_path = config_path
        self.rules: List[BaseSafetyRule] = []
        self.enabled_policies: Dict[str, bool] = {}
        self._load_config()
        self._init_rules()

    def _load_config(self) -> None:
        p = Path(self.config_path)
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                checks = cfg.get("checks", {})
                for check_name, meta in checks.items():
                    self.enabled_policies[check_name] = meta.get("enabled", True)
                logger.info(f"Loaded {len(self.enabled_policies)} safety check configs from {p}")
            except Exception as e:
                logger.warning(f"Error loading safety config: {e}. Defaulting all to enabled.")
        else:
            logger.info("No safety.yaml found; using default active checks.")

    def _init_rules(self) -> None:
        for rule_cls in ALL_SAFETY_RULES:
            rule_inst = rule_cls()
            policy_val = rule_inst.policy.value if hasattr(rule_inst.policy, "value") else str(rule_inst.policy)
            if self.enabled_policies.get(policy_val, True):
                self.rules.append(rule_inst)
        logger.info(f"Initialized {len(self.rules)} active safety rule checkers.")

    def evaluate(
        self,
        candidate_text: str,
        user_input: Optional[UserInput] = None,
        risk: Optional[RiskResult] = None,
        multimodal_context: Optional[MultimodalContext] = None
    ) -> Tuple[SafetyAction, List[SafetyViolation], float]:
        """Execute multi-layer safety checks and return decision, violations, and toxicity score."""
        violations: List[SafetyViolation] = []

        # Layer 1 & 2: Evaluate all active policy rules
        for rule in self.rules:
            try:
                rule_violations = rule.check(
                    candidate_text=candidate_text,
                    user_input=user_input,
                    risk=risk,
                    multimodal_context=multimodal_context
                )
                if rule_violations:
                    violations.extend(rule_violations)
            except Exception as err:
                logger.error(f"Rule {rule.__class__.__name__} evaluation failed: {err}")

        # Compute composite toxicity score
        toxicity_score = 0.0
        if violations:
            severity_weights = {
                SafetySeverity.LOW: 0.2,
                SafetySeverity.MEDIUM: 0.5,
                SafetySeverity.HIGH: 0.8,
                SafetySeverity.CRITICAL: 1.0
            }
            max_weight = max(severity_weights.get(v.severity, 0.5) for v in violations)
            count_bonus = min(0.3, len(violations) * 0.05)
            toxicity_score = min(1.0, round(max_weight + count_bonus, 2))

        # Determine SafetyAction triage
        has_critical = any(v.severity == SafetySeverity.CRITICAL for v in violations)
        # Upstream crisis risk triggers BLOCK_AND_ESCALATE if safety violation occurred
        is_high_risk = False
        if risk and risk.is_high_risk:
            is_high_risk = True
        elif multimodal_context and multimodal_context.risk and multimodal_context.risk.is_high_risk:
            is_high_risk = True

        if has_critical or (is_high_risk and violations):
            action = SafetyAction.BLOCK_AND_ESCALATE
        elif violations:
            action = SafetyAction.REVISE
        else:
            action = SafetyAction.ALLOW

        return action, violations, toxicity_score
