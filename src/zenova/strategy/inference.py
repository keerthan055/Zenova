"""Inference engine for emotional support strategy recommendation."""
import os
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from zenova.schemas.strategy import SupportStrategy, DialogStage
from zenova.strategy.taxonomy import StrategyTaxonomy
from zenova.strategy.baseline import StrategyTfidfBaseline
from zenova.strategy.trainer import StrategyTransformerTrainer
from zenova.core.logging import get_logger

logger = get_logger("zenova.strategy.inference")


class StrategyInferenceEngine:
    """Production inference engine for support strategy classification."""

    def __init__(
        self,
        model_dir: str = "models/strategy",
        use_transformer: bool = True,
        device: str = "cpu"
    ):
        self.model_dir = model_dir
        self.use_transformer = use_transformer
        self.device = device
        self.transformer_trainer: Optional[StrategyTransformerTrainer] = None
        self.baseline_model: Optional[StrategyTfidfBaseline] = None
        self.taxonomy: StrategyTaxonomy = StrategyTaxonomy.default()

        self._load_model()

    def _load_model(self) -> None:
        p = Path(self.model_dir)
        ckpt_pt = p / "transformer_checkpoint.pt"
        base_joblib = p / "tfidf_baseline.joblib"

        if self.use_transformer and ckpt_pt.exists():
            try:
                self.transformer_trainer = StrategyTransformerTrainer.load(str(p), device=self.device)
                self.taxonomy = self.transformer_trainer.taxonomy
                logger.info(f"Loaded Strategy Transformer from {ckpt_pt}")
                return
            except Exception as e:
                logger.warning(f"Failed to load transformer checkpoint: {e}. Falling back to baseline...")

        if base_joblib.exists():
            try:
                self.baseline_model = StrategyTfidfBaseline.load(str(p))
                self.taxonomy = self.baseline_model.taxonomy
                self.use_transformer = False
                logger.info(f"Loaded Strategy TF-IDF baseline from {base_joblib}")
                return
            except Exception as e:
                logger.warning(f"Failed to load baseline model: {e}")

        logger.warning(f"No trained strategy model found in {p}. Running in uninitialized state.")

    def format_input_text(
        self,
        text: str,
        emotion: Optional[str] = None,
        situation: Optional[str] = None,
        problem_type: Optional[str] = None
    ) -> str:
        """Format raw text with multimodal context tags."""
        emo_tag = f"[EMOTION: {emotion.strip()}]" if emotion else ""
        sit_tag = f"[SITUATION: {situation.strip()}]" if situation else ""
        prob_tag = f"[PROBLEM: {problem_type.strip()}]" if problem_type else ""
        prefix = " ".join([p for p in [emo_tag, sit_tag, prob_tag] if p])
        if prefix:
            return f"{prefix} {text}"
        return text

    def predict_strategy(
        self,
        text: str,
        emotion: Optional[str] = None,
        situation: Optional[str] = None,
        problem_type: Optional[str] = None,
        stage: Optional[DialogStage] = None
    ) -> Dict[str, Any]:
        """Generate structured strategy recommendation with ranked alternatives."""
        formatted_text = self.format_input_text(
            text=text,
            emotion=emotion,
            situation=situation,
            problem_type=problem_type
        )

        probs_dict: Dict[str, float] = {}

        if self.use_transformer and self.transformer_trainer and self.transformer_trainer.model:
            model = self.transformer_trainer.model
            tok = self.transformer_trainer.tokenizer
            tax = self.transformer_trainer.taxonomy
            input_ids, mask = tok.encode(formatted_text)
            input_ids = input_ids.to(self.device)
            mask = mask.to(self.device)

            with torch.no_grad():
                logits = model(input_ids, src_key_padding_mask=mask)
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

            for i, p in enumerate(probs):
                strat_name = tax.id2label.get(i, SupportStrategy.OTHERS.value)
                probs_dict[strat_name] = round(float(p), 4)

        elif self.baseline_model and self.baseline_model.is_trained:
            probs_list = self.baseline_model.predict_proba([formatted_text])
            probs_dict = probs_list[0] if probs_list else {}
        else:
            # Heuristic default fallback
            probs_dict = {s: 0.125 for s in self.taxonomy.strategies}
            probs_dict[SupportStrategy.QUESTION.value] = 0.5

        # Rank strategies
        ranked = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
        top_strategy_name, top_conf = ranked[0]
        try:
            top_strategy = SupportStrategy(top_strategy_name)
        except ValueError:
            top_strategy = SupportStrategy.OTHERS

        # Build alternatives (excluding top 1)
        alternatives = [
            {"strategy": strat, "confidence": conf, "description": self.taxonomy.get_description(strat)}
            for strat, conf in ranked[1:4]
        ]

        # Stage inference if not provided
        dialog_stage = stage or DialogStage.COMFORTING
        rationale = f"{self.taxonomy.get_description(top_strategy.value)} (Confidence: {top_conf:.1%})"

        return {
            "selected_strategy": top_strategy,
            "confidence": float(top_conf),
            "stage": dialog_stage,
            "probabilities": probs_dict,
            "ranked_strategies": [{"strategy": s, "confidence": c} for s, c in ranked],
            "alternatives": alternatives,
            "rationale": rationale,
            "model_version": "strategy-transformer-v1" if self.use_transformer else "strategy-baseline-v1"
        }
