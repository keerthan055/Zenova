"""ESConv dataset parser, turn extraction, and conversation-level splitting."""
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
from sklearn.model_selection import train_test_split

from zenova.schemas.strategy import SupportStrategy, DialogStage
from zenova.strategy.taxonomy import StrategyTaxonomy
from zenova.core.logging import get_logger

logger = get_logger("zenova.strategy.dataset")

STRATEGY_MAP: Dict[str, SupportStrategy] = {
    "Question": SupportStrategy.QUESTION,
    "Restatement or Paraphrasing": SupportStrategy.RESTATEMENT_OR_PARAPHRASING,
    "Reflection of feelings": SupportStrategy.REFLECTION_OF_FEELINGS,
    "Affirmation and Reassurance": SupportStrategy.AFFIRMATION_AND_REASSURANCE,
    "Self-disclosure": SupportStrategy.SELF_DISCLOSURE,
    "Providing Suggestions": SupportStrategy.PROVIDING_SUGGESTIONS,
    "Information": SupportStrategy.INFORMATION,
    "Others": SupportStrategy.OTHERS,
}


class ESConvTurnSample:
    """Represents an individual supporter turn strategy prediction sample."""

    def __init__(
        self,
        dialog_id: int,
        turn_index: int,
        situation: str,
        emotion_type: str,
        problem_type: str,
        dialog_history: List[Dict[str, str]],
        target_strategy: SupportStrategy,
        supporter_response: str,
        stage: DialogStage,
        current_user_message: str = ""
    ):
        self.dialog_id = dialog_id
        self.turn_index = turn_index
        self.situation = situation
        self.emotion_type = emotion_type
        self.problem_type = problem_type
        self.dialog_history = dialog_history
        if not current_user_message and dialog_history:
            for turn in reversed(dialog_history):
                if turn.get("speaker") == "seeker":
                    current_user_message = turn.get("content", "")
        self.current_user_message = current_user_message
        self.target_strategy = target_strategy
        self.supporter_response = supporter_response
        self.stage = stage

    def format_conversation_text(self, max_history_turns: int = 6) -> str:
        """Format recent dialogue history into sequential speaker text."""
        recent = self.dialog_history[-max_history_turns:]
        lines = []
        for t in recent:
            spk = "Seeker" if t.get("speaker") in ("seeker", "user") else "Supporter"
            cnt = t.get("content", "").strip()
            if cnt:
                lines.append(f"{spk}: {cnt}")
        return " \n ".join(lines)

    def format_input(self, condition: str = "C", max_history_turns: int = 6) -> str:
        """Format input text for controlled experiments.

        Condition A: Conversation only
        Condition B: Conversation + Emotion
        Condition C: Conversation + Emotion + Situation/Problem context
        """
        conv_text = self.format_conversation_text(max_history_turns=max_history_turns)

        if condition.upper() == "A":
            return conv_text
        elif condition.upper() == "B":
            emo_tag = f"[EMOTION: {self.emotion_type.strip()}]" if self.emotion_type else "[EMOTION: neutral]"
            return f"{emo_tag} {conv_text}"
        else:  # Condition C (Full Context)
            emo_tag = f"[EMOTION: {self.emotion_type.strip()}]" if self.emotion_type else "[EMOTION: neutral]"
            sit_tag = f"[SITUATION: {self.situation.strip()}]" if self.situation else ""
            prob_tag = f"[PROBLEM: {self.problem_type.strip()}]" if self.problem_type else ""
            prefix = " ".join([p for p in [emo_tag, sit_tag, prob_tag] if p])
            return f"{prefix} {conv_text}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dialog_id": self.dialog_id,
            "turn_index": self.turn_index,
            "situation": self.situation,
            "emotion_type": self.emotion_type,
            "problem_type": self.problem_type,
            "dialog_history": self.dialog_history,
            "current_user_message": self.current_user_message,
            "target_strategy": self.target_strategy.value,
            "supporter_response": self.supporter_response,
            "stage": self.stage.value,
        }


def infer_dialog_stage(turn_idx: int, total_turns: int) -> DialogStage:
    """Heuristic stage determination based on Hill's helping model: Exploration -> Comforting -> Action."""
    if total_turns <= 0:
        return DialogStage.COMFORTING
    progress = turn_idx / float(total_turns)
    if progress < 0.33:
        return DialogStage.EXPLORATION
    elif progress < 0.70:
        return DialogStage.COMFORTING
    else:
        return DialogStage.ACTION


class ESConvDatasetLoader:
    """Loads, validates, and prepares ESConv dialogue turns for strategy modeling."""

    def __init__(
        self,
        raw_data_path: str = "data/raw/esconv/ESConv.json",
        taxonomy: Optional[StrategyTaxonomy] = None
    ):
        self.raw_data_path = raw_data_path
        self.taxonomy = taxonomy or StrategyTaxonomy.default()

    def load_raw(self) -> List[Dict[str, Any]]:
        """Read the raw ESConv JSON array."""
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"ESConv dataset not found at {self.raw_data_path}")
        with open(self.raw_data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} raw ESConv dialogues from {self.raw_data_path}")
        return data

    def extract_strategy_samples(self, dialogues: Optional[List[Dict[str, Any]]] = None) -> List[ESConvTurnSample]:
        """Extract turns where supporter acts, recording context and true strategy."""
        if dialogues is None:
            dialogues = self.load_raw()

        samples: List[ESConvTurnSample] = []
        for d_idx, item in enumerate(dialogues):
            situation = item.get("situation", "")
            emotion_type = item.get("emotion_type", "")
            problem_type = item.get("problem_type", "")
            dialog = item.get("dialog", [])
            total_turns = len(dialog)

            history: List[Dict[str, str]] = []
            last_seeker_msg = ""

            for t_idx, turn in enumerate(dialog):
                speaker = turn.get("speaker")
                content = turn.get("content", "").strip()
                annotation = turn.get("annotation", {})
                raw_strategy = annotation.get("strategy")

                if speaker == "seeker":
                    last_seeker_msg = content

                if speaker == "supporter" and raw_strategy:
                    # Strategy normalization using configurable taxonomy
                    norm_label = self.taxonomy.normalize_label(raw_strategy)
                    norm_strategy = SupportStrategy(norm_label)
                    stage = infer_dialog_stage(t_idx, total_turns)

                    sample = ESConvTurnSample(
                        dialog_id=d_idx,
                        turn_index=t_idx,
                        situation=situation,
                        emotion_type=emotion_type,
                        problem_type=problem_type,
                        dialog_history=list(history),
                        current_user_message=last_seeker_msg,
                        target_strategy=norm_strategy,
                        supporter_response=content,
                        stage=stage
                    )
                    samples.append(sample)

                # Append to history
                history.append({"speaker": speaker, "content": content})

        logger.info(f"Extracted {len(samples)} supporter turn strategy samples across {len(dialogues)} dialogues.")
        return samples

    def create_splits(
        self,
        samples: List[ESConvTurnSample],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42
    ) -> Tuple[List[ESConvTurnSample], List[ESConvTurnSample], List[ESConvTurnSample]]:
        """Create reproducible conversation-level train/val/test splits.

        CRITICAL REQUIREMENT:
        Separation is strictly at the CONVERSATION level (by dialog_id).
        Zero turns from the same conversation leak across train, val, and test splits.
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"

        # 1. Identify all unique conversation IDs
        unique_dialog_ids = sorted(list(set(s.dialog_id for s in samples)))
        total_dialogs = len(unique_dialog_ids)

        # 2. Split conversation IDs
        temp_ratio = val_ratio + test_ratio
        train_ids, temp_ids = train_test_split(
            unique_dialog_ids,
            test_size=temp_ratio,
            random_state=seed
        )

        test_sub_ratio = test_ratio / temp_ratio
        val_ids, test_ids = train_test_split(
            temp_ids,
            test_size=test_sub_ratio,
            random_state=seed
        )

        train_id_set = set(train_ids)
        val_id_set = set(val_ids)
        test_id_set = set(test_ids)

        # 3. Assert zero leakage across conversation boundaries
        assert len(train_id_set & val_id_set) == 0, "Leakage detected between train and val dialogues!"
        assert len(train_id_set & test_id_set) == 0, "Leakage detected between train and test dialogues!"
        assert len(val_id_set & test_id_set) == 0, "Leakage detected between val and test dialogues!"

        # 4. Filter turn samples by conversation split
        train_samples = [s for s in samples if s.dialog_id in train_id_set]
        val_samples = [s for s in samples if s.dialog_id in val_id_set]
        test_samples = [s for s in samples if s.dialog_id in test_id_set]

        logger.info(
            f"Conversation-level splits generated (seed={seed}, total_dialogs={total_dialogs}):\n"
            f"  Train: {len(train_ids)} dialogues ({len(train_samples)} turns)\n"
            f"  Val:   {len(val_ids)} dialogues ({len(val_samples)} turns)\n"
            f"  Test:  {len(test_ids)} dialogues ({len(test_samples)} turns)"
        )
        return train_samples, val_samples, test_samples
