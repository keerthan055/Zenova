"""Configurable Support Strategy Taxonomy for ZENOVA.

Allows decoupling the strategy classification system from the ESConv dataset
so future datasets (e.g. Hill's Helping Skills, Motivational Interviewing,
or clinical trial taxonomies) can be plugged in without restructuring ZENOVA.
"""
from typing import List, Dict, Any, Optional
from enum import Enum
from zenova.schemas.strategy import SupportStrategy


# Canonical 8 ESConv strategies mapped to standardized SupportStrategy enum
DEFAULT_ESCONV_STRATEGIES: List[str] = [
    SupportStrategy.QUESTION.value,
    SupportStrategy.RESTATEMENT_OR_PARAPHRASING.value,
    SupportStrategy.REFLECTION_OF_FEELINGS.value,
    SupportStrategy.AFFIRMATION_AND_REASSURANCE.value,
    SupportStrategy.SELF_DISCLOSURE.value,
    SupportStrategy.PROVIDING_SUGGESTIONS.value,
    SupportStrategy.INFORMATION.value,
    SupportStrategy.OTHERS.value,
]

DEFAULT_STRATEGY_DESCRIPTIONS: Dict[str, str] = {
    SupportStrategy.QUESTION.value: (
        "Open or closed exploratory questions seeking clarification and deeper user reflection."
    ),
    SupportStrategy.RESTATEMENT_OR_PARAPHRASING.value: (
        "Restating or paraphrasing the user's statements to show active listening and understanding."
    ),
    SupportStrategy.REFLECTION_OF_FEELINGS.value: (
        "Articulating and validating the user's underlying emotional state and feelings."
    ),
    SupportStrategy.AFFIRMATION_AND_REASSURANCE.value: (
        "Offering emotional affirmation, validation of strengths, and gentle encouragement."
    ),
    SupportStrategy.SELF_DISCLOSURE.value: (
        "Sharing relatable perspectives or experiences to build therapeutic rapport."
    ),
    SupportStrategy.PROVIDING_SUGGESTIONS.value: (
        "Collaboratively proposing constructive coping mechanisms, resources, or problem-solving actions."
    ),
    SupportStrategy.INFORMATION.value: (
        "Providing factual, psychoeducational, or structural wellbeing information."
    ),
    SupportStrategy.OTHERS.value: (
        "Conversational chit-chat, greetings, check-ins, or supportive transitions."
    ),
}


class StrategyTaxonomy:
    """Configurable Strategy Taxonomy manager."""

    def __init__(
        self,
        taxonomy_name: str = "ESConv-8",
        strategies: Optional[List[str]] = None,
        descriptions: Optional[Dict[str, str]] = None,
        alias_mapping: Optional[Dict[str, str]] = None
    ):
        self.taxonomy_name = taxonomy_name
        self.strategies = list(strategies or DEFAULT_ESCONV_STRATEGIES)
        self.descriptions = dict(descriptions or DEFAULT_STRATEGY_DESCRIPTIONS)
        self.alias_mapping = dict(alias_mapping or {})

        # Build index lookups
        self.label2id = {s: i for i, s in enumerate(self.strategies)}
        self.id2label = {i: s for i, s in enumerate(self.strategies)}

    @property
    def num_classes(self) -> int:
        return len(self.strategies)

    def normalize_label(self, raw_label: str) -> str:
        """Map raw dataset label or alias to the canonical strategy name."""
        if not raw_label:
            return SupportStrategy.OTHERS.value

        # Check direct match
        if raw_label in self.label2id:
            return raw_label

        # Check aliases
        if raw_label in self.alias_mapping:
            canonical = self.alias_mapping[raw_label]
            if canonical in self.label2id:
                return canonical

        # Case-insensitive search
        lower_map = {s.lower(): s for s in self.strategies}
        if raw_label.lower() in lower_map:
            return lower_map[raw_label.lower()]

        # Default fallback
        return SupportStrategy.OTHERS.value

    def get_description(self, strategy_label: str) -> str:
        return self.descriptions.get(
            strategy_label,
            "Emotional support strategy."
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "taxonomy_name": self.taxonomy_name,
            "num_classes": self.num_classes,
            "strategies": self.strategies,
            "descriptions": self.descriptions
        }

    @classmethod
    def default(cls) -> "StrategyTaxonomy":
        return cls(taxonomy_name="ESConv-8")
