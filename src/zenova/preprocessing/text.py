"""Text preprocessing utilities for ZENOVA."""
import re
import unicodedata
from typing import List


def clean_text(text: str) -> str:
    """Normalize whitespace and unicode characters in input text."""
    if not text:
        return ""
    # Normalize unicode (NFKC)
    text = unicodedata.normalize("NFKC", text)
    # Replace carriage returns and excessive whitespace
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def truncate_turns(dialogue_history: List[str], max_turns: int = 5) -> List[str]:
    """Retain the most recent N conversational turns for context window control."""
    if not dialogue_history:
        return []
    return dialogue_history[-max_turns:]
