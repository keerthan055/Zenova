"""Unit tests for ZENOVA preprocessing."""
from zenova.preprocessing.text import clean_text, truncate_turns


def test_clean_text():
    raw = "  Hello   world! \r\n\r\n\r\n  This is    a test.  "
    cleaned = clean_text(raw)
    assert "Hello world!" in cleaned
    assert "This is a test." in cleaned
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_truncate_turns():
    history = [f"turn_{i}" for i in range(10)]
    truncated = truncate_turns(history, max_turns=3)
    assert len(truncated) == 3
    assert truncated == ["turn_7", "turn_8", "turn_9"]
    assert truncate_turns([]) == []
