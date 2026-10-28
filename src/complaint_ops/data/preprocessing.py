"""Text preprocessing and query validation helpers."""

import re
from typing import List, Tuple, Union
import pandas as pd

ValidationResult = Tuple[bool, str]


def clean_text(text: str) -> str:
    """Normalize whitespace and strip unprintable characters."""
    if not isinstance(text, str):
        return ""
    # Strip invisible/control characters except standard whitespace
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess_series(series: Union[pd.Series, List[str]]) -> List[str]:
    """Clean a sequence of text strings."""
    return [clean_text(str(x)) for x in series]


def validate_input_query(
    query: str, min_chars: int = 2, max_chars: int = 2000
) -> ValidationResult:
    """Validate query content and constraints."""
    if not query or not isinstance(query, str):
        return False, "Query must be a non-empty string."
    cleaned = clean_text(query)
    if len(cleaned) < min_chars:
        return False, f"Query is too short (min {min_chars} characters required)."
    if len(cleaned) > max_chars:
        return False, f"Query exceeds maximum character limit of {max_chars}."
    return True, cleaned

