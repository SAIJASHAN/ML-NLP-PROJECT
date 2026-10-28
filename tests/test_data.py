"""Unit tests for data ingestion, cleaning, and validation."""

import pytest
from complaint_ops.data.ingest import inspect_dataset, load_banking77
from complaint_ops.data.preprocessing import clean_text, validate_input_query


def test_clean_text_basic():
    assert clean_text("  hello   world  ") == "hello world"
    assert clean_text("Check balance\n\tplease! ") == "Check balance please!"
    assert clean_text(None) == ""
    assert clean_text(12345) == ""


def test_clean_text_control_characters():
    dirty = "my card\x00\x08 is blocked"
    cleaned = clean_text(dirty)
    assert cleaned == "my card is blocked"


def test_validate_input_query():
    valid, res = validate_input_query("Where is my card?")
    assert valid is True
    assert res == "Where is my card?"

    too_short, err = validate_input_query("a", min_chars=3)
    assert too_short is False
    assert "too short" in err

    empty, err = validate_input_query("")
    assert empty is False
    assert "non-empty" in err


def test_inspect_dataset(sample_train_df, sample_test_df, sample_categories):
    summary = inspect_dataset(sample_train_df, sample_test_df, sample_categories)
    assert summary["train_samples"] == 15
    assert summary["test_samples"] == 5
    assert summary["num_categories"] == 5
    assert summary["train_null_text"] == 0
    assert summary["test_null_text"] == 0
    assert len(summary["unseen_test_categories"]) == 0
