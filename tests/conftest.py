"""Pytest test configuration and shared fixtures."""

import sys
from pathlib import Path
import pandas as pd
import pytest

# Ensure src directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture
def sample_categories():
    return [
        "card_arrival",
        "card_linking",
        "exchange_rate",
        "pin_blocked",
        "transfer_timing",
    ]


@pytest.fixture
def sample_train_df(sample_categories):
    data = [
        {"text": "When will my new debit card arrive in the mail?", "category": "card_arrival"},
        {"text": "How long until I receive my card?", "category": "card_arrival"},
        {"text": "My card still hasn't arrived yet.", "category": "card_arrival"},
        {"text": "How do I link a new card to my account?", "category": "card_linking"},
        {"text": "Can I connect my credit card to the app?", "category": "card_linking"},
        {"text": "Link existing card to profile.", "category": "card_linking"},
        {"text": "What is the exchange rate for USD to EUR?", "category": "exchange_rate"},
        {"text": "Foreign exchange rate today.", "category": "exchange_rate"},
        {"text": "Currency conversion rates in the app.", "category": "exchange_rate"},
        {"text": "I blocked my PIN by typing it incorrectly.", "category": "pin_blocked"},
        {"text": "PIN is blocked, please unblock it.", "category": "pin_blocked"},
        {"text": "Incorrect PIN entered 3 times, now locked.", "category": "pin_blocked"},
        {"text": "How long do international transfers take?", "category": "transfer_timing"},
        {"text": "When will the funds reach the recipient account?", "category": "transfer_timing"},
        {"text": "Transfer pending for two business days.", "category": "transfer_timing"},
    ]
    return pd.DataFrame(data)


@pytest.fixture
def sample_test_df(sample_categories):
    data = [
        {"text": "Has my replacement card been dispatched?", "category": "card_arrival"},
        {"text": "Steps to link my bank card.", "category": "card_linking"},
        {"text": "Show me the exchange fee and rate for GBP.", "category": "exchange_rate"},
        {"text": "My PIN got blocked today.", "category": "pin_blocked"},
        {"text": "Why hasn't my money transfer arrived?", "category": "transfer_timing"},
    ]
    return pd.DataFrame(data)
