import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

# The real-tokenizer tests read the project's Hugging Face cache when no HF_HOME is set.
if "HF_HOME" not in os.environ and (ROOT / ".cache" / "huggingface").is_dir():
    os.environ["HF_HOME"] = str(ROOT / ".cache" / "huggingface")


@pytest.fixture
def request_data():
    return {
        "context": "Duplicate charge",
        "question": "Choose a team",
        "options": [{"id": "sales", "text": "Sales"}, {"id": "billing", "text": "Billing"}],
    }
