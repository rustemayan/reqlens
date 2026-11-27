import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

EXAMPLE = ROOT / "examples" / "requirements.md"


@pytest.fixture
def example_path():
    return EXAMPLE


@pytest.fixture
def example_doc():
    from reqlens.loaders import load_document

    return load_document(EXAMPLE)
