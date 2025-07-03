import pytest


@pytest.fixture(scope="session")
def test_model_name():
    """Common test model name for sentence transformers."""
    return "all-MiniLM-L6-v2"
