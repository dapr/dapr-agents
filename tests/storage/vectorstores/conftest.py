import pytest


@pytest.fixture(scope="session")
def test_model_name():
    """Common test model name for sentence transformers."""
    return "all-MiniLM-L6-v2"


@pytest.fixture(scope="session")
def test_collection_name():
    """Common test collection name for vector stores."""
    return "test_collection"
