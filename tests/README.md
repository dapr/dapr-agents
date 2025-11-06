# Tests

## Running Tests

### Run all tests
```bash
pytest tests/ -v
```

### Run tests with coverage
```bash
pytest tests/ -v --cov=dapr_agents --cov-report=term-missing --cov-report=html
```

### Run specific test file
```bash
pytest tests/document/embedder/test_sentence.py -v
```

### Run specific test class
```bash
pytest tests/document/embedder/test_sentence.py::TestSentenceTransformerEmbedder -v
```

### Run specific test method
```bash
pytest tests/document/embedder/test_sentence.py::TestSentenceTransformerEmbedder::test_embedder_creation -v
```

## Test Organization

Tests are organized by module/class functionality, 
and we try to mimic the folder structure of the repo.
