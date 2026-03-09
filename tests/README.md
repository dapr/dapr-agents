<!--
Copyright 2026 The Dapr Authors
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

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
