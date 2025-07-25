# DSPy Production Examples

This repository contains comprehensive examples of using DSPy framework with Pydantic for production AI applications.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API key
   ```

3. **Run basic examples:**
   ```bash
   python src/basic_examples.py
   ```

4. **Run tests:**
   ```bash
   python -m pytest tests/ -v
   ```

## Project Structure

```
src/
├── basic_examples.py         # Core DSPy concepts and basic usage
├── pydantic_integration.py   # Integration with Pydantic for data validation
├── advanced_patterns.py      # Advanced patterns for production use
├── production_examples.py    # Production deployment examples
├── dspy_setup.py            # Common DSPy configuration
└── util.py                  # Utility functions

tests/
└── test_pipelines.py        # Comprehensive test suite

data/
└── sample_data.json         # Sample data for training and testing
```

## Usage Examples

### Basic Question Answering

```python
from basic_examples import BasicPipeline
from src.dspy_setup import setup_dspy_basic

setup_dspy_basic()
pipeline = BasicPipeline()
result = pipeline(question="What is machine learning?")
print(result.answer)
```

### With Pydantic Validation

```python
from src.pydantic_integration import AnalysisModule
from src.dspy_setup import setup_dspy_basic

setup_dspy_basic()
analyzer = AnalysisModule()
result = analyzer("This is a positive message about technology.")
print(f"Sentiment: {result.sentiment}, Confidence: {result.confidence}")
```

### Advanced Patterns

```python
from src.advanced_patterns import ResilientQAPipeline, CachedRAGPipeline
from src.dspy_setup import setup_dspy

setup_dspy()

# Resilient pipeline with retry logic
resilient = ResilientQAPipeline(max_retries=3)
result = resilient(context="Context here", question="Your question?")

# Cached pipeline for performance
cached = CachedRAGPipeline(cache_size=100)
result = cached(question="What is AI?")
print(f"Cache stats: {cached.get_cache_stats()}")
```

## Environment Variables

Create a `.env` file with:

```env
OPENAI_API_KEY=your_api_key_here
MODEL_NAME=gpt-4o-mini
MAX_TOKENS=1000
CACHE_SIZE=1000
ENABLE_MONITORING=true
LOG_LEVEL=INFO
```

## Testing

### Run the full test suite:

```bash
python -m pytest tests/ -v --tb=short
```

### Run specific test categories:

```bash
# Basic functionality
python -m pytest tests/test_pipelines.py::TestBasicPipelines -v

# Pydantic integration
python -m pytest tests/test_pipelines.py::TestPydanticIntegration -v

# Advanced patterns
python -m pytest tests/test_pipelines.py::TestAdvancedPatterns -v

# Async functionality (if available)
python -m pytest tests/test_pipelines.py::TestAsyncPipeline -v
```

## Features

- **Basic DSPy Usage**: Simple Q&A and RAG patterns
- **Pydantic Integration**: Structured data validation and parsing
- **Advanced Patterns**: Resilient pipelines, caching, monitoring
- **Production Ready**: Error handling, metrics, async support
- **Comprehensive Testing**: Unit tests for all components

## Requirements

- Python 3.8+
- OpenAI API key
- See `requirements.txt` for full dependency list

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite
5. Submit a pull request

## License

MIT License - see LICENSE file for details
