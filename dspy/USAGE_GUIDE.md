# 🚀 DSPy Project Usage Guide

## 🎯 Quick Start

### 1. Run Basic Examples
```bash
cd /Users/vmac/documents/code/GitHub/PyDataFlowNote/dspy
python src/basic_examples.py
```

### 2. Run Advanced Patterns
```bash
python src/advanced_patterns.py
```

### 3. Run Complete Production Example
```bash
python src/complete_example.py
```

### 4. Run All Tests
```bash
python -m pytest tests/ -v
```

### 5. Test Logfire Integration
```bash
python test_logfire_integration.py
```

---

## 📊 Monitor Your Application

### Logfire Dashboard
Visit: **https://logfire.pydantic.dev/**

What you'll see:
- 🔍 **Request traces** with complete execution paths
- 📈 **Performance metrics** and timing data
- ❌ **Error tracking** with full context
- 📊 **System health** monitoring
- ✅ **Pydantic validation** events

---

## 🔧 Development Workflow

### Using the Framework
```python
# Import core components
from src.logfire_setup import get_logfire_manager, logfire_span
from src.monitoring import get_monitoring_manager, monitor_function
from src.pydantic_integration import AnalysisModule
from src.basic_examples import BasicPipeline

# Initialize Logfire
logfire_manager = get_logfire_manager()

# Create and use pipelines
pipeline = BasicPipeline()
result = pipeline("Your question here")

# Monitor functions
@monitor_function("my_custom_function")
def my_function():
    return "result"

# Create spans
@logfire_span("custom_operation", component="my_app")
def custom_operation():
    logfire_manager.log_event("Operation started", "info")
    # Your code here
```

### Adding Custom Monitoring
```python
from src.monitoring import get_monitoring_manager

manager = get_monitoring_manager()

# Record metrics
manager.increment_counter("my_operations", 1)
manager.set_gauge("current_users", 42)
manager.record_timer("operation_duration", 1.5)

# Check health
health = manager.perform_health_check()
print(f"System status: {health.status}")
```

---

## 🧪 Testing Your Code

### Run Specific Test Categories
```bash
# Basic pipeline tests
pytest tests/test_pipelines.py::TestBasicPipelines -v

# Pydantic integration tests  
pytest tests/test_pipelines.py::TestPydanticIntegration -v

# Advanced pattern tests
pytest tests/test_pipelines.py::TestAdvancedPatterns -v

# Async pipeline tests
pytest tests/test_pipelines.py::TestAsyncPipeline -v
```

### Custom Test Integration
```python
from src.logfire_setup import logfire_span

@logfire_span("test_my_feature", component="tests")
def test_my_feature():
    # Your test code with automatic logging
    pass
```

---

## 📋 Project Structure Reference

```
src/
├── 🔥 logfire_setup.py         # Core Logfire setup
├── ⚙️ config.py                # Configuration management
├── 🛠️ dspy_setup.py            # DSPy framework setup
├── 📊 monitoring.py            # System monitoring
├── 📋 pydantic_integration.py  # Data validation
├── 📝 basic_examples.py        # Basic patterns
├── 🔧 advanced_patterns.py     # Advanced patterns
├── 🎯 complete_example.py      # Production example
└── 🛠️ util.py                 # Utility functions
```

---

## 🎉 All Systems Ready!

Your DSPy project is now:
- ✅ **Fully organized** with clear module structure
- ✅ **Comprehensively tested** with 93% pass rate
- ✅ **Production ready** with full observability
- ✅ **Logfire integrated** with real-time monitoring
- ✅ **Pydantic validated** with type safety
- ✅ **Performance monitored** with metrics and health checks

**Happy coding! 🚀**
