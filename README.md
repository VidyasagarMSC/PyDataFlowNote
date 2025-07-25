# 🐍 PyDataFlowNote

**A comprehensive collection of Python data science, AI, and observability examples**

> ⭐ **Star & Watch this repository for updates!**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](License)
[![Colab](https://img.shields.io/badge/Colab-Ready-yellow.svg)](https://colab.research.google.com/)

---

## 🎯 What is PyDataFlowNote?

This repository combines elements from **Python (Py)**, data processing libraries like **NumPy**, **SciPy**, **scikit-learn**, **TensorFlow**, **PyTorch**, and the integration of **Jupyter notebooks**. It reflects the flow of data and computations in a notebook environment, covering:

- 📊 **Data Processing & Analysis** - Modern libraries beyond pandas
- 🔢 **Linear Algebra & Mathematics** - SymPy, NumPy, and computational mathematics
- 🤖 **AI Framework Integration** - DSPy with production-ready patterns
- 🔍 **Observability & Monitoring** - AgentOps, Langfuse, and Logfire
- ✅ **Data Validation** - Pydantic models and type safety
- 📓 **Interactive Learning** - All examples available as Colab notebooks

---

## 📂 Project Structure

```
PyDataFlowNote/
├── 📊 data_processing/              # Modern data processing libraries
│   └── 📓 libraries.ipynb          # Benchmarking beyond pandas
├── 🔢 linear-algebra/              # Mathematical computations
│   └── 📓 norm-1D.ipynb            # Vector norms in Python libraries
├── 🧮 sympy/                       # Symbolic mathematics
│   └── 📓 intro.ipynb              # SymPy introduction and examples
├── ✅ pydantic/                     # Data validation and type safety
│   └── 📓 intro.ipynb              # Pydantic models and validation
├── 🔍 observability/               # AI agent monitoring and tracking
│   ├── 🤖 agentops_example.py      # AgentOps integration
│   ├── 📊 langfuse_example.py      # Langfuse observability
│   ├── 👥 multiagent.py            # Multi-agent monitoring
│   └── 📋 requirements.txt         # Dependencies
└── 🚀 dspy/                        # Production DSPy framework
    ├── 📂 src/                     # Core DSPy modules
    ├── 🧪 tests/                   # Comprehensive test suite
    ├── 📊 data/                    # Sample datasets
    └── 📖 README.md                # Detailed DSPy guide
```

---

## 📚 Complete Code Samples & Notebooks

### 📊 Data Processing & Analysis

| 🎯 Topic | 📓 Jupyter Notebook | 🚀 Colab | 📄 Article | 📝 Description |
|----------|---------------------|-----------|-----------|----------------|
| **Modern Data Libraries** | [libraries.ipynb](data_processing/libraries.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/VidyasagarMSC/PyDataFlowNote/blob/main/data_processing/libraries.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | [DZone Article](https://dzone.com/articles/modern-data-processing-libraries-beyond-pandas) | Benchmarking and comparison of modern data processing libraries beyond pandas |

### 🔢 Linear Algebra & Mathematics

| 🎯 Topic | 📓 Jupyter Notebook | 🚀 Colab | 📄 Article | 📝 Description |
|----------|---------------------|-----------|-----------|----------------|
| **Vector Norms** | [norm-1D.ipynb](linear-algebra/norm-1D.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/VidyasagarMSC/PyDataFlowNote/blob/main/linear-algebra/norm-1D.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | [DZone Article](https://dzone.com/articles/norm-of-a-one-dimensional-tensor-in-python-libraries) | Computing norms of 1D tensors across different Python libraries |
| **Symbolic Math** | [intro.ipynb](sympy/intro.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/VidyasagarMSC/PyDataFlowNote/blob/main/sympy/intro.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | - | Introduction to SymPy for symbolic mathematics and equation solving |

### ✅ Data Validation & Type Safety

| 🎯 Topic | 📓 Jupyter Notebook | 🚀 Colab | 📄 Article | 📝 Description |
|----------|---------------------|-----------|-----------|----------------|
| **Pydantic Models** | [intro.ipynb](pydantic/intro.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/VidyasagarMSC/PyDataFlowNote/blob/main/pydantic/intro.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | - | Complete guide to Pydantic data validation, nested models, and custom validators |

### 🔍 AI Observability & Monitoring

| 🎯 Topic | 💻 Python Script | 📚 Documentation | 📝 Description |
|----------|------------------|------------------|----------------|
| **AgentOps Integration** | [agentops_example.py](observability/agentops_example.py) | [Setup Guide](observability/README.md) | Complete AgentOps integration for AI agent monitoring |
| **Langfuse Observability** | [langfuse_example.py](observability/langfuse_example.py) | [Setup Guide](observability/README.md) | Langfuse integration for LLM application observability |
| **Multi-Agent Systems** | [multiagent.py](observability/multiagent.py) | [Setup Guide](observability/README.md) | Monitoring and tracking for complex multi-agent workflows |

### 🚀 Production AI Framework (DSPy)

| 🎯 Component | 💻 Python Module | 📚 Documentation | 📝 Description |
|--------------|------------------|------------------|----------------|
| **Basic Examples** | [basic_examples.py](dspy/src/basic_examples.py) | [DSPy README](dspy/README.md) | Core DSPy patterns and basic usage examples |
| **Advanced Patterns** | [advanced_patterns.py](dspy/src/advanced_patterns.py) | [DSPy README](dspy/README.md) | Production patterns with caching, retry logic, and monitoring |
| **Pydantic Integration** | [pydantic_integration.py](dspy/src/pydantic_integration.py) | [DSPy README](dspy/README.md) | Type-safe DSPy with Pydantic validation |
| **Complete Production** | [complete_example.py](dspy/src/complete_example.py) | [DSPy README](dspy/README.md) | Full production example with Logfire observability |
| **Configuration** | [config.py](dspy/src/config.py) | [DSPy README](dspy/README.md) | Environment-based configuration with Pydantic |
| **Monitoring** | [monitoring.py](dspy/src/monitoring.py) | [DSPy README](dspy/README.md) | System health checks and performance monitoring |
| **Logfire Setup** | [logfire_setup.py](dspy/src/logfire_setup.py) | [DSPy README](dspy/README.md) | Complete Logfire observability integration |
| **Test Suite** | [test_pipelines.py](dspy/tests/test_pipelines.py) | [DSPy README](dspy/README.md) | Comprehensive testing (13/14 tests passing) |

---

## ⚡ Quick Start Guides

### 🔥 For Notebooks (Colab Ready)
1. **Click any Colab badge** above to run notebooks instantly
2. **Or clone locally:**
   ```bash
   git clone https://github.com/VidyasagarMSC/PyDataFlowNote.git
   cd PyDataFlowNote
   jupyter lab
   ```

### 🤖 For AI Observability
```bash
cd observability
pip install -r requirements.txt
cp .env.template .env
# Edit .env with your API keys
python agentops_example.py
```

### 🚀 For DSPy Production Framework
```bash
cd dspy
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
python src/basic_examples.py
```

---

## 🎓 Learning Paths

### 🆕 **Beginners**: Start with Fundamentals
1. 📊 [Data Processing Libraries](data_processing/libraries.ipynb) - Learn modern alternatives to pandas
2. 🔢 [Linear Algebra Basics](linear-algebra/norm-1D.ipynb) - Understanding vector operations
3. 🧮 [Symbolic Mathematics](sympy/intro.ipynb) - Solve equations symbolically
4. ✅ [Data Validation](pydantic/intro.ipynb) - Build type-safe applications

### 🚀 **Intermediate**: Production AI Systems
1. 🔍 [AI Observability](observability/) - Monitor your AI applications
2. 🤖 [DSPy Basics](dspy/src/basic_examples.py) - Learn DSPy fundamentals
3. 🏭 [Advanced Patterns](dspy/src/advanced_patterns.py) - Production-ready patterns
4. 📊 [Full Integration](dspy/src/complete_example.py) - Complete observability pipeline

### 👨‍💻 **Advanced**: Enterprise Development
1. 🧪 [Testing Strategies](dspy/tests/) - Comprehensive test coverage
2. ⚙️ [Configuration Management](dspy/src/config.py) - Environment-based setup
3. 📈 [Performance Monitoring](dspy/src/monitoring.py) - System health tracking
4. 🔥 [Observability Integration](dspy/src/logfire_setup.py) - Production monitoring

---

## 🛠️ Technologies Covered

### 📊 **Data Science Stack**
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation (plus alternatives)
- **SciPy** - Scientific computing
- **SymPy** - Symbolic mathematics
- **Polars** - Fast DataFrames
- **DuckDB** - In-process SQL OLAP database

### 🤖 **AI & Machine Learning**
- **DSPy** - Programming—not prompting—language models
- **OpenAI** - GPT models integration
- **Anthropic** - Claude models integration
- **Pydantic** - Data validation for AI applications

### 🔍 **Observability & Monitoring**
- **Logfire** - Pydantic's observability platform
- **AgentOps** - AI agent monitoring
- **Langfuse** - LLM application observability
- **Structlog** - Structured logging
- **psutil** - System monitoring

### 🧪 **Development Tools**
- **Pytest** - Testing framework
- **Black** - Code formatting
- **MyPy** - Type checking
- **Jupyter** - Interactive development
- **Google Colab** - Cloud notebooks

---

## 📈 Project Status

| 📂 Component | 🟢 Status | 📊 Coverage | 🚀 Colab Ready |
|--------------|------------|-------------|----------------|
| **Data Processing** | ✅ Complete | 100% | ✅ Yes |
| **Linear Algebra** | ✅ Complete | 100% | ✅ Yes |
| **SymPy Examples** | ✅ Complete | 100% | ✅ Yes |
| **Pydantic Guide** | ✅ Complete | 100% | ✅ Yes |
| **Observability** | ✅ Complete | 3 Examples | ❌ Local Only |
| **DSPy Framework** | ✅ Complete | 93% Tests | ❌ Local Only |

---

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-addition`
3. **Add your notebook or code** with comprehensive documentation
4. **Ensure Colab compatibility** for notebooks
5. **Add entry to this README** in the appropriate table
6. **Submit a pull request** with detailed description

### 📝 Contribution Guidelines
- **Notebooks**: Must be Colab-compatible with clear explanations
- **Code**: Include type hints and comprehensive docstrings
- **Documentation**: Update README tables with new content
- **Testing**: Add tests for new functionality
- **Examples**: Provide practical, real-world examples

---

## 📚 External Resources

### 📄 **Published Articles**
- [Norm of 1D Tensors in Python Libraries](https://dzone.com/articles/norm-of-a-one-dimensional-tensor-in-python-libraries) - DZone
- [Modern Data Processing Libraries Beyond Pandas](https://dzone.com/articles/modern-data-processing-libraries-beyond-pandas) - DZone

### 🔗 **Useful Links**
- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [Pydantic Documentation](https://pydantic.dev/)
- [Logfire Platform](https://logfire.pydantic.dev/)
- [Google Colab](https://colab.research.google.com/)

---

## 📄 License

**MIT License** - see [License](License) file for details

---

## 🆘 Support

- **🐛 Issues**: Create GitHub issues for bugs and feature requests
- **💡 Discussions**: Use GitHub Discussions for questions
- **📧 Contact**: Open an issue for direct communication
- **⭐ Star**: Show support by starring the repository

---

**🎯 Status: ACTIVELY MAINTAINED** ✅

*This repository is continuously updated with new examples, patterns, and best practices for Python data science and AI development.*
