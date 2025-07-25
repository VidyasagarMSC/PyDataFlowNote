"""
DSPy Production Examples Package

A comprehensive collection of DSPy examples demonstrating:
- Basic DSPy concepts and patterns
- Pydantic integration for data validation
- Advanced patterns for production use
- Testing and deployment strategies
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Core modules
from .dspy_setup import setup_dspy, setup_dspy_basic, get_model_config, validate_setup
from .basic_examples import BasicPipeline, RAGPipeline
from .pydantic_integration import AnalysisModule, QueryInput, ValidatedRAGPipeline, AnalysisResult
from .advanced_patterns import ResilientQAPipeline, CachedRAGPipeline, MonitoredDSPyModule

# Utility functions
try:
    from .util import load_sample_data, create_training_examples, simple_metric
except ImportError:
    # Util module might not exist yet
    pass

__all__ = [
    # Setup functions
    'setup_dspy',
    'setup_dspy_basic', 
    'get_model_config',
    'validate_setup',
    
    # Core pipelines
    'BasicPipeline',
    'RAGPipeline',
    
    # Pydantic integration
    'AnalysisModule',
    'QueryInput', 
    'ValidatedRAGPipeline',
    'AnalysisResult',
    
    # Advanced patterns
    'ResilientQAPipeline',
    'CachedRAGPipeline', 
    'MonitoredDSPyModule',
    
    # Utilities (if available)
    'load_sample_data',
    'create_training_examples',
    'simple_metric',
]
