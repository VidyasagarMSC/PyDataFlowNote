"""
Utility Functions for DSPy Project

Common utility functions for text processing, validation, and formatting.
"""

import json
import time
import hashlib
from typing import Any, Dict, List
from datetime import datetime

# Handle Logfire import
try:
    try:
        from .logfire_setup import get_logfire_manager, logfire_span
    except ImportError:
        from logfire_setup import get_logfire_manager, logfire_span
    logfire_manager = get_logfire_manager()
    LOGFIRE_AVAILABLE = True
except ImportError:
    logfire_manager = None
    LOGFIRE_AVAILABLE = False
    def logfire_span(name, **kwargs):
        def decorator(func):
            return func
        return decorator


def create_request_id(prefix: str = "req") -> str:
    """Create unique request ID."""
    return f"{prefix}_{int(time.time() * 1000)}"


def normalize_text(text: str) -> str:
    """Normalize text for processing."""
    return text.strip().lower() if text else ""


def create_hash(data: str) -> str:
    """Create MD5 hash for data."""
    return hashlib.md5(data.encode()).hexdigest()


def format_timestamp(timestamp: float = None) -> str:
    """Format timestamp for logging."""
    return datetime.fromtimestamp(timestamp or time.time()).isoformat()


def safe_json_serialize(data: Any) -> str:
    """Safely serialize data to JSON."""
    try:
        return json.dumps(data, indent=2, default=str)
    except Exception as e:
        return f'{{"error": "Serialization failed: {str(e)}"}}'


def validate_question(question: str, min_len: int = 5, max_len: int = 500) -> bool:
    """Validate question meets requirements."""
    if not question or not isinstance(question, str):
        return False
    return min_len <= len(question.strip()) <= max_len


def extract_key_terms(text: str, max_terms: int = 10) -> List[str]:
    """Extract key terms from text."""
    if not text:
        return []
    
    words = text.lower().split()
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    terms = [word.strip('.,!?;:') for word in words 
             if word not in stop_words and len(word) > 2]
    
    return list(dict.fromkeys(terms))[:max_terms]


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity score between texts."""
    if not text1 or not text2:
        return 0.0
    
    terms1 = set(extract_key_terms(text1))
    terms2 = set(extract_key_terms(text2))
    
    if not terms1 and not terms2:
        return 1.0
    if not terms1 or not terms2:
        return 0.0
    
    intersection = len(terms1.intersection(terms2))
    union = len(terms1.union(terms2))
    
    return intersection / union if union > 0 else 0.0


def format_response(success: bool, data: Any = None, error: str = None) -> Dict[str, Any]:
    """Format standard API response."""
    response = {'success': success, 'timestamp': time.time()}
    
    if success and data is not None:
        response['data'] = data
    elif not success and error:
        response['error'] = error
    
    return response


# Sample data loading functions
@logfire_span("load_sample_data", component="util")
def load_sample_data(file_path: str = "data/sample_data.json") -> Dict[str, Any]:
    """Load sample data from JSON file with Logfire monitoring."""
    import os
    
    if LOGFIRE_AVAILABLE:
        logfire_manager.log_event("Loading sample data", "info", file_path=file_path)
    
    # Try different paths if file not found
    possible_paths = [
        file_path,
        os.path.join(os.path.dirname(__file__), '..', file_path),
        os.path.join(os.getcwd(), file_path)
    ]
    
    for path in possible_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if LOGFIRE_AVAILABLE:
                    logfire_manager.log_event("Sample data loaded successfully", "info", 
                                            path=path, 
                                            data_size=len(json.dumps(data)))
                return data
        except FileNotFoundError:
            continue
        except Exception as e:
            if LOGFIRE_AVAILABLE:
                logfire_manager.log_error(e, f"Error loading sample data from {path}")
            print(f"Error loading sample data from {path}: {e}")
            continue
    
    # Return default data if file not found
    if LOGFIRE_AVAILABLE:
        logfire_manager.log_event("Using default sample data", "warning", file_path=file_path)
    return get_default_sample_data()


def get_default_sample_data() -> Dict[str, Any]:
    """Get default sample data if file not available."""
    return {
        "training_examples": [
            {
                "question": "What is artificial intelligence?",
                "context": "AI refers to simulation of human intelligence in machines.",
                "answer": "AI is the simulation of human intelligence in machines."
            },
            {
                "question": "What is machine learning?",
                "context": "ML is a subset of AI that learns from data.",
                "answer": "ML is algorithms that learn patterns from data."
            }
        ],
        "test_examples": [
            {
                "question": "What is deep learning?",
                "expected_topics": ["neural networks", "layers", "learning"]
            }
        ],
        "contexts": {
            "ai_context": "AI encompasses various technologies including ML and NLP.",
            "ml_context": "ML includes supervised, unsupervised, and reinforcement learning."
        }
    }


@logfire_span("create_training_examples", component="util")
def create_training_examples(sample_data: Dict[str, Any] = None):
    """Create DSPy training examples from sample data with Logfire monitoring."""
    import dspy
    
    if sample_data is None:
        sample_data = load_sample_data()
    
    if LOGFIRE_AVAILABLE:
        logfire_manager.log_event("Creating training examples", "info", 
                                 has_sample_data=sample_data is not None)
    
    examples = []
    for item in sample_data.get('training_examples', []):
        example = dspy.Example(
            question=item['question'],
            answer=item['answer']
        ).with_inputs('question')
        examples.append(example)
    
    # Add optimization examples if available
    for item in sample_data.get('optimization_examples', []):
        example = dspy.Example(
            question=item['question'],
            answer=item['answer']
        ).with_inputs('question')
        examples.append(example)
    
    if LOGFIRE_AVAILABLE:
        logfire_manager.log_event("Training examples created", "info", 
                                 example_count=len(examples))
    
    return examples


@logfire_span("simple_metric", component="util")
def simple_metric(example, pred, trace=None) -> float:
    """Simple evaluation metric for DSPy optimization with Logfire monitoring."""
    if not hasattr(pred, 'answer') or not pred.answer:
        if LOGFIRE_AVAILABLE:
            logfire_manager.log_event("Metric evaluation: no answer", "debug", score=0.0)
        return 0.0
    
    answer = pred.answer.strip()
    if len(answer) < 10:
        if LOGFIRE_AVAILABLE:
            logfire_manager.log_event("Metric evaluation: answer too short", "debug", 
                                     answer_length=len(answer), score=0.0)
        return 0.0
    
    # Check if answer contains any relevant terms from question
    question_terms = set(extract_key_terms(example.question))
    answer_terms = set(extract_key_terms(answer))
    
    if question_terms.intersection(answer_terms):
        score = 1.0
    else:
        score = 0.5  # Partial credit for reasonable length
    
    if LOGFIRE_AVAILABLE:
        logfire_manager.log_event("Metric evaluation completed", "debug", 
                                 score=score,
                                 answer_length=len(answer),
                                 has_common_terms=bool(question_terms.intersection(answer_terms)))
    
    return score


def get_test_cases(sample_data: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Get test cases from sample data."""
    if sample_data is None:
        sample_data = load_sample_data()
    
    return sample_data.get('test_examples', [])
