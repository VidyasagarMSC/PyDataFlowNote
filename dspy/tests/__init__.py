"""
Test Suite for DSPy Production Examples

This module contains comprehensive tests for:
- Basic DSPy functionality
- Pydantic integration
- Advanced patterns
- Error handling and edge cases
"""

import os
import sys

# Add src directory to Python path for imports
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Test configuration
TEST_CONFIG = {
    'skip_if_no_api_key': True,
    'max_test_retries': 2,
    'test_timeout': 30,
}

# Common test utilities
def skip_if_no_openai_key():
    """Skip test if OpenAI API key is not available."""
    import pytest
    import os
    if not os.getenv('OPENAI_API_KEY'):
        pytest.skip("OpenAI API key not available")

def get_test_cases():
    """Get standard test cases for validation."""
    return [
        {
            "question": "What is machine learning?",
            "expected_topics": ["algorithms", "data", "learning", "artificial", "intelligence"]
        },
        {
            "question": "How does photosynthesis work?", 
            "expected_topics": ["plants", "sunlight", "energy", "carbon", "oxygen"]
        },
        {
            "question": "What is Python programming?",
            "expected_topics": ["programming", "language", "code", "software", "development"]
        }
    ]
