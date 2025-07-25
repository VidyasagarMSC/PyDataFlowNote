"""
Common DSPy Setup Module

Centralized configuration for DSPy: environment setup, API keys, and model config.
"""

import os
import dspy
from dotenv import load_dotenv
from typing import Optional, Dict, Any

# Load environment variables
load_dotenv()


def setup_dspy(model: str = "openai/gpt-4o-mini",
               api_key: Optional[str] = None,
               max_tokens: int = 1000,
               temperature: float = 0.7,
               **kwargs) -> dspy.LM:
    """Configure DSPy parameters."""
    api_key = api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key is required.")
    lm = dspy.LM(model=model, api_key=api_key, max_tokens=max_tokens, temperature=temperature, **kwargs)
    dspy.settings.configure(lm=lm)
    return lm


def setup_dspy_basic() -> dspy.LM:
    """Quick basic setup for DSPy."""
    return setup_dspy(max_tokens=500, temperature=0.7)


def get_model_config() -> Dict[str, Any]:
    """Retrieve current model configuration."""
    if hasattr(dspy.settings, 'lm') and dspy.settings.lm:
        lm = dspy.settings.lm
        return {
            'model': getattr(lm, 'model', 'unknown'),
            'max_tokens': getattr(lm, 'max_tokens', 'unknown'),
            'temperature': getattr(lm, 'temperature', 'unknown'),
            'configured': True
        }
    return {'configured': False}


def validate_setup() -> bool:
    """Validate DSPy setup."""
    return get_model_config().get('configured', False)


if __name__ == "__main__":
    if not validate_setup():
        setup_dspy_basic()
    print("DSPy setup complete.")
