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


@logfire_span("setup_dspy", component="dspy_setup")
def setup_dspy(model: str = "openai/gpt-4o-mini",
               api_key: Optional[str] = None,
               max_tokens: int = 1000,
               temperature: float = 0.7,
               **kwargs) -> dspy.LM:
    """Configure DSPy parameters with Logfire monitoring."""
    if LOGFIRE_AVAILABLE:
        logfire_manager.log_event("Starting DSPy setup", "info", model=model, max_tokens=max_tokens, temperature=temperature)
    
    api_key = api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        if LOGFIRE_AVAILABLE:
            logfire_manager.log_error(ValueError("OpenAI API key is required"), "DSPy setup failed")
        raise ValueError("OpenAI API key is required.")
    
    try:
        lm = dspy.LM(model=model, api_key=api_key, max_tokens=max_tokens, temperature=temperature, **kwargs)
        dspy.settings.configure(lm=lm)
        
        if LOGFIRE_AVAILABLE:
            logfire_manager.log_event("DSPy setup completed successfully", "info", 
                                     model=model, configured=True)
        return lm
    except Exception as e:
        if LOGFIRE_AVAILABLE:
            logfire_manager.log_error(e, "DSPy configuration failed", model=model)
        raise


@logfire_span("setup_dspy_basic", component="dspy_setup")
def setup_dspy_basic() -> dspy.LM:
    """Quick basic setup for DSPy with Logfire monitoring."""
    if LOGFIRE_AVAILABLE:
        logfire_manager.log_event("Starting basic DSPy setup", "info")
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


@logfire_span("validate_setup", component="dspy_setup")
def validate_setup() -> bool:
    """Validate DSPy setup with Logfire monitoring."""
    config = get_model_config()
    is_configured = config.get('configured', False)
    
    if LOGFIRE_AVAILABLE:
        logfire_manager.log_event("DSPy setup validation", "info", 
                                 is_configured=is_configured,
                                 config=config)
    
    return is_configured


if __name__ == "__main__":
    if not validate_setup():
        setup_dspy_basic()
    print("DSPy setup complete.")
