"""
Logfire Setup and Configuration

This module provides comprehensive Logfire setup for observability,
including auto-instrumentation, custom spans, and dashboard integration.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, Callable
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

try:
    import logfire
    from logfire import configure, instrument_openai, instrument_anthropic, StructlogProcessor
    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False
    logfire = None

import structlog
from dotenv import load_dotenv

# Handle configuration import
try:
    from .config import get_config
except ImportError:
    from config import get_config

# Load environment variables
load_dotenv()


class LogfireManager:
    """Centralized Logfire management and configuration"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self._initialized = False
        self._logfire_client = None
        self._spans = {}
        
    def initialize(self) -> bool:
        """Initialize Logfire with comprehensive configuration"""
        if not LOGFIRE_AVAILABLE:
            logging.warning("Logfire not available - install with: pip install logfire")
            return False
            
        if self._initialized:
            return True
            
        try:
            # Get token from environment or config
            token = os.getenv("LOGFIRE_TOKEN") or self.config.logfire_token
            if not token:
                logging.warning("No Logfire token found - set LOGFIRE_TOKEN environment variable")
                return False
                
            # Configure Logfire with comprehensive settings
            logfire.configure(
                token=token,
                service_name="dspy-pipeline",
                service_version="1.0.0",
                send_to_logfire=self.config.logfire_send_to_logfire,
                console=logfire.ConsoleOptions(
                    colors=True,
                    include_timestamps=True,
                    verbose=self.config.log_level == "DEBUG"
                )
            )
            
            # Instrument Pydantic separately
            try:
                logfire.instrument_pydantic(record="all")
                logfire.info("Pydantic instrumentation enabled")
            except Exception as e:
                logfire.info(f"Pydantic instrumentation failed: {e}")
            
            # Auto-instrument common libraries
            self._setup_auto_instrumentation()
            
            # Setup structured logging integration
            self._setup_structured_logging()
            
            self._initialized = True
            self._logfire_client = logfire
            
            # Log successful initialization
            logfire.info(
                "Logfire initialized successfully",
                project=self.config.logfire_project,
                environment=self.config.logfire_environment,
                service="dspy-pipeline"
            )
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize Logfire: {e}")
            return False
    
    def _setup_auto_instrumentation(self):
        """Setup auto-instrumentation for various libraries"""
        try:
            # Instrument OpenAI if available
            try:
                import openai
                instrument_openai(openai, suppress_other_instrumentation=False)
                logfire.info("OpenAI instrumentation enabled")
            except ImportError:
                pass
                
            # Instrument Anthropic if available
            try:
                import anthropic
                instrument_anthropic(anthropic, suppress_other_instrumentation=False)
                logfire.info("Anthropic instrumentation enabled")
            except ImportError:
                pass
                
            # Instrument requests if available
            try:
                if hasattr(logfire, 'instrument_requests'):
                    logfire.instrument_requests()
                    logfire.info("Requests instrumentation enabled")
            except Exception as e:
                logfire.info(f"Requests instrumentation not available: {e}")
                
            # Instrument system metrics if available
            try:
                if hasattr(logfire, 'instrument_system_metrics'):
                    logfire.instrument_system_metrics()
                    logfire.info("System metrics instrumentation enabled")
            except Exception as e:
                logfire.info(f"System metrics instrumentation not available: {e}")
                
        except Exception as e:
            logging.warning(f"Some auto-instrumentation failed: {e}")
    
    def _setup_structured_logging(self):
        """Setup structured logging with Logfire integration"""
        try:
            # Configure structlog with Logfire processor
            processors = [
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                StructlogProcessor(),  # Logfire processor
                structlog.processors.JSONRenderer()
            ]
            
            structlog.configure(
                processors=processors,
                wrapper_class=structlog.stdlib.BoundLogger,
                logger_factory=structlog.stdlib.LoggerFactory(),
                cache_logger_on_first_use=True,
            )
            
            logfire.info("Structured logging configured with Logfire integration")
            
        except Exception as e:
            logging.warning(f"Failed to setup structured logging: {e}")
    
    @contextmanager
    def span(self, name: str, **attributes):
        """Create a Logfire span with attributes"""
        if not self._initialized or not self._logfire_client:
            yield None
            return
            
        with logfire.span(name, **attributes) as span:
            span_id = id(span)
            self._spans[span_id] = span
            try:
                yield span
            finally:
                self._spans.pop(span_id, None)
    
    def log_event(self, message: str, level: str = "info", **attributes):
        """Log an event with attributes"""
        if not self._initialized or not self._logfire_client:
            logging.log(getattr(logging, level.upper()), message)
            return
        
        # Merge context with attributes
        merged_attributes = {**self.get_context(), **attributes}
            
        log_func = getattr(logfire, level.lower(), logfire.info)
        log_func(message, **merged_attributes)
    
    def log_error(self, error: Exception, context: str = "", **attributes):
        """Log an error with full context"""
        error_attrs = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            **self.get_context(),
            **attributes
        }
        
        if not self._initialized or not self._logfire_client:
            logging.error(f"Error in {context}: {error}", extra=error_attrs)
            return
            
        logfire.error(f"Error in {context}", **error_attrs)
    
    def log_metrics(self, metrics: Dict[str, Any], **attributes):
        """Log metrics to Logfire"""
        if not self._initialized or not self._logfire_client:
            logging.info(f"Metrics: {metrics}")
            return
        
        merged_attributes = {**self.get_context(), **attributes}
        logfire.info("Metrics collected", metrics=metrics, **merged_attributes)
    
    def log_model_call(self, model: str, prompt: str, response: str, 
                      duration: float, **attributes):
        """Log model API calls"""
        call_attrs = {
            "model": model,
            "prompt_length": len(prompt),
            "response_length": len(response),
            "duration_seconds": duration,
            **attributes
        }
        
        if not self._initialized or not self._logfire_client:
            logging.info(f"Model call: {model}", extra=call_attrs)
            return
            
        logfire.info("Model API call", **call_attrs)
    
    def is_initialized(self) -> bool:
        """Check if Logfire is initialized"""
        return self._initialized
    
    def get_client(self):
        """Get the Logfire client"""
        return self._logfire_client if self._initialized else None
    
    def set_context(self, **kwargs):
        """Set logging context (for compatibility - Logfire handles context differently)"""
        # Store context for use in logging
        if not hasattr(self, '_context'):
            self._context = {}
        self._context.update(kwargs)
    
    def clear_context(self):
        """Clear logging context"""
        if hasattr(self, '_context'):
            self._context.clear()
    
    def get_context(self) -> Dict[str, Any]:
        """Get current logging context"""
        return getattr(self, '_context', {})


# Global Logfire manager instance
_logfire_manager: Optional[LogfireManager] = None


def get_logfire_manager() -> LogfireManager:
    """Get global Logfire manager instance"""
    global _logfire_manager
    if _logfire_manager is None:
        _logfire_manager = LogfireManager()
        _logfire_manager.initialize()
    return _logfire_manager


def logfire_span(name: str, **span_attributes):
    """Decorator to wrap functions with Logfire spans"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_logfire_manager()
            
            # Add function metadata to span
            attributes = {
                "function_name": func.__name__,
                "module": func.__module__,
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys()),
                **span_attributes
            }
            
            with manager.span(name or func.__name__, **attributes) as span:
                try:
                    result = func(*args, **kwargs)
                    if span:
                        span.set_attribute("success", True)
                        span.set_attribute("result_type", type(result).__name__)
                    return result
                except Exception as e:
                    if span:
                        span.set_attribute("success", False)
                        span.set_attribute("error_type", type(e).__name__)
                        span.set_attribute("error_message", str(e))
                    manager.log_error(e, f"Function {func.__name__}")
                    raise
                    
        return wrapper
    return decorator


def logfire_log(level: str = "info", message: str = None, **attributes):
    """Decorator to log function calls"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_logfire_manager()
            
            log_message = message or f"Executing {func.__name__}"
            log_attrs = {
                "function": func.__name__,
                "module": func.__module__,
                **attributes
            }
            
            manager.log_event(f"Starting: {log_message}", level, **log_attrs)
            
            try:
                result = func(*args, **kwargs)
                manager.log_event(f"Completed: {log_message}", level, 
                                success=True, **log_attrs)
                return result
            except Exception as e:
                manager.log_error(e, log_message, **log_attrs)
                raise
                
        return wrapper
    return decorator


# Initialize Logfire on module import
def initialize_logfire():
    """Initialize Logfire if not already done"""
    manager = get_logfire_manager()
    if manager.is_initialized():
        print("✅ Logfire initialized and ready for dashboard logging")
        return True
    else:
        print("❌ Logfire initialization failed - check configuration")
        return False


# Auto-initialize when module is imported
if __name__ != "__main__":
    initialize_logfire()
