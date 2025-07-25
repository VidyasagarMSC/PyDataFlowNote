"""
Advanced Logging Configuration with Logfire Integration

Provides structured logging, observability, and monitoring capabilities
using Logfire for the DSPy production system.
"""

import os
import sys
import json
import time
import logging
import structlog
from typing import Dict, Any, Optional, Union, Callable
from datetime import datetime
from contextlib import contextmanager
from functools import wraps

try:
    import logfire
    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False
    logfire = None

# Handle configuration import
try:
    from .config import get_config
except ImportError:
    from config import get_config


class LogfireHandler(logging.Handler):
    """Custom logging handler that sends logs to Logfire"""
    
    def __init__(self, logfire_client=None):
        super().__init__()
        self.logfire_client = logfire_client
        
    def emit(self, record):
        """Emit log record to Logfire"""
        if not self.logfire_client:
            return
            
        try:
            log_data = {
                'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                'level': record.levelname,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno,
            }
            
            # Add extra fields if present
            if hasattr(record, 'extra_data'):
                log_data.update(record.extra_data)
                
            self.logfire_client.info(
                "DSPy Log Entry",
                **log_data
            )
        except Exception as e:
            # Don't let logging errors break the application
            print(f"Logfire logging error: {e}")


class DSPyLogger:
    """Enhanced logger with Logfire integration and structured logging"""
    
    def __init__(self, name: str, config=None):
        self.name = name
        self.config = config or get_config()
        self.logger = self._setup_logger()
        self.logfire_client = self._setup_logfire()
        self._request_context = {}
        
    def _setup_logger(self) -> logging.Logger:
        """Setup structured logger with appropriate handlers"""
        logger = logging.getLogger(self.name)
        logger.setLevel(getattr(logging, self.config.log_level.value))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler with structured format
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler for persistent logging
        if self.config.log_dir:
            os.makedirs(self.config.log_dir, exist_ok=True)
            file_handler = logging.FileHandler(
                os.path.join(self.config.log_dir, f"{self.name}.log")
            )
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        # Add Logfire handler if available and configured
        if LOGFIRE_AVAILABLE and self.logfire_client:
            logfire_handler = LogfireHandler(self.logfire_client)
            logger.addHandler(logfire_handler)
            
        return logger
    
    def _setup_logfire(self):
        """Setup Logfire client if available"""
        if not LOGFIRE_AVAILABLE:
            self.logger.warning("Logfire not available - install logfire package for advanced observability")
            return None
            
        if not self.config.logfire_send_to_logfire:
            return None
            
        try:
            # Configure Logfire
            logfire.configure(
                token=self.config.logfire_token,
                project_name=self.config.logfire_project,
                service_name="dspy-pipeline",
                service_version="1.0.0",
                environment=self.config.logfire_environment,
            )
            
            self.logger.info("Logfire configured successfully")
            return logfire
            
        except Exception as e:
            self.logger.error(f"Failed to configure Logfire: {e}")
            return None
    
    def set_context(self, **kwargs):
        """Set logging context for all subsequent logs"""
        self._request_context.update(kwargs)
    
    def clear_context(self):
        """Clear logging context"""
        self._request_context.clear()
    
    def _log_with_context(self, level: str, message: str, **kwargs):
        """Log message with context"""
        log_data = {**self._request_context, **kwargs}
        getattr(self.logger, level.lower())(message, extra={'extra_data': log_data})
        
        # Send to Logfire if available
        if self.logfire_client and self.config.enable_tracing:
            getattr(self.logfire_client, level.lower())(message, **log_data)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self._log_with_context('DEBUG', message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self._log_with_context('INFO', message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self._log_with_context('WARNING', message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self._log_with_context('ERROR', message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self._log_with_context('CRITICAL', message, **kwargs)
    
    @contextmanager
    def span(self, operation_name: str, **attributes):
        """Create a tracing span for an operation"""
        start_time = time.time()
        span_id = f"{operation_name}_{int(start_time * 1000)}"
        
        # Set span context
        original_context = self._request_context.copy()
        self.set_context(span_id=span_id, operation=operation_name, **attributes)
        
        self.info(f"Starting {operation_name}", 
                 span_id=span_id, 
                 operation_type="start",
                 **attributes)
        
        try:
            # Use Logfire span if available
            if self.logfire_client:
                with self.logfire_client.span(operation_name, **attributes) as span:
                    yield span
            else:
                yield None
                
        except Exception as e:
            self.error(f"Error in {operation_name}: {str(e)}", 
                      span_id=span_id,
                      operation_type="error",
                      error_type=type(e).__name__)
            raise
        finally:
            duration = time.time() - start_time
            self.info(f"Completed {operation_name}", 
                     span_id=span_id,
                     operation_type="complete",
                     duration_seconds=duration)
            
            # Restore original context
            self._request_context = original_context
    
    def log_function_call(self, func_name: str, args: tuple, kwargs: dict, result: Any = None, error: Exception = None):
        """Log function call details"""
        log_data = {
            'function': func_name,
            'args_count': len(args),
            'kwargs_keys': list(kwargs.keys()),
        }
        
        if error:
            self.error(f"Function {func_name} failed", 
                      error=str(error), 
                      error_type=type(error).__name__,
                      **log_data)
        else:
            self.info(f"Function {func_name} succeeded", 
                     has_result=result is not None,
                     **log_data)
    
    def log_metrics(self, metrics: Dict[str, Union[int, float, str]]):
        """Log performance metrics"""
        if self.config.metrics_enabled:
            self.info("Performance metrics", **metrics)
            
            if self.logfire_client:
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        self.logfire_client.metric(metric_name, metric_value)


def get_logger(name: str) -> DSPyLogger:
    """Get a configured logger instance"""
    return DSPyLogger(name)


def log_execution_time(logger: DSPyLogger = None):
    """Decorator to log function execution time"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)
                
            start_time = time.time()
            func_name = f"{func.__module__}.{func.__name__}"
            
            try:
                with logger.span(f"execute_{func.__name__}", function=func_name):
                    result = func(*args, **kwargs)
                    
                execution_time = time.time() - start_time
                logger.log_function_call(func_name, args, kwargs, result)
                logger.log_metrics({
                    'execution_time_seconds': execution_time,
                    'function_name': func_name
                })
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.log_function_call(func_name, args, kwargs, error=e)
                logger.log_metrics({
                    'execution_time_seconds': execution_time,
                    'function_name': func_name,
                    'error': True
                })
                raise
                
        return wrapper
    return decorator


def log_api_call(logger: DSPyLogger = None):
    """Decorator to log API calls with request/response details"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)
                
            request_id = f"api_{int(time.time() * 1000)}"
            
            with logger.span("api_call", 
                           request_id=request_id,
                           function=func.__name__,
                           api_endpoint=func.__name__):
                
                logger.info("API call started", 
                           request_id=request_id,
                           endpoint=func.__name__)
                
                try:
                    result = func(*args, **kwargs)
                    logger.info("API call succeeded", 
                               request_id=request_id,
                               has_result=result is not None)
                    return result
                    
                except Exception as e:
                    logger.error("API call failed", 
                                request_id=request_id,
                                error=str(e),
                                error_type=type(e).__name__)
                    raise
                    
        return wrapper  
    return decorator


# Global logger instance
main_logger = get_logger("dspy.main")
