"""
Configuration Management with Pydantic Settings

Centralized configuration for the DSPy project using Pydantic Settings
for validation, type safety, and environment variable handling.
"""

import os
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
from enum import Enum
from pathlib import Path


class LogLevel(str, Enum):
    """Supported log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ModelProvider(str, Enum):
    """Supported model providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"


class DSPyConfig(BaseSettings):
    """DSPy framework configuration"""
    
    # API Configuration
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(None, description="Anthropic API key")
    azure_api_key: Optional[str] = Field(None, description="Azure API key")
    azure_endpoint: Optional[str] = Field(None, description="Azure endpoint URL")
    
    # Model Configuration
    model_provider: ModelProvider = Field(ModelProvider.OPENAI, description="Primary model provider")
    model_name: str = Field("gpt-4o-mini", description="Model name to use")
    max_tokens: int = Field(1000, ge=1, le=8192, description="Maximum tokens per request")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Model temperature")
    timeout: int = Field(30, ge=1, le=300, description="Request timeout in seconds")
    
    # Performance Configuration
    cache_size: int = Field(1000, ge=0, le=10000, description="Cache size for responses")
    max_retries: int = Field(3, ge=0, le=10, description="Maximum retry attempts")
    rate_limit_per_minute: int = Field(100, ge=1, le=1000, description="Rate limit per minute")
    batch_size: int = Field(10, ge=1, le=100, description="Batch processing size")
    
    # Monitoring and Observability
    enable_monitoring: bool = Field(True, description="Enable performance monitoring")
    enable_tracing: bool = Field(True, description="Enable request tracing")
    log_level: LogLevel = Field(LogLevel.INFO, description="Logging level")
    metrics_enabled: bool = Field(True, description="Enable metrics collection")
    
    # Logfire Configuration
    logfire_token: Optional[str] = Field(None, description="Logfire authentication token")
    logfire_project: Optional[str] = Field("dspy-production", description="Logfire project name")
    logfire_environment: str = Field("development", description="Environment name")
    logfire_send_to_logfire: bool = Field(True, description="Send logs to Logfire")
    
    # Data and Storage
    data_dir: str = Field("data", description="Data directory path")
    cache_dir: str = Field(".cache", description="Cache directory path")
    log_dir: str = Field("logs", description="Log directory path")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "env_prefix": "",
        "extra": "ignore"  # Ignore extra fields
    }
        
    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v, info):
        """Validate model name based on provider"""
        data = info.data if info else {}
        provider = data.get("model_provider")
        if provider == ModelProvider.OPENAI:
            valid_models = ["gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
            if not any(valid in v for valid in valid_models):
                raise ValueError(f"Invalid OpenAI model: {v}")
        elif provider == ModelProvider.ANTHROPIC:
            if not v.startswith("claude"):
                raise ValueError(f"Invalid Anthropic model: {v}")
        return v
    
    @field_validator("openai_api_key", "anthropic_api_key", mode="before")
    @classmethod
    def validate_api_keys(cls, v):
        """Validate API key format"""
        if v and len(v.strip()) < 10:
            raise ValueError("API key too short")
        return v.strip() if v else v
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration dictionary"""
        return {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout": self.timeout,
        }
    
    def get_api_key(self) -> Optional[str]:
        """Get appropriate API key based on provider"""
        if self.model_provider == ModelProvider.OPENAI:
            return self.openai_api_key
        elif self.model_provider == ModelProvider.ANTHROPIC:
            return self.anthropic_api_key
        elif self.model_provider == ModelProvider.AZURE:
            return self.azure_api_key
        return None
    
    def ensure_directories(self):
        """Ensure required directories exist"""
        for dir_path in [self.data_dir, self.cache_dir, self.log_dir]:
            os.makedirs(dir_path, exist_ok=True)


class PipelineConfig(BaseSettings):
    """Pipeline-specific configuration"""
    
    # Pipeline Settings
    enable_optimization: bool = Field(True, description="Enable pipeline optimization")
    optimization_rounds: int = Field(3, ge=1, le=10, description="Optimization rounds")
    validation_split: float = Field(0.2, ge=0.1, le=0.5, description="Validation split ratio")
    
    # RAG Settings
    max_context_length: int = Field(4000, ge=100, le=8000, description="Maximum context length")
    chunk_size: int = Field(512, ge=100, le=2000, description="Text chunk size")
    chunk_overlap: int = Field(50, ge=0, le=200, description="Chunk overlap size")
    top_k_results: int = Field(5, ge=1, le=20, description="Top K retrieval results")
    
    # Pydantic Integration
    strict_validation: bool = Field(True, description="Enable strict Pydantic validation")
    validation_retries: int = Field(2, ge=0, le=5, description="Validation retry attempts")
    fallback_enabled: bool = Field(True, description="Enable fallback on validation errors")
    
    model_config = {
        "env_file": ".env",
        "env_prefix": "PIPELINE_",
        "extra": "ignore"  # Ignore extra fields
    }


# Global configuration instances
_config: Optional[DSPyConfig] = None
_pipeline_config: Optional[PipelineConfig] = None


def get_config() -> DSPyConfig:
    """Get global configuration instance"""
    global _config
    if _config is None:
        _config = DSPyConfig()
        _config.ensure_directories()
    return _config


def get_pipeline_config() -> PipelineConfig:
    """Get pipeline configuration instance"""
    global _pipeline_config
    if _pipeline_config is None:
        _pipeline_config = PipelineConfig()
    return _pipeline_config


def reload_config():
    """Reload configuration from environment"""
    global _config, _pipeline_config
    _config = None
    _pipeline_config = None
    return get_config(), get_pipeline_config()


# Export commonly used configs
config = get_config()
pipeline_config = get_pipeline_config()
