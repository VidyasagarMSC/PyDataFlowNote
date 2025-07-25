"""
Comprehensive Monitoring and Metrics Module

This module provides monitoring capabilities using Pydantic for data validation
and Logfire for observability, including custom metrics, health checks, and
performance monitoring.
"""

import time
import threading
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict, deque
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum
import psutil
import json

try:
    from .logfire_setup import get_logfire_manager, logfire_span
    from .config import get_config
except ImportError:
    from logfire_setup import get_logfire_manager, logfire_span
    from config import get_config


class HealthStatus(str, Enum):
    """System health status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class MetricType(str, Enum):
    """Types of metrics we can collect"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class SystemMetrics(BaseModel):
    """System-level metrics with Pydantic validation"""
    model_config = ConfigDict(
        plugin_settings={'logfire': {'record': 'all'}},
        validate_assignment=True
    )
    
    timestamp: datetime = Field(default_factory=datetime.now)
    cpu_percent: float = Field(..., ge=0.0, le=100.0)
    memory_percent: float = Field(..., ge=0.0, le=100.0)
    memory_used_mb: float = Field(..., ge=0.0)
    memory_available_mb: float = Field(..., ge=0.0)
    disk_usage_percent: float = Field(..., ge=0.0, le=100.0)
    active_threads: int = Field(..., ge=0)
    
    
class ApplicationMetrics(BaseModel):
    """Application-specific metrics"""
    model_config = ConfigDict(
        plugin_settings={'logfire': {'record': 'all'}},
        validate_assignment=True
    )
    
    timestamp: datetime = Field(default_factory=datetime.now)
    total_requests: int = Field(default=0, ge=0)
    successful_requests: int = Field(default=0, ge=0)
    failed_requests: int = Field(default=0, ge=0)
    average_response_time: float = Field(default=0.0, ge=0.0)
    requests_per_minute: float = Field(default=0.0, ge=0.0)
    cache_hits: int = Field(default=0, ge=0)
    cache_misses: int = Field(default=0, ge=0)
    validation_errors: int = Field(default=0, ge=0)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        total = self.total_requests
        if total == 0:
            return 100.0
        return (self.successful_requests / total) * 100.0
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate percentage"""
        total_cache_ops = self.cache_hits + self.cache_misses
        if total_cache_ops == 0:
            return 0.0
        return (self.cache_hits / total_cache_ops) * 100.0


class HealthCheck(BaseModel):
    """Health check result model"""
    model_config = ConfigDict(
        plugin_settings={'logfire': {'record': 'all'}},
        validate_assignment=True
    )
    
    timestamp: datetime = Field(default_factory=datetime.now)
    status: HealthStatus = Field(...)
    uptime_seconds: float = Field(..., ge=0.0)
    system_metrics: SystemMetrics = Field(...)
    app_metrics: ApplicationMetrics = Field(...)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    @property
    def is_healthy(self) -> bool:
        """Check if system is healthy"""
        return self.status == HealthStatus.HEALTHY


class MetricEvent(BaseModel):
    """Individual metric event"""
    model_config = ConfigDict(
        plugin_settings={'logfire': {'record': 'all'}},
        validate_assignment=True
    )
    
    name: str = Field(..., min_length=1)
    value: Union[int, float] = Field(...)
    metric_type: MetricType = Field(...)
    timestamp: datetime = Field(default_factory=datetime.now)
    tags: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MonitoringManager:
    """Comprehensive monitoring manager with Logfire integration"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.logfire_manager = get_logfire_manager()
        self.start_time = time.time()
        
        # Metrics storage
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._timers: Dict[str, List[float]] = defaultdict(list)
        
        # Request tracking
        self._request_times: deque = deque(maxlen=1000)
        self._error_count = 0
        self._success_count = 0
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Health check thresholds
        self.cpu_threshold = 80.0
        self.memory_threshold = 85.0
        self.disk_threshold = 90.0
        self.response_time_threshold = 5.0
        
    @logfire_span("collect_system_metrics", component="monitoring")
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect system-level metrics"""
        try:
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Get memory info
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / (1024 * 1024)
            memory_available_mb = memory.available / (1024 * 1024)
            
            # Get disk usage
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            
            # Get thread count
            active_threads = threading.active_count()
            
            metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_mb=memory_used_mb,
                memory_available_mb=memory_available_mb,
                disk_usage_percent=disk_usage_percent,
                active_threads=active_threads
            )
            
            # Log to Logfire
            self.logfire_manager.log_metrics(
                {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "memory_used_mb": memory_used_mb,
                    "disk_usage_percent": disk_usage_percent,
                    "active_threads": active_threads
                },
                metric_type="system_metrics"
            )
            
            return metrics
            
        except Exception as e:
            self.logfire_manager.log_error(e, "Failed to collect system metrics")
            # Return default metrics on error
            return SystemMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_mb=0.0,
                memory_available_mb=0.0,
                disk_usage_percent=0.0,
                active_threads=0
            )
    
    @logfire_span("collect_app_metrics", component="monitoring")
    def collect_application_metrics(self) -> ApplicationMetrics:
        """Collect application-specific metrics"""
        with self._lock:
            total_requests = self._success_count + self._error_count
            
            # Calculate average response time
            avg_response_time = 0.0
            if self._request_times:
                avg_response_time = sum(self._request_times) / len(self._request_times)
            
            # Calculate requests per minute
            current_time = time.time()
            uptime_minutes = max((current_time - self.start_time) / 60, 1)
            requests_per_minute = total_requests / uptime_minutes
            
            metrics = ApplicationMetrics(
                total_requests=total_requests,
                successful_requests=self._success_count,
                failed_requests=self._error_count,
                average_response_time=avg_response_time,
                requests_per_minute=requests_per_minute,
                cache_hits=self._counters.get('cache_hits', 0),
                cache_misses=self._counters.get('cache_misses', 0),
                validation_errors=self._counters.get('validation_errors', 0)
            )
            
            # Log application metrics to Logfire
            self.logfire_manager.log_metrics(
                {
                    "total_requests": total_requests,
                    "success_rate": metrics.success_rate,
                    "avg_response_time": avg_response_time,
                    "requests_per_minute": requests_per_minute,
                    "cache_hit_rate": metrics.cache_hit_rate
                },
                metric_type="application_metrics"
            )
            
            return metrics
    
    @logfire_span("health_check", component="monitoring")
    def perform_health_check(self) -> HealthCheck:
        """Perform comprehensive health check"""
        system_metrics = self.collect_system_metrics()
        app_metrics = self.collect_application_metrics()
        
        errors = []
        warnings = []
        status = HealthStatus.HEALTHY
        
        # Check system health
        if system_metrics.cpu_percent > self.cpu_threshold:
            errors.append(f"High CPU usage: {system_metrics.cpu_percent:.1f}%")
            status = HealthStatus.UNHEALTHY
        elif system_metrics.cpu_percent > self.cpu_threshold * 0.8:
            warnings.append(f"Elevated CPU usage: {system_metrics.cpu_percent:.1f}%")
            if status == HealthStatus.HEALTHY:
                status = HealthStatus.DEGRADED
        
        if system_metrics.memory_percent > self.memory_threshold:
            errors.append(f"High memory usage: {system_metrics.memory_percent:.1f}%")
            status = HealthStatus.UNHEALTHY
        elif system_metrics.memory_percent > self.memory_threshold * 0.8:
            warnings.append(f"Elevated memory usage: {system_metrics.memory_percent:.1f}%")
            if status == HealthStatus.HEALTHY:
                status = HealthStatus.DEGRADED
        
        if system_metrics.disk_usage_percent > self.disk_threshold:
            errors.append(f"High disk usage: {system_metrics.disk_usage_percent:.1f}%")
            status = HealthStatus.UNHEALTHY
        
        # Check application health
        if app_metrics.success_rate < 95.0 and app_metrics.total_requests > 10:
            errors.append(f"Low success rate: {app_metrics.success_rate:.1f}%")
            status = HealthStatus.UNHEALTHY
        elif app_metrics.success_rate < 98.0 and app_metrics.total_requests > 10:
            warnings.append(f"Reduced success rate: {app_metrics.success_rate:.1f}%")
            if status == HealthStatus.HEALTHY:
                status = HealthStatus.DEGRADED
        
        if app_metrics.average_response_time > self.response_time_threshold:
            errors.append(f"High response time: {app_metrics.average_response_time:.2f}s")
            status = HealthStatus.UNHEALTHY
        
        uptime = time.time() - self.start_time
        
        health_check = HealthCheck(
            status=status,
            uptime_seconds=uptime,
            system_metrics=system_metrics,
            app_metrics=app_metrics,
            errors=errors,
            warnings=warnings
        )
        
        # Log health check to Logfire
        self.logfire_manager.log_event(
            f"Health check completed: {status.value}",
            "info" if status == HealthStatus.HEALTHY else "warning",
            status=status.value,
            error_count=len(errors),
            warning_count=len(warnings),
            uptime_seconds=uptime,
            success_rate=app_metrics.success_rate
        )
        
        return health_check
    
    def record_request(self, duration: float, success: bool = True):
        """Record a request with its duration and outcome"""
        with self._lock:
            self._request_times.append(duration)
            if success:
                self._success_count += 1
            else:
                self._error_count += 1
    
    def increment_counter(self, name: str, value: int = 1, **tags):
        """Increment a counter metric"""
        with self._lock:
            self._counters[name] += value
        
        # Create metric event
        event = MetricEvent(
            name=name,
            value=value,
            metric_type=MetricType.COUNTER,
            tags=tags
        )
        
        # Log to Logfire
        self.logfire_manager.log_event(
            f"Counter incremented: {name}",
            "debug",
            counter_name=name,
            counter_value=self._counters[name],
            increment=value,
            **tags
        )
    
    def set_gauge(self, name: str, value: float, **tags):
        """Set a gauge metric value"""
        with self._lock:
            self._gauges[name] = value
        
        # Create metric event
        event = MetricEvent(
            name=name,
            value=value,
            metric_type=MetricType.GAUGE,
            tags=tags
        )
        
        # Log to Logfire
        self.logfire_manager.log_event(
            f"Gauge updated: {name}",
            "debug",
            gauge_name=name,
            gauge_value=value,
            **tags
        )
    
    def record_timer(self, name: str, duration: float, **tags):
        """Record a timer metric"""
        with self._lock:
            self._timers[name].append(duration)
            # Keep only last 100 measurements
            if len(self._timers[name]) > 100:
                self._timers[name] = self._timers[name][-100:]
        
        # Create metric event
        event = MetricEvent(
            name=name,
            value=duration,
            metric_type=MetricType.TIMER,
            tags=tags
        )
        
        # Log to Logfire
        self.logfire_manager.log_event(
            f"Timer recorded: {name}",
            "debug",
            timer_name=name,
            duration=duration,
            **tags
        )
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics"""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "timers": {
                    name: {
                        "count": len(times),
                        "avg": sum(times) / len(times) if times else 0,
                        "min": min(times) if times else 0,
                        "max": max(times) if times else 0
                    }
                    for name, times in self._timers.items()
                },
                "requests": {
                    "total": self._success_count + self._error_count,
                    "successful": self._success_count,
                    "failed": self._error_count,
                    "success_rate": (self._success_count / max(self._success_count + self._error_count, 1)) * 100
                }
            }


# Global monitoring manager instance
_monitoring_manager: Optional[MonitoringManager] = None


def get_monitoring_manager() -> MonitoringManager:
    """Get global monitoring manager instance"""
    global _monitoring_manager
    if _monitoring_manager is None:
        _monitoring_manager = MonitoringManager()
    return _monitoring_manager


def monitor_function(name: str = None, record_args: bool = False):
    """Decorator to monitor function execution"""
    def decorator(func):
        @logfire_span(f"monitored_function_{func.__name__}", component="monitoring")
        def wrapper(*args, **kwargs):
            manager = get_monitoring_manager()
            func_name = name or f"{func.__module__}.{func.__name__}"
            
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record successful execution
                manager.record_request(duration, success=True)
                manager.record_timer(f"function_{func_name}", duration)
                manager.increment_counter(f"function_calls_{func_name}")
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                # Record failed execution
                manager.record_request(duration, success=False)
                manager.record_timer(f"function_{func_name}_error", duration)
                manager.increment_counter(f"function_errors_{func_name}")
                
                raise
                
        return wrapper
    return decorator


if __name__ == "__main__":
    # Demo the monitoring system
    manager = get_monitoring_manager()
    
    # Simulate some activity
    import random
    
    for i in range(10):
        duration = random.uniform(0.1, 2.0)
        success = random.choice([True, True, True, False])  # 75% success rate
        manager.record_request(duration, success)
        time.sleep(0.1)
    
    # Perform health check
    health = manager.perform_health_check()
    print(f"Health Status: {health.status}")
    print(f"Success Rate: {health.app_metrics.success_rate:.1f}%")
    print(f"Average Response Time: {health.app_metrics.average_response_time:.2f}s")
