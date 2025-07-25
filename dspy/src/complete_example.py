"""
Complete Example: Pydantic + Logfire Integration with DSPy

This example demonstrates the complete integration of Pydantic for data validation
and Logfire for observability in a DSPy application, showing how all components
work together to provide production-ready monitoring and logging.
"""

import time
import asyncio
from typing import List, Dict, Any
from pydantic import BaseModel, Field, ValidationError

try:
    from .logfire_setup import get_logfire_manager, logfire_span, initialize_logfire
    from .monitoring import get_monitoring_manager, monitor_function
    from .pydantic_integration import AnalysisModule, QueryInput, ValidatedRAGPipeline
    from .dspy_setup import setup_dspy_basic
    from .config import get_config
except ImportError:
    from logfire_setup import get_logfire_manager, logfire_span, initialize_logfire
    from monitoring import get_monitoring_manager, monitor_function
    from pydantic_integration import AnalysisModule, QueryInput, ValidatedRAGPipeline
    from dspy_setup import setup_dspy_basic
    from config import get_config


class ProcessingRequest(BaseModel):
    """Request model for text processing with full Pydantic validation"""
    text: str = Field(..., min_length=10, max_length=5000, description="Text to process")
    analysis_type: str = Field(default="sentiment", description="Type of analysis to perform")
    include_metadata: bool = Field(default=True, description="Include processing metadata")
    priority: int = Field(default=1, ge=1, le=5, description="Processing priority")
    user_id: str = Field(..., min_length=1, description="User identifier")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "This is an example text for sentiment analysis processing.",
                "analysis_type": "sentiment",
                "include_metadata": True,
                "priority": 2,
                "user_id": "user123"
            }
        }
    }


class ProcessingResponse(BaseModel):
    """Response model with comprehensive metadata and validation"""
    request_id: str = Field(..., description="Unique request identifier")
    user_id: str = Field(..., description="User identifier")
    result: Dict[str, Any] = Field(..., description="Processing results")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Processing metadata")
    processing_time: float = Field(..., ge=0.0, description="Total processing time")
    success: bool = Field(..., description="Whether processing was successful")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")
    warnings: List[str] = Field(default_factory=list, description="Any warnings generated")


class ProductionPipeline:
    """Production-ready pipeline with full observability"""
    
    def __init__(self):
        # Initialize all components
        self.config = get_config()
        self.logfire_manager = get_logfire_manager()
        self.monitoring_manager = get_monitoring_manager()
        
        # Initialize DSPy
        setup_dspy_basic()
        
        # Initialize processing modules
        self.analysis_module = AnalysisModule()
        self.rag_pipeline = ValidatedRAGPipeline()
        
        # Set up logging context
        self.logfire_manager.set_context(
            service="production_pipeline",
            version="1.0.0",
            environment=self.config.logfire_environment
        )
        
        self.logfire_manager.log_event(
            "Production pipeline initialized",
            "info",
            modules_loaded=["analysis", "rag"],
            logfire_enabled=self.logfire_manager.is_initialized()
        )
    
    @logfire_span("process_request", component="production_pipeline")
    @monitor_function("pipeline_processing")
    def process_request(self, request: ProcessingRequest) -> ProcessingResponse:
        """Process a request with full observability and error handling"""
        request_id = f"req_{int(time.time() * 1000)}_{request.user_id}"
        start_time = time.time()
        
        # Set request context for logging
        self.logfire_manager.set_context(
            request_id=request_id,
            user_id=request.user_id,
            analysis_type=request.analysis_type,
            priority=request.priority
        )
        
        self.logfire_manager.log_event(
            "Processing request started",
            "info",
            text_length=len(request.text),
            word_count=len(request.text.split())
        )
        
        errors = []
        warnings = []
        result = {}
        
        try:
            # Validate request
            with self.logfire_manager.span("request_validation") as span:
                # Request is already validated by Pydantic, but we can add business logic validation
                if request.priority > 3:
                    warnings.append("High priority request - expedited processing")
                
                if len(request.text.split()) > 1000:
                    warnings.append("Large text detected - may take longer to process")
                
                if span:
                    span.set_attribute("validation_warnings", len(warnings))
            
            # Perform text analysis
            with self.logfire_manager.span("text_analysis") as span:
                try:
                    analysis_result = self.analysis_module.forward(request.text, request_id)
                    result["analysis"] = {
                        "sentiment": analysis_result.sentiment,
                        "confidence": analysis_result.confidence,
                        "themes": analysis_result.key_themes,
                        "summary": analysis_result.summary,
                        "word_count": analysis_result.word_count,
                        "processing_time": analysis_result.processing_time
                    }
                    
                    # Record metrics
                    self.monitoring_manager.record_timer("analysis_duration", analysis_result.processing_time or 0)
                    self.monitoring_manager.set_gauge("last_confidence_score", analysis_result.confidence)
                    
                    if span:
                        span.set_attribute("analysis_success", True)
                        span.set_attribute("sentiment", analysis_result.sentiment)
                        span.set_attribute("confidence", analysis_result.confidence)
                
                except Exception as e:
                    errors.append(f"Analysis failed: {str(e)}")
                    self.logfire_manager.log_error(e, "Text analysis failed")
                    self.monitoring_manager.increment_counter("analysis_errors")
                    
                    if span:
                        span.set_attribute("analysis_success", False)
                        span.set_attribute("error", str(e))
            
            # Perform RAG processing if requested
            if request.analysis_type in ["rag", "question_answering"]:
                with self.logfire_manager.span("rag_processing") as span:
                    try:
                        # Create a query from the text
                        query_input = QueryInput(
                            question=f"What is the main topic of this text: {request.text[:200]}...",
                            max_results=3,
                            include_reasoning=True
                        )
                        
                        rag_result = self.rag_pipeline(query_input)
                        result["rag"] = rag_result
                        
                        self.monitoring_manager.record_timer("rag_duration", time.time() - start_time)
                        
                        if span:
                            span.set_attribute("rag_success", True)
                            span.set_attribute("results_count", len(str(rag_result)))
                    
                    except Exception as e:
                        errors.append(f"RAG processing failed: {str(e)}")
                        self.logfire_manager.log_error(e, "RAG processing failed")
                        self.monitoring_manager.increment_counter("rag_errors")
                        
                        if span:
                            span.set_attribute("rag_success", False)
                            span.set_attribute("error", str(e))
            
            # Calculate final processing time
            processing_time = time.time() - start_time
            
            # Create metadata
            metadata = {}
            if request.include_metadata:
                metadata = {
                    "processing_time": processing_time,
                    "timestamp": time.time(),
                    "model_config": self.config.get_model_config(),
                    "text_stats": {
                        "length": len(request.text),
                        "word_count": len(request.text.split()),
                        "sentence_count": request.text.count('.') + request.text.count('!') + request.text.count('?')
                    },
                    "request_info": {
                        "priority": request.priority,
                        "analysis_type": request.analysis_type
                    }
                }
            
            # Record final metrics
            success = len(errors) == 0
            self.monitoring_manager.record_request(processing_time, success)
            
            if success:
                self.monitoring_manager.increment_counter("successful_requests", user_id=request.user_id)
            else:
                self.monitoring_manager.increment_counter("failed_requests", user_id=request.user_id)
            
            # Create response
            response = ProcessingResponse(
                request_id=request_id,
                user_id=request.user_id,
                result=result,
                metadata=metadata,
                processing_time=processing_time,
                success=success,
                errors=errors,
                warnings=warnings
            )
            
            # Log completion
            self.logfire_manager.log_event(
                "Request processing completed",
                "info" if success else "error",
                success=success,
                error_count=len(errors),
                warning_count=len(warnings),
                processing_time=processing_time,
                result_keys=list(result.keys())
            )
            
            return response
            
        except Exception as e:
            # Handle unexpected errors
            processing_time = time.time() - start_time
            self.logfire_manager.log_error(e, "Unexpected error in request processing")
            self.monitoring_manager.record_request(processing_time, success=False)
            self.monitoring_manager.increment_counter("unexpected_errors", user_id=request.user_id)
            
            return ProcessingResponse(
                request_id=request_id,
                user_id=request.user_id,
                result={},
                metadata={"error_occurred": True},
                processing_time=processing_time,
                success=False,
                errors=[f"Unexpected error: {str(e)}"],
                warnings=[]
            )
        
        finally:
            # Clear request context
            self.logfire_manager.clear_context()
    
    @logfire_span("health_check", component="production_pipeline")
    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health = self.monitoring_manager.perform_health_check()
        
        return {
            "status": health.status.value,
            "timestamp": health.timestamp.isoformat(),
            "uptime_seconds": health.uptime_seconds,
            "system_metrics": {
                "cpu_percent": health.system_metrics.cpu_percent,
                "memory_percent": health.system_metrics.memory_percent,
                "disk_usage_percent": health.system_metrics.disk_usage_percent,
                "active_threads": health.system_metrics.active_threads
            },
            "application_metrics": {
                "total_requests": health.app_metrics.total_requests,
                "success_rate": health.app_metrics.success_rate,
                "average_response_time": health.app_metrics.average_response_time,
                "requests_per_minute": health.app_metrics.requests_per_minute,
                "cache_hit_rate": health.app_metrics.cache_hit_rate
            },
            "errors": health.errors,
            "warnings": health.warnings,
            "logfire_status": "enabled" if self.logfire_manager.is_initialized() else "disabled"
        }
    
    def get_metrics_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive metrics for dashboard display"""
        metrics_summary = self.monitoring_manager.get_metrics_summary()
        health = self.health_check()
        
        return {
            "health": health,
            "metrics": metrics_summary,
            "logfire_info": {
                "initialized": self.logfire_manager.is_initialized(),
                "project": self.config.logfire_project,
                "environment": self.config.logfire_environment
            },
            "configuration": {
                "model_provider": self.config.model_provider.value,
                "model_name": self.config.model_name,
                "log_level": self.config.log_level.value,
                "monitoring_enabled": self.config.enable_monitoring
            }
        }


def run_demo():
    """Run a comprehensive demo of the system"""
    print("🚀 Starting Pydantic + Logfire + DSPy Integration Demo")
    print("=" * 60)
    
    # Initialize Logfire
    if not initialize_logfire():
        print("⚠️  Logfire not initialized - some features may not work")
    
    # Create pipeline
    pipeline = ProductionPipeline()
    
    # Create sample requests
    sample_requests = [
        ProcessingRequest(
            text="This is an amazing product that has completely transformed my workflow. I'm incredibly happy with the results and would highly recommend it to anyone looking for efficiency improvements.",
            analysis_type="sentiment",
            user_id="demo_user_1",
            priority=2
        ),
        ProcessingRequest(
            text="The service was okay, nothing special but not terrible either. It does what it promises but lacks the wow factor that would make me truly excited about it.",
            analysis_type="sentiment",
            user_id="demo_user_2",
            priority=1
        ),
        ProcessingRequest(
            text="Unfortunately, this experience was quite disappointing. The product failed to meet expectations and caused more problems than it solved. Very frustrating overall.",
            analysis_type="sentiment",
            user_id="demo_user_3",
            priority=3
        )
    ]
    
    print(f"📝 Processing {len(sample_requests)} sample requests...")
    
    # Process requests
    results = []
    for i, request in enumerate(sample_requests, 1):
        print(f"\n🔄 Processing request {i}/{len(sample_requests)}...")
        
        try:
            response = pipeline.process_request(request)
            results.append(response)
            
            print(f"   ✅ Success: {response.success}")
            print(f"   ⏱️  Time: {response.processing_time:.2f}s")
            if response.result.get("analysis"):
                analysis = response.result["analysis"]
                print(f"   😊 Sentiment: {analysis['sentiment']} ({analysis['confidence']:.2f})")
                print(f"   🏷️  Themes: {', '.join(analysis['themes'][:3])}")
            
            if response.errors:
                print(f"   ❌ Errors: {len(response.errors)}")
            if response.warnings:
                print(f"   ⚠️  Warnings: {len(response.warnings)}")
                
        except ValidationError as e:
            print(f"   ❌ Validation Error: {e}")
        except Exception as e:
            print(f"   ❌ Unexpected Error: {e}")
    
    # Show health check
    print(f"\n🏥 System Health Check:")
    health = pipeline.health_check()
    print(f"   Status: {health['status'].upper()}")
    print(f"   Uptime: {health['uptime_seconds']:.1f}s")
    print(f"   Success Rate: {health['application_metrics']['success_rate']:.1f}%")
    print(f"   Avg Response Time: {health['application_metrics']['average_response_time']:.2f}s")
    print(f"   CPU Usage: {health['system_metrics']['cpu_percent']:.1f}%")
    print(f"   Memory Usage: {health['system_metrics']['memory_percent']:.1f}%")
    
    if health['errors']:
        print(f"   ❌ Errors: {', '.join(health['errors'])}")
    if health['warnings']:
        print(f"   ⚠️  Warnings: {', '.join(health['warnings'])}")
    
    # Show dashboard summary
    print(f"\n📊 Dashboard Summary:")
    dashboard = pipeline.get_metrics_dashboard()
    print(f"   Total Requests: {dashboard['metrics']['requests']['total']}")
    print(f"   Success Rate: {dashboard['metrics']['requests']['success_rate']:.1f}%")
    print(f"   Logfire Status: {dashboard['logfire_info']['initialized']}")
    print(f"   Model: {dashboard['configuration']['model_name']}")
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"🔍 Check your Logfire dashboard for detailed observability data")
    
    return results, health, dashboard


if __name__ == "__main__":
    # Run the demo
    results, health, dashboard = run_demo()
    
    # Optional: Print detailed results for debugging
    if False:  # Set to True for verbose output
        print("\n" + "="*60)
        print("DETAILED RESULTS")
        print("="*60)
        
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"  Request ID: {result.request_id}")
            print(f"  Success: {result.success}")
            print(f"  Processing Time: {result.processing_time:.3f}s")
            if result.result:
                print(f"  Result Keys: {list(result.result.keys())}")
            if result.errors:
                print(f"  Errors: {result.errors}")
            if result.metadata:
                print(f"  Metadata Keys: {list(result.metadata.keys())}")
