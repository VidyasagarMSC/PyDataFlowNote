import os
import logging
import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from dotenv import load_dotenv
from advanced_patterns import MonitoredDSPyModule, CachedRAGPipeline
from pydantic_integration import QueryInput, ValidatedRAGPipeline
from dspy_setup import setup_dspy_basic

# Load environment variables
load_dotenv()

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dspy_production.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


@dataclass
class ProductionConfig:
    """Configuration for production deployment"""
    openai_api_key: str
    model_name: str = "gpt-3.5-turbo"
    max_tokens: int = 500
    cache_size: int = 1000
    max_retries: int = 3
    request_timeout: int = 30
    rate_limit_per_minute: int = 100
    enable_monitoring: bool = True
    log_level: str = "INFO"


class ProductionPipelineManager:
    """Production-ready pipeline manager with full observability"""

    def __init__(self, config: ProductionConfig):
        self.config = config
        self._setup_logging()
        self._setup_dspy()
        self._setup_pipelines()
        self._request_count = 0
        self._start_time = time.time()

    def _setup_logging(self):
        """Configure production logging"""
        logging.getLogger().setLevel(getattr(logging, self.config.log_level))
        logger.info("Production pipeline manager initializing...")

    def _setup_dspy(self):
        """Initialize DSPy with production configuration"""
        try:
            setup_dspy_basic()
            logger.info(f"DSPy configured with model: {self.config.model_name}")
        except Exception as e:
            logger.error(f"Failed to configure DSPy: {e}")
            raise

    def _setup_pipelines(self):
        """Initialize production pipelines with monitoring"""
        try:
            # Base pipelines
            base_rag = CachedRAGPipeline(cache_size=self.config.cache_size)
            base_validated = ValidatedRAGPipeline()

            # Add monitoring if enabled
            if self.config.enable_monitoring:
                self.rag_pipeline = MonitoredDSPyModule(base_rag, logger)
                self.validated_pipeline = MonitoredDSPyModule(base_validated, logger)
            else:
                self.rag_pipeline = base_rag
                self.validated_pipeline = base_validated

            logger.info("Production pipelines initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize pipelines: {e}")
            raise

    @contextmanager
    def request_context(self, request_id: str):
        """Context manager for request tracking"""
        start_time = time.time()
        self._request_count += 1

        logger.info(f"Request {request_id} started (total requests: {self._request_count})")

        try:
            yield
        except Exception as e:
            logger.error(f"Request {request_id} failed: {e}")
            raise
        finally:
            duration = time.time() - start_time
            logger.info(f"Request {request_id} completed in {duration:.2f}s")

    def process_query(self, query_data: Dict[str, Any], request_id: str = None) -> Dict[str, Any]:
        """Process a query with full production safeguards"""
        if request_id is None:
            request_id = f"req_{int(time.time() * 1000)}"

        with self.request_context(request_id):
            try:
                # Validate input using Pydantic
                query_input = QueryInput(**query_data)

                # Process with validated pipeline
                result = self.validated_pipeline(query_input)

                # Add metadata
                result['request_id'] = request_id
                result['timestamp'] = time.time()
                result['model_used'] = self.config.model_name

                logger.info(f"Query processed successfully: {request_id}")
                return {
                    'success': True,
                    'data': result,
                    'request_id': request_id
                }

            except Exception as e:
                logger.error(f"Query processing failed: {e}")
                return {
                    'success': False,
                    'error': str(e),
                    'request_id': request_id,
                    'timestamp': time.time()
                }

    def process_rag_query(self, question: str, context: str = None, request_id: str = None) -> Dict[str, Any]:
        """Process RAG query with caching and monitoring"""
        if request_id is None:
            request_id = f"rag_{int(time.time() * 1000)}"

        with self.request_context(request_id):
            try:
                result = self.rag_pipeline(question=question, context=context)

                return {
                    'success': True,
                    'data': {
                        'question': question,
                        'answer': result.answer,
                        'confidence': getattr(result, 'confidence', 0.0),
                        'from_cache': getattr(result, 'from_cache', False),
                        'context_used': getattr(result, 'context', None),
                        'request_id': request_id,
                        'timestamp': time.time()
                    }
                }

            except Exception as e:
                logger.error(f"RAG query processing failed: {e}")
                return {
                    'success': False,
                    'error': str(e),
                    'request_id': request_id,
                    'timestamp': time.time()
                }

    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health metrics"""
        uptime = time.time() - self._start_time

        health_data = {
            'status': 'healthy',
            'uptime_seconds': uptime,
            'total_requests': self._request_count,
            'requests_per_minute': (self._request_count / max(uptime / 60, 1)),
            'configuration': {
                'model_name': self.config.model_name,
                'cache_size': self.config.cache_size,
                'max_retries': self.config.max_retries,
                'monitoring_enabled': self.config.enable_monitoring
            }
        }

        # Add pipeline-specific metrics if monitoring is enabled
        if self.config.enable_monitoring:
            try:
                if hasattr(self.rag_pipeline, 'get_metrics'):
                    health_data['rag_pipeline_metrics'] = self.rag_pipeline.get_metrics()

                if hasattr(self.validated_pipeline, 'get_metrics'):
                    health_data['validated_pipeline_metrics'] = self.validated_pipeline.get_metrics()

                # Add cache statistics
                if hasattr(self.rag_pipeline.base_module, 'get_cache_stats'):
                    health_data['cache_stats'] = self.rag_pipeline.base_module.get_cache_stats()

            except Exception as e:
                logger.warning(f"Failed to collect some metrics: {e}")
                health_data['metrics_warning'] = str(e)

        return health_data

    def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down production pipeline manager...")

        # Log final statistics
        final_health = self.get_system_health()
        logger.info(f"Final system statistics: {json.dumps(final_health, indent=2)}")

        # Cleanup resources
        if hasattr(self.rag_pipeline, '__del__'):
            self.rag_pipeline.__del__()

        logger.info("Shutdown complete")


class ProductionAPIServer:
    """Simple production API server simulation"""

    def __init__(self, config: ProductionConfig):
        self.pipeline_manager = ProductionPipelineManager(config)
        self.is_running = False

    def start(self):
        """Start the production server"""
        self.is_running = True
        logger.info("Production API server started")

        # Simulate server startup checks
        health = self.pipeline_manager.get_system_health()
        if health['status'] == 'healthy':
            logger.info("All systems operational")
        else:
            logger.warning("Some systems may have issues")

    def handle_query_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming query requests"""
        if not self.is_running:
            return {
                'success': False,
                'error': 'Server is not running',
                'timestamp': time.time()
            }

        return self.pipeline_manager.process_query(request_data)

    def handle_rag_request(self, question: str, context: str = None) -> Dict[str, Any]:
        """Handle RAG-specific requests"""
        if not self.is_running:
            return {
                'success': False,
                'error': 'Server is not running',
                'timestamp': time.time()
            }

        return self.pipeline_manager.process_rag_query(question, context)

    def handle_health_check(self) -> Dict[str, Any]:
        """Handle health check requests"""
        return self.pipeline_manager.get_system_health()

    def stop(self):
        """Stop the production server"""
        self.is_running = False
        self.pipeline_manager.shutdown()
        logger.info("Production API server stopped")


def create_production_config() -> ProductionConfig:
    """Create production configuration from environment"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")

    return ProductionConfig(
        openai_api_key=api_key,
        model_name=os.getenv('MODEL_NAME', 'gpt-3.5-turbo'),
        max_tokens=int(os.getenv('MAX_TOKENS', '500')),
        cache_size=int(os.getenv('CACHE_SIZE', '1000')),
        max_retries=int(os.getenv('MAX_RETRIES', '3')),
        request_timeout=int(os.getenv('REQUEST_TIMEOUT', '30')),
        enable_monitoring=os.getenv('ENABLE_MONITORING', 'true').lower() == 'true',
        log_level=os.getenv('LOG_LEVEL', 'INFO')
    )


def demonstrate_production_deployment():
    """Demonstrate production deployment patterns"""
    print("=== Production Deployment Demo ===")

    try:
        # Create production configuration
        config = create_production_config()

        # Initialize production server
        server = ProductionAPIServer(config)
        server.start()

        # Simulate various requests
        print("\n--- Testing Query Requests ---")

        # Test 1: Valid query request
        query_request = {
            'question': 'What is artificial intelligence?',
            'max_results': 3,
            'include_reasoning': True
        }

        result1 = server.handle_query_request(query_request)
        print(f"Query Request Result: {result1['success']}")
        if result1['success']:
            print(f"Answer Length: {len(result1['data']['answer'])} characters")

        # Test 2: RAG request
        print("\n--- Testing RAG Requests ---")

        rag_result = server.handle_rag_request(
            question="How does machine learning work?",
            context="Machine learning uses algorithms to learn patterns from data."
        )
        print(f"RAG Request Result: {rag_result['success']}")
        if rag_result['success']:
            print(f"From Cache: {rag_result['data'].get('from_cache', False)}")

        # Test 3: Repeat RAG request to test caching
        print("\n--- Testing Cache Performance ---")

        rag_result2 = server.handle_rag_request(
            question="How does machine learning work?"  # Same question
        )
        if rag_result2['success']:
            print(f"Second Request From Cache: {rag_result2['data'].get('from_cache', False)}")

        # Test 4: Health check
        print("\n--- System Health Check ---")

        health = server.handle_health_check()
        print(f"System Status: {health['status']}")
        print(f"Total Requests: {health['total_requests']}")
        print(f"Uptime: {health['uptime_seconds']:.1f} seconds")

        if 'cache_stats' in health:
            cache_stats = health['cache_stats']
            print(f"Cache Hit Rate: {cache_stats.get('hit_rate', 'N/A')}")

        # Test 5: Error handling
        print("\n--- Testing Error Handling ---")

        try:
            invalid_request = {
                'question': 'Hi',  # Too short, should fail validation
                'max_results': 25  # Too large, should fail validation
            }

            error_result = server.handle_query_request(invalid_request)
            print(f"Invalid Request Handled: {not error_result['success']}")
            if not error_result['success']:
                print(f"Error Message: {error_result['error'][:100]}...")

        except Exception as e:
            print(f"Error handling test failed: {e}")

        # Shutdown
        print("\n--- Shutting Down ---")
        server.stop()

        print("Production deployment demo completed successfully!")

    except Exception as e:
        print(f"Production demo failed: {e}")
        print("Make sure you have set OPENAI_API_KEY in your .env file")


def demonstrate_monitoring_and_observability():
    """Demonstrate comprehensive monitoring capabilities"""
    print("\n=== Monitoring and Observability Demo ===")

    try:
        config = create_production_config()
        pipeline_manager = ProductionPipelineManager(config)

        # Process several requests to generate metrics
        test_queries = [
            {'question': 'What is deep learning?', 'max_results': 3},
            {'question': 'How do neural networks work?', 'max_results': 5},
            {'question': 'What is natural language processing?', 'max_results': 2},
            {'question': 'What is deep learning?', 'max_results': 3}  # Repeat for cache test
        ]

        print("Processing test queries for metrics generation...")

        for i, query in enumerate(test_queries):
            result = pipeline_manager.process_query(query, f"test_req_{i}")
            print(f"Query {i + 1}: {'Success' if result['success'] else 'Failed'}")

        # Get comprehensive health metrics
        print("\n--- System Health Metrics ---")
        health = pipeline_manager.get_system_health()

        print(json.dumps(health, indent=2, default=str))

        # Cleanup
        pipeline_manager.shutdown()

    except Exception as e:
        print(f"Monitoring demo failed: {e}")


if __name__ == "__main__":
    # Run production demonstrations
    demonstrate_production_deployment()
    demonstrate_monitoring_and_observability()