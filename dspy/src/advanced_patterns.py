import dspy
import hashlib
import asyncio
import logging
import time
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
# Handle both module import and direct script execution
try:
    from .dspy_setup import setup_dspy_basic
    from .logfire_setup import get_logfire_manager, logfire_span, logfire_log
except ImportError:
    from dspy_setup import setup_dspy_basic
    from logfire_setup import get_logfire_manager, logfire_span, logfire_log

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Logfire manager
logfire_manager = get_logfire_manager()


@dataclass
class PipelineMetrics:
    """Pipeline performance metrics."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_latency: float = 0.0
    average_latency: float = 0.0

class BasicQA(dspy.Signature):
    """Basic QA signature."""
    context = dspy.InputField()
    question = dspy.InputField()
    answer = dspy.OutputField()


class ResilientQAPipeline(dspy.Module):
    """QA Pipeline with error handling and Logfire monitoring."""
    
    def __init__(self, max_retries: int = 3):
        super().__init__()
        self.qa = dspy.ChainOfThought(BasicQA)
        self.fallback_qa = dspy.Predict(BasicQA)
        self.max_retries = max_retries
        self.metrics = PipelineMetrics()
        logfire_manager.log_event("ResilientQA pipeline initialized", "info", component="resilient_qa_pipeline")
    
    @logfire_span("resilient_qa_forward", component="advanced_patterns")
    def forward(self, context: str, question: str) -> dspy.Prediction:
        """Process question with retries."""
        logfire_manager.log_event("Processing question with ResilientQA", "info", question=question[:100], context_length=len(context), max_retries=self.max_retries)
        
        for attempt in range(self.max_retries):
            try:
                self.metrics.total_calls += 1
                start_time = time.time()
                
                qa_module = self.qa if attempt == 0 else self.fallback_qa
                result = qa_module(context=context, question=question)
                
                if self._is_valid_answer(result.answer):
                    latency = time.time() - start_time
                    self.metrics.successful_calls += 1
                    self.metrics.total_latency += latency
                    self.metrics.average_latency = self.metrics.total_latency / self.metrics.successful_calls
                    
                    logfire_manager.log_event("Successful answer processed", "info", attempt=attempt, latency=latency)
                    
                    return dspy.Prediction(
                        answer=result.answer,
                        confidence=1.0 - (attempt * 0.2),
                        method_used="primary" if attempt == 0 else "fallback",
                        attempts=attempt + 1
                    )
                    
            except Exception as e:
                logfire_manager.log_error(e, "Error processing question", attempt=attempt)
                if attempt == self.max_retries - 1:
                    self.metrics.failed_calls += 1
                    return dspy.Prediction(
                        answer="Unable to provide answer at this time.",
                        confidence=0.0,
                        method_used="error",
                        attempts=self.max_retries
                    )
        
        return dspy.Prediction(
            answer="Service unavailable.",
            confidence=0.0,
            method_used="final_fallback",
            attempts=self.max_retries
        )
    
    def _is_valid_answer(self, answer: str) -> bool:
        """Validate answer quality."""
        if not answer or len(answer.strip()) < 10:
            return False
        
        failure_patterns = ["i don't know", "cannot answer", "not enough information"]
        return not any(pattern in answer.lower() for pattern in failure_patterns)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        success_rate = (self.metrics.successful_calls / max(self.metrics.total_calls, 1)) * 100
        return {
            'total_calls': self.metrics.total_calls,
            'successful_calls': self.metrics.successful_calls,
            'success_rate': f"{success_rate:.1f}%",
            'average_latency': f"{self.metrics.average_latency:.2f}s"
        }

class CachedRAGPipeline(dspy.Module):
    """RAG Pipeline with caching and Logfire monitoring."""
    
    def __init__(self, cache_size: int = 100):
        super().__init__()
        self.qa_pipeline = ResilientQAPipeline()
        self._cache: Dict[str, dspy.Prediction] = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        logfire_manager.log_event("CachedRAG pipeline initialized", "info", component="cached_rag_pipeline", cache_size=cache_size)
    
    @logfire_span("cached_rag_forward", component="advanced_patterns")
    def forward(self, question: str, context: Optional[str] = None) -> dspy.Prediction:
        """Process question with caching."""
        context = context or "General knowledge about technology and AI."
        cache_key = self._create_cache_key(question, context)
        
        logfire_manager.log_event("Processing cached RAG query", "info", question=question[:100], has_context=context is not None)
        
        if cache_key in self._cache:
            self.cache_hits += 1
            result = self._cache[cache_key]
            result.from_cache = True
            logfire_manager.log_event("Cache hit", "info", cache_hits=self.cache_hits, hit_rate=self.cache_hits/(self.cache_hits + self.cache_misses))
            return result
        
        self.cache_misses += 1
        logfire_manager.log_event("Cache miss - processing new query", "info", cache_misses=self.cache_misses)
        
        start_time = time.time()
        result = self.qa_pipeline(context=context, question=question)
        processing_time = time.time() - start_time
        
        self._manage_cache_size()
        self._cache[cache_key] = result
        result.from_cache = False
        
        logfire_manager.log_event("Query processed and cached", "info", processing_time=processing_time, cache_size=len(self._cache))
        
        return result

    def _create_cache_key(self, question: str, context: str) -> str:
        """Create cache key."""
        combined = f"{question.lower().strip()}|{context.lower().strip()[:200]}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _manage_cache_size(self):
        """Remove oldest entry if cache is full."""
        if len(self._cache) >= self.cache_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / max(total, 1)) * 100
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': f"{hit_rate:.1f}%",
            'cache_size': len(self._cache)
        }

class MonitoredDSPyModule(dspy.Module):
    """Module wrapper with monitoring."""
    
    def __init__(self, base_module: dspy.Module, logger: logging.Logger = None):
        super().__init__()
        self.base_module = base_module
        self.logger = logger or logging.getLogger(__name__)
        self.metrics = PipelineMetrics()
    
    def forward(self, *args, **kwargs):
        """Execute module with monitoring."""
        start_time = time.time()
        self.metrics.total_calls += 1
        
        try:
            result = self.base_module(*args, **kwargs)
            
            self.metrics.successful_calls += 1
            latency = time.time() - start_time
            self.metrics.total_latency += latency
            self.metrics.average_latency = self.metrics.total_latency / self.metrics.successful_calls
            
            return result
            
        except Exception as e:
            self.metrics.failed_calls += 1
            self.logger.error(f"Module failed: {str(e)}")
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get monitoring metrics."""
        success_rate = (self.metrics.successful_calls / max(self.metrics.total_calls, 1)) * 100
        return {
            'total_calls': self.metrics.total_calls,
            'successful_calls': self.metrics.successful_calls,
            'success_rate': f"{success_rate:.1f}%"
        }

class AsyncRAGPipeline(dspy.Module):
    """Async RAG pipeline for concurrent processing."""
    
    def __init__(self, max_workers: int = 4):
        super().__init__()
        self.rag = CachedRAGPipeline()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def forward_async(self, questions: List[str], contexts: Optional[List[str]] = None) -> List[Dict]:
        """Process multiple questions concurrently."""
        contexts = contexts or [None] * len(questions)
        loop = asyncio.get_event_loop()
        
        tasks = [
            loop.run_in_executor(self.executor, self._process_question, q, c)
            for q, c in zip(questions, contexts)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return [
            {
                'question': questions[i],
                'answer': result.answer if not isinstance(result, Exception) else None,
                'success': not isinstance(result, Exception),
                'error': str(result) if isinstance(result, Exception) else None
            }
            for i, result in enumerate(results)
        ]
    
    def _process_question(self, question: str, context: Optional[str] = None) -> dspy.Prediction:
        """Process single question."""
        return self.rag(question=question, context=context)
    
    def __del__(self):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)

@logfire_span("demonstrate_advanced_patterns", component="advanced_patterns")
def demonstrate_advanced_patterns():
    """Demo advanced DSPy patterns with Logfire monitoring."""
    logfire_manager.log_event("Starting advanced patterns demo", "info")
    setup_dspy_basic()
    
    print("=== Advanced Patterns Demo ===")
    
    # Resilient pipeline
    resilient = ResilientQAPipeline(max_retries=2)
    context = "Python is a programming language created by Guido van Rossum in 1991."
    
    result = resilient(context=context, question="Who created Python?")
    print(f"Resilient Answer: {result.answer}")
    print(f"Metrics: {resilient.get_metrics()}")
    logfire_manager.log_event("Resilient pipeline demo completed", "info", answer_quality="high" if len(result.answer) > 20 else "low")
    
    # Cached pipeline
    cached = CachedRAGPipeline(cache_size=10)
    
    # Test caching
    question = "What is machine learning?"
    result1 = cached(question=question)
    result2 = cached(question=question)  # Should hit cache
    
    print(f"\nCache test - First call from cache: {getattr(result1, 'from_cache', False)}")
    print(f"Cache test - Second call from cache: {getattr(result2, 'from_cache', False)}")
    print(f"Cache stats: {cached.get_cache_stats()}")
    
    logfire_manager.log_event("Cache demo completed", "info", cache_stats=cached.get_cache_stats())


@logfire_span("demonstrate_async", component="advanced_patterns")
async def demonstrate_async():
    """Demo async processing with Logfire monitoring."""
    logfire_manager.log_event("Starting async demo", "info")
    setup_dspy_basic()
    
    async_pipeline = AsyncRAGPipeline(max_workers=2)
    questions = ["What is AI?", "What is ML?", "What is NLP?"]
    
    start_time = time.time()
    results = await async_pipeline.forward_async(questions)
    processing_time = time.time() - start_time
    
    success_count = sum(1 for r in results if r['success'])
    
    logfire_manager.log_event(
        "Async processing completed", 
        "info", 
        total_questions=len(questions),
        successful_results=success_count,
        processing_time=processing_time
    )
    
    for result in results:
        print(f"Q: {result['question']} | Success: {result['success']}")


@logfire_span("run_all_demos", component="advanced_patterns")
def run_all_demos():
    """Run all demos with Logfire monitoring."""
    logfire_manager.log_event("Starting all advanced pattern demos", "info")
    start_time = time.time()
    
    try:
        demonstrate_advanced_patterns()
        asyncio.run(demonstrate_async())
        
        total_time = time.time() - start_time
        logfire_manager.log_event("All demos completed successfully", "info", total_duration=total_time)
        
    except Exception as e:
        logfire_manager.log_error(e, "Demo execution failed")
        print(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    run_all_demos()
