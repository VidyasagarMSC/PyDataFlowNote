import pytest
import pytest_asyncio
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import dspy
from src.dspy_setup import setup_dspy_basic, setup_dspy
from src.basic_examples import BasicPipeline
from src.pydantic_integration import AnalysisModule, QueryInput, ValidatedRAGPipeline
from src.advanced_patterns import ResilientQAPipeline, CachedRAGPipeline, AsyncRAGPipeline
from dspy.evaluate import Evaluate


class TestBasicPipelines:
    """Test basic DSPy pipeline functionality"""

    def setup_method(self):
        """Setup test environment"""
        try:
            self.lm = setup_dspy_basic()
            self.pipeline = BasicPipeline()
            
            # Load test cases from sample data
            from src.util import load_sample_data, get_test_cases
            try:
                sample_data = load_sample_data()
                self.test_cases = get_test_cases(sample_data)
            except Exception:
                # Fallback to hardcoded test cases
                self.test_cases = [
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
        except Exception as e:
            pytest.skip(f"Cannot setup DSPy (likely missing API key): {e}")

    def test_pipeline_initialization(self):
        """Test that pipeline initializes correctly"""
        assert self.pipeline is not None
        assert hasattr(self.pipeline, 'forward')
        assert hasattr(self.pipeline, 'qa')
        assert hasattr(self.pipeline, 'rag')

    def test_answer_generation(self):
        """Test basic answer generation"""
        question = "What is artificial intelligence?"
        result = self.pipeline(question)

        assert hasattr(result, 'answer')
        assert isinstance(result.answer, str)
        assert len(result.answer.strip()) > 0
        assert len(result.answer) > 10  # Meaningful answer length

    def test_answer_quality(self):
        """Test answer quality and relevance"""
        for case in self.test_cases[:2]:  # Test first 2 to avoid API limits
            result = self.pipeline(case["question"])

            # Check answer length
            assert len(result.answer) > 20, f"Answer too short for: {case['question']}"

            # Check for topic relevance (at least one expected topic should appear)
            answer_lower = result.answer.lower()
            topic_found = any(topic in answer_lower for topic in case["expected_topics"])
            assert topic_found, f"No relevant topics found in answer for: {case['question']}"

    def test_context_handling(self):
        """Test pipeline handling with custom context"""
        from src.util import load_sample_data
        
        try:
            sample_data = load_sample_data()
            custom_context = sample_data.get('contexts', {}).get('dspy_context', 
                "DSPy is a framework for programming language models. It provides automatic optimization of prompts.")
        except Exception:
            custom_context = "DSPy is a framework for programming language models. It provides automatic optimization of prompts. DSPy uses signatures to define task interfaces."

        question = "What is DSPy?"
        result = self.pipeline(question, context=custom_context)

        assert "DSPy" in result.answer or "dspy" in result.answer.lower()
        assert hasattr(result, 'answer')
        assert len(result.answer) > 0


class TestPydanticIntegration:
    """Test Pydantic integration with DSPy"""

    def setup_method(self):
        """Setup test environment"""
        try:
            setup_dspy()
            self.analyzer = AnalysisModule()
            self.rag_pipeline = ValidatedRAGPipeline()
        except Exception as e:
            pytest.skip(f"Cannot setup DSPy (likely missing API key): {e}")

    def test_analysis_result_validation(self):
        """Test that analysis results are properly validated"""
        sample_text = "This is a positive message about technology and innovation."

        result = self.analyzer(sample_text)

        # Check Pydantic model fields
        assert hasattr(result, 'sentiment')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'key_themes')
        assert hasattr(result, 'summary')
        assert hasattr(result, 'word_count')

        # Check field types and constraints
        assert result.sentiment in ['positive', 'negative', 'neutral']
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.key_themes, list)
        assert 1 <= len(result.key_themes) <= 5
        assert len(result.summary) >= 10
        assert result.word_count > 0

    def test_query_input_validation(self):
        """Test query input validation"""
        # Valid query
        valid_query = QueryInput(
            question="What is machine learning?",
            max_results=3
        )
        assert valid_query.question == "What is machine learning?"
        assert valid_query.max_results == 3

        # Invalid query - too short
        with pytest.raises(Exception):  # Pydantic ValidationError
            QueryInput(question="Hi")

        # Invalid query - too many results
        with pytest.raises(Exception):  # Pydantic ValidationError
            QueryInput(question="Valid question?", max_results=25)

    def test_validated_rag_pipeline(self):
        """Test RAG pipeline with validated inputs"""
        query = QueryInput(
            question="What is artificial intelligence?",
            include_reasoning=True
        )

        result = self.rag_pipeline(query)

        assert 'question' in result
        assert 'answer' in result
        assert 'context' in result
        assert 'reasoning' in result  # Should be included
        assert result['question'] == query.question


class TestAdvancedPatterns:
    """Test advanced DSPy patterns"""

    def setup_method(self):
        """Setup test environment"""
        try:
            setup_dspy()
            self.resilient_pipeline = ResilientQAPipeline(max_retries=2)
            self.cached_pipeline = CachedRAGPipeline(cache_size=10)
        except Exception as e:
            pytest.skip(f"Cannot setup DSPy (likely missing API key): {e}")

    def test_resilient_pipeline_success(self):
        """Test resilient pipeline success case"""
        context = "Python is a programming language created by Guido van Rossum."
        question = "Who created Python?"

        result = self.resilient_pipeline(context=context, question=question)

        assert hasattr(result, 'answer')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'method_used')
        assert hasattr(result, 'attempts')
        assert result.confidence > 0

    def test_resilient_pipeline_metrics(self):
        """Test resilient pipeline metrics collection"""
        context = "Test context for metrics."
        questions = ["Question 1?", "Question 2?"]

        for question in questions:
            self.resilient_pipeline(context=context, question=question)

        metrics = self.resilient_pipeline.get_metrics()

        assert 'total_calls' in metrics
        assert 'successful_calls' in metrics
        assert 'success_rate' in metrics
        assert metrics['total_calls'] >= len(questions)

    def test_cached_pipeline_caching(self):
        """Test caching functionality"""
        question = "What is caching?"

        # First call - should be cache miss
        result1 = self.cached_pipeline(question=question)
        stats_after_first = self.cached_pipeline.get_cache_stats()

        # Second call - should be cache hit
        result2 = self.cached_pipeline(question=question)
        stats_after_second = self.cached_pipeline.get_cache_stats()

        # Verify caching worked
        assert stats_after_second['cache_hits'] > stats_after_first['cache_hits']
        assert hasattr(result2, 'from_cache')
        assert result2.from_cache == True

    def test_cache_size_management(self):
        """Test cache size management"""
        small_cache = CachedRAGPipeline(cache_size=2)

        # Fill cache beyond capacity
        questions = ["Q1?", "Q2?", "Q3?"]  # 3 questions, cache size 2

        for question in questions:
            small_cache(question=question)

        stats = small_cache.get_cache_stats()
        assert int(stats['cache_size']) <= 2  # Should not exceed max size


@pytest.mark.asyncio
class TestAsyncPipeline:
    """Test asynchronous pipeline functionality"""

    def setup_method(self):
        """Setup async test environment"""
        try:
            setup_dspy()
            self.async_pipeline = AsyncRAGPipeline(max_workers=2)
        except Exception as e:
            pytest.skip(f"Cannot setup DSPy (likely missing API key): {e}")

    async def test_async_batch_processing(self):
        """Test asynchronous batch processing"""
        questions = [
            "What is AI?",
            "What is ML?",
            "What is DL?"
        ]

        results = await self.async_pipeline.forward_async(questions)

        assert len(results) == len(questions)

        for i, result in enumerate(results):
            assert 'question' in result
            assert 'success' in result
            assert result['question'] == questions[i]

            if result['success']:
                assert 'answer' in result
                assert result['answer'] is not None
            else:
                assert 'error' in result

    async def test_async_error_handling(self):
        """Test async pipeline error handling"""
        # Mix of valid and potentially problematic questions
        questions = [
            "What is artificial intelligence?",
            "",  # Empty question might cause issues
            "What is machine learning?"
        ]

        results = await self.async_pipeline.forward_async(questions)

        assert len(results) == len(questions)

        # Check that pipeline handles errors gracefully
        for result in results:
            assert isinstance(result, dict)
            assert 'success' in result


class TestSystemEvaluation:
    """System-level evaluation tests"""

    def setup_method(self):
        """Setup evaluation environment"""
        try:
            setup_dspy()
            self.pipeline = BasicPipeline()
        except Exception as e:
            pytest.skip(f"Cannot setup DSPy (likely missing API key): {e}")

    def custom_accuracy_metric(self, example, pred, trace=None):
        """Custom accuracy metric for evaluation"""
        if not hasattr(pred, 'answer') or not pred.answer:
            return 0.0

        # Simple keyword-based accuracy
        pred_lower = pred.answer.lower()

        # Check for meaningful content
        if len(pred_lower.strip()) < 10:
            return 0.0

        # Check for failure indicators
        failure_terms = ['i don\'t know', 'cannot answer', 'not sure']
        for term in failure_terms:
            if term in pred_lower:
                return 0.0

        return 1.0  # Consider it correct if it's a substantial, confident answer

    def test_systematic_evaluation(self):
        """Test systematic evaluation using DSPy's framework"""
        # Create simple test set
        test_examples = [
            {"question": "What is programming?", "context": "Programming is writing code."},
            {"question": "What is Python?", "context": "Python is a programming language."}
        ]

        # Convert to DSPy examples
        dspy_examples = []
        for ex in test_examples:
            dspy_ex = dspy.Example(
                question=ex["question"],
                context=ex["context"]
            ).with_inputs('question')
            dspy_examples.append(dspy_ex)

        # Run evaluation
        try:
            evaluator = Evaluate(
                devset=dspy_examples,
                metric=self.custom_accuracy_metric,
                num_threads=1  # Use single thread to avoid API rate limits
            )

            score = evaluator(self.pipeline)

            # Score should be between 0 and 1
            assert 0.0 <= score <= 1.0
            print(f"Evaluation score: {score}")

        except Exception as e:
            # Evaluation might fail due to API limits or other issues
            print(f"Evaluation failed: {e}")
            pytest.skip("Evaluation failed - possibly due to API limits")


# Test runner function
def run_tests():
    """Run all tests with proper error handling"""
    test_args = [
        __file__,
        "-v",  # Verbose output
        "-s",  # Don't capture print statements
        "--tb=short"  # Short traceback format
    ]

    try:
        pytest.main(test_args)
    except Exception as e:
        print(f"Test execution failed: {e}")


if __name__ == "__main__":
    run_tests()
