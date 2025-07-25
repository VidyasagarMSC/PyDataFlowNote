"""
Basic DSPy Examples

Simple examples demonstrating core DSPy concepts:
- Basic Q&A
- Chain of thought reasoning
- RAG (Retrieval Augmented Generation)
- Pipeline optimization
"""

import dspy
import time
from typing import Optional, List
from dspy.teleprompt import BootstrapFewShot

# Handle both module import and direct script execution
try:
    from .dspy_setup import setup_dspy_basic
    from .logfire_setup import get_logfire_manager, logfire_span, logfire_log
except ImportError:
    from dspy_setup import setup_dspy_basic
    from logfire_setup import get_logfire_manager, logfire_span, logfire_log

# Initialize Logfire manager
logfire_manager = get_logfire_manager()



# === Core Signatures ===

class BasicQA(dspy.Signature):
    """Simple question answering."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Clear, helpful answer")


class ChainOfThoughtQA(dspy.Signature):
    """Question answering with reasoning."""
    question = dspy.InputField()
    reasoning = dspy.OutputField(desc="Step-by-step reasoning")
    answer = dspy.OutputField(desc="Final answer")


class RAG(dspy.Signature):
    """Answer questions using provided context."""
    context = dspy.InputField(desc="Relevant information")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Answer based on context")



# === Pipeline Modules ===

class BasicPipeline(dspy.Module):
    """Simple pipeline that can handle basic Q&A and RAG with Logfire monitoring."""
    
    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought(BasicQA)
        self.rag = dspy.ChainOfThought(RAG)
        logfire_manager.log_event("BasicPipeline initialized", "info", component="basic_pipeline")
    
    @logfire_span("basic_pipeline_forward", component="basic_examples")
    def forward(self, question: str, context: Optional[str] = None):
        """Answer question with or without context."""
        logfire_manager.log_event(
            "Processing question in BasicPipeline",
            "info",
            question_length=len(question),
            has_context=context is not None,
            pipeline_type="rag" if context else "qa"
        )
        
        start_time = time.time()
        try:
            if context:
                result = self.rag(context=context, question=question)
                logfire_manager.log_event(
                    "RAG processing completed",
                    "info",
                    processing_time=time.time() - start_time,
                    context_length=len(context)
                )
            else:
                result = self.qa(question=question)
                logfire_manager.log_event(
                    "QA processing completed",
                    "info",
                    processing_time=time.time() - start_time
                )
            return result
        except Exception as e:
            logfire_manager.log_error(e, "BasicPipeline forward failed", 
                                     question=question[:100],
                                     has_context=context is not None)
            raise


class RAGPipeline(dspy.Module):
    """RAG pipeline with document handling and Logfire monitoring."""

    def __init__(self, max_docs: int = 3):
        super().__init__()
        self.rag = dspy.ChainOfThought(RAG)
        self.max_docs = max_docs
        logfire_manager.log_event(
            "RAGPipeline initialized", 
            "info", 
            component="rag_pipeline",
            max_docs=max_docs
        )

    @logfire_span("rag_pipeline_forward", component="basic_examples")
    def forward(self, question: str, documents: Optional[List[str]] = None):
        """Answer question using document context."""
        if not documents:
            documents = [
                "DSPy is a framework for programming language models.",
                "It provides signatures, modules, and optimizers.",
                "DSPy enables automatic optimization of prompts."
            ]
            logfire_manager.log_event(
                "Using default documents", 
                "info", 
                default_docs_count=len(documents)
            )
        
        start_time = time.time()
        
        logfire_manager.log_event(
            "Processing RAG query",
            "info",
            question_length=len(question),
            document_count=len(documents),
            max_docs=self.max_docs
        )
        
        try:
            context = " ".join(documents[:self.max_docs])
            result = self.rag(context=context, question=question)
            
            processing_time = time.time() - start_time
            logfire_manager.log_event(
                "RAG pipeline completed",
                "info",
                processing_time=processing_time,
                context_length=len(context),
                documents_used=min(len(documents), self.max_docs)
            )
            
            return dspy.Prediction(
                context=context,
                answer=result.answer
            )
        except Exception as e:
            logfire_manager.log_error(e, "RAGPipeline forward failed",
                                     question=question[:100],
                                     document_count=len(documents) if documents else 0)
            raise



# === Example Functions ===

@logfire_span("example_basic_qa", component="basic_examples")
def example_basic_qa():
    """Example 1: Basic question answering."""
    print("\n=== Example 1: Basic Q&A ===")
    logfire_manager.log_event("Starting basic QA example", "info", example="basic_qa")
    
    pipeline = BasicPipeline()
    question = "What is artificial intelligence?"
    result = pipeline(question=question)
    
    logfire_manager.log_event(
        "Basic QA example completed", 
        "info", 
        question=question,
        answer_length=len(result.answer) if hasattr(result, 'answer') else 0
    )
    
    print(f"Question: {question}")
    print(f"Answer: {result.answer}")


@logfire_span("example_rag", component="basic_examples")
def example_rag():
    """Example 2: RAG with context."""
    print("\n=== Example 2: RAG with Context ===")
    logfire_manager.log_event("Starting RAG example", "info", example="rag")
    
    pipeline = BasicPipeline()
    context = "Machine learning is a subset of AI that uses algorithms to learn from data."
    question = "What is machine learning?"
    result = pipeline(question=question, context=context)
    
    logfire_manager.log_event(
        "RAG example completed",
        "info",
        question=question,
        context_length=len(context),
        answer_length=len(result.answer) if hasattr(result, 'answer') else 0
    )
    
    print(f"Question: {question}")
    print(f"Context: {context}")
    print(f"Answer: {result.answer}")


@logfire_span("example_chain_of_thought", component="basic_examples")
def example_chain_of_thought():
    """Example 3: Chain of thought reasoning."""
    print("\n=== Example 3: Chain of Thought ===")
    logfire_manager.log_event("Starting chain of thought example", "info", example="chain_of_thought")
    
    cot = dspy.ChainOfThought(ChainOfThoughtQA)
    question = "Why is renewable energy important?"
    result = cot(question=question)
    
    logfire_manager.log_event(
        "Chain of thought example completed",
        "info",
        question=question,
        reasoning_length=len(result.reasoning) if hasattr(result, 'reasoning') else 0
    )
    
    print(f"Question: {question}")
    print(f"Reasoning: {result.reasoning}")
    print(f"Answer: {result.answer}")


@logfire_span("example_rag_pipeline", component="basic_examples")
def example_rag_pipeline():
    """Example 4: RAG pipeline with documents."""
    print("\n=== Example 4: RAG Pipeline ===")
    logfire_manager.log_event("Starting RAG pipeline example", "info", example="rag_pipeline")
    
    rag_pipeline = RAGPipeline()
    question = "What does DSPy provide?"
    result = rag_pipeline(question=question)
    
    logfire_manager.log_event(
        "RAG pipeline example completed",
        "info",
        question=question,
        answer_length=len(result.answer) if hasattr(result, 'answer') else 0,
        context_used_len=len(result.context) if hasattr(result, 'context') else 0
    )
    
    print(f"Question: {question}")
    print(f"Answer: {result.answer}")
    print(f"Context used: {result.context[:100]}...")


# === Optimization Example ===

def create_sample_dataset():
    """Create simple dataset for optimization using sample data."""
    try:
        from .util import create_training_examples
    except ImportError:
        from util import create_training_examples
    
    try:
        return create_training_examples()
    except Exception as e:
        print(f"Could not load sample data: {e}")
        # Fallback to hardcoded examples
        return [
            dspy.Example(
                question="What is AI?",
                answer="AI is artificial intelligence - machines simulating human intelligence."
            ).with_inputs('question'),
            dspy.Example(
                question="What is ML?", 
                answer="ML is machine learning - algorithms that learn from data."
            ).with_inputs('question'),
            dspy.Example(
                question="What is NLP?",
                answer="NLP is natural language processing - AI for understanding text."
            ).with_inputs('question')
        ]


def simple_metric(example, pred, trace=None):
    """Simple evaluation metric."""
    try:
        from .util import simple_metric as util_metric
    except ImportError:
        from util import simple_metric as util_metric
    return util_metric(example, pred, trace)


@logfire_span("example_optimization", component="basic_examples")
def example_optimization():
    """Example 5: Pipeline optimization."""
    print("\n=== Example 5: Pipeline Optimization ===")
    logfire_manager.log_event("Starting optimization example", "info", example="optimization")
    
    # Create dataset and pipeline
    dataset = create_sample_dataset()
    pipeline = BasicPipeline()
    
    print("Before optimization:")
    result = pipeline(question="What is AI?")
    logfire_manager.log_event("Before optimization", "info", initial_answer=result.answer)
    print(f"Answer: {result.answer}")
    
    # Optimize pipeline
    try:
        optimizer = BootstrapFewShot(metric=simple_metric, max_bootstrapped_demos=2)
        optimized = optimizer.compile(pipeline, trainset=dataset[:2])
        
        print("\nAfter optimization:")
        result = optimized(question="What is AI?")
        logfire_manager.log_event("After optimization", "info", optimized_answer=result.answer)
        print(f"Answer: {result.answer}")
        
    except Exception as e:
        logfire_manager.log_error(e, "Optimization failed in example_optimization")
        print(f"Optimization failed: {e}")


# === Main Execution ===

@logfire_span("run_all_examples", component="basic_examples")
def run_all_examples():
    """Run all examples in sequence."""
    logfire_manager.log_event("Starting all basic examples", "info", total_examples=5)
    
    start_time = time.time()
    setup_dspy_basic()
    
    print("DSPy Basic Examples")
    print("=" * 50)
    
    try:
        example_basic_qa()
        example_rag()
        example_chain_of_thought()
        example_rag_pipeline()
        example_optimization()
        
        total_time = time.time() - start_time
        logfire_manager.log_event(
            "All examples completed successfully",
            "info",
            total_duration=total_time,
            examples_completed=5
        )
        
        print("\n" + "=" * 50)
        print("All examples completed!")
        
    except Exception as e:
        logfire_manager.log_error(e, "Failed during basic examples execution")
        print(f"\nError during examples: {e}")
        raise


if __name__ == "__main__":
    run_all_examples()
    