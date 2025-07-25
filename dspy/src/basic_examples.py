"""
Basic DSPy Examples

Simple examples demonstrating core DSPy concepts:
- Basic Q&A
- Chain of thought reasoning
- RAG (Retrieval Augmented Generation)
- Pipeline optimization
"""

import dspy
from typing import Optional, List
from dspy_setup import setup_dspy_basic
from dspy.teleprompt import BootstrapFewShot



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
    """Simple pipeline that can handle basic Q&A and RAG."""
    
    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought(BasicQA)
        self.rag = dspy.ChainOfThought(RAG)
    
    def forward(self, question: str, context: Optional[str] = None):
        """Answer question with or without context."""
        if context:
            return self.rag(context=context, question=question)
        return self.qa(question=question)


class RAGPipeline(dspy.Module):
    """RAG pipeline with document handling."""

    def __init__(self, max_docs: int = 3):
        super().__init__()
        self.rag = dspy.ChainOfThought(RAG)
        self.max_docs = max_docs

    def forward(self, question: str, documents: Optional[List[str]] = None):
        """Answer question using document context."""
        if not documents:
            documents = [
                "DSPy is a framework for programming language models.",
                "It provides signatures, modules, and optimizers.",
                "DSPy enables automatic optimization of prompts."
            ]
        
        context = " ".join(documents[:self.max_docs])
        result = self.rag(context=context, question=question)
        
        return dspy.Prediction(
            context=context,
            answer=result.answer
        )



# === Example Functions ===

def example_basic_qa():
    """Example 1: Basic question answering."""
    print("\n=== Example 1: Basic Q&A ===")
    
    pipeline = BasicPipeline()
    question = "What is artificial intelligence?"
    result = pipeline(question=question)
    
    print(f"Question: {question}")
    print(f"Answer: {result.answer}")


def example_rag():
    """Example 2: RAG with context."""
    print("\n=== Example 2: RAG with Context ===")
    
    pipeline = BasicPipeline()
    context = "Machine learning is a subset of AI that uses algorithms to learn from data."
    question = "What is machine learning?"
    result = pipeline(question=question, context=context)
    
    print(f"Question: {question}")
    print(f"Context: {context}")
    print(f"Answer: {result.answer}")


def example_chain_of_thought():
    """Example 3: Chain of thought reasoning."""
    print("\n=== Example 3: Chain of Thought ===")
    
    cot = dspy.ChainOfThought(ChainOfThoughtQA)
    question = "Why is renewable energy important?"
    result = cot(question=question)
    
    print(f"Question: {question}")
    print(f"Reasoning: {result.reasoning}")
    print(f"Answer: {result.answer}")


def example_rag_pipeline():
    """Example 4: RAG pipeline with documents."""
    print("\n=== Example 4: RAG Pipeline ===")
    
    rag_pipeline = RAGPipeline()
    question = "What does DSPy provide?"
    result = rag_pipeline(question=question)
    
    print(f"Question: {question}")
    print(f"Answer: {result.answer}")
    print(f"Context used: {result.context[:100]}...")


# === Optimization Example ===

def create_sample_dataset():
    """Create simple dataset for optimization using sample data."""
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
    from util import simple_metric as util_metric
    return util_metric(example, pred, trace)


def example_optimization():
    """Example 5: Pipeline optimization."""
    print("\n=== Example 5: Pipeline Optimization ===")
    
    # Create dataset and pipeline
    dataset = create_sample_dataset()
    pipeline = BasicPipeline()
    
    print("Before optimization:")
    result = pipeline(question="What is AI?")
    print(f"Answer: {result.answer}")
    
    # Optimize pipeline
    try:
        optimizer = BootstrapFewShot(metric=simple_metric, max_bootstrapped_demos=2)
        optimized = optimizer.compile(pipeline, trainset=dataset[:2])
        
        print("\nAfter optimization:")
        result = optimized(question="What is AI?")
        print(f"Answer: {result.answer}")
        
    except Exception as e:
        print(f"Optimization failed: {e}")


# === Main Execution ===

def run_all_examples():
    """Run all examples in sequence."""
    setup_dspy_basic()
    
    print("DSPy Basic Examples")
    print("=" * 50)
    
    example_basic_qa()
    example_rag()
    example_chain_of_thought()
    example_rag_pipeline()
    example_optimization()
    
    print("\n" + "=" * 50)
    print("All examples completed!")


if __name__ == "__main__":
    run_all_examples()
    