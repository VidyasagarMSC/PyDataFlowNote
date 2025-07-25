import dspy
import time
from pydantic import BaseModel, Field, field_validator, ValidationError, ConfigDict
from typing import List, Optional, Dict, Any, Union
from dotenv import load_dotenv
from enum import Enum
from datetime import datetime

# Handle both module import and direct script execution
try:
    from .dspy_setup import setup_dspy_basic
    from .logfire_setup import get_logfire_manager, logfire_span
except ImportError:
    from dspy_setup import setup_dspy_basic
    from logfire_setup import get_logfire_manager, logfire_span

# Load environment variables
load_dotenv()


class SentimentEnum(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class AnalysisResult(BaseModel):
    """Pydantic model for structured text analysis results with Logfire integration"""
    model_config = ConfigDict(
        # Enable Logfire integration
        plugin_settings={
            'logfire': {'record': 'all'}
        },
        # Additional validation settings
        validate_assignment=True,
        str_strip_whitespace=True,
        frozen=False
    )
    sentiment: SentimentEnum = Field(..., description="Sentiment classification")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    key_themes: List[str] = Field(..., min_length=1, max_length=5, description="Main themes")
    summary: str = Field(..., min_length=10, max_length=200, description="Brief summary")
    word_count: int = Field(..., gt=0, description="Number of words in original text")
    # Additional metadata fields
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    model_used: Optional[str] = Field(None, description="Model used for analysis")
    timestamp: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")
    request_id: Optional[str] = Field(None, description="Unique request identifier")

    @field_validator('key_themes')
    def validate_themes(cls, v):
        """Clean and validate themes"""
        cleaned = [theme.strip().lower() for theme in v if theme.strip()]
        if not cleaned:
            raise ValueError("At least one theme is required")
        return cleaned[:5]  # Limit to 5 themes

    @field_validator('summary')
    def validate_summary(cls, v):
        """Ensure summary is meaningful"""
        cleaned = v.strip()
        if len(cleaned.split()) < 3:
            raise ValueError("Summary must contain at least 3 words")
        return cleaned


class TextAnalysis(dspy.Signature):
    """Analyze text for sentiment, themes, and provide summary."""
    text = dspy.InputField(desc="Text to analyze")
    sentiment = dspy.OutputField(desc="Sentiment: positive, negative, or neutral")
    confidence = dspy.OutputField(desc="Confidence score between 0 and 1")
    themes = dspy.OutputField(desc="Key themes as comma-separated list")
    summary = dspy.OutputField(desc="Brief summary of the text")


class AnalysisModule(dspy.Module):
    """DSPy module that produces Pydantic-validated outputs with Logfire observability"""

    def __init__(self):
        super().__init__()
        self.analyzer = dspy.ChainOfThought(TextAnalysis)
        self.logfire_manager = get_logfire_manager()

    @logfire_span("text_analysis", component="analysis_module")
    def forward(self, text: str, request_id: str = None) -> AnalysisResult:
        """Analyze text and return validated results with full observability"""
        start_time = time.time()
        
        # Generate request ID if not provided
        if not request_id:
            request_id = f"analysis_{int(time.time() * 1000)}"
        
        # Log analysis start
        self.logfire_manager.log_event(
            "Starting text analysis",
            "info",
            request_id=request_id,
            text_length=len(text),
            word_count=len(text.split())
        )
        
        try:
            # Get raw prediction from DSPy with span
            with self.logfire_manager.span("dspy_prediction", request_id=request_id) as span:
                prediction = self.analyzer(text=text)
                if span:
                    span.set_attribute("prediction_generated", True)

            # Parse and validate the output using Pydantic
            with self.logfire_manager.span("pydantic_validation", request_id=request_id) as span:
                result_data = self._parse_analysis_output(prediction, text)
                result_data.update({
                    'processing_time': time.time() - start_time,
                    'request_id': request_id,
                    'model_used': getattr(prediction, '_model_name', 'unknown')
                })
                
                validated_result = AnalysisResult(**result_data)
                
                if span:
                    span.set_attribute("validation_successful", True)
                    span.set_attribute("sentiment", validated_result.sentiment)
                    span.set_attribute("confidence", validated_result.confidence)
            
            # Log successful completion
            self.logfire_manager.log_event(
                "Text analysis completed successfully",
                "info",
                request_id=request_id,
                sentiment=validated_result.sentiment,
                confidence=validated_result.confidence,
                processing_time=validated_result.processing_time
            )
            
            return validated_result
            
        except ValidationError as e:
            # Handle validation errors with fallback
            self.logfire_manager.log_error(
                e, 
                "Pydantic validation failed",
                request_id=request_id,
                error_details=e.errors()
            )
            return self._handle_validation_error(text, e, request_id)
            
        except Exception as e:
            # Handle other errors
            self.logfire_manager.log_error(
                e,
                "General analysis error", 
                request_id=request_id
            )
            return self._handle_general_error(text, e, request_id)

    @staticmethod
    def _parse_analysis_output(prediction, original_text: str) -> Dict[str, Any]:
        """Parse DSPy output into structured format"""
        # Extract sentiment
        sentiment_raw = getattr(prediction, 'sentiment', 'neutral').lower().strip()
        if sentiment_raw not in ['positive', 'negative', 'neutral']:
            sentiment_raw = 'neutral'

        # Extract confidence (with fallback)
        try:
            confidence_raw = getattr(prediction, 'confidence', '0.5')
            if isinstance(confidence_raw, str):
                # Try to extract number from string
                confidence = float(''.join(c for c in confidence_raw if c.isdigit() or c == '.'))
                confidence = max(0.0, min(1.0, confidence))
            else:
                confidence = float(confidence_raw)
        except:
            confidence = 0.5

        # Extract themes
        themes_raw = getattr(prediction, 'themes', 'general topic')
        if isinstance(themes_raw, str):
            themes = [theme.strip() for theme in themes_raw.split(',')]
        else:
            themes = ['general topic']

        # Extract summary with fallback
        summary = getattr(prediction, 'summary', f"Analysis of {len(original_text.split())} word text")
        if len(summary.strip()) < 10:
            summary = f"This text contains {len(original_text.split())} words and discusses various topics."

        return {
            'sentiment': sentiment_raw,
            'confidence': confidence,
            'key_themes': themes,
            'summary': summary,
            'word_count': len(original_text.split())
        }

    def _handle_validation_error(self, text: str, error: ValidationError, request_id: str) -> AnalysisResult:
        """Handle Pydantic validation errors with safe fallback and logging"""
        self.logfire_manager.log_error(
            error,
            "Pydantic validation failed - using fallback",
            request_id=request_id,
            error_count=len(error.errors()),
            fallback_used=True
        )

        # Create safe fallback result
        return AnalysisResult(
            sentiment=SentimentEnum.NEUTRAL,
            confidence=0.5,
            key_themes=['general content'],
            summary=f"Text analysis with {len(text.split())} words",
            word_count=len(text.split()),
            request_id=request_id,
            model_used='fallback'
        )

    def _handle_general_error(self, text: str, error: Exception, request_id: str) -> AnalysisResult:
        """Handle general errors with safe fallback and logging"""
        self.logfire_manager.log_error(
            error,
            "General processing error - using fallback",
            request_id=request_id,
            fallback_used=True
        )

        return AnalysisResult(
            sentiment=SentimentEnum.NEUTRAL,
            confidence=0.0,
            key_themes=['error processing'],
            summary="Unable to process text due to an error",
            word_count=len(text.split()),
            request_id=request_id,
            model_used='error_fallback'
        )


class QueryInput(BaseModel):
    """Pydantic model for validating user queries"""
    question: str = Field(..., min_length=5, max_length=500, description="User question")
    context_filter: Optional[str] = Field(None, max_length=100, description="Optional context filter")
    max_results: int = Field(default=5, ge=1, le=20, description="Maximum number of results")
    include_reasoning: bool = Field(default=True, description="Include reasoning in response")

    @field_validator('question')
    def sanitize_question(cls, v):
        """Sanitize and validate question"""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Question cannot be empty")

        # Remove potentially harmful patterns (basic sanitization)
        harmful_patterns = ['<script>', '</script>', 'javascript:', 'eval(']
        for pattern in harmful_patterns:
            if pattern.lower() in cleaned.lower():
                raise ValueError("Question contains potentially harmful content")

        return cleaned

    @field_validator('context_filter')
    def validate_context_filter(cls, v):
        """Validate context filter if provided"""
        if v is not None:
            return v.strip()
        return v


class ValidatedRAGPipeline(dspy.Module):
    """RAG pipeline with Pydantic input validation"""

    def __init__(self):
        super().__init__()
        self.basic_qa = dspy.ChainOfThought(
            dspy.make_signature("context, question -> reasoning, answer")
        )

    def forward(self, query_input: QueryInput) -> Dict[str, Any]:
        """Process validated query input"""
        # Input is already validated by Pydantic
        context = self._get_context(query_input.context_filter)

        # Generate answer
        prediction = self.basic_qa(
            context=context,
            question=query_input.question
        )

        # Structure response
        response = {
            'question': query_input.question,
            'answer': getattr(prediction, 'answer', 'No answer generated'),
            'context': context,
            'max_results': query_input.max_results
        }

        if query_input.include_reasoning:
            response['reasoning'] = getattr(prediction, 'reasoning', 'No reasoning provided')

        return response

    @staticmethod
    def _get_context(filter_term: Optional[str] = None) -> str:
        """Get context based on filter"""
        contexts = {
            'ai': "Artificial Intelligence is the simulation of human intelligence in machines.",
            'ml': "Machine Learning is a subset of AI that learns from data.",
            'python': "Python is a versatile programming language popular in data science.",
            None: "General knowledge context about technology and programming."
        }

        return contexts.get(filter_term, contexts[None])


def demonstrate_pydantic_integration():
    """Demonstrate Pydantic integration with DSPy"""
    setup_dspy_basic()

    print("=== Text Analysis with Pydantic Validation ===")

    # Create analysis module
    analyzer = AnalysisModule()

    # Test with sample text
    sample_text = """
    DSPy is an innovative framework that revolutionizes how we work with language models.
    It provides a systematic approach to building reliable AI applications through
    declarative programming and automatic optimization. This makes development more
    efficient and results more predictable.
    """

    try:
        result = analyzer(sample_text.strip())
        print(f"Text: {sample_text.strip()}")
        print(f"Sentiment: {result.sentiment}")
        print(f"Confidence: {result.confidence}")
        print(f"Themes: {result.key_themes}")
        print(f"Summary: {result.summary}")
        print(f"Word Count: {result.word_count}")

    except Exception as e:
        print(f"Analysis failed: {e}")

    print("\n=== RAG Pipeline with Input Validation ===")

    # Create validated RAG pipeline
    rag_pipeline = ValidatedRAGPipeline()

    # Test with valid input
    try:
        valid_query = QueryInput(
            question="What is machine learning?",
            context_filter="ml",
            max_results=3,
            include_reasoning=True
        )

        result = rag_pipeline(valid_query)
        print("Valid Query Results:")
        for key, value in result.items():
            print(f"  {key}: {value}")

    except ValidationError as e:
        print(f"Validation error: {e}")

    # Test with invalid input
    print("\n=== Testing Input Validation ===")
    try:
        invalid_query = QueryInput(
            question="Hi",  # Too short
            max_results=25  # Too large
        )
    except ValidationError as e:
        print("Caught validation errors as expected:")
        for error in e.errors():
            print(f"  - {error['loc'][0]}: {error['msg']}")


if __name__ == "__main__":
    demonstrate_pydantic_integration()
