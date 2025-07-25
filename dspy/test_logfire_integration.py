#!/usr/bin/env python3
"""
Comprehensive Test Suite for Logfire Integration

This test suite validates all Logfire integrations across the project,
including setup, Pydantic models, monitoring, and complete workflows.
"""

import sys
import time
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_logfire_setup():
    """Test basic Logfire setup and initialization"""
    print("🔧 Testing Logfire Setup...")
    
    from logfire_setup import get_logfire_manager, initialize_logfire
    
    # Test initialization
    result = initialize_logfire()
    assert result, "Logfire initialization failed"
    print("   ✅ Logfire initialized successfully")
    
    # Test manager creation
    manager = get_logfire_manager()
    assert manager.is_initialized(), "Logfire manager not initialized"
    print("   ✅ Logfire manager created and initialized")
    
    # Test context management
    manager.set_context(test_session="logfire_test", user="test_user")
    context = manager.get_context()
    assert "test_session" in context, "Context not set properly"
    print("   ✅ Context management working")
    
    # Test logging
    manager.log_event("Test event from test suite", "info", test_attribute="test_value")
    manager.log_metrics({"test_metric": 42, "another_metric": 3.14})
    print("   ✅ Event and metrics logging working")
    
    manager.clear_context()
    print("✅ Logfire Setup Test: PASSED\n")


def test_pydantic_integration():
    """Test Pydantic models with Logfire integration"""
    print("📋 Testing Pydantic Integration...")
    
    from pydantic_integration import AnalysisResult, SentimentEnum, QueryInput
    
    # Test AnalysisResult model
    result = AnalysisResult(
        sentiment=SentimentEnum.POSITIVE,
        confidence=0.9,
        key_themes=["test", "integration", "success"],
        summary="This is a test summary for validating Pydantic integration with Logfire",
        word_count=12,
        request_id="test_pydantic_001"
    )
    assert result.sentiment == SentimentEnum.POSITIVE, "Sentiment not set correctly"
    print("   ✅ AnalysisResult model validation working")
    
    # Test QueryInput model
    query = QueryInput(
        question="How does Pydantic integrate with Logfire for observability?",
        max_results=5,
        include_reasoning=True
    )
    assert len(query.question) >= 5, "Query validation failed"
    print("   ✅ QueryInput model validation working")
    
    print("✅ Pydantic Integration Test: PASSED\n")


def test_monitoring_integration():
    """Test monitoring system with Logfire"""
    print("📊 Testing Monitoring Integration...")
    
    from monitoring import get_monitoring_manager, monitor_function
    
    # Test monitoring manager
    manager = get_monitoring_manager()
    assert manager is not None, "Monitoring manager creation failed"
    print("   ✅ Monitoring manager created")
    
    # Test metrics recording
    manager.increment_counter("test_operations", 3)
    manager.set_gauge("test_temperature", 25.5)
    manager.record_timer("test_operation_time", 1.5)
    print("   ✅ Metrics recording working")
    
    # Test decorated function monitoring
    @monitor_function("test_monitored_function")
    def test_function(delay=0.1):
        time.sleep(delay)
        return "test_result"
    
    result = test_function(0.05)
    assert result == "test_result", "Monitored function failed"
    print("   ✅ Function monitoring decorator working")
    
    # Test health check
    health = manager.perform_health_check()
    assert health.status.value in ["healthy", "degraded", "unhealthy"], "Health check failed"
    print(f"   ✅ Health check working (status: {health.status.value})")
    
    print("✅ Monitoring Integration Test: PASSED\n")


def test_complete_workflow():
    """Test complete workflow integration"""
    print("🔄 Testing Complete Workflow...")
    
    from complete_example import ProcessingRequest, ProductionPipeline
    
    # Create pipeline
    pipeline = ProductionPipeline()
    print("   ✅ Production pipeline created")
    
    # Create test request
    request = ProcessingRequest(
        text="This is a comprehensive test of the entire Logfire and Pydantic integration system. It validates that all components work together seamlessly.",
        analysis_type="sentiment",
        user_id="test_user_workflow",
        priority=2
    )
    print("   ✅ Processing request created and validated")
    
    # Process request
    response = pipeline.process_request(request)
    assert response.success, f"Request processing failed: {response.errors}"
    assert "analysis" in response.result, "Analysis result missing"
    print("   ✅ Request processed successfully")
    
    # Test health check
    health = pipeline.health_check()
    assert health["status"] in ["healthy", "degraded", "unhealthy"], "Health check failed"
    print(f"   ✅ Pipeline health check working (status: {health['status']})")
    
    print("✅ Complete Workflow Test: PASSED\n")


def test_error_handling():
    """Test error handling and logging"""
    print("⚠️ Testing Error Handling...")
    
    from logfire_setup import get_logfire_manager
    from pydantic_integration import AnalysisResult, SentimentEnum
    from pydantic import ValidationError
    
    manager = get_logfire_manager()
    
    # Test error logging
    try:
        raise ValueError("Test error for validation")
    except Exception as e:
        manager.log_error(e, "Test error context", test_info="error_handling_test")
    print("   ✅ Error logging working")
    
    # Test Pydantic validation errors
    try:
        # This should fail validation
        result = AnalysisResult(
            sentiment="invalid_sentiment",  # Invalid enum value
            confidence=1.5,  # Out of range
            key_themes=[],  # Empty list (min_length=1)
            summary="Short",  # Too short (min_length=10)
            word_count=-1  # Negative number
        )
    except ValidationError as e:
        print("   ✅ Pydantic validation errors caught correctly")
        manager.log_error(e, "Pydantic validation test", validation_errors=len(e.errors()))
    
    print("✅ Error Handling Test: PASSED\n")


def test_performance_monitoring():
    """Test performance monitoring capabilities"""
    print("⚡ Testing Performance Monitoring...")
    
    from logfire_setup import logfire_span, logfire_log
    from monitoring import get_monitoring_manager
    
    manager = get_monitoring_manager()
    
    # Test span decorator performance
    @logfire_span("performance_test_span", component="test_suite")
    def performance_test_function():
        time.sleep(0.1)
        manager.increment_counter("performance_test_operations")
        return "performance_test_complete"
    
    start_time = time.time()
    result = performance_test_function()
    duration = time.time() - start_time
    
    assert result == "performance_test_complete", "Performance test function failed"
    assert duration >= 0.1, "Performance test didn't take expected time"
    print(f"   ✅ Performance monitoring working (duration: {duration:.3f}s)")
    
    # Test metrics collection
    metrics_summary = manager.get_metrics_summary()
    assert "counters" in metrics_summary, "Metrics summary missing counters"
    assert "requests" in metrics_summary, "Metrics summary missing requests"
    print("   ✅ Metrics collection and summary working")
    
    print("✅ Performance Monitoring Test: PASSED\n")


def run_all_tests():
    """Run all Logfire integration tests"""
    print("🚀 Starting Comprehensive Logfire Integration Test Suite")
    print("=" * 60)
    
    tests = [
        test_logfire_setup,
        test_pydantic_integration,
        test_monitoring_integration,
        test_complete_workflow,
        test_error_handling,
        test_performance_monitoring
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} FAILED: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
            print()
    
    print("=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All Logfire integration tests PASSED!")
        print("🔍 Check your Logfire dashboard at: https://logfire.pydantic.dev/")
        print("   You should see logs, metrics, and spans from this test suite")
    else:
        print("💥 Some tests FAILED - check the output above for details")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
