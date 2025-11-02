"""
Quick test to verify the Q&A endpoint fix for None context handling.

This script tests that the Q&A endpoint doesn't crash with 'NoneType' subscriptable error
when request.context is None.
"""

import requests
import json
import sys

def test_qa_with_none_context():
    """Test Q&A endpoint with None context (default case)."""

    url = "http://localhost:8000/research/ask"

    # Test 1: Minimal request (context is None by default)
    print("Test 1: Q&A with no context field (defaults to None)")
    payload = {
        "question": "What is machine learning?"
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
        print(f"Status code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"✅ SUCCESS: Answer received")
            print(f"   Answer: {data['answer'][:100]}...")
            print(f"   Sources: {len(data['sources'])} sources")
            print(f"   Confidence: {data['confidence_score']:.2f}")
            return True
        else:
            print(f"❌ FAILED: Status {response.status_code}")
            print(f"   Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ EXCEPTION: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_qa_with_explicit_none_context():
    """Test Q&A endpoint with explicitly null context."""

    url = "http://localhost:8000/research/ask"

    print("\nTest 2: Q&A with explicit null context")
    payload = {
        "question": "What are neural networks?",
        "context": None,
        "user_id": "test_user",
        "conversation_id": "test_conv_123"
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
        print(f"Status code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"✅ SUCCESS: Answer received")
            print(f"   Answer: {data['answer'][:100]}...")
            return True
        else:
            print(f"❌ FAILED: Status {response.status_code}")
            print(f"   Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ EXCEPTION: {type(e).__name__}: {str(e)}")
        return False


def test_qa_with_valid_context():
    """Test Q&A endpoint with valid context."""

    url = "http://localhost:8000/research/ask"

    print("\nTest 3: Q&A with valid context string")
    payload = {
        "question": "How do transformers work?",
        "context": "I am researching attention mechanisms in deep learning",
        "user_id": "test_user",
        "conversation_id": "test_conv_456"
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
        print(f"Status code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"✅ SUCCESS: Answer received")
            print(f"   Answer: {data['answer'][:100]}...")
            return True
        else:
            print(f"❌ FAILED: Status {response.status_code}")
            print(f"   Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ EXCEPTION: {type(e).__name__}: {str(e)}")
        return False


def test_health_check():
    """Test that the API is running."""

    print("Checking API health...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print(f"✅ API is healthy")
            return True
        else:
            print(f"⚠️  API returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("   Make sure the research_api container is running:")
        print("   docker-compose ps research_api")
        return False


if __name__ == "__main__":
    print("=" * 80)
    print("Q&A Endpoint Fix Verification")
    print("=" * 80)
    print()

    # Check if API is running
    if not test_health_check():
        print("\n❌ API is not accessible. Exiting.")
        sys.exit(1)

    print()
    print("-" * 80)

    # Run all tests
    results = []
    results.append(("None context (default)", test_qa_with_none_context()))
    results.append(("Explicit null context", test_qa_with_explicit_none_context()))
    results.append(("Valid context", test_qa_with_valid_context()))

    print()
    print("=" * 80)
    print("Test Results Summary")
    print("=" * 80)

    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")

    print()

    # Overall result
    all_passed = all(result[1] for result in results)
    if all_passed:
        print("🎉 All tests passed! The Q&A endpoint handles None context correctly.")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed. Review the output above.")
        sys.exit(1)
