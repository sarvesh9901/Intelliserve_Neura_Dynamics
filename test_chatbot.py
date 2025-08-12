# test_chatbot.py
import re
import pytest
import datetime
from main import call_graph  # Assuming your main script is named main.py

# Storage for test results
test_results = []
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_result(test_name, query, response, status):
    """Store test results for later writing to a file."""
    test_results.append({
        "test_name": test_name,
        "query": query,
        "response": response,
        "status": status
    })


def test_weather_in_mumbai():
    """Test that the chatbot returns Mumbai weather information."""
    query = "tell me weather in mumbai"
    response = call_graph(query)
    status = "PASS"
    try:
        assert isinstance(response, str)
        assert "mumbai" in response.lower()
        assert any(unit in response.lower() for unit in ["°c", "celsius", "temperature"])
    except AssertionError:
        status = "FAIL"
        raise
    finally:
        log_result("test_weather_in_mumbai", query, response, status)


def test_weather_in_invalid_city():
    """Test that the chatbot handles unknown city gracefully."""
    query = "tell me weather in asdfghjkl"
    response = call_graph(query)
    status = "PASS"
    try:
        assert isinstance(response, str)
        assert "error" in response.lower() or "not found" in response.lower()
    except AssertionError:
        status = "FAIL"
        raise
    finally:
        log_result("test_weather_in_invalid_city", query, response, status)


def test_find_match_known_query():
    """Test that the chatbot retrieves known answers from the vector DB."""
    query = "What is fact table?"
    response = call_graph(query)
    status = "PASS"
    try:
        assert isinstance(response, str)
        assert "i don't know" not in response.lower()
    except AssertionError:
        status = "FAIL"
        raise
    finally:
        log_result("test_find_match_known_query", query, response, status)


def test_find_match_unknown_query():
    """Test that the chatbot returns 'I don't know.' for unrelated queries."""
    query = "Tell me about aliens on Mars?"
    response = call_graph(query)
    status = "PASS"
    try:
        assert isinstance(response, str)
        assert "i don't know" in response.lower()
    except AssertionError:
        status = "FAIL"
        raise
    finally:
        log_result("test_find_match_unknown_query", query, response, status)


@pytest.mark.parametrize("query", [
    "What is the weather in Pune?",
    "What is claim procedure for car insurance?",
])
def test_multiple_queries(query):
    """Test that multiple queries return non-empty answers."""
    response = call_graph(query)
    status = "PASS"
    try:
        assert isinstance(response, str)
        assert len(response.strip()) > 0
    except AssertionError:
        status = "FAIL"
        raise
    finally:
        log_result("test_multiple_queries", query, response, status)


@pytest.fixture(scope="session", autouse=True)
def write_results_to_file(request):
    """Write test results to a Markdown file after all tests finish."""
    def finalize():
        output_file = "test_results.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"# Chatbot Test Results ({timestamp})\n\n")
            for i, r in enumerate(test_results, start=1):
                f.write(f"## Test {i}: {r['test_name']}\n")
                f.write(f"**Query:** {r['query']}\n\n")
                f.write(f"**Response:** {r['response']}\n\n")
                f.write(f"**Status:** {r['status']}\n\n")
                f.write("---\n\n")
        print(f"\n✅ Test results saved to {output_file}")
    request.addfinalizer(finalize)
