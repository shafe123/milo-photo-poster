"""
Integration tests for Weather API (OpenWeatherMap)
These tests make real API calls and should be run with: pytest -m integration
"""

import sys
import os
from unittest.mock import patch, MagicMock
import json

# Mark all tests in this file as integration tests
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import function_app
from function_app import get_current_weather


@pytest.mark.integration
def test_weather_api_real_call():
    """
    Integration test: Make a real API call to OpenWeatherMap One Call API 3.0

    Prerequisites:
    - WEATHER_API_KEY environment variable set
    - WEATHER_LAT environment variable set (optional, defaults to "40.4406" for Pittsburgh)
    - WEATHER_LON environment variable set (optional, defaults to "-79.9959" for Pittsburgh)

    Run with: pytest -m integration tests/test_weather_api.py::test_weather_api_real_call -v
    """
    # Load configuration from environment
    api_key = os.environ.get("WEATHER_API_KEY")
    lat = os.environ.get("WEATHER_LAT", "40.4406")
    lon = os.environ.get("WEATHER_LON", "-79.9959")

    # Skip test if API key is missing
    if not api_key:
        pytest.skip("Weather API key not configured in environment")

    # Set up function_app configuration
    function_app.WEATHER_API_KEY = api_key
    function_app.WEATHER_LAT = lat
    function_app.WEATHER_LON = lon

    print("\nTesting OpenWeatherMap One Call API 3.0")
    print(f"Coordinates: ({lat}, {lon})")

    # Make the actual API call
    weather_info = get_current_weather()

    # Verify the response
    assert weather_info is not None, "Weather API should return data"

    # Check that all expected fields are present
    assert "description" in weather_info, "Weather info should include description"
    assert "temperature" in weather_info, "Weather info should include temperature"
    assert "feels_like" in weather_info, "Weather info should include feels_like"

    # Verify data types
    assert isinstance(weather_info["description"], str), (
        "Description should be a string"
    )
    assert isinstance(weather_info["temperature"], (int, float)), (
        "Temperature should be numeric"
    )
    assert isinstance(weather_info["feels_like"], (int, float)), (
        "Feels_like should be numeric"
    )

    # Verify reasonable temperature ranges (in Fahrenheit)
    assert -100 <= weather_info["temperature"] <= 150, (
        f"Temperature {weather_info['temperature']}°F seems unrealistic"
    )
    assert -100 <= weather_info["feels_like"] <= 150, (
        f"Feels-like temperature {weather_info['feels_like']}°F seems unrealistic"
    )

    # Verify description is not empty
    assert len(weather_info["description"]) > 0, "Description should not be empty"

    print("\nResults:")
    print(f"  Description: {weather_info['description']}")
    print(f"  Temperature: {weather_info['temperature']}°F")
    print(f"  Feels Like: {weather_info['feels_like']}°F")


@pytest.mark.integration
def test_weather_api_different_location():
    """
    Integration test: Test weather API with a different location (New York)

    Prerequisites:
    - WEATHER_API_KEY environment variable set

    Run with: pytest -m integration tests/test_weather_api.py::test_weather_api_different_location -v
    """
    # Load configuration from environment
    api_key = os.environ.get("WEATHER_API_KEY")

    # Skip test if API key is missing
    if not api_key:
        pytest.skip("Weather API key not configured in environment")

    # Set up function_app configuration with New York coordinates
    function_app.WEATHER_API_KEY = api_key
    function_app.WEATHER_LAT = "40.7128"  # New York City latitude
    function_app.WEATHER_LON = "-74.0060"  # New York City longitude

    print("\nTesting OpenWeatherMap One Call API 3.0 for New York")

    # Make the actual API call
    weather_info = get_current_weather()

    # Verify the response
    assert weather_info is not None, "Weather API should return data for New York"

    print("\nResults for New York:")
    print(f"  Description: {weather_info['description']}")
    print(f"  Temperature: {weather_info['temperature']}°F")


def test_weather_api_no_key():
    """
    Unit test: Verify behavior when API key is not configured

    Run with: pytest tests/test_weather_api.py::test_weather_api_no_key -v
    """
    # Set up function_app configuration with no API key
    original_key = function_app.WEATHER_API_KEY
    function_app.WEATHER_API_KEY = None

    try:
        # Make the call
        weather_info = get_current_weather()

        # Verify it returns None gracefully
        assert weather_info is None, (
            "Weather API should return None when API key is not configured"
        )

    finally:
        # Restore original key
        function_app.WEATHER_API_KEY = original_key


@patch("function_app.requests.get")
def test_weather_api_network_error(mock_get):
    """
    Unit test: Verify behavior when API request fails

    Run with: pytest tests/test_weather_api.py::test_weather_api_network_error -v
    """
    # Set up function_app configuration
    function_app.WEATHER_API_KEY = "test_key"
    function_app.WEATHER_LAT = "40.4406"
    function_app.WEATHER_LON = "-79.9959"

    # Mock a network error
    mock_get.side_effect = Exception("Network error")

    # Make the call
    weather_info = get_current_weather()

    # Verify it returns None gracefully
    assert weather_info is None, "Weather API should return None on network error"


@patch("function_app.requests.get")
def test_weather_api_invalid_json(mock_get):
    """
    Unit test: Verify behavior when API returns invalid JSON

    Run with: pytest tests/test_weather_api.py::test_weather_api_invalid_json -v
    """
    # Set up function_app configuration
    function_app.WEATHER_API_KEY = "test_key"
    function_app.WEATHER_LAT = "40.4406"
    function_app.WEATHER_LON = "-79.9959"

    # Mock a response with invalid JSON
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
    mock_get.return_value = mock_response

    # Make the call
    weather_info = get_current_weather()

    # Verify it returns None gracefully
    assert weather_info is None, "Weather API should return None on JSON decode error"


@patch("function_app.requests.get")
def test_weather_api_successful_response(mock_get):
    """
    Unit test: Verify parsing of a successful One Call API 3.0 response

    Run with: pytest tests/test_weather_api.py::test_weather_api_successful_response -v
    """
    # Set up function_app configuration
    function_app.WEATHER_API_KEY = "test_key"
    function_app.WEATHER_LAT = "40.4406"
    function_app.WEATHER_LON = "-79.9959"

    # Mock a successful One Call API 3.0 response
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "lat": 40.4406,
        "lon": -79.9959,
        "timezone": "America/New_York",
        "current": {
            "dt": 1684929490,
            "temp": 72.5,
            "feels_like": 70.3,
            "weather": [{"description": "clear sky", "main": "Clear"}],
        },
    }
    mock_get.return_value = mock_response

    # Make the call
    weather_info = get_current_weather()

    # Verify the parsed response
    assert weather_info is not None
    assert weather_info["description"] == "clear sky"
    assert (
        weather_info["temperature"] == 72
    )  # rounded (72.5 rounds to 72 using Python's banker's rounding)
    assert weather_info["feels_like"] == 70  # rounded


@patch("function_app.requests.get")
def test_weather_api_http_error(mock_get):
    """
    Unit test: Verify behavior when API returns HTTP error

    Run with: pytest tests/test_weather_api.py::test_weather_api_http_error -v
    """
    # Set up function_app configuration
    function_app.WEATHER_API_KEY = "invalid_key"
    function_app.WEATHER_LAT = "40.4406"
    function_app.WEATHER_LON = "-79.9959"

    # Mock an HTTP error (401 Unauthorized)
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = Exception("401 Unauthorized")
    mock_get.return_value = mock_response

    # Make the call
    weather_info = get_current_weather()

    # Verify it returns None gracefully
    assert weather_info is None, "Weather API should return None on HTTP error"


if __name__ == "__main__":
    # Allow running this file directly for quick testing
    # Load environment variables from local.settings.json
    try:
        with open("local.settings.json", "r") as f:
            settings = json.load(f)
        values = settings.get("Values", {})
        for k, v in values.items():
            os.environ[k] = v

        print("Running integration test with real API call...")
        test_weather_api_real_call()
        print("\n✓ Integration test passed!")

    except FileNotFoundError:
        print("Error: local.settings.json not found")
    except Exception as e:
        print(f"Error: {e}")
