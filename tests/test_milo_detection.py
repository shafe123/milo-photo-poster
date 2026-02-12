"""
Tests for Milo detection using Azure Custom Vision
"""
import sys
import os
from unittest.mock import Mock, MagicMock, patch
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import function_app
from function_app import check_milo_in_photo


def test_milo_detection_with_cached_result():
    """Test that cached Milo detection result is used when available"""
    # Create mock blob client with cached metadata
    mock_blob_client = MagicMock()
    mock_blob_client.blob_name = "test_photo.jpg"
    
    mock_properties = Mock()
    mock_properties.metadata = {
        function_app.MILO_DETECTED_KEY: "true",
        function_app.MILO_CONFIDENCE_KEY: "0.95"
    }
    mock_blob_client.get_blob_properties.return_value = mock_properties
    
    # Call function
    is_milo_present, confidence = check_milo_in_photo(
        mock_blob_client, 
        "http://example.com/test.jpg"
    )
    
    # Verify cached result is returned
    assert is_milo_present is True
    assert confidence == 0.95
    # Verify no API call was made (set_blob_metadata not called)
    mock_blob_client.set_blob_metadata.assert_not_called()


def test_milo_detection_api_call_milo_detected():
    """Test Custom Vision API call when Milo is detected"""
    # Mock blob client without cached result
    mock_blob_client = MagicMock()
    mock_blob_client.blob_name = "test_photo.jpg"
    
    # First call: no cached metadata
    mock_properties_initial = Mock()
    mock_properties_initial.metadata = {}
    
    # Second call: for caching result
    mock_properties_for_cache = Mock()
    mock_properties_for_cache.metadata = {}
    
    mock_blob_client.get_blob_properties.side_effect = [
        mock_properties_initial,
        mock_properties_for_cache
    ]
    
    # Mock Custom Vision API response with Milo detected
    mock_response = Mock()
    mock_response.json.return_value = {
        "predictions": [
            {"tagName": "milo", "probability": 0.92},
            {"tagName": "emilio", "probability": 0.05},
            {"tagName": "neither", "probability": 0.02},
            {"tagName": "both", "probability": 0.01}
        ]
    }
    mock_response.raise_for_status = Mock()
    
    with patch('requests.post', return_value=mock_response):
        with patch.dict(os.environ, {
            'CUSTOM_VISION_PREDICTION_ENDPOINT': 'https://test.cognitiveservices.azure.com/',
            'CUSTOM_VISION_PREDICTION_KEY': 'test-key',
            'CUSTOM_VISION_PROJECT_ID': 'test-project-id',
            'CUSTOM_VISION_ITERATION_NAME': 'Iteration1',
            'MILO_CONFIDENCE_THRESHOLD': '0.7'
        }):
            # Force reload configuration
            function_app.CUSTOM_VISION_PREDICTION_ENDPOINT = 'https://test.cognitiveservices.azure.com/'
            function_app.CUSTOM_VISION_PREDICTION_KEY = 'test-key'
            function_app.CUSTOM_VISION_PROJECT_ID = 'test-project-id'
            function_app.CUSTOM_VISION_ITERATION_NAME = 'Iteration1'
            function_app.MILO_CONFIDENCE_THRESHOLD = 0.7
            
            # Call function
            is_milo_present, confidence = check_milo_in_photo(
                mock_blob_client,
                "http://example.com/test.jpg"
            )
    
    # Verify Milo was detected with correct confidence
    assert is_milo_present is True
    assert confidence == 0.92
    
    # Verify result was cached
    mock_blob_client.set_blob_metadata.assert_called_once()
    cached_metadata = mock_blob_client.set_blob_metadata.call_args[0][0]
    assert cached_metadata[function_app.MILO_DETECTED_KEY] == "true"
    assert float(cached_metadata[function_app.MILO_CONFIDENCE_KEY]) == 0.92


def test_milo_detection_api_call_both_cats_detected():
    """Test Custom Vision API call when both cats are detected"""
    # Mock blob client without cached result
    mock_blob_client = MagicMock()
    mock_blob_client.blob_name = "test_photo.jpg"
    
    mock_properties_initial = Mock()
    mock_properties_initial.metadata = {}
    
    mock_properties_for_cache = Mock()
    mock_properties_for_cache.metadata = {}
    
    mock_blob_client.get_blob_properties.side_effect = [
        mock_properties_initial,
        mock_properties_for_cache
    ]
    
    # Mock Custom Vision API response with both cats detected
    mock_response = Mock()
    mock_response.json.return_value = {
        "predictions": [
            {"tagName": "both", "probability": 0.88},
            {"tagName": "milo", "probability": 0.08},
            {"tagName": "emilio", "probability": 0.03},
            {"tagName": "neither", "probability": 0.01}
        ]
    }
    mock_response.raise_for_status = Mock()
    
    with patch('requests.post', return_value=mock_response):
        with patch.dict(os.environ, {
            'CUSTOM_VISION_PREDICTION_ENDPOINT': 'https://test.cognitiveservices.azure.com',  # No trailing slash
            'CUSTOM_VISION_PREDICTION_KEY': 'test-key',
            'CUSTOM_VISION_PROJECT_ID': 'test-project-id',
            'CUSTOM_VISION_ITERATION_NAME': 'Iteration1',
            'MILO_CONFIDENCE_THRESHOLD': '0.7'
        }):
            # Force reload configuration
            function_app.CUSTOM_VISION_PREDICTION_ENDPOINT = 'https://test.cognitiveservices.azure.com'
            function_app.CUSTOM_VISION_PREDICTION_KEY = 'test-key'
            function_app.CUSTOM_VISION_PROJECT_ID = 'test-project-id'
            function_app.CUSTOM_VISION_ITERATION_NAME = 'Iteration1'
            function_app.MILO_CONFIDENCE_THRESHOLD = 0.7
            
            # Call function
            is_milo_present, confidence = check_milo_in_photo(
                mock_blob_client,
                "http://example.com/test.jpg"
            )
    
    # Verify Milo was detected (both cats = Milo present)
    assert is_milo_present is True
    assert confidence == 0.88


def test_milo_detection_api_call_milo_not_detected():
    """Test Custom Vision API call when Milo is not detected (only Emilio)"""
    # Mock blob client without cached result
    mock_blob_client = MagicMock()
    mock_blob_client.blob_name = "test_photo.jpg"
    
    mock_properties_initial = Mock()
    mock_properties_initial.metadata = {}
    
    mock_properties_for_cache = Mock()
    mock_properties_for_cache.metadata = {}
    
    mock_blob_client.get_blob_properties.side_effect = [
        mock_properties_initial,
        mock_properties_for_cache
    ]
    
    # Mock Custom Vision API response with only Emilio detected
    mock_response = Mock()
    mock_response.json.return_value = {
        "predictions": [
            {"tagName": "emilio", "probability": 0.94},
            {"tagName": "milo", "probability": 0.03},
            {"tagName": "both", "probability": 0.02},
            {"tagName": "neither", "probability": 0.01}
        ]
    }
    mock_response.raise_for_status = Mock()
    
    with patch('requests.post', return_value=mock_response):
        with patch.dict(os.environ, {
            'CUSTOM_VISION_PREDICTION_ENDPOINT': 'https://test.cognitiveservices.azure.com/',
            'CUSTOM_VISION_PREDICTION_KEY': 'test-key',
            'CUSTOM_VISION_PROJECT_ID': 'test-project-id',
            'CUSTOM_VISION_ITERATION_NAME': 'Iteration1',
            'MILO_CONFIDENCE_THRESHOLD': '0.7'
        }):
            # Force reload configuration
            function_app.CUSTOM_VISION_PREDICTION_ENDPOINT = 'https://test.cognitiveservices.azure.com/'
            function_app.CUSTOM_VISION_PREDICTION_KEY = 'test-key'
            function_app.CUSTOM_VISION_PROJECT_ID = 'test-project-id'
            function_app.CUSTOM_VISION_ITERATION_NAME = 'Iteration1'
            function_app.MILO_CONFIDENCE_THRESHOLD = 0.7
            
            # Call function
            is_milo_present, confidence = check_milo_in_photo(
                mock_blob_client,
                "http://example.com/test.jpg"
            )
    
    # Verify Milo was not detected
    assert is_milo_present is False
    assert confidence == 0.03
    
    # Verify result was cached
    mock_blob_client.set_blob_metadata.assert_called_once()
    cached_metadata = mock_blob_client.set_blob_metadata.call_args[0][0]
    assert cached_metadata[function_app.MILO_DETECTED_KEY] == "false"


def test_milo_detection_below_threshold():
    """Test when Milo confidence is below threshold"""
    # Mock blob client without cached result
    mock_blob_client = MagicMock()
    mock_blob_client.blob_name = "test_photo.jpg"
    
    mock_properties_initial = Mock()
    mock_properties_initial.metadata = {}
    
    mock_properties_for_cache = Mock()
    mock_properties_for_cache.metadata = {}
    
    mock_blob_client.get_blob_properties.side_effect = [
        mock_properties_initial,
        mock_properties_for_cache
    ]
    
    # Mock Custom Vision API response with low Milo confidence
    mock_response = Mock()
    mock_response.json.return_value = {
        "predictions": [
            {"tagName": "neither", "probability": 0.60},
            {"tagName": "milo", "probability": 0.25},  # Below 0.7 threshold
            {"tagName": "emilio", "probability": 0.10},
            {"tagName": "both", "probability": 0.05}
        ]
    }
    mock_response.raise_for_status = Mock()
    
    with patch('requests.post', return_value=mock_response):
        with patch.dict(os.environ, {
            'CUSTOM_VISION_PREDICTION_ENDPOINT': 'https://test.cognitiveservices.azure.com/',
            'CUSTOM_VISION_PREDICTION_KEY': 'test-key',
            'CUSTOM_VISION_PROJECT_ID': 'test-project-id',
            'CUSTOM_VISION_ITERATION_NAME': 'Iteration1',
            'MILO_CONFIDENCE_THRESHOLD': '0.7'
        }):
            # Force reload configuration
            function_app.CUSTOM_VISION_PREDICTION_ENDPOINT = 'https://test.cognitiveservices.azure.com/'
            function_app.CUSTOM_VISION_PREDICTION_KEY = 'test-key'
            function_app.CUSTOM_VISION_PROJECT_ID = 'test-project-id'
            function_app.CUSTOM_VISION_ITERATION_NAME = 'Iteration1'
            function_app.MILO_CONFIDENCE_THRESHOLD = 0.7
            
            # Call function
            is_milo_present, confidence = check_milo_in_photo(
                mock_blob_client,
                "http://example.com/test.jpg"
            )
    
    # Verify Milo was not detected (below threshold)
    assert is_milo_present is False
    assert confidence == 0.25


def test_milo_detection_missing_configuration():
    """Test behavior when Custom Vision is not configured"""
    # Mock blob client without cached result
    mock_blob_client = MagicMock()
    mock_blob_client.blob_name = "test_photo.jpg"
    
    mock_properties = Mock()
    mock_properties.metadata = {}
    mock_blob_client.get_blob_properties.return_value = mock_properties
    
    # Clear Custom Vision configuration
    with patch.dict(os.environ, {}, clear=True):
        function_app.CUSTOM_VISION_PREDICTION_ENDPOINT = None
        function_app.CUSTOM_VISION_PREDICTION_KEY = None
        function_app.CUSTOM_VISION_PROJECT_ID = None
        
        # Call function
        is_milo_present, confidence = check_milo_in_photo(
            mock_blob_client,
            "http://example.com/test.jpg"
        )
    
    # Verify function assumes Milo is present when not configured
    assert is_milo_present is True
    assert confidence == 1.0


def test_milo_detection_api_error_handling():
    """Test error handling when Custom Vision API call fails"""
    # Mock blob client without cached result
    mock_blob_client = MagicMock()
    mock_blob_client.blob_name = "test_photo.jpg"
    
    mock_properties = Mock()
    mock_properties.metadata = {}
    mock_blob_client.get_blob_properties.return_value = mock_properties
    
    # Mock Custom Vision API to raise an error
    mock_response = Mock()
    mock_response.raise_for_status.side_effect = Exception("API Error")
    
    with patch('requests.post', return_value=mock_response):
        with patch.dict(os.environ, {
            'CUSTOM_VISION_PREDICTION_ENDPOINT': 'https://test.cognitiveservices.azure.com/',
            'CUSTOM_VISION_PREDICTION_KEY': 'test-key',
            'CUSTOM_VISION_PROJECT_ID': 'test-project-id',
            'CUSTOM_VISION_ITERATION_NAME': 'Iteration1',
            'MILO_CONFIDENCE_THRESHOLD': '0.7'
        }):
            # Force reload configuration
            function_app.CUSTOM_VISION_PREDICTION_ENDPOINT = 'https://test.cognitiveservices.azure.com/'
            function_app.CUSTOM_VISION_PREDICTION_KEY = 'test-key'
            function_app.CUSTOM_VISION_PROJECT_ID = 'test-project-id'
            function_app.CUSTOM_VISION_ITERATION_NAME = 'Iteration1'
            function_app.MILO_CONFIDENCE_THRESHOLD = 0.7
            
            # Call function
            is_milo_present, confidence = check_milo_in_photo(
                mock_blob_client,
                "http://example.com/test.jpg"
            )
    
    # Verify function assumes Milo is present on error (fail-safe)
    assert is_milo_present is True
    assert confidence == 1.0


def test_milo_detection_url_trailing_slash_handling():
    """Test that trailing slash in endpoint URL is handled correctly"""
    # Mock blob client without cached result
    mock_blob_client = MagicMock()
    mock_blob_client.blob_name = "test_photo.jpg"
    
    mock_properties_initial = Mock()
    mock_properties_initial.metadata = {}
    
    mock_properties_for_cache = Mock()
    mock_properties_for_cache.metadata = {}
    
    mock_blob_client.get_blob_properties.side_effect = [
        mock_properties_initial,
        mock_properties_for_cache
    ]
    
    # Mock Custom Vision API response
    mock_response = Mock()
    mock_response.json.return_value = {
        "predictions": [
            {"tagName": "milo", "probability": 0.85}
        ]
    }
    mock_response.raise_for_status = Mock()
    
    with patch('requests.post', return_value=mock_response) as mock_post:
        with patch.dict(os.environ, {
            'CUSTOM_VISION_PREDICTION_ENDPOINT': 'https://test.cognitiveservices.azure.com/',  # Trailing slash
            'CUSTOM_VISION_PREDICTION_KEY': 'test-key',
            'CUSTOM_VISION_PROJECT_ID': 'test-project-id',
            'CUSTOM_VISION_ITERATION_NAME': 'Iteration1',
            'MILO_CONFIDENCE_THRESHOLD': '0.7'
        }):
            # Force reload configuration
            function_app.CUSTOM_VISION_PREDICTION_ENDPOINT = 'https://test.cognitiveservices.azure.com/'
            function_app.CUSTOM_VISION_PREDICTION_KEY = 'test-key'
            function_app.CUSTOM_VISION_PROJECT_ID = 'test-project-id'
            function_app.CUSTOM_VISION_ITERATION_NAME = 'Iteration1'
            function_app.MILO_CONFIDENCE_THRESHOLD = 0.7
            
            # Call function
            is_milo_present, confidence = check_milo_in_photo(
                mock_blob_client,
                "http://example.com/test.jpg"
            )
    
    # Verify the URL was constructed correctly (no double slash)
    called_url = mock_post.call_args[0][0]
    assert '//customvision' not in called_url
    assert '/customvision/v3.0/Prediction/' in called_url
    assert is_milo_present is True


if __name__ == "__main__":
    # Run tests with pytest
    import pytest
    pytest.main([__file__, "-v"])
