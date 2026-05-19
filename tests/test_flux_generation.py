"""
Tests for AI image generation with FLUX.
Run with: pytest tests/test_flux_generation.py -v
"""

import os
import sys
import pytest
import json
import importlib
from unittest.mock import Mock, patch
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up environment variables for testing"""
    monkeypatch.setenv("OPENAI_IMAGE_API_KEY", "test-api-key")
    monkeypatch.setenv("FLUX_API_URL", "https://example.com/flux")

    # Load from local.settings.json if available
    settings_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "local.settings.json"
    )
    
    if os.path.exists(settings_path):
        with open(settings_path, 'r') as f:
            settings = json.load(f)
            for key, value in settings.get('Values', {}).items():
                monkeypatch.setenv(key, value)


@pytest.fixture
def function_app_module(mock_env_vars):
    """Reload function_app after environment setup so module-level config is refreshed."""
    import function_app

    return importlib.reload(function_app)


@pytest.fixture
def mock_blob_client():
    """Mock Azure Blob Storage client"""
    client = Mock()
    blob_client = Mock()
    blob_client.download_blob().readall.return_value = b"fake_image_bytes"
    client.get_blob_client.return_value = blob_client
    return client


class TestFluxGeneration:
    """Unit tests for FLUX image generation (mocked API calls)"""
    
    def test_generate_ai_image_loads_description(self, function_app_module, mock_blob_client):
        """Test that the function loads Milo's description from file"""
        with patch('function_app.requests.post') as mock_post:
            # Mock successful FLUX response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'data': [{'b64_json': 'ZmFrZV9pbWFnZV9ieXRlcw=='}]  # base64 "fake_image_bytes"
            }
            mock_post.return_value = mock_response
            
            # Call the function
            result = function_app_module.generate_ai_image(
                client=Mock(),
                image_model="FLUX.2-pro",
                blob_service_client=mock_blob_client,
                container_name="test-container"
            )
            
            # Verify it succeeded
            assert result is not None
            assert isinstance(result, bytes)
            
            # Verify the API was called with correct structure
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            payload = call_args[1]['json']
            
            assert payload['model'] == 'FLUX.2-pro'
            assert payload['width'] == 1024
            assert payload['height'] == 1024
            assert 'prompt' in payload
            assert len(payload['prompt']) > 100  # Should have detailed description
    
    def test_generate_ai_image_includes_reference_image(self, function_app_module, mock_blob_client):
        """Test that reference image is included when blob storage is available"""
        with patch('function_app.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'data': [{'b64_json': 'ZmFrZV9pbWFnZV9ieXRlcw=='}]
            }
            mock_post.return_value = mock_response
            
            result = function_app_module.generate_ai_image(
                client=Mock(),
                image_model="FLUX.2-pro",
                blob_service_client=mock_blob_client,
                container_name="test-container"
            )
            
            assert result is not None
            
            # Verify reference image was included
            payload = mock_post.call_args[1]['json']
            assert 'image_prompt' in payload
            # Verify it loaded the first reference photo
            mock_blob_client.get_blob_client.assert_called_once()
    
    def test_generate_ai_image_without_reference_image(self, function_app_module):
        """Test that generation works without reference image"""
        with patch('function_app.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'data': [{'b64_json': 'ZmFrZV9pbWFnZV9ieXRlcw=='}]
            }
            mock_post.return_value = mock_response
            
            result = function_app_module.generate_ai_image(
                client=Mock(),
                image_model="FLUX.2-pro",
                blob_service_client=None,
                container_name=None
            )
            
            assert result is not None
            
            # Verify reference image was NOT included
            payload = mock_post.call_args[1]['json']
            assert 'image_prompt' not in payload
    
    def test_generate_ai_image_handles_api_error(self, function_app_module):
        """Test error handling for FLUX API failures"""
        with patch('function_app.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
            mock_post.return_value = mock_response
            
            result = function_app_module.generate_ai_image(
                client=Mock(),
                image_model="FLUX.2-pro"
            )
            
            assert result is None


@pytest.mark.integration
class TestFluxIntegration:
    """Integration tests for FLUX (makes real API calls)"""
    
    def test_flux_generation_with_real_api(self, mock_env_vars):
        """Test actual FLUX API call - only run when specifically testing integration"""
        from function_app import generate_ai_image
        from azure.storage.blob import BlobServiceClient
        
        # Get real credentials
        connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        container_name = os.environ.get("BLOB_CONTAINER_NAME", "milo-photos")
        
        if not connection_string:
            pytest.skip("AZURE_STORAGE_CONNECTION_STRING not configured")
        
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        result = generate_ai_image(
            client=Mock(),  # Not used for FLUX
            image_model="FLUX.2-pro",
            blob_service_client=blob_service_client,
            container_name=container_name
        )
        
        assert result is not None
        assert isinstance(result, bytes)
        assert len(result) > 10000  # Should be a reasonable image size
        
        # Optionally save the test image
        output_path = os.path.join(
            os.path.dirname(__file__),
            f"flux_integration_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        with open(output_path, 'wb') as f:
            f.write(result)
        print(f"\nTest image saved to: {output_path}")
