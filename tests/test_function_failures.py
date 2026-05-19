"""
Tests for Azure Function failure behavior.
Verifies that the function raises exceptions instead of silently failing.
Run with: pytest tests/test_function_failures.py -v
"""

import sys
import os
import importlib
import pytest
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestFunctionFailures:
    """Tests that verify the function fails loudly instead of silently"""

    @staticmethod
    def _load_daily_milo_post():
        """Reload function_app so module-level environment config is refreshed per test."""
        import function_app

        return importlib.reload(function_app).daily_milo_post
    
    @patch.dict(os.environ, {
        'AZURE_STORAGE_CONNECTION_STRING': 'test',
        'COMPUTER_VISION_ENDPOINT': 'test',
        'COMPUTER_VISION_KEY': 'test',
        'OPENAI_TEXT_API_KEY': 'test',
        'OPENAI_IMAGE_API_KEY': 'test',
        'OPENAI_IMAGE_ENDPOINT': 'test',
        'OPENAI_TEXT_ENDPOINT': 'test',
        'POSTLY_API_KEY': 'test',
        'POSTLY_WORKSPACE_ID': 'test',
        'OPENAI_TEXT_MODEL': 'gpt-4',
        'OPENAI_IMAGE_MODEL': 'FLUX.2-pro',
        'BLOB_CONTAINER_NAME': 'test-container',
        'DAYS_TO_CHECK': '7',
    })
    def test_raises_exception_when_postly_fails(self):
        """Test that function raises exception when Postly posting fails"""
        daily_milo_post = self._load_daily_milo_post()
        
        with patch('function_app.BlobServiceClient'), \
             patch('function_app.ComputerVisionClient'), \
             patch('function_app.AzureOpenAI'), \
             patch('function_app.select_best_photo') as mock_select, \
             patch('function_app.generate_witty_caption') as mock_caption, \
             patch('function_app.post_to_postly') as mock_post:
            
            # Mock successful photo selection
            mock_select.return_value = (b"fake_image_data", "test.jpg", "Test image")
            mock_caption.return_value = "Test caption"
            
            # Mock Postly failure
            mock_post.return_value = False
            
            # Create mock timer
            mock_timer = Mock()
            
            # Should raise RuntimeError
            with pytest.raises(RuntimeError, match="Failed to post to Postly"):
                daily_milo_post(mock_timer)
    
    @patch.dict(os.environ, {
        'AZURE_STORAGE_CONNECTION_STRING': 'test',
        'COMPUTER_VISION_ENDPOINT': 'test',
        'COMPUTER_VISION_KEY': 'test',
        'OPENAI_TEXT_API_KEY': 'test',
        'OPENAI_IMAGE_API_KEY': 'test',
        'OPENAI_IMAGE_ENDPOINT': 'test',
        'OPENAI_TEXT_ENDPOINT': 'test',
        'POSTLY_API_KEY': 'test',
        'POSTLY_WORKSPACE_ID': 'test',
        'OPENAI_TEXT_MODEL': 'gpt-4',
        'OPENAI_IMAGE_MODEL': 'FLUX.2-pro',
        'BLOB_CONTAINER_NAME': 'test-container',
        'DAYS_TO_CHECK': '7',
    })
    def test_raises_exception_when_no_image_found(self):
        """Test that function raises exception when no image can be obtained"""
        daily_milo_post = self._load_daily_milo_post()
        
        with patch('function_app.BlobServiceClient'), \
             patch('function_app.ComputerVisionClient'), \
             patch('function_app.AzureOpenAI'), \
             patch('function_app.select_best_photo') as mock_select, \
             patch('function_app.generate_ai_image') as mock_generate:
            
            # Mock no photo found
            mock_select.return_value = None
            # Mock AI generation failure
            mock_generate.return_value = None
            
            # Create mock timer
            mock_timer = Mock()
            
            # Should raise RuntimeError
            with pytest.raises(RuntimeError, match="Failed to obtain image"):
                daily_milo_post(mock_timer)
    
    def test_raises_exception_when_config_missing(self):
        """Test that function raises exception when required config is missing or invalid"""
        # Clear all environment variables
        with patch.dict(os.environ, {}, clear=True):
            daily_milo_post = self._load_daily_milo_post()
            # Create mock timer
            mock_timer = Mock()
            
            # Should raise RuntimeError or ValueError for missing/invalid config
            with pytest.raises((RuntimeError, ValueError)):
                daily_milo_post(mock_timer)
    
    @patch.dict(os.environ, {
        'AZURE_STORAGE_CONNECTION_STRING': 'test',
        'COMPUTER_VISION_ENDPOINT': 'test',
        'COMPUTER_VISION_KEY': 'test',
        'OPENAI_TEXT_API_KEY': 'test',
        'OPENAI_IMAGE_API_KEY': 'test',
        'OPENAI_IMAGE_ENDPOINT': 'test',
        'OPENAI_TEXT_ENDPOINT': 'test',
        'POSTLY_API_KEY': 'test',
        'POSTLY_WORKSPACE_ID': 'test',
        'OPENAI_TEXT_MODEL': 'gpt-4',
        'OPENAI_IMAGE_MODEL': 'FLUX.2-pro',
        'BLOB_CONTAINER_NAME': 'test-container',
        'DAYS_TO_CHECK': '7',
    })
    def test_success_does_not_raise_exception(self):
        """Test that function completes successfully when everything works"""
        daily_milo_post = self._load_daily_milo_post()
        
        with patch('function_app.BlobServiceClient') as mock_blob_client, \
             patch('function_app.ComputerVisionClient'), \
             patch('function_app.AzureOpenAI'), \
             patch('function_app.select_best_photo') as mock_select, \
             patch('function_app.generate_witty_caption') as mock_caption, \
             patch('function_app.post_to_postly') as mock_post, \
             patch('function_app.mark_blob_as_posted'):
            
            # Mock successful photo selection
            mock_select.return_value = (b"fake_image_data", "test.jpg", "Test image")
            mock_caption.return_value = "Test caption"
            mock_post.return_value = True
            
            # Mock blob client for marking as posted
            mock_container = Mock()
            mock_blob = Mock()
            mock_blob_client.from_connection_string.return_value.get_container_client.return_value = mock_container
            mock_container.get_blob_client.return_value = mock_blob
            
            # Create mock timer
            mock_timer = Mock()
            
            # Should complete without raising exception
            try:
                daily_milo_post(mock_timer)
            except Exception as e:
                pytest.fail(f"Function should not raise exception on success, but raised: {e}")
