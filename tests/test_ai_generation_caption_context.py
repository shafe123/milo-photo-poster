"""Tests for wiring caption context into AI image generation fallback."""

import os
import sys
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_daily_milo_post_passes_caption_context_to_ai_generation():
    """When fallback AI generation is used, caption text should guide image generation."""
    from function_app import daily_milo_post

    with patch('function_app.AZURE_STORAGE_CONNECTION_STRING', 'test'), \
         patch('function_app.COMPUTER_VISION_ENDPOINT', 'test'), \
         patch('function_app.COMPUTER_VISION_KEY', 'test'), \
         patch('function_app.OPENAI_TEXT_API_KEY', 'test'), \
         patch('function_app.OPENAI_IMAGE_ENDPOINT', 'test'), \
         patch('function_app.POSTLY_API_KEY', 'test'), \
         patch('function_app.POSTLY_WORKSPACE_ID', 'test'), \
         patch('function_app.OPENAI_TEXT_ENDPOINT', 'test'), \
         patch('function_app.OPENAI_TEXT_MODEL', 'gpt-4'), \
         patch('function_app.OPENAI_IMAGE_MODEL', 'FLUX.2-pro'), \
         patch('function_app.BLOB_CONTAINER_NAME', 'test-container'), \
         patch('function_app.DAYS_TO_CHECK', 7), \
         patch('function_app.BlobServiceClient') as mock_blob_service, \
         patch('function_app.ComputerVisionClient'), \
         patch('function_app.AzureOpenAI'), \
         patch('function_app.select_best_photo', return_value=None), \
         patch('function_app.get_current_context', return_value={"day_of_week": "Monday"}), \
         patch('function_app.generate_witty_caption', return_value='Rainy day window watching'), \
         patch('function_app.generate_ai_image', return_value=b'image-bytes') as mock_generate_ai_image, \
         patch('function_app.post_to_postly', return_value=True):
        mock_blob_service.from_connection_string.return_value = Mock()
        daily_milo_post(Mock())

        assert mock_generate_ai_image.call_count == 1
        assert (
            mock_generate_ai_image.call_args.kwargs['caption_context']
            == 'Rainy day window watching'
        )
