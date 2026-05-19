import sys
import os
import json
import pytest
from unittest.mock import patch, Mock, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from function_app import post_to_postly, delete_postly_post


# Load environment variables from local.settings.json  
def load_local_settings(path="local.settings.json"):
    with open(path, "r") as f:
        settings = json.load(f)
    return settings.get("Values", {})


# Fixture to safely load settings with proper isolation
@pytest.fixture(scope="function")
def env_with_settings(monkeypatch):
    """Load settings for tests that need real credentials"""
    settings = load_local_settings()
    for k, v in settings.items():
        monkeypatch.setenv(k, v)
    return settings


# Test configuration - loaded lazily by tests that need them
def get_test_config():
    """Get test configuration - only call in integration tests"""
    settings = load_local_settings()
    return {
        "POSTLY_API_KEY": settings.get("POSTLY_API_KEY"),
        "POSTLY_WORKSPACE_ID": settings.get("POSTLY_WORKSPACE_ID"),
        "POSTLY_TARGET_PLATFORMS": settings.get("POSTLY_TARGET_PLATFORMS", "all"),
    }


class TestPostlyAPI:
    """Test suite for Postly API integration"""

    # Test constants (not real credentials - tests use mocks)
    TEST_API_KEY = "test-api-key-12345"
    TEST_WORKSPACE_ID = "test-workspace-id-67890"
    TEST_TARGET_PLATFORMS = "bluesky,instagram"

    @pytest.fixture
    def sample_image_data(self):
        """Create sample image data for testing"""
        # Create a minimal valid JPEG header
        return b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 100

    @pytest.fixture
    def sample_caption(self):
        """Sample caption for testing"""
        return "Daily Milo! 😾 Testing the Postly API integration #Milo #Cats #GrumpyCat"

    @patch("function_app.requests.post")
    def test_post_to_postly_success(
        self, mock_post, sample_image_data, sample_caption
    ):
        """Test successful posting to Postly API"""
        # Mock the upload response
        mock_upload_response = Mock()
        mock_upload_response.status_code = 200
        mock_upload_response.json.return_value = {
            "data": {"url": "https://storage.postly.ai/test-image.jpg"}
        }
        mock_upload_response.raise_for_status = Mock()

        # Mock the post creation response
        mock_create_response = Mock()
        mock_create_response.status_code = 200
        mock_create_response.json.return_value = {
            "data": {"id": "test-post-id", "status": "published"}
        }
        mock_create_response.raise_for_status = Mock()

        # Set up the mock to return different responses for each call
        mock_post.side_effect = [mock_upload_response, mock_create_response]

        # Call the function
        result = post_to_postly(
            api_key=self.TEST_API_KEY,
            workspace_id=self.TEST_WORKSPACE_ID,
            image_data=sample_image_data,
            caption=sample_caption,
            target_platforms=self.TEST_TARGET_PLATFORMS,
        )

        # Assertions
        assert result is True
        assert mock_post.call_count == 2

        # Check upload call
        upload_call = mock_post.call_args_list[0]
        assert upload_call[0][0] == "https://openapi.postly.ai/v1/files"
        assert "X-API-KEY" in upload_call[1]["headers"]
        assert "X-File-Size" in upload_call[1]["headers"]
        assert upload_call[1]["headers"]["X-File-Size"] == str(len(sample_image_data))

        # Check post creation call
        post_call = mock_post.call_args_list[1]
        assert post_call[0][0] == "https://openapi.postly.ai/v1/posts"
        post_data = post_call[1]["json"]
        assert post_data["workspace"] == self.TEST_WORKSPACE_ID
        assert post_data["text"] == sample_caption
        assert post_data["post_now"] is True
        assert len(post_data["media"]) == 1
        assert post_data["media"][0]["url"] == "https://storage.postly.ai/test-image.jpg"

    @patch("function_app.requests.post")
    def test_post_to_postly_upload_failure(
        self, mock_post, sample_image_data, sample_caption
    ):
        """Test handling of upload failure"""
        # Mock a failed upload response
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = Exception("Upload failed")

        mock_post.return_value = mock_response

        # Call the function
        result = post_to_postly(
            api_key=self.TEST_API_KEY,
            workspace_id=self.TEST_WORKSPACE_ID,
            image_data=sample_image_data,
            caption=sample_caption,
        )

        # Assertions
        assert result is False
        assert mock_post.call_count == 1

    @patch("function_app.requests.post")
    def test_post_to_postly_no_url_in_response(
        self, mock_post, sample_image_data, sample_caption
    ):
        """Test handling when upload doesn't return a URL"""
        # Mock upload response without URL
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": {}}  # No URL in response
        mock_response.raise_for_status = Mock()

        mock_post.return_value = mock_response

        # Call the function
        result = post_to_postly(
            api_key=self.TEST_API_KEY,
            workspace_id=self.TEST_WORKSPACE_ID,
            image_data=sample_image_data,
            caption=sample_caption,
        )

        # Assertions
        assert result is False
        assert mock_post.call_count == 1

    @patch("function_app.requests.post")
    def test_post_to_postly_post_creation_failure(
        self, mock_post, sample_image_data, sample_caption
    ):
        """Test handling of post creation failure"""
        # Mock successful upload
        mock_upload_response = Mock()
        mock_upload_response.status_code = 200
        mock_upload_response.json.return_value = {
            "data": {"url": "https://storage.postly.ai/test-image.jpg"}
        }
        mock_upload_response.raise_for_status = Mock()

        # Mock failed post creation
        mock_post_response = Mock()
        mock_post_response.status_code = 400
        mock_post_response.text = "Bad Request"
        mock_post_response.raise_for_status.side_effect = Exception(
            "Post creation failed"
        )

        mock_post.side_effect = [mock_upload_response, mock_post_response]

        # Call the function
        result = post_to_postly(
            api_key=self.TEST_API_KEY,
            workspace_id=self.TEST_WORKSPACE_ID,
            image_data=sample_image_data,
            caption=sample_caption,
        )

        # Assertions
        assert result is False
        assert mock_post.call_count == 2

    @patch("function_app.requests.post")
    def test_post_to_postly_with_specific_platforms(
        self, mock_post, sample_image_data, sample_caption
    ):
        """Test posting with specific target platforms"""
        # Mock successful responses
        mock_upload_response = Mock()
        mock_upload_response.status_code = 200
        mock_upload_response.json.return_value = {
            "data": {"url": "https://storage.postly.ai/test-image.jpg"}
        }
        mock_upload_response.raise_for_status = Mock()

        mock_create_response = Mock()
        mock_create_response.status_code = 200
        mock_create_response.json.return_value = {
            "data": {"id": "test-post-id", "status": "published"}
        }
        mock_create_response.raise_for_status = Mock()

        mock_post.side_effect = [mock_upload_response, mock_create_response]

        # Call with specific platforms
        specific_platforms = "platform1,platform2"
        result = post_to_postly(
            api_key=self.TEST_API_KEY,
            workspace_id=self.TEST_WORKSPACE_ID,
            image_data=sample_image_data,
            caption=sample_caption,
            target_platforms=specific_platforms,
        )

        # Assertions
        assert result is True
        post_call = mock_post.call_args_list[1]
        post_data = post_call[1]["json"]
        assert post_data["target_platforms"] == specific_platforms


@pytest.mark.integration
class TestPostlyAPIIntegration:
    """Integration tests that actually call the Postly API"""

    @pytest.fixture
    def sample_image_data(self):
        """Create sample image data for testing"""
        return b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 100

    @pytest.fixture
    def sample_caption(self):
        """Sample caption for testing"""
        return "Daily Milo! 😾 [TEST POST - Please ignore] Testing the Postly API integration #Milo #Cats #GrumpyCat"

    @pytest.fixture
    def future_schedule(self):
        """Create a schedule far in the future to prevent immediate posting"""
        from datetime import datetime, timedelta
        future_date = datetime.now() + timedelta(days=365)
        return {
            "one_off_date": future_date.strftime("%Y-%m-%d"),
            "time": future_date.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "timezone": "UTC"
        }

    @pytest.mark.integration
    def test_post_to_postly_real_api(self, sample_image_data, sample_caption, future_schedule, env_with_settings):
        """
        Integration test that actually calls Postly API.
        IMPORTANT: Schedules post 1 year in the future (safe - won't publish).
        Note: Cleanup via deletion is not working due to API limitations with scheduled posts.
        The post will remain scheduled but won't publish for a year.
        Skip by default - run with: pytest -m integration
        """
        config = get_test_config()
        
        # Schedule the post and get the post ID
        success, post_id = post_to_postly(
            api_key=config["POSTLY_API_KEY"],
            workspace_id=config["POSTLY_WORKSPACE_ID"],
            image_data=sample_image_data,
            caption=sample_caption,
            target_platforms=config["POSTLY_TARGET_PLATFORMS"],
            schedule=future_schedule,
            return_post_id=True,
        )

        assert success is True, "Failed to schedule post on Postly API"
        assert post_id is not None, "Post ID was not returned"
        
        # Note: Deletion of scheduled posts via key_id is not supported by the API
        # The post will remain scheduled for 1 year in the future (safe, won't publish)
        # If manual cleanup is needed, scheduled posts can be viewed/deleted in Postly dashboard


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
