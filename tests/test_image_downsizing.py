"""
Tests for image downsizing functionality
"""

import sys
import os
import io
from unittest.mock import Mock, MagicMock
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from function_app import (
    downsize_image_if_needed,
    MAX_IMAGE_SIZE_BYTES,
    MAX_IMAGE_DIMENSION,
)


def create_test_image(width, height, format="JPEG"):
    """Helper function to create a test image"""
    img = Image.new("RGB", (width, height), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    buffer.seek(0)
    return buffer.getvalue()


def test_downsize_not_needed_for_small_image():
    """Test that small images are not downsized"""
    # Create a small image (should be well under 4MB)
    image_data = create_test_image(800, 600)

    # Create mock blob client
    mock_blob_client = MagicMock()
    mock_properties = Mock()
    mock_properties.size = len(image_data)  # Small size
    mock_blob_client.get_blob_properties.return_value = mock_properties

    # Call downsize function
    result = downsize_image_if_needed(
        mock_blob_client, max_size_bytes=MAX_IMAGE_SIZE_BYTES
    )

    # Should return False (no downsizing needed)
    assert result is False
    # Should not have uploaded
    assert not mock_blob_client.upload_blob.called


def test_downsize_needed_for_large_image():
    """Test that large images are downsized"""
    # Create a large image
    large_image_data = create_test_image(5000, 4000)

    # Create mock blob client
    mock_blob_client = MagicMock()
    mock_blob_client.blob_name = "test_large.jpg"

    # Mock properties to return large size
    mock_properties = Mock()
    mock_properties.size = 10 * 1024 * 1024  # 10 MB (over the limit)
    mock_blob_client.get_blob_properties.return_value = mock_properties

    # Mock download to return the large image
    mock_download = Mock()
    mock_download.readall.return_value = large_image_data
    mock_blob_client.download_blob.return_value = mock_download

    # Call downsize function
    result = downsize_image_if_needed(
        mock_blob_client, max_size_bytes=MAX_IMAGE_SIZE_BYTES
    )

    # Should return True (downsizing happened)
    assert result is True
    # Should have uploaded the downsized image
    assert mock_blob_client.upload_blob.called

    # Check that uploaded data is smaller than the limit
    upload_call = mock_blob_client.upload_blob.call_args
    uploaded_data = upload_call[0][0]
    uploaded_data.seek(0, 2)  # Seek to end
    uploaded_size = uploaded_data.tell()
    assert uploaded_size <= MAX_IMAGE_SIZE_BYTES


def test_downsize_maintains_aspect_ratio():
    """Test that downsizing maintains the aspect ratio"""
    # Create a very large image
    large_image_data = create_test_image(8000, 6000)  # 4:3 aspect ratio

    # Create mock blob client
    mock_blob_client = MagicMock()
    mock_blob_client.blob_name = "test_aspect.jpg"

    mock_properties = Mock()
    mock_properties.size = 10 * 1024 * 1024  # Over limit
    mock_blob_client.get_blob_properties.return_value = mock_properties

    mock_download = Mock()
    mock_download.readall.return_value = large_image_data
    mock_blob_client.download_blob.return_value = mock_download

    # Call downsize function
    result = downsize_image_if_needed(
        mock_blob_client, max_size_bytes=MAX_IMAGE_SIZE_BYTES
    )

    assert result is True
    assert mock_blob_client.upload_blob.called

    # Get the uploaded image and check dimensions
    upload_call = mock_blob_client.upload_blob.call_args
    uploaded_data = upload_call[0][0]
    uploaded_data.seek(0)

    # Open the uploaded image and check aspect ratio
    img = Image.open(uploaded_data)
    width, height = img.size

    # Should be at or under MAX_IMAGE_DIMENSION
    assert width <= MAX_IMAGE_DIMENSION
    assert height <= MAX_IMAGE_DIMENSION

    # Aspect ratio should be maintained (4:3 = 1.333...)
    aspect_ratio = width / height
    expected_ratio = 8000 / 6000
    assert (
        abs(aspect_ratio - expected_ratio) < 0.01
    )  # Allow small floating point difference


def test_downsize_handles_rgba_images():
    """Test that RGBA images (with transparency) are converted properly"""
    # Create an RGBA image
    img = Image.new("RGBA", (5000, 5000), color=(255, 0, 0, 128))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    rgba_data = buffer.getvalue()

    # Create mock blob client
    mock_blob_client = MagicMock()
    mock_blob_client.blob_name = "test_rgba.png"

    mock_properties = Mock()
    mock_properties.size = 10 * 1024 * 1024
    mock_blob_client.get_blob_properties.return_value = mock_properties

    mock_download = Mock()
    mock_download.readall.return_value = rgba_data
    mock_blob_client.download_blob.return_value = mock_download

    # Call downsize function
    result = downsize_image_if_needed(
        mock_blob_client, max_size_bytes=MAX_IMAGE_SIZE_BYTES
    )

    assert result is True
    assert mock_blob_client.upload_blob.called

    # Get the uploaded image and verify it's RGB (not RGBA)
    upload_call = mock_blob_client.upload_blob.call_args
    uploaded_data = upload_call[0][0]
    uploaded_data.seek(0)

    img = Image.open(uploaded_data)
    assert img.mode == "RGB"  # Should be converted to RGB


def test_downsize_with_custom_limits():
    """Test downsizing with custom size and dimension limits"""
    # Create a moderately sized image
    image_data = create_test_image(2000, 1500)

    # Create mock blob client
    mock_blob_client = MagicMock()
    mock_blob_client.blob_name = "test_custom.jpg"

    # Set size to be over custom limit (1MB)
    mock_properties = Mock()
    mock_properties.size = 2 * 1024 * 1024  # 2 MB
    mock_blob_client.get_blob_properties.return_value = mock_properties

    mock_download = Mock()
    mock_download.readall.return_value = image_data
    mock_blob_client.download_blob.return_value = mock_download

    # Call with custom limits
    custom_max_size = 1 * 1024 * 1024  # 1 MB
    custom_max_dim = 1024
    result = downsize_image_if_needed(
        mock_blob_client, max_size_bytes=custom_max_size, max_dimension=custom_max_dim
    )

    assert result is True
    assert mock_blob_client.upload_blob.called

    # Verify uploaded image respects custom limits
    upload_call = mock_blob_client.upload_blob.call_args
    uploaded_data = upload_call[0][0]
    uploaded_data.seek(0, 2)
    uploaded_size = uploaded_data.tell()
    assert uploaded_size <= custom_max_size

    uploaded_data.seek(0)
    img = Image.open(uploaded_data)
    assert max(img.size) <= custom_max_dim


def test_downsize_error_handling():
    """Test that errors in downsizing are handled gracefully"""
    # Create mock blob client that will fail
    mock_blob_client = MagicMock()
    mock_blob_client.blob_name = "test_error.jpg"
    mock_blob_client.get_blob_properties.side_effect = Exception("Network error")

    # Should not raise exception, should return False
    result = downsize_image_if_needed(mock_blob_client)

    assert result is False
    assert not mock_blob_client.upload_blob.called


if __name__ == "__main__":
    print("Running test_downsize_not_needed_for_small_image...")
    test_downsize_not_needed_for_small_image()
    print("✓ test_downsize_not_needed_for_small_image passed")

    print("\nRunning test_downsize_needed_for_large_image...")
    test_downsize_needed_for_large_image()
    print("✓ test_downsize_needed_for_large_image passed")

    print("\nRunning test_downsize_maintains_aspect_ratio...")
    test_downsize_maintains_aspect_ratio()
    print("✓ test_downsize_maintains_aspect_ratio passed")

    print("\nRunning test_downsize_handles_rgba_images...")
    test_downsize_handles_rgba_images()
    print("✓ test_downsize_handles_rgba_images passed")

    print("\nRunning test_downsize_with_custom_limits...")
    test_downsize_with_custom_limits()
    print("✓ test_downsize_with_custom_limits passed")

    print("\nRunning test_downsize_error_handling...")
    test_downsize_error_handling()
    print("✓ test_downsize_error_handling passed")

    print("\nAll tests passed!")
