"""
Tests for Bluesky image optimization functionality
"""
import sys
import os
import io
from unittest.mock import Mock, MagicMock, patch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from function_app import create_bluesky_optimized_image, MAX_BLUESKY_SIZE_BYTES


def create_test_image(width, height, format='JPEG', color='red'):
    """Helper function to create a test image"""
    img = Image.new('RGB', (width, height), color=color)
    buffer = io.BytesIO()
    img.save(buffer, format=format, quality=95)
    buffer.seek(0)
    return buffer.getvalue()


def test_optimize_large_image_for_bluesky():
    """Test that large images are optimized to fit Bluesky's size limit"""
    # Create a large high-quality image (should be > 900KB)
    large_image_data = create_test_image(3000, 2000)
    original_size = len(large_image_data)
    
    # Optimize for Bluesky
    optimized_data = create_bluesky_optimized_image(large_image_data)
    optimized_size = len(optimized_data)
    
    # Should be smaller than original
    assert optimized_size < original_size, f"Optimized size {optimized_size} should be less than original {original_size}"
    
    # Should be under Bluesky's limit
    assert optimized_size <= MAX_BLUESKY_SIZE_BYTES, f"Optimized size {optimized_size} exceeds Bluesky limit {MAX_BLUESKY_SIZE_BYTES}"
    
    # Should still be a valid image
    img = Image.open(io.BytesIO(optimized_data))
    assert img.size[0] > 0 and img.size[1] > 0


def test_small_image_stays_small():
    """Test that images already under the limit are handled efficiently"""
    # Create a small image
    small_image_data = create_test_image(800, 600)
    original_size = len(small_image_data)
    
    # Optimize for Bluesky
    optimized_data = create_bluesky_optimized_image(small_image_data)
    optimized_size = len(optimized_data)
    
    # Should be under Bluesky's limit
    assert optimized_size <= MAX_BLUESKY_SIZE_BYTES
    
    # Should be a valid image
    img = Image.open(io.BytesIO(optimized_data))
    assert img.size[0] > 0 and img.size[1] > 0


def test_optimize_maintains_aspect_ratio():
    """Test that optimization maintains the aspect ratio when resizing"""
    # Create a wide image (16:9 aspect ratio)
    wide_image_data = create_test_image(4000, 2250)
    
    # Optimize for Bluesky
    optimized_data = create_bluesky_optimized_image(wide_image_data)
    
    # Check the result
    img = Image.open(io.BytesIO(optimized_data))
    width, height = img.size
    
    # Calculate aspect ratios
    original_ratio = 4000 / 2250
    optimized_ratio = width / height
    
    # Should maintain aspect ratio (allow small floating point difference)
    assert abs(original_ratio - optimized_ratio) < 0.02, \
        f"Aspect ratio changed from {original_ratio:.3f} to {optimized_ratio:.3f}"


def test_optimize_handles_rgba_images():
    """Test that RGBA images (with transparency) are converted properly"""
    # Create an RGBA image
    img = Image.new('RGBA', (2000, 1500), color=(255, 0, 0, 128))
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    rgba_data = buffer.getvalue()
    
    # Optimize for Bluesky
    optimized_data = create_bluesky_optimized_image(rgba_data)
    
    # Should be under limit
    assert len(optimized_data) <= MAX_BLUESKY_SIZE_BYTES
    
    # Get the optimized image and verify it's RGB (JPEG doesn't support transparency)
    img = Image.open(io.BytesIO(optimized_data))
    assert img.mode == 'RGB', f"Expected RGB mode, got {img.mode}"


def test_optimize_very_large_image():
    """Test optimization of a very large image that needs aggressive compression"""
    # Create a very large image (should require dimension reduction)
    very_large_image = create_test_image(5000, 4000)
    
    # Optimize for Bluesky
    optimized_data = create_bluesky_optimized_image(very_large_image)
    optimized_size = len(optimized_data)
    
    # Must be under Bluesky's limit
    assert optimized_size <= MAX_BLUESKY_SIZE_BYTES, \
        f"Failed to optimize very large image: {optimized_size / 1024:.1f}KB > {MAX_BLUESKY_SIZE_BYTES / 1024:.1f}KB"
    
    # Should be a valid image
    img = Image.open(io.BytesIO(optimized_data))
    assert img.format == 'JPEG'


def test_optimize_with_custom_limit():
    """Test optimization with a custom size limit"""
    # Create an image
    image_data = create_test_image(2000, 1500)
    
    # Optimize with a smaller limit (500KB)
    custom_limit = 500 * 1024
    optimized_data = create_bluesky_optimized_image(image_data, max_size_bytes=custom_limit)
    
    # Should be under the custom limit
    assert len(optimized_data) <= custom_limit, \
        f"Optimized size {len(optimized_data)} exceeds custom limit {custom_limit}"


def test_optimize_error_handling():
    """Test that errors are handled gracefully"""
    # Pass invalid data
    invalid_data = b"not an image"
    
    # Should not raise exception, should return original data
    result = create_bluesky_optimized_image(invalid_data)
    
    # Should return the same data as fallback
    assert result == invalid_data


if __name__ == "__main__":
    print("Running test_optimize_large_image_for_bluesky...")
    test_optimize_large_image_for_bluesky()
    print("✓ test_optimize_large_image_for_bluesky passed")
    
    print("\nRunning test_small_image_stays_small...")
    test_small_image_stays_small()
    print("✓ test_small_image_stays_small passed")
    
    print("\nRunning test_optimize_maintains_aspect_ratio...")
    test_optimize_maintains_aspect_ratio()
    print("✓ test_optimize_maintains_aspect_ratio passed")
    
    print("\nRunning test_optimize_handles_rgba_images...")
    test_optimize_handles_rgba_images()
    print("✓ test_optimize_handles_rgba_images passed")
    
    print("\nRunning test_optimize_very_large_image...")
    test_optimize_very_large_image()
    print("✓ test_optimize_very_large_image passed")
    
    print("\nRunning test_optimize_with_custom_limit...")
    test_optimize_with_custom_limit()
    print("✓ test_optimize_with_custom_limit passed")
    
    print("\nRunning test_optimize_error_handling...")
    test_optimize_error_handling()
    print("✓ test_optimize_error_handling passed")
    
    print("\nAll tests passed!")
